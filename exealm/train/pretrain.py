"""Pretraining: next-token prediction, with TONN's architecture search early on, the retrieval index refreshed as the
model's own encoder learns, and every batch split into swarms of agents that share a board (docs/swarms.md).

Smoke test, no download needed (CPU, about a minute):
    python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --datastore_chunks 2000

The fair comparison every architecture change must win (same data, same steps, same parameter count):
    python -m exealm.train.pretrain --preset tiny --data synthetic --steps 2000 --model tonn --out runs/tonn
    python -m exealm.train.pretrain --preset tiny --data synthetic --steps 2000 --model llama --out runs/llama

On real data, after exealm.data.pack:
    python -m exealm.train.pretrain --preset 100m --data data/packed/train --val data/packed/val --seq 2048 --batch 16

A mix of sources with weights, each source in many packed directories (the form is in exealm/data/loader.py: open_data):
    python -m exealm.train.pretrain --preset 100m --data mix.json --val data/packed/val --datastore_data data/packed/library

Resuming: --save_every N writes the full training state to <out>/ckpt.pt every N steps (the model, the optimizer, the
current architecture, every random number generator, the search's state and the datastore's keys). Pick the run up
with --resume <out>/ckpt.pt, or --resume auto to use <out>/ckpt.pt if it exists. With the same arguments, a resumed run
ends with exactly the weights an uninterrupted run would have (tested on CPU). --stop_at N saves and stops after step N
without changing the learning-rate schedule, for sessions with a time limit such as Colab:
    python -m exealm.train.pretrain --preset small --data data/packed/train --steps 20000 --save_every 500 \
        --stop_at 5000 --resume auto      # run the same line again in the next session to continue

Several processes (DistributedDataParallel): start the same command with torchrun, one process per GPU (NCCL), or
several on a CPU (gloo). --batch stays the total over all processes: every process draws the same batch and keeps its
own rows, whole swarms only, so N processes train on exactly the data one process would. Process 0 runs the search,
evaluates, logs and saves; after every search round the others copy its circuits, weights and optimizer state, and
after every datastore refresh its keys. --resume auto works the same way.
    torchrun --nproc_per_node 2 -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --device cpu
  TODO(infra): FSDP, and the datastore search sharded across GPUs (every process holds the whole datastore today).
  TODO(infra): torch.compile.

--wandb PROJECT also sends every eval, search and resume record to Weights & Biases (pip install wandb). A resumed run
continues the same W&B run: its id is kept in <out>/wandb_id.txt.
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace

import numpy as np
import torch

from ..config import ModelConfig, get_config
from ..data.loader import TokenShards, open_data
from ..memory import Datastore, MemoryContext
from ..model import ExeaLM, Llama, count_params
from ..model.search import Searcher


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dist_setup(dev):
    """Several processes (torchrun sets WORLD_SIZE, RANK and LOCAL_RANK): join the process group, with one GPU per
    process on CUDA (NCCL) or gloo on the CPU. Returns (rank, number of processes, device); a plain run gives
    (0, 1, dev)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, dev
    import torch.distributed as dist
    if dev.type == "cuda":
        dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(dev)
    elif dev.type != "cpu":
        raise ValueError(f"several processes run on CUDA or the CPU, not on {dev.type}")
    if not dist.is_initialized():
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")
    return dist.get_rank(), dist.get_world_size(), dev


def broadcast_(t, dev):
    """Overwrite t with process 0's copy (tensors that live off the device, such as Adam's step, go through it)."""
    import torch.distributed as dist
    if t.device == dev:
        dist.broadcast(t, src=0)
    else:
        buf = t.to(dev)
        dist.broadcast(buf, src=0)
        t.copy_(buf)


@torch.no_grad()
def sync_from_rank0(model, opt, dev):
    """After process 0 ran a search round: every process takes its circuits, then its weights, buffers and optimizer
    state. A replaced slot gets fresh random weights and cleared Adam rows on process 0 only, so copying is what keeps
    all processes training one and the same model."""
    import torch.distributed as dist
    box = [model.specs]
    dist.broadcast_object_list(box, src=0)
    for lay, row in zip(model.layers, box[0]):
        for k, spec in enumerate(row):
            if spec != lay.specs[k]:
                lay.set_spec(k, spec)
    for t in list(model.parameters()) + list(model.buffers()):
        broadcast_(t.data, dev)
    params = list(model.parameters())
    shape = [[(n, tuple(v.shape)) for n, v in opt.state[p].items() if torch.is_tensor(v)] if p in opt.state else []
             for p in params]
    box = [shape]
    dist.broadcast_object_list(box, src=0)
    if box[0] != shape:
        raise RuntimeError("the optimizer state differs between processes; they no longer train the same model")
    for p in params:
        for v in (opt.state[p].values() if p in opt.state else ()):
            if torch.is_tensor(v):
                broadcast_(v, dev)


def lr_at(step, total, peak, warmup, decay_frac, floor=0.1):
    """Warmup-stable-decay (WSD): linear warmup, flat, then a linear decay over the last decay_frac of training.
    WSD lets us branch off the flat part to try a different final data mix without restarting (docs/training.md)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return peak
    return peak * (floor + (1 - floor) * (1 - (step - d0) / max(1, total - d0)))


def board_summary(usage):
    """usage: per layer, a list of exealm.model.board.usage_stats dicts (one per validation batch). Returns each stat
    averaged over everything, plus read_nothing and post_strength per layer (where the board is used differs a lot
    between layers)."""
    flat = [d for per_layer in usage for d in per_layer]
    out = {k: round(sum(d[k] for d in flat) / len(flat), 4) for k in flat[0]}
    for k in ("read_nothing", "post_strength"):
        out[k + "_per_layer"] = [round(sum(d[k] for d in u) / len(u), 4) if u else None for u in usage]
    return out


def wandb_record(rec, prefix=""):
    """A log record's numbers as flat "eval/val_bits_per_token"-style keys: W&B plots numbers, so nested dicts are
    flattened, lists become one key per item, and text and missing values are left out."""
    out = {}
    if not prefix:
        prefix = str(rec.get("event", "log")) + "/"
    for k, v in rec.items():
        if k in ("event", "step") and prefix.count("/") == 1:
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[prefix + k] = v
        elif isinstance(v, dict):
            out.update(wandb_record(v, prefix + k + "/"))
        elif isinstance(v, (list, tuple)):
            out.update(wandb_record({str(i): x for i, x in enumerate(v)}, prefix + k + "/"))
    return out


def make_optimizer(model, lr, wd):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    return torch.optim.AdamW([dict(params=decay, weight_decay=wd), dict(params=rest, weight_decay=0.0)],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)


def save(path, model, opt, step, cfg, kind, state=None):
    """state: what else an exact resume needs (random generators, search, datastore keys, see main()). Written to a
    temporary file first, so a run killed mid-save never leaves a broken checkpoint behind."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, cfg=asdict(cfg), kind=kind,
                    specs=getattr(model, "specs", None), state=state), tmp)
    os.replace(tmp, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    model = ExeaLM(cfg, ck["specs"]) if ck["kind"] == "tonn" else Llama(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--config", default=None, help="JSON file of ModelConfig fields to override")
    ap.add_argument("--model", default="tonn", choices=["tonn", "llama"])
    ap.add_argument("--data", default="synthetic",
                    help="'synthetic', a packed directory, a pattern of packed directories, or a mix .json with "
                         "per-source weights (exealm/data/loader.py: open_data)")
    ap.add_argument("--val", default=None, help="validation split, same forms as --data (default: windows of --data)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--swarm_policy", default="mixed", choices=["mixed", "split", "random"],
                    help="what each swarm reads (docs/swarms.md)")
    ap.add_argument("--search", type=int, default=1)
    ap.add_argument("--search_every", type=int, default=100)
    ap.add_argument("--search_until", type=float, default=0.5)
    ap.add_argument("--datastore_chunks", type=int, default=0, help="0: no retrieval (RG slots output zeros)")
    ap.add_argument("--datastore_data", default=None,
                    help="one packed directory to sample the retrieval library from (default: --data, which must "
                         "then be one packed directory)")
    ap.add_argument("--refresh_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--save_every", type=int, default=0, help="write <out>/ckpt.pt every this many steps (0: never)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from, or 'auto' for <out>/ckpt.pt if it exists")
    ap.add_argument("--stop_at", type=int, default=0,
                    help="save <out>/ckpt.pt and stop after this step, keeping the --steps schedule (for time limits)")
    ap.add_argument("--wandb", default=None, help="Weights & Biases project to log to (default: no W&B logging)")
    ap.add_argument("--wandb_name", default=None, help="name of the W&B run (default: the --out folder's name)")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    rank, world, dev = dist_setup(pick_device(a.device))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else None
    data = open_data(a.data)
    val = data if a.val is None else open_data(a.val)
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    resume = ckpt_path if a.resume == "auto" else a.resume
    if a.resume == "auto" and not os.path.exists(ckpt_path):
        resume = None
    ck = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if ck is not None:
        if ck["kind"] != a.model:
            raise ValueError(f"checkpoint is a {ck['kind']} model, but --model is {a.model}")
        if ck.get("state") is None:
            raise ValueError(f"{resume} has no training state to resume from (saved before --save_every existed)")
        cfg = ModelConfig.from_dict(ck["cfg"])                          # the run's own config and architecture
        model = (ExeaLM(cfg, ck["specs"]) if a.model == "tonn" else Llama(cfg)).to(dev)
        model.load_state_dict(ck["model"])
    else:
        over = json.load(open(a.config)) if a.config else {}
        over["vocab_size"] = int(data.meta["vocab_size"])
        over.setdefault("seed", a.seed)                                # --seed also draws the starting circuits
        if a.model == "llama":                                         # the baseline is a plain Llama unless the
            over.setdefault("attn_gate", False)                        # --config asks for the Qwen3.5 parts
            over.setdefault("qk_norm", False)
        cfg = get_config(a.preset, **over)
        model = (ExeaLM(cfg) if a.model == "tonn" else Llama(cfg)).to(dev)
    opt = make_optimizer(model, a.lr, a.wd)
    if ck is not None:
        opt.load_state_dict(ck["opt"])
    warmup = a.warmup if a.warmup is not None else max(1, min(2000, a.steps // 20))
    swarm = cfg.swarm_size if a.model == "tonn" else 1
    if world > 1 and a.batch % (world * swarm):
        raise ValueError(f"--batch {a.batch} is the total over {world} processes and must split into whole swarms of "
                         f"{swarm}: use a multiple of {world * swarm}")
    per = a.batch // world                                              # rows each process trains on
    rng = np.random.default_rng(a.seed)

    ds = None
    if a.model == "tonn" and a.datastore_chunks:
        if a.datastore_data:
            ds = Datastore.from_shards(TokenShards(a.datastore_data), a.datastore_chunks, cfg.rag_chunk_len,
                                       seed=a.seed + 1)
            ds.doc_ids += int(data.meta["n_docs"])                     # library documents are not training documents
        elif hasattr(data, "tokens"):
            ds = Datastore.from_shards(data, a.datastore_chunks, cfg.rag_chunk_len, seed=a.seed + 1)
        else:
            raise ValueError("--data is a mix or has many shards: name the retrieval library with --datastore_data")
        ds = ds.to(dev)
        ds.refresh(model.encoder, model.emb, step=0)
        if world > 1:
            broadcast_(ds.keys, dev)                                    # every process searches the same keys

    def mem_for(docs):
        return MemoryContext(datastore=ds, doc_ids=docs.to(dev), swarm_size=swarm)

    def fixed_batches(src, n, seed):
        r = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            x, y, d = src.batch(a.batch, a.seq, r, swarm, a.swarm_policy)
            out.append((x.to(dev), y.to(dev), mem_for(d)))
        return out

    val_batches = fixed_batches(val, a.eval_batches, 10_000)
    sel_batches = fixed_batches(data, a.eval_batches, 20_000)            # the search's own held-out windows
    searcher = Searcher(every=a.search_every, until=a.search_until, seed=a.seed) if (a.search and a.model == "tonn") \
        else None

    @torch.no_grad()
    def evaluate():
        """Validation bits. For swarms, also with the board cut (every agent alone): the difference is what the
        agents gained from each other."""
        model.eval()
        bpt = float(data.meta.get("bytes_per_token", 1.0))
        boards = [lay.ag for lay in model.layers] if a.model == "tonn" else []
        usage = [[] for _ in boards]                                     # per layer, per batch: board_usage stats
        loss = 0.0
        for b in boards:                                                 # recorded on the shared-board pass only
            b.record = True
        with torch.autocast(**amp) if amp else torch.no_grad():
            for x, y, m in val_batches:
                loss += float(model(x, m, y)[1]) / len(val_batches)
                for u, b in zip(usage, boards):
                    if b.last is not None:
                        u.append(b.last)
            for b in boards:
                b.record, b.last = False, None
            cut = None
            if swarm > 1:
                cut = sum(float(model(x, replace(m, swarm_size=1), y)[1]) for x, y, m in val_batches) / len(val_batches)
        model.train()
        bits = loss / math.log(2)
        out = dict(val_bits_per_token=round(bits, 4), val_bits_per_byte=round(bits / bpt, 4))
        if cut is not None:
            out["val_bits_per_token_board_cut"] = round(cut / math.log(2), 4)
            out["board_gain_bits_per_token"] = round((cut - loss) / math.log(2), 4)
        if any(usage):
            out["board"] = board_summary(usage)
        return out

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a") if rank == 0 else None     # process 0 logs for all of them

    run = None
    if a.wandb and rank == 0:
        import wandb
        id_file = os.path.join(a.out, "wandb_id.txt")                    # so a resumed run continues the same W&B run
        run_id = open(id_file).read().strip() if resume and os.path.exists(id_file) else None
        run = wandb.init(project=a.wandb, name=a.wandb_name or os.path.basename(os.path.normpath(a.out)), id=run_id,
                         resume="allow", config=dict(vars(a), model_cfg=asdict(cfg), params=count_params(model)))
        with open(id_file, "w") as f:
            f.write(str(run.id))

    def emit(rec):
        if rank:
            return
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if run is not None and "step" in rec:
            run.log(wandb_record(rec), step=rec["step"])

    def train_state(losses, secs):
        """Everything besides the weights and the optimizer that the next step depends on."""
        return dict(numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    searcher=None if searcher is None else dict(round=searcher.round, born=dict(searcher.born),
                                                                rng=searcher.rng.getstate()),
                    datastore=None if ds is None else dict(keys=ds.keys.cpu(), refreshed_at=ds.refreshed_at),
                    losses=[float(l) for l in losses], secs=secs, total_steps=a.steps)

    start, losses, done_secs = 0, [], 0.0
    if ck is not None:
        st = ck["state"]
        start, done_secs = ck["step"], st["secs"]
        rng.bit_generator.state = st["numpy_rng"]
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        if searcher is not None and st["searcher"] is not None:
            searcher.round, searcher.born = st["searcher"]["round"], dict(st["searcher"]["born"])
            searcher.rng.setstate(st["searcher"]["rng"])
        if ds is not None and st["datastore"] is not None:
            ds.keys, ds.refreshed_at = st["datastore"]["keys"].to(dev), st["datastore"]["refreshed_at"]
        losses = [torch.tensor(l) for l in st["losses"]]
        emit(dict(event="resume", step=start, path=resume, total_steps=a.steps,
                  saved_total_steps=st["total_steps"]))
    else:
        emit(dict(event="start", model=a.model, params=count_params(model), device=str(dev), cfg=asdict(cfg),
                  architecture=model.architecture() if a.model == "tonn" else None,
                  **(dict(processes=world) if world > 1 else {})))
    net = model                                                         # what trains; model itself evaluates and saves
    if world > 1:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        net = DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None,
                                      find_unused_parameters=True)     # e.g. RG slots read nothing without a library
    t0 = time.time() - done_secs
    for step in range(start, a.steps):
        if searcher is not None and searcher.due(step, a.steps):
            rec = searcher.maybe_round(model, opt, step, a.steps, sel_batches, a.seq) if rank == 0 else None
            if world > 1:
                sync_from_rank0(model, opt, dev)
            if rec:
                emit(dict(event="search", **rec))
        if ds is not None and step > 0 and step % a.refresh_every == 0:
            ds.refresh(model.encoder, model.emb, step=step)
            if world > 1:
                broadcast_(ds.keys, dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, warmup, a.decay_frac)
        x, y, d = data.batch(a.batch, a.seq, rng, swarm, a.swarm_policy)
        if world > 1:                                                   # this process's rows: whole swarms only
            x, y, d = x[rank * per:(rank + 1) * per], y[rank * per:(rank + 1) * per], d[rank * per:(rank + 1) * per]
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(**amp) if amp else torch.enable_grad():
            loss = net(x, mem_for(d), y)[1]
        mean = loss.detach()
        if world > 1:
            # Each process's loss is a mean over its own targets, and DDP averages gradients over processes. Weighting
            # by each process's share of the targets makes the update the one a single process would take on the
            # whole batch, even when masked seam targets make the shares unequal.
            n = (y != -100).sum().float()
            tot = torch.stack([n, mean.float() * n])
            dist.all_reduce(tot)
            mean = tot[1] / tot[0]                                      # the whole batch's loss, for the log
            loss = loss * (n * world / tot[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        losses.append(mean)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            if rank == 0:
                rec = dict(event="eval", step=step + 1, train_loss=round(float(torch.stack(losses).mean()), 4),
                           lr=round(opt.param_groups[0]["lr"], 6), secs=round(time.time() - t0, 1), **evaluate())
                emit(rec)
            losses = []
        if rank == 0 and step + 1 < a.steps and (
                (a.save_every and (step + 1) % a.save_every == 0) or step + 1 == a.stop_at):
            save(ckpt_path, model, opt, step + 1, cfg, a.model, train_state(losses, time.time() - t0))
        if step + 1 == a.stop_at and step + 1 < a.steps:
            emit(dict(event="stop", step=step + 1, ckpt=ckpt_path))
            if run is not None:
                run.finish()
            if world > 1:
                dist.barrier()                                          # the checkpoint is written before anyone exits
                dist.destroy_process_group()
            return model
    if a.model == "tonn":
        emit(dict(event="end", architecture=model.architecture()))
    if rank == 0:
        save(os.path.join(a.out, "model.pt"), model, opt, a.steps, cfg, a.model,
             train_state(losses, time.time() - t0))
    if run is not None:
        run.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return model


if __name__ == "__main__":
    main()
