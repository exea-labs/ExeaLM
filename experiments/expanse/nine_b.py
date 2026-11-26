"""The 9B model's training speed and memory, sharded with PyTorch's FSDP2 on 1, 2 and 4 Expanse nodes: the price of the
9B run in the funding application (plan.py, NINE_B; docs/budget.md, "The 9B model").

With AdamW, a model needs 16 bytes per parameter for its weights, gradients and Adam's two statistics in 32-bit: 146 GB
at 9B, more than one 80 GB GPU holds, so DDP (all pretrain.py has) cannot train it. Under FSDP2
(torch.distributed.fsdp.fully_shard) every GPU keeps only its share of all four, gathers a layer's full weights in bf16
just before the layer runs and drops them after it, and the gradients are averaged in 32-bit. Each layer (its attention
or TONN layer and its MLP) is one unit, then the root: the embedding, tied to the output head, and the last norm.

    torchrun --nproc_per_node 4 experiments/expanse/nine_b.py --batch 4                 # one node, 4 rows per GPU
    sbatch --nodes=2 experiments/expanse/nine_b.sb                                       # what the study runs

The model is ExeaLM as the study trains it since 26 November (docs/decisions.md): the 9b preset with
exealm_mid_9b.json (no memory slots, one TONN layer at the middle layer, 21 of 42, value residual and XSA) and the 32k
tokenizer: 9.28B parameters. --checkpoint 1 (the default) recomputes
each layer's activations in the backward pass instead of keeping them. AdamW takes its fused step when this torch can
take it on sharded parameters (tried once at the start), else the default one.

A step is what throughput.py times: forward, backward, gradient clip and AdamW, on random tokens, --batch rows of --seq
tokens on every GPU. After --warmup untimed steps, process 0 adds one row to --out (a JSON list, as throughput.py
writes): nodes, GPUs, parameters, rows per GPU, the median seconds per step (of the slowest process), tokens per second
in all and per GPU, effective TFLOP/s per GPU (6 x parameters x tokens per second / GPUs, so the recomputation
checkpointing adds is not counted), the peak memory of the fullest GPU, and whether checkpointing was on. The same batch
is read every step, so loss_end below loss_start shows the sharded optimizer trains. --batch is taken as given: if a
process runs out of memory, it prints an error row (process 0 also writes it to --out) and stops at once, and torchrun
stops the others. The others are waiting for it inside a collective call, so nothing is retried.

--hsdp 1 shards within each node and keeps one copy per node (HSDP): only the gradients cross between nodes, once a
step, instead of every layer's weights twice and its gradients once. It needs the 146 GB to fit one node's GPUs.

Three things the model needs under FSDP, done by prepare(); none changes what it computes (tests/test_nine_b.py):
  - FSDP rebuilds the dicts among a module's arguments (to cast them to bf16 and to hook the backward pass), so each
    attention layer would get a copy of the dict they share, the first layer's values would never reach the others,
    and value residual would be silently off. Each layer is called with its input alone and gets the rest beside it.
  - Every TONN layer builds retrieval and board circuits whether a slot uses them or not, and their scalar parameters
    are refused by fully_shard (torch 2.6). This model has no memory slots, so they go.
  - A TONN layer replaces its running output sizes (a buffer) in its forward pass. Recomputed under checkpointing, it
    would update them twice and run from different sizes than the forward pass did, giving a different gradient. The
    recomputation starts from the sizes the forward pass saw and leaves the ones it left.

One process (no torchrun) trains the model unsharded, with 32-bit weights under bf16 autocast as throughput.py does: for
the CPU test (FSDP2 does not train on the CPU in torch 2.6) and small presets. FSDP2 itself has not run on a GPU yet.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as recompute

try:
    import fcntl
except ImportError:                                                     # Windows: rows are added without a lock
    fcntl = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..")]
from exealm import ExeaLM, MemoryContext, count_params, get_config  # noqa: E402
from exealm.train.pretrain import dist_setup, make_optimizer, pick_device  # noqa: E402

CORE = os.path.join(HERE, "exealm_core.json")
MEMORY = ("RG", "AG", "WA")                                             # the circuits with outside memory


def build(preset="9b", vocab=32768, **extra):
    """ExeaLM as the study trains it: exealm_core.json's fields with one TONN layer at the middle layer instead of one
    every fourth layer, as pretrain's --config applies exealm_mid_<preset>.json. extra: more fields (the test uses more
    layers of the tiny preset)."""
    core = {k: tuple(v) if isinstance(v, list) else v for k, v in json.load(open(CORE)).items() if k != "tonn_every"}
    n = get_config(preset, **extra).n_layers
    core.update(tonn_every=0, attn_layers=tuple(i for i in range(n) if i != n // 2))
    return ExeaLM(get_config(preset, **dict(core, vocab_size=vocab, **extra)))


class Unit(nn.Module):
    """One layer's sequence mixer or MLP, called with its input alone: what FSDP sees and shards, together with the
    other half of its layer, and what checkpointing recomputes. Its other arguments are in .rest (set by Wrap)."""

    def __init__(self, inner, checkpoint):
        super().__init__()
        self.inner, self.checkpoint, self.rest = inner, checkpoint, ()

    def forward(self, x):
        inner, rest = self.inner, self.rest
        if not self.checkpoint:
            return inner(x, *rest)
        sizes = getattr(inner, "scale", None)              # a TONN layer's running output sizes, replaced each pass
        seen = None if sizes is None else sizes.clone()
        calls = []

        def run(y):
            calls.append(1)
            if sizes is None or len(calls) == 1:           # the forward pass
                return inner(y, *rest)
            left = inner.scale                             # the recomputation in the backward pass: from the sizes
            inner.scale = seen.clone()                     # the forward pass saw, then back to those it left
            try:
                return inner(y, *rest)
            finally:
                inner.scale = left

        return recompute(run, x, use_reentrant=False)


class Wrap(nn.Module):
    """What the model calls in place of a layer's sequence mixer or MLP: passes the input to the Unit and hands it the
    other arguments beside the call, so FSDP never copies the dict the attention layers share (value residual) or a
    MemoryContext's stream state."""

    def __init__(self, inner, checkpoint):
        super().__init__()
        self.unit = Unit(inner, checkpoint)

    def forward(self, x, *rest):
        self.unit.rest = rest
        try:
            return self.unit(x)
        finally:
            self.unit.rest = ()


def prepare(model, checkpoint):
    """What the model needs before FSDP shards it (the module docstring says why). Returns each layer's two units,
    (sequence mixer, MLP), for shard()."""
    cfg = model.cfg
    if model.mlps is None:
        raise ValueError("nine_b.py shards each layer's mixer with its MLP: a config without MLPs is not handled")
    for lay in model.layers:
        used = {s["tmpl"] for s in lay.specs} & set(MEMORY)
        if used:
            raise ValueError(f"layer {lay.index} has {sorted(used)} slots: nine_b.py measures the memory-free model")
        lay.rg = lay.ag = lay.wa = None                    # built by every TONN layer, called by none of these slots
    layers, ti, ai = [], 0, 0
    for i in range(cfg.n_layers):
        if i in cfg.attn_layers:
            model.attn[ai] = mixer = Wrap(model.attn[ai], checkpoint)
            ai += 1
        else:
            model.layers[ti] = mixer = Wrap(model.layers[ti], checkpoint)
            ti += 1
        model.mlps[i] = mlp = Wrap(model.mlps[i], checkpoint)
        layers.append((mixer.unit, mlp.unit))
    return layers


def make_mesh(dev, world, per_node, hsdp):
    """The GPUs to shard over: all of them (FSDP), or each node's, with one copy per node (HSDP)."""
    from torch.distributed.device_mesh import init_device_mesh
    if hsdp:
        return init_device_mesh(dev.type, (world // per_node, per_node), mesh_dim_names=("replicate", "shard"))
    return init_device_mesh(dev.type, (world,), mesh_dim_names=("shard",))


def shard(model, layers, mesh, param_dtype=torch.bfloat16, reshard_after_forward=True):
    """FSDP2: each layer's two units as one, then the root. Weights are gathered in param_dtype and gradients averaged
    in 32-bit. The last layer keeps its weights after the forward pass, since the backward pass starts with it."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    for i, units in enumerate(layers):
        fully_shard(list(units), mesh=mesh, mp_policy=mp,
                    reshard_after_forward=reshard_after_forward and i < len(layers) - 1)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    return model


def fused_adamw_works(device_type, world):
    """Does AdamW's fused step run on sharded parameters (DTensors) in this torch, and give the default step's numbers?
    Tried on a small sharded tensor, the same way on every process."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    mesh = init_device_mesh(device_type, (world,))
    start = torch.arange(4.0 * world, device=device_type).view(2 * world, 2)
    grad = torch.full_like(start, 0.5)
    want = start.clone().requires_grad_()
    want.grad = grad.clone()
    torch.optim.AdamW([want], lr=0.1).step()
    p = nn.Parameter(distribute_tensor(start, mesh, [Shard(0)]))
    p.grad = distribute_tensor(grad, mesh, [Shard(0)])
    try:
        torch.optim.AdamW([p], lr=0.1, fused=True).step()
    except Exception:                                      # whatever the reason, the default step is the answer
        return False
    return bool(torch.allclose(p.full_tensor(), want.detach()))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def measure(model, opt, batch, seq, warmup, steps, dev, seed=0, where=None):
    """Seconds of each timed step, and the loss of the first and last step. One batch of random tokens, read every step;
    the forward call and the loss are throughput.py's. where["stage"] says which step is running."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, model.cfg.vocab_size, (batch, seq + 1), generator=g).to(dev)
    times, losses = [], []
    for i in range(warmup + steps):
        if where is not None:
            where["stage"] = f"step {i + 1}"
        sync(dev)
        t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            loss = model(x[:, :-1], MemoryContext(), x[:, 1:])[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync(dev)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
        losses.append(float(loss))
    return times, (losses[0], losses[-1])


def rates(params, batch, seq, gpus, s):
    """The row's speed: s seconds per step with batch rows of seq tokens on each of gpus GPUs."""
    tokens = batch * seq * gpus / s
    return dict(params_M=round(params / 1e6, 1), s_per_step=round(s, 4), tokens_per_s=round(tokens),
                tokens_per_s_per_gpu=round(tokens / gpus),
                effective_TFLOPs_per_gpu=round(6 * params * tokens / gpus / 1e12, 1))


def append(path, rows):
    """Add rows to the JSON list at path. The jobs on 1, 2 and 4 nodes may finish at the same time, so the file is
    locked while it is read and rewritten, where the file system has locks."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as f:
        if fcntl is not None:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
            except OSError:
                pass
        f.seek(0)
        text = f.read()
        old = json.loads(text) if text.strip() else []
        f.seek(0)
        f.truncate()
        json.dump(old + rows, f, indent=1)
    print("wrote", path, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="The 9B model's training speed and memory under FSDP2")
    ap.add_argument("--preset", default="9b")
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=2, help="rows per GPU in a step, taken as given")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--checkpoint", type=int, default=1,
                    help="1: recompute each layer's activations in the backward pass")
    ap.add_argument("--hsdp", type=int, default=0, help="1: shard within each node, one copy per node")
    ap.add_argument("--warmup", type=int, default=5, help="untimed steps first")
    ap.add_argument("--steps", type=int, default=20, help="timed steps")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None,
                    help="JSON file process 0 adds the row to (e.g. results/expanse/nine_b.json)")
    a = ap.parse_args(argv)
    if a.batch < 1:
        raise SystemExit("--batch is rows per GPU: 1 or more")

    rank, world, dev = dist_setup(pick_device(a.device))
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    base = dict(nodes=max(1, world // per_node), gpus=world, preset=a.preset, batch_per_gpu=a.batch, seq=a.seq,
                checkpointing=bool(a.checkpoint), sharding="none" if world == 1 else "hsdp" if a.hsdp else "fsdp")
    where = dict(stage="building the model")
    try:
        torch.manual_seed(0)                               # the same starting weights on every process
        with torch.device(dev):                            # on the GPU from the start: 36 GB in 32-bit, then sharded
            model = build(a.preset, a.vocab)
        params = count_params(model)
        layers = prepare(model, a.checkpoint)
        fused = dev.type == "cuda"
        if world > 1:
            where["stage"] = "sharding"
            shard(model, layers, make_mesh(dev, world, per_node, a.hsdp))
            fused = fused and fused_adamw_works(dev.type, world)
        opt = make_optimizer(model, 1e-4, 0.1, fused=fused)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)        # the steps' peak, Adam's statistics included
        times, (first, last) = measure(model, opt, a.batch, a.seq, a.warmup, a.steps, dev, seed=rank, where=where)
    except torch.cuda.OutOfMemoryError:
        rec = dict(base, error=f"out of memory on process {rank} at {where['stage']}")
        print(json.dumps(rec), flush=True)
        if rank == 0 and a.out:
            append(a.out, [rec])
        if world > 1:
            os._exit(1)                                    # the others wait in a collective call this one will not join
        raise SystemExit(1)

    s = sorted(times)[len(times) // 2]
    peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0
    held = torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else 0
    if world > 1:                                          # the slowest process and the fullest GPU
        import torch.distributed as dist
        t = torch.tensor([s, peak, held], dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        s, peak, held = t.tolist()
    rec = None
    if rank == 0:
        gb = (lambda b: round(b / 2 ** 30, 1)) if dev.type == "cuda" else (lambda b: None)
        rec = dict(base, **rates(params, a.batch, a.seq, world, s), peak_mem_gb=gb(peak), peak_reserved_gb=gb(held),
                   loss_start=round(first, 4), loss_end=round(last, 4), optimizer="adamw fused" if fused else "adamw",
                   warmup=a.warmup, steps=a.steps, vocab=a.vocab,
                   gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type, torch=torch.__version__)
        print(json.dumps(rec), flush=True)
        if a.out:
            append(a.out, [rec])
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return rec


if __name__ == "__main__":
    main()
