"""Speed benchmark: training and generation tokens per second, TONN against the Llama baseline of the same size.

"tonn" is the preset's ExeaLM: for "small" and "100m" that is the default hybrid (attention layers with TONN layers
among them), for "small-tonn" and "100m-tonn" pure TONN. experiments/speed.py measures training speed only, at full
size (results/README.md, section 5); this script adds generation and runs on any preset, so a faster kernel
(exealm/model/scan.py) or a recurrent generation cache can show what it buys on your own hardware. Both models get the
same preset, batch, sequence length and optimizer; TONN can also be timed with retrieval switched on
(--datastore_chunks) and its swarms (--swarm).

    python -m exealm.eval.speed --preset small --batch 8 --seq 512            # training and generation, both models
    python -m exealm.eval.speed --preset tiny --device cpu --steps 5          # a quick check

Training is timed over forward + backward + optimizer step, after a few warmup steps that are not counted. Generation
is timed with the same greedy loop for both models, which re-reads the whole context for every new token, as
ExeaLM.generate does today; a recurrent state cache should make TONN's number jump.
Results print as one JSON line per measurement, then a table with TONN's slowdown against Llama.
"""
import argparse
import json
import time

import torch

from ..config import get_config, llama_baseline
from ..memory import Datastore, MemoryContext
from ..model import ExeaLM, Llama, count_params
from ..train.pretrain import make_optimizer, pick_device


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def build(kind, cfg, dev, datastore_chunks=0, seed=0):
    """A fresh model and the datastore it reads. TONN with datastore_chunks > 0 retrieves from a random datastore of
    that many chunks, so the timing includes the search and the live re-encoding of the retrieved chunks."""
    torch.manual_seed(seed)
    model = (ExeaLM(cfg) if kind == "tonn" else Llama(llama_baseline(cfg))).to(dev)
    ds = None
    if kind == "tonn" and datastore_chunks:
        g = torch.Generator().manual_seed(seed)
        toks = torch.randint(0, cfg.vocab_size, (datastore_chunks, cfg.rag_chunk_len), generator=g)
        ds = Datastore(toks, torch.arange(datastore_chunks) + 1).to(dev)          # doc 0 is the text being read
        ds.refresh(model.encoder, model.emb)
    return model, ds


def time_training(model, ds, cfg, dev, batch, seq, steps, warmup, swarm, amp):
    opt = make_optimizer(model, 1e-4, 0.1)
    x = torch.randint(0, cfg.vocab_size, (batch, seq + 1), device=dev)
    docs = torch.zeros(batch, seq, dtype=torch.long, device=dev)
    mem = MemoryContext(datastore=ds, doc_ids=docs, swarm_size=swarm)
    model.train()
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            sync(dev)
            t0 = time.perf_counter()
        with torch.autocast(**amp) if amp else torch.enable_grad():
            loss = model(x[:, :-1], mem, x[:, 1:])[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync(dev)
    secs = time.perf_counter() - t0
    return dict(train_tokens_per_sec=round(batch * seq * steps / secs, 1),
                train_ms_per_step=round(1000 * secs / steps, 2))


@torch.no_grad()
def greedy(model, idx, new_tokens, ds, swarm):
    """Greedy decoding that re-reads the whole context per token. The baseline has no generate(), so both models are
    timed with this one loop."""
    for _ in range(new_tokens):
        docs = torch.zeros_like(idx)
        nxt = model(idx, MemoryContext(datastore=ds, doc_ids=docs, swarm_size=swarm))[0][:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], 1)
    return idx


@torch.no_grad()
def time_generation(model, ds, cfg, dev, batch, prompt, new_tokens, swarm):
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (batch, prompt), device=dev)
    greedy(model, idx, 2, ds, swarm)                                    # warmup
    sync(dev)
    t0 = time.perf_counter()
    greedy(model, idx, new_tokens, ds, swarm)
    sync(dev)
    secs = time.perf_counter() - t0
    return dict(gen_tokens_per_sec=round(batch * new_tokens / secs, 1))


def run(preset="tiny", models=("tonn", "llama"), batch=8, seq=128, steps=10, warmup=3, gen_batch=1, gen_prompt=64,
        gen_tokens=32, datastore_chunks=0, swarm=1, device="auto", generation=True, overrides=None):
    """Returns one dict per model with its parameter count and speeds."""
    dev = pick_device(device)
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else None
    cfg = get_config(preset, **(overrides or {}))
    out = []
    for kind in models:
        model, ds = build(kind, cfg, dev, datastore_chunks)
        sw = swarm if kind == "tonn" else 1
        rec = dict(model=kind, preset=preset, params=count_params(model), device=str(dev), batch=batch, seq=seq,
                   retrieval=ds is not None, swarm=sw)
        rec.update(time_training(model, ds, cfg, dev, batch, seq, steps, warmup, sw, amp))
        if generation:
            gsw = sw if gen_batch % sw == 0 else 1
            rec.update(time_generation(model, ds, cfg, dev, gen_batch, gen_prompt, gen_tokens, gsw))
        out.append(rec)
        del model, ds
    return out


def table(recs):
    """A plain-text table, with each model's slowdown against the Llama baseline when there is one."""
    base = next((r for r in recs if r["model"] == "llama"), None)
    cols = ["model", "params", "train_tokens_per_sec", "train_ms_per_step", "gen_tokens_per_sec"]
    lines = [" | ".join(cols + (["slower than llama (train)", "slower than llama (gen)"] if base else []))]
    for r in recs:
        row = [str(r.get(c, "-")) for c in cols]
        if base:
            row.append(f"{base['train_tokens_per_sec'] / r['train_tokens_per_sec']:.2f}x")
            g = r.get("gen_tokens_per_sec")
            row.append(f"{base['gen_tokens_per_sec'] / g:.2f}x" if g and base.get("gen_tokens_per_sec") else "-")
        lines.append(" | ".join(row))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="small")
    ap.add_argument("--models", default="tonn,llama")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--steps", type=int, default=10, help="timed training steps")
    ap.add_argument("--warmup", type=int, default=3, help="untimed training steps first")
    ap.add_argument("--gen_batch", type=int, default=1)
    ap.add_argument("--gen_prompt", type=int, default=64)
    ap.add_argument("--gen_tokens", type=int, default=32)
    ap.add_argument("--no_generation", action="store_true")
    ap.add_argument("--datastore_chunks", type=int, default=0, help="time TONN with retrieval on (0: RG outputs zeros)")
    ap.add_argument("--swarm", type=int, default=1, help="agents per swarm for TONN (batch must be a multiple)")
    ap.add_argument("--vocab_size", type=int, default=None, help="override the preset's vocabulary size")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="also append the JSON lines to this file")
    a = ap.parse_args(argv)
    recs = run(a.preset, tuple(a.models.split(",")), a.batch, a.seq, a.steps, a.warmup, a.gen_batch, a.gen_prompt,
               a.gen_tokens, a.datastore_chunks, a.swarm, a.device, not a.no_generation,
               dict(vocab_size=a.vocab_size) if a.vocab_size else None)
    for r in recs:
        print(json.dumps(r))
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    print()
    print(table(recs))
    return recs


if __name__ == "__main__":
    main()
