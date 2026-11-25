"""Training throughput on one GPU, measured the way results/funding/throughput_mi300x.json was, so the compute plan can
be rescaled to the machine the hours are on.

    python experiments/throughput.py --out results/expanse                     # 0.1B and 0.5B, both models, bf16 and 32-bit
    python experiments/throughput.py --sizes 1B,3B --precision bf16            # the two larger shapes
    python experiments/throughput.py --sizes 0.5B --precision bf16 --compile 1 # what torch.compile buys
    python experiments/throughput.py --sizes 0.5B --precision bf16 --tf32 1    # what TF32 buys in the 32-bit parts

    python experiments/throughput.py --models hybrid,llama,qwen --precision bf16    # the three models the plan trains

A row is one model at one size and precision: seconds per training step (forward, backward, gradient clip, optimizer)
on random tokens, tokens per second, effective TFLOP/s (6 x parameters x tokens per second) and the peak GPU memory.
If a batch does not fit the GPU, it is halved and the row says so. The models:

    hybrid   ExeaLM as the plan trains it (exealm_core.json): no memory circuits, a TONN layer every fourth layer,
             value residual and XSA
    llama    the plain same-size Llama, the plan's reference below 3B (the other MI300X rows)
    qwen     the plan's baseline: the base Qwen3.5-style model, gated attention and QK-norm, Gated DeltaNet in 3 of
             every 4 layers
    exealm   the preset with its board and library slots, swarms of 4, no datastore
    gated    a Llama with gated attention and QK-norm

With bf16 rows the script also prints the compute plan (experiments/expanse/plan.py) at the measured rates. These are
single-GPU rates. For a 4-GPU run, time the real loop (experiments/expanse/README.md).
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "expanse")]
import plan  # noqa: E402  (experiments/expanse/plan.py)
from exealm import ExeaLM, Llama, MemoryContext, count_params, get_config, llama_baseline  # noqa: E402
from exealm.config import MENU  # noqa: E402
from exealm.train.pretrain import make_optimizer, pick_device  # noqa: E402

PRESET = {"0.1B": "100m", "0.25B": "250m", "0.5B": "500m", "1B": "1b", "3B": "3b", "tiny": "tiny"}
BATCH = {"0.1B": 32, "0.25B": 32, "0.5B": 16, "1B": 16, "3B": 8, "tiny": 8}            # 0.1B and 0.5B: as measured on the MI300X
KINDS = ("hybrid", "llama", "exealm", "gated", "qwen")


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def build(size, kind, vocab, dev):
    """The model, and what its forward pass is given: rows alone, or (for "exealm") swarms that share a board."""
    if kind not in KINDS:
        raise SystemExit(f"--models takes any of {', '.join(KINDS)}, not {kind}")
    over = dict(vocab_size=vocab) if size != "tiny" else {}
    if kind == "exealm":
        cfg = get_config(PRESET[size], **over)
        return ExeaLM(cfg).to(dev), MemoryContext(swarm_size=cfg.swarm_size)
    cfg = get_config(PRESET[size], pinned={}, menu=tuple(t for t in MENU if t not in ("RG", "AG")), **over)
    if kind == "hybrid":                         # ExeaLM's layout as the study trains it (exealm_core.json)
        import json, os
        core = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expanse", "exealm_core.json")
        core = json.load(open(core)) if os.path.exists(core) else {}
        keep = {k: core[k] for k in ("tonn_every", "value_residual", "xsa") if k in core}
        if keep:
            cfg = get_config(PRESET[size], pinned={}, menu=cfg.menu, **keep, **over)
    if kind in ("gated", "qwen"):
        cfg = replace(cfg, attn_gate=True, qk_norm=True)
    if kind == "qwen":                                              # Gated DeltaNet in 3 of every 4 layers
        cfg = replace(cfg, delta_layers=tuple(i for i in range(cfg.n_layers) if i % 4 != 3))
    model = ExeaLM(cfg) if kind == "hybrid" else Llama(cfg if kind in ("gated", "qwen") else llama_baseline(cfg))
    return model.to(dev), MemoryContext()


def measure(model, mem, batch, a, dev, bf16):
    """Median seconds per step over a.steps steps, after a.warmup steps that are not timed."""
    opt = make_optimizer(model, 1e-4, 0.1, fused=dev.type == "cuda")
    fwd = torch.compile(model) if a.compile else model
    x = torch.randint(0, model.cfg.vocab_size, (batch, a.seq + 1), device=dev)
    times = []
    for i in range(a.warmup + a.steps):
        sync(dev)
        t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if bf16 else torch.enable_grad():
            loss = fwd(x[:, :-1], mem, x[:, 1:])[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync(dev)
        if i >= a.warmup:
            times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def row(size, kind, precision, a, dev):
    batch = a.batch or BATCH[size]
    rec = dict(size=size, model=kind, precision=precision, seq=a.seq)
    while True:
        model = None
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            torch.manual_seed(0)
            model, mem = build(size, kind, a.vocab, dev)
            s = measure(model, mem, batch, a, dev, precision == "bf16")
            params = count_params(model)
            rec.update(params_M=round(params / 1e6, 1), batch=batch, s_per_step=round(s, 4),
                       tokens_per_s=round(batch * a.seq / s),
                       effective_TFLOPs=round(6 * params * batch * a.seq / s / 1e12, 1))
            if dev.type == "cuda":
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 1)
            return rec
        except torch.cuda.OutOfMemoryError:
            del model
            rec["did_not_fit_batch"] = batch
            batch //= 2
            if batch == 0:
                return dict(rec, error="out of memory at batch 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="0.1B,0.5B", help="any of " + ", ".join(PRESET))
    ap.add_argument("--models", default="hybrid,llama", help="any of " + ", ".join(KINDS))
    ap.add_argument("--precision", default="bf16,fp32")
    ap.add_argument("--batch", type=int, default=0, help="rows per step (default: 32, 32, 16, 16 and 8 for the five sizes)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--steps", type=int, default=20, help="timed steps")
    ap.add_argument("--warmup", type=int, default=5, help="untimed steps first (use 20 or more with --compile 1)")
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--tf32", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="folder for throughput_<tag>.json")
    ap.add_argument("--tag", default=None, help="name of the output file (default: from the GPU's name)")
    a = ap.parse_args()
    dev = pick_device(a.device)
    if a.tf32 and dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)
    rows = []
    for size in a.sizes.split(","):
        for kind in a.models.split(","):
            for precision in a.precision.split(","):
                rows.append(dict(row(size, kind, precision, a, dev), gpu=gpu, torch=torch.__version__,
                                 compile=a.compile, tf32=a.tf32))
                print(json.dumps(rows[-1]), flush=True)
    if any(r["precision"] == "bf16" and r["size"] in plan.SIZE_NAME and r.get("effective_TFLOPs", 0) > 0
           for r in rows) and not (a.compile or a.tf32):
        print("\nThe compute plan at these single-GPU rates:\n")
        plan.table(*plan.measured(rows), plan.defaults())
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = a.tag or "".join(c if c.isalnum() else "_" for c in gpu.lower()).strip("_")
        path = os.path.join(a.out, f"throughput_{tag}.json")
        old = json.load(open(path)) if os.path.exists(path) else []
        json.dump(old + rows, open(path, "w"), indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
