"""Does the flash-linear-attention kernel train the same Qwen3.5-style model as our chunk_gated_delta, and how much
faster is it? For each preset: one model, one batch of random tokens, forward and backward under bf16 autocast with
each kernel on the same weights; compares the loss and every gradient (cosine and relative size), then times forward
plus backward (median of 5 after 2 warm-up). Writes results/expanse/fla_check.json. One H100, a few minutes:

    sbatch experiments/expanse/fla_check.sb

The kernels round differently in bf16, so the losses should agree to about 1e-3 and each gradient's cosine with the
other kernel's should be above about 0.99.
"""
import argparse
import json
import statistics
import time

import torch

from exealm.config import get_config
from exealm.model import Llama, count_params
from exealm.model import delta


def step(model, idx, tgt):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(idx, targets=tgt)
    loss.backward()
    return loss.item()


def grads(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}


def seconds(model, idx, tgt, reps=5, warm=2):
    for _ in range(warm):
        step(model, idx, tgt)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        step(model, idx, tgt)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def check(preset, batch, seq, vocab):
    over = json.load(open(f"experiments/expanse/qwen_{preset}.json"))
    model = Llama(get_config(preset, vocab_size=vocab, seed=0, **over)).cuda()
    g = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(0, vocab, (batch, seq), device="cuda", generator=g)
    tgt = torch.randint(0, vocab, (batch, seq), device="cuda", generator=g)
    row = dict(preset=preset, params_m=round(count_params(model) / 1e6, 1), batch=batch, seq=seq)
    delta.USE_FLA = False
    loss_torch = step(model, idx, tgt)
    g_torch = grads(model)
    delta.USE_FLA = True
    loss_fla = step(model, idx, tgt)
    g_fla = grads(model)
    cos, rel = [], []
    for n, a in g_torch.items():
        b = g_fla[n]
        cos.append((torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm() + 1e-30)).item())
        rel.append(((a - b).norm() / (a.norm() + 1e-30)).item())
    row.update(loss_torch=round(loss_torch, 5), loss_fla=round(loss_fla, 5),
               loss_diff=round(abs(loss_torch - loss_fla), 6),
               grad_cos_min=round(min(cos), 5), grad_cos_median=round(statistics.median(cos), 6),
               grad_rel_diff_max=round(max(rel), 5), grad_rel_diff_median=round(statistics.median(rel), 6))
    delta.USE_FLA = False
    row["seconds_torch"] = round(seconds(model, idx, tgt), 4)
    delta.USE_FLA = True
    row["seconds_fla"] = round(seconds(model, idx, tgt), 4)
    row["speedup"] = round(row["seconds_torch"] / row["seconds_fla"], 2)
    row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vocab", type=int, default=32768, help="the study's tokenizer")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default="results/expanse/fla_check.json")
    a = ap.parse_args()
    delta.fla_kernel()
    rows = [check("100m", 8, a.seq, a.vocab), check("500m", 8, a.seq, a.vocab), check("3b", 4, a.seq, a.vocab)]
    rec = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows,
               note="one batch of random tokens per preset, bf16 autocast, forward and backward; the windows per GPU "
                    "the study trains with at 3B (4)")
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
