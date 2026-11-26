"""Does the flash-linear-attention kernel train the same Qwen3.5-style model as our chunk_gated_delta, and how much
faster is it? For each preset: one model, one batch of random tokens, forward and backward on the same weights four
ways: our kernel in fp32 without autocast (the reference), our kernel and the fla kernel under bf16 autocast (as the
study trains), and the fla kernel in fp32. Each is compared with the reference: the loss, and every gradient's cosine
and relative size. Then forward plus backward is timed under bf16 (median of 5 after 2 warm-up). Writes
results/expanse/fla_check.json. One H100, a few minutes:

    sbatch -A fli100 experiments/expanse/fla_check.sb

The fla kernel in fp32 should match the reference to rounding (same recurrence). Under bf16 the fla kernel should be
about as far from the reference as our own kernel is.
"""
import argparse
import json
import statistics
import time

import torch

from exealm.config import get_config
from exealm.model import Llama, count_params
from exealm.model import delta


def step(model, idx, tgt, amp):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        _, loss = model(idx, targets=tgt)
    loss.backward()
    return loss.item()


def grads(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}


def seconds(model, idx, tgt, reps=5, warm=2):
    for _ in range(warm):
        step(model, idx, tgt, True)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        step(model, idx, tgt, True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def against(ref_loss, ref, loss, g):
    cos, rel = [], []
    for n, a in ref.items():
        b = g[n]
        cos.append((torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm() + 1e-30)).item())
        rel.append(((a - b).norm() / (a.norm() + 1e-30)).item())
    return dict(loss=round(loss, 5), loss_diff=round(abs(loss - ref_loss), 6), grad_cos_min=round(min(cos), 5),
                grad_cos_median=round(statistics.median(cos), 6), grad_rel_diff_median=round(statistics.median(rel), 6),
                grad_rel_diff_max=round(max(rel), 5))


def check(preset, compare_batch, time_batch, seq, vocab):
    over = json.load(open(f"experiments/expanse/qwen_{preset}.json"))
    model = Llama(get_config(preset, vocab_size=vocab, seed=0, **over)).cuda()
    gen = torch.Generator(device="cuda").manual_seed(0)
    row = dict(preset=preset, params_m=round(count_params(model) / 1e6, 1), seq=seq)
    if compare_batch:
        idx = torch.randint(0, vocab, (compare_batch, seq), device="cuda", generator=gen)
        tgt = torch.randint(0, vocab, (compare_batch, seq), device="cuda", generator=gen)
        delta.USE_FLA = False
        ref_loss = step(model, idx, tgt, False)
        ref = grads(model)
        row.update(compare_batch=compare_batch, reference_loss=round(ref_loss, 5))
        for name, fla, amp in (("ours_bf16", False, True), ("fla_bf16", True, True), ("fla_fp32", True, False)):
            delta.USE_FLA = fla
            loss = step(model, idx, tgt, amp)
            row[name] = against(ref_loss, ref, loss, grads(model))
        del ref
    idx = torch.randint(0, vocab, (time_batch, seq), device="cuda", generator=gen)
    tgt = torch.randint(0, vocab, (time_batch, seq), device="cuda", generator=gen)
    row["time_batch"] = time_batch
    delta.USE_FLA = False
    row["seconds_ours"] = round(seconds(model, idx, tgt), 4)
    delta.USE_FLA = True
    row["seconds_fla"] = round(seconds(model, idx, tgt), 4)
    row["speedup"] = round(row["seconds_ours"] / row["seconds_fla"], 2)
    row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
    model.zero_grad(set_to_none=True)
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
    rows = [check("100m", 8, 8, a.seq, a.vocab), check("500m", 2, 8, a.seq, a.vocab), check("3b", 0, 2, a.seq, a.vocab)]
    rec = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows,
               note="one batch of random tokens per preset; compare_batch windows for the comparison with the fp32 "
                    "reference, time_batch windows for the timing (3B: 2, half of what each GPU trains with; it does "
                    "not fit 4 without the optimizer split)")
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
