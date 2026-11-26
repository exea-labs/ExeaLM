"""How many times ExeaLM's training steps the Qwen3.5-style model needs to reach the same held-out score.

For each in-run evaluation of an ExeaLM run (512 fixed held-out windows, the same for every model of the study), the
step at which the paired Qwen3.5-style run of the same size and seed first reaches that score, interpolated between
its evaluations. Every step is the same number of tokens for both, so this is a token multiplier; in FLOPs it is the
same up to the two models' parameter counts (within 1%). Read from the jobs' logs on Expanse, while runs train:

    python experiments/expanse/step_multiplier.py

Comparisons are fair in the constant-rate part of the schedule; in the last 10% (the anneal) a run's score falls
faster, so a multiplier read against a finished baseline's anneal is an underestimate.
"""
import glob
import json
import sys

import numpy as np


def evals(run):
    out = {}
    for f in sorted(glob.glob(f"logs/expanse/exealm-{run}-*.out")):
        for line in open(f, errors="ignore"):
            if line.startswith("{") and '"event": "eval"' in line:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if "val_bits_per_byte" in e:
                    out[e["step"]] = e["val_bits_per_byte"]
    return dict(sorted(out.items()))


LAST = "--last" in sys.argv                         # only the latest evaluation of each pair (the 15-minute status)
for size, tpp in (("100m", "111x"), ("250m", "4.1x"), ("500m", "4.1x"), ("3b", "4.1x")):
    e, q = evals(f"exealm_{size}_{tpp}_s0"), evals(f"qwen_{size}_{tpp}_s0")
    if not e or not q:
        continue
    qs, qv = np.array(list(q)), np.array(list(q.values()))
    if not LAST:
        print(f"== {size} {tpp}: Qwen evaluated to step {qs.max()}")
    for s, v in (list(e.items())[-1:] if LAST else e.items()):
        if s < 1000:
            continue
        hit = np.where(qv <= v)[0]
        if len(hit) == 0:
            print(f"  {size} {tpp}: ExeaLM step {s}: {v:.3f}; Qwen not there by step {qs.max()} ({qv.min():.3f}): over {qs.max() / s:.2f}x")
            continue
        i = hit[0]
        qstep = qs[0] if i == 0 else qs[i - 1] + (qs[i] - qs[i - 1]) * (qv[i - 1] - v) / (qv[i - 1] - qv[i])
        print(f"  {size} {tpp}: ExeaLM step {s}: {v:.3f}; Qwen reaches it at step {qstep:.0f}: {qstep / s:.2f}x the steps")
