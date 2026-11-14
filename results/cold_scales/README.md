# Unmeasured-circuit normalization: correctness fix, mixed pilot

Status: draft, not a demonstrated loss improvement. The contribution rule requires
a fair comparison before a model/training change goes in; this pilot does not
establish a win. No claim about large models, real text or reported swarm gains.

The old unmeasured-slot fallback divides a slot by RMS across the current batch
and time axis. In a one-layer integrator model with initially zero scales, changing
a future token changed earlier logits by 0.00384018; changing an unrelated row
changed the first row's logits by 0.0493513. The constant prior reduces these to
1.49e-7 (FFT-level numerical noise) and 0 respectively. The experiment reproduces
both perturbations. Regression tests cover training/eval modes, newly replaced
slots, and bitwise equality of outputs/statistic updates for already measured slots.

The patch uses scale 1 only while no past measurement exists. The first training
forward still records RMS for the next batch, and later EMA updates remain the
same. Evaluation does not update these measurements. No checkpoint layout changes.
Warm-slot behavior is unchanged; cold/newly active slots intentionally behave
differently. This does not retroactively remove leakage from an old run.

## Matched CPU pilot

Two seeds, tiny preset, 127,432 parameters per run, 100 steps, batch 4, length 64:
25,600 input tokens per run. Each arm for a seed uses identical initial config,
synthetic data, batch RNG, held-out windows, optimizer and step budget. Search runs
every 20 steps; retrieval has 100 chunks and refreshes every 40. Evaluation uses
two fixed batches. Seeds change initial weights/circuit choices but are matched
between arms. No benchmark or real-text corpus was used.

| Normalization | Seed 0 validation bits/token | Seed 1 validation bits/token |
|---|---:|---:|
| previous batch RMS fallback | 5.0181 | 5.1083 |
| constant prior, proposed | 5.0003 | 5.1593 |
| causal prefix RMS, rejected candidate | 5.0668 | 5.1539 |

The proposed constant prior improves seed 0 by 0.0178 but worsens seed 1 by
0.0510; the mean is 0.0166 worse. The prefix candidate removed leakage too but
worsened both seeds and is not in production code. It is retained as an experiment
control so negative results are visible and reproducible. This tiny pilot is not a
substitute for a longer real-text comparison and does not justify promoting the
model change. Resolve the loss tradeoff before marking the PR ready for merge.

Reproduce all three arms and both seeds:

```bash
python -m experiments.cold_scale_smoke --out runs/cold-scale-smoke
```

The output directory must be new. The script embeds the old normalization as a
frozen reference from main 04d2289 and restores the production method after running.
The JSON records full model configs, environment and unrounded perturbation deltas.
Weights and intermediate training files are not committed.

The existing exact-resume test previously relied on learning a useless circuit by
chance after a particular step. Changing early normalization can make every slot
useful and prevent that scenario. This branch includes the deterministic fixture
already proposed in PR #43: real worth evaluation still runs, then test-only
negative scores guarantee actual replacements. Architecture and bitwise equality
of every resumed state tensor remain asserted.
