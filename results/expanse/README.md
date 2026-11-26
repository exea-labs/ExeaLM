# The scaling study on SDSC Expanse: results

Results of the NAIRR Pilot allocation (2,000 GPU-hours on NVIDIA H100s, project `fli100`), in the order they came in.
The plan is [docs/budget.md](../../docs/budget.md); how every run is submitted, [docs/runbook.md](../../docs/runbook.md).
Raw files are in this folder and on Hugging Face (`exea-labs/exealm-nairr-runs`, `results/expanse/`).

## 1. The first job (job 56074172, 25 November 2026, 56 minutes on one node)

### Correctness on the H100: every check passed

`gpu_check.txt`. On an NVIDIA H100 80GB HBM3 (torch 2.15.0, CUDA 13.2):

- Every circuit type, the attention block, the Gated DeltaNet block, and each whole model (ExeaLM, Llama, Qwen3.5-style)
  compute on the GPU what they compute on the CPU, to 1e-5 or better in 32-bit; bf16 differs from 32-bit by 5e-3 at
  most.
- A short run trains (validation bits per token 5.61 to 2.68 in 150 steps), and a run stopped and resumed from its
  checkpoint is identical to an uninterrupted one, to the last bit.
- Two processes on two GPUs (DDP) stay in sync at every evaluation, and their resume is identical to the last bit.

### Training speed on one H100 (bf16, no torch.compile)

`throughput_h100.json`. Effective TFLOP/s is 6 x parameters x tokens per second. MI300X figures, which the plan was
first priced at, in brackets.

| Size | ExeaLM | Llama | Qwen3.5-style |
|---|---|---|---|
| 0.1B | 74.8 (56.2) | 135.4 (85.1) | 41.8 |
| 0.25B | 109.8 | 201.2 | 68.3 |
| 0.5B | 119.2 (81.0) | 217.1 (124.8) | 69.8 |
| 3B | 150.1 | 287.0 | 82.7 |

- ExeaLM runs at about half the plain Llama's speed per parameter, and about twice the Qwen3.5-style model's. The
  Qwen3.5-style model's Gated DeltaNet is plain PyTorch, not the flash-linear-attention kernels Qwen itself uses, so
  its speed is ours, not the design's: compare it with ExeaLM in FLOPs, not in GPU-hours.
- torch.compile at 0.5B: ExeaLM 296.8 TFLOP/s (2.5 times eager), Llama 295.4 (1.4 times). Not yet used in training:
  `compile_check.sb` tests it in the real training loop.
- TF32 changes nothing (120.1 and 218.5 at 0.5B): the models run in bf16.
- Four GPUs train 3.68 times as fast as one (0.5B ExeaLM, the real training loop, `scaling_500m.json`): 92% efficient.
- In the real training loop at 0.1B (the lead check below, 32 windows a step), Llama trained at 244,000 tokens a second
  and the Qwen3.5-style model at 73,000, as in the throughput test. ExeaLM trained at about 127,000 in its first 80%
  and about 68,000 while reading streams (its last 20%): streaming runs four windows one after another in rows a
  quarter as many.

At these rates (`plan_at_h100_rates.txt`), the 28 runs of the plan cost 1,309 GPU-hours if four GPUs were four times
one, about 1,410 at the measured 3.68; the reserve is about 390 GPU-hours. **A 9B ExeaLM on 1T tokens costs about
101,000 GPU-hours at ExeaLM's measured 3B rate** (it was 187,000 at the MI300X rate); the 9B measurement
(`nine_b.sb`) will replace this with the 9B model's own speed.

### The lead check: ExeaLM leads both baselines on all 18 held-out sets, on every seed

`lead_100m.json`. At 0.1B (ExeaLM 93.8M, Qwen3.5-style 95.7M, Llama 89.7M parameters), 3,000 steps of 16,384 tokens
(49M tokens), learning rate 1e-3 for all, 3 seeds each, on the study's data (`slice-30b-r2`). ExeaLM read streams in
its last 20% of steps, as the plan trains it; the baselines read one window per row throughout. Scored by
`eval_text.py` on the same fixed windows for every model, each window read alone, on the held-out set of each of the
18 sources. Bits per byte, mean of 3 seeds, lower is better:

| Held-out set | ExeaLM | Qwen3.5-style | Llama | ExeaLM ahead of Qwen3.5-style |
|---|---|---|---|---|
| research_python | **0.936** | 0.974 | 1.033 | 0.037 (3 of 3 seeds) |
| code_reasoning | **0.881** | 0.910 | 0.964 | 0.029 (3 of 3 seeds) |
| algebraic_stack | **1.252** | 1.283 | 1.355 | 0.031 (3 of 3 seeds) |
| pretrain_behaviors | **1.393** | 1.412 | 1.468 | 0.020 (3 of 3 seeds) |
| starcoder | **1.270** | 1.304 | 1.372 | 0.034 (3 of 3 seeds) |
| stackexchange | **1.273** | 1.306 | 1.367 | 0.033 (3 of 3 seeds) |
| pes2o | **1.278** | 1.301 | 1.349 | 0.022 (3 of 3 seeds) |
| arxiv | **1.434** | 1.457 | 1.511 | 0.022 (3 of 3 seeds) |
| wikipedia_en | **1.608** | 1.627 | 1.685 | 0.019 (3 of 3 seeds) |
| megamath | **1.356** | 1.380 | 1.433 | 0.023 (3 of 3 seeds) |
| open_web_math | **1.483** | 1.503 | 1.558 | 0.021 (3 of 3 seeds) |
| math_reasoning | **1.155** | 1.187 | 1.242 | 0.032 (3 of 3 seeds) |
| dclm | **1.648** | 1.662 | 1.710 | 0.014 (3 of 3 seeds) |
| txt360_web | **1.665** | 1.686 | 1.741 | 0.021 (3 of 3 seeds) |
| fineweb2_ja | **1.729** | 1.758 | 1.800 | 0.029 (3 of 3 seeds) |
| wikipedia_ja | **1.930** | 1.981 | 2.042 | 0.051 (3 of 3 seeds) |
| txt360_qa | **1.422** | 1.444 | 1.496 | 0.022 (3 of 3 seeds) |
| sft_pretrain | **1.360** | 1.394 | 1.456 | 0.034 (3 of 3 seeds) |
| **Mean of the 18 sets** | **1.393** | 1.420 | 1.477 | 0.027 |
| **Weighted as the training mix** | **1.281** | 1.309 | 1.367 | 0.027 |

Per seed, mean of the 18 sets: ExeaLM 1.395, 1.393, 1.391; Qwen3.5-style 1.424, 1.420, 1.418; Llama 1.483, 1.475,
1.472. ExeaLM is ahead of the Qwen3.5-style model on all 54 set-and-seed pairs, by 0.014 to 0.051, and further ahead of
Llama on all of them. The largest gaps are on research Python (0.037), Japanese Wikipedia (0.051), worked code
solutions (0.029) and math reasoning (0.032).

What this does not show: one size (0.1B), 49M training tokens, one learning rate shared by all three models (the
learning-rate checks set each model's own), and ExeaLM's two attention changes (value residual, XSA) and its stream
phase are part of ExeaLM and not of the baselines. `ablate.sb` measures what each of them adds, and the baselines with
the two attention changes.

### A bug found and fixed: in-run validation of streamed runs

Training's own validation score (`val_bits_per_byte` in each run's log) drew `--batch` rows, and a streamed run's row
is four windows, so ExeaLM was scored on 64 windows and the baselines on 256: different samples of an 18-source mix.
In this check it read ExeaLM 0.10 bits per byte worse than eval_text did (1.385 against 1.282 weighted as the mix, seed
2), and the baselines right (Llama 1.372 against 1.367). Fixed in commit 039f62b: every model is now scored on the same
windows. The learning-rate checks compare a model only with itself, so they were not affected; the study's results come
from eval_text.py in any case.

## 2. torch.compile in the real training loop (job 56074612, 21 minutes)

`compile_check.json`. 0.5B, 400 steps of 16 windows of 512 per GPU, real data, eager against compiled; tokens per
second over steps 200 to 400, after compiling; held-out bits per byte at step 400 (in-run).

| Model | Eager tokens/s | Compiled | Speed-up | Bits per byte, compiled minus eager |
|---|---|---|---|---|
| Qwen3.5-style, 1 GPU | 22,760 | 54,883 | 2.41x | -0.024 |
| Llama, 1 GPU | 75,931 | 102,202 | 1.35x | +0.003 |
| ExeaLM, 4 GPUs, streams from step 200 | 70,891 | 50,691 | 0.72x | -0.000 |

- Compiled training learns the same model: the scores agree within the noise of 400 steps.
- The baselines are much faster compiled, so the plan trains them compiled (commit after 5e0a6e9). The Qwen3.5-style
  model's 3B run should cost about 300 GPU-hours instead of 721.
- ExeaLM is slower compiled in its stream phase, and its first 200 steps (one window per row) gained nothing either
  (46,103 tokens/s compiled including about 90 s of compiling, 141,809 eager). The bare step compiled 2.5 times as fast
  in the first job, before TONN's running-size update was moved outside the compiled graph (commit fdef3ec) to fix a
  crash in compiled streaming; that move breaks the graph at every TONN layer. ExeaLM trains eager; making it compile
  well is left for later.
- The 1-GPU ExeaLM pair of this job ran code from before that fix: its compiled run stopped at step 200 with the crash
  the fix addresses, so that row is not a measurement.
- ExeaLM reads streams at half its speed (70,891 against 141,809 tokens/s on 4 GPUs): each window's forward pass sees
  a quarter of the rows. Its last 20% costs about 1.2 times what the throughput test suggests.

## 3. Ablations at 0.1B (job 56074735, 40 minutes)

`ablation_100m.json`. The lead check's setting (0.1B, 3,000 steps of 16,384 tokens, warm-up 200, learning rate 1e-3,
the study's data), one change at a time, scored on the same fixed windows as `lead_100m.json`. Bits per byte, mean of
the 18 held-out sets, lower is better:

| Model | Seeds | Bits per byte | Per seed |
|---|---|---|---|
| Llama with gated attention, QK-norm, value residual and XSA | 3 | **1.349** | 1.349, 1.351, 1.348 |
| ExeaLM without its stream phase | 3 | 1.376 | 1.378, 1.376, 1.374 |
| the same at learning rate 2e-3 | 1 | 1.379 | |
| the same with a TONN layer every third layer | 1 | 1.382 | |
| ExeaLM as planned (streams in the last 20%; lead check) | 3 | 1.393 | 1.395, 1.393, 1.391 |
| ExeaLM without streams, learning rate 5e-4 | 1 | 1.397 | |
| ExeaLM without streams, value residual or XSA | 3 | 1.401 | 1.397, 1.406, 1.401 |
| Qwen3.5-style with value residual and XSA | 1 | 1.403 | |
| Qwen3.5-style as planned (lead check) | 3 | 1.420 | 1.424, 1.420, 1.418 |
| Llama as planned (lead check) | 3 | 1.477 | 1.483, 1.475, 1.472 |

What it shows:

- **A Llama with the four attention changes beats every ExeaLM, on all 18 held-out sets** (by 0.016 to 0.064 against
  ExeaLM without streams; 0.027 on the mean, 3 seeds each). At this scale and token count, attention layers with these
  changes do better than TONN layers.
- **TONN layers beat Gated DeltaNet layers as a hybrid's non-attention layer, with or without the two changes, on all
  18 sets.** Without value residual and XSA: ExeaLM 1.401 against the Qwen3.5-style model's 1.420 (3 seeds each). With
  them: ExeaLM 1.376 (3 seeds) against 1.403 (1 seed).
- **The stream phase costs 0.017 on windows read alone** (1.393 against 1.376, 3 seeds each). On the MI300X it gained
  0.023 on the fourth window of long documents read as a stream; the study's single-window comparisons pay for it.
- Value residual and XSA give ExeaLM 0.026 (1.401 to 1.376), the Qwen3.5-style model 0.017 (one seed), and the Llama
  0.128 together with the gate and QK-norm (1.477 to 1.349).
- A TONN layer every fourth layer beats every third (1.376 against 1.382, one seed for every third), as on the MI300X.
  More attention helps; the Llama above is the limit of that trend.
- ExeaLM's learning rate: 1e-3 and 2e-3 are within 0.003 of each other; 5e-4 is 0.021 worse.

What it does not show: one size, 49M training tokens (far below the plan's 4.1 tokens per parameter at 0.1B, 0.38B
tokens, and the long pair's 111), one learning rate for the baselines, single seeds for four rows.

## 4. Next

The learning-rate checks (`go.sh lr`), the ablations at 0.1B (`ablate.sb`), and the compile check
(`compile_check.sb`), then the curve.
