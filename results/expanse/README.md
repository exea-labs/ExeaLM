# The scaling study on SDSC Expanse: results

Results of the NAIRR Pilot allocation (2,000 GPU-hours on NVIDIA H100s, project `fli100`). The summary first; the
sections after it are the record, in the order results came in, with what each does not show. The plan is
[docs/budget.md](../../docs/budget.md); every command, [docs/runbook.md](../../docs/runbook.md); the decisions,
[docs/decisions.md](../../docs/decisions.md). The finished models are public on Hugging Face:
[exea-labs/exealm-nairr](https://huggingface.co/exea-labs/exealm-nairr).

## Summary (27 November 2026, 2 a.m.; the 3B pair is training)

ExeaLM: attention layers with a gate, QK-norm, value residual and XSA, one TONN layer (searched signal-processing
circuits) at the middle layer, and streams of four windows in the last 20% of training. Every comparison below is at
the same size, data, tokens and recipe; bits per byte on held-out text, lower is better.

| What | Result | Basis | Section |
|---|---|---|---|
| Ahead of the Qwen3.5 design | 0.1B: 1.157 against 1.190; 0.25B: 1.014 against 1.026 (plain Llama 1.200, 1.041); 0.5B: 0.933 against 0.939 (Llama 0.958) on windows alone, 0.873 against 0.906 on long documents | 0.1B and 0.25B: 3 seeds, all 54 set-and-seed pairs at each size; 0.5B: 2 seeds, all 36 pairs | 9, 13, 20 |
| Long documents | 4,096 tokens read as a stream: 0.1B 1.096 against 1.157; 0.25B 0.955 against 0.997. 16,384 tokens (8 times the streams it trained on): 0.1B 1.073 against 1.134; 0.25B 0.947 against 0.996, the lead growing along the document | 4,096: 3 seeds, all 54 pairs at each size. 16,384: 3 seeds at 0.1B, 1 at 0.25B | 10, 15, 19 |
| Learns more per token | at 111 tokens per parameter (the 9B's), the Qwen3.5-style model needs 1.34 and 1.41 times ExeaLM's steps to match it at step 20,000, rising through training, and never reaches its final score. Finished: 0.958 against 0.963 on windows alone, 0.902 against 0.946 on long documents | 2 seeds, 0.1B, both ahead on every reading | 12, 24 |
| A library distilled from a larger model | held-out research questions 1.645 to 0.590 with the library and a BM25 search; 0.110 to 0.116 ahead of the Qwen3.5-style model with the same library | 3 seeds, 0.1B | 7, 8, 11 |
| Against the released Qwen3.5-9B | 0.1B ExeaLM with its library 0.590 against Qwen3.5-9B without one 1.031 on held-out research questions (a system against a model; with the same library Qwen3.5-9B scores 0.313). On text alone Qwen3.5-9B is far ahead (0.674 against 1.011 for ExeaLM 0.25B) | 1 seed | 18 |
| Where the TONN layer goes | one TONN layer at the middle layer beats one in every fourth layer (1.348 against 1.376, 54 of 54 pairs); one at layer 0 costs 0.038 | 12 layouts, 0.1B | 4, 5 |
| What the TONN layer itself adds | about 0.1%: against the same model with attention in its place, at 0.1B, 1.1566 against 1.1574 on windows alone, 1.0955 against 1.0968 on 4,096-token streams, 1.0726 against 1.0737 on 16,384-token streams. The lead over the Qwen3.5 design comes from the attention changes and the streams | 3 seeds (ahead in 2, 3 and 3 of them); 0.25B and the long run are training | 5, 19 |
| The 9B, priced | 206 effective TFLOP/s per H100 on one node; about 75,000 GPU-hours for 1T tokens | 1 node; 2 and 4 nodes measured, being written up | 16 |
| Streams without most of their cost | mixing windows alone into the stream phase gives back half of what it cost on windows alone (1.1532 against 1.1566; none 1.1496) for 0.0014 of its long-document gain (1.0969 against 1.0955); now the recipe | 3 seeds, 0.1B | 21 |
| Distillation into the weights | 30% teacher text (Qwen3.5-9B) in pretraining: worse, 1.1592 against 1.1549 | 1 seed, 0.1B | 22 |
| Benchmarks at 0.5B | mean of 7 zero-shot tasks 0.429 against 0.421 (Llama 0.417), ahead in both seeds and on 5 of 7 tasks; small: two tasks near chance | 2 seeds | 20 |
| Did not work | ExeaLM's own retrieval circuit (0.1% of questions find their entry); three 0.1B agents sharing a board lose to one model reading as a stream | | 7, 14 |
| Void | the first 0.5B runs and the 3B pair's first 1,100 steps: a multi-GPU bug skipped gradient averaging; rerun with the fix | | 17 |

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

## 4. Layouts with more attention (job 56077551, 35 minutes)

`layout_100m.json`. Same setting as §3, no streams, one seed unless stated. Bits per byte, mean of the 18 sets:

| ExeaLM layout (TONN layers of 14) | Seeds | Bits per byte | Against the Llama with the four attention changes (1.3493, 3 seeds) |
|---|---|---|---|
| only layer 7 (the middle) | 1 | **1.3488** | -0.0005; ahead on 13 of 18 sets |
| only layer 13 (the last) | 1 | 1.3512 | +0.0019 |
| every sixth: 0, 6, 12 | 1 | 1.3728 | +0.0235 |
| every fourth: 0, 4, 8, 12 (§3) | 3 | 1.3758 | +0.0264 |
| every seventh: 0, 7 | 1 | 1.3777 | +0.0284 |
| every eighth: 0, 8 | 2 | 1.3805 | +0.0312 |
| only layer 0 (the first) | 1 | 1.3867 | +0.0374 |

The Llama with the four changes at learning rate 2e-3 scored 1.3553, worse than at 1e-3.

- A TONN layer at layer 0 costs about 0.035, and every "every Nth" layout has one there. That, more than the share of
  attention, is why those layouts trail.
- One TONN layer in the middle ties the Llama with the four attention changes (one seed, inside its seeds' spread of
  1.3477 to 1.3509) and is ahead of it on 13 of 18 sets. It is 0.072 ahead of the Qwen3.5-style model.
- `layout2.sb` follows up: more seeds of the middle layer, and more TONN layers with none at layer 0.

## 5. One TONN layer in the middle beats the Llama with four attention changes (job 56081126)

`layout2_100m.json`, with §3 and §4 for the other rows. Same setting, no streams unless stated. Mean of the 18 sets:

| ExeaLM layout (TONN layers of 14) | Seeds | Bits per byte | Per seed | Against the Llama (1.3493) | Sets ahead |
|---|---|---|---|---|---|
| one, layer 7 (the middle) | 3 | **1.3484** | 1.3488, 1.3501, 1.3464 | -0.0009 | 14 of 18 |
| one, layer 6 | 1 | 1.3497 | | +0.0004 | 9 |
| one, layer 13 | 1 | 1.3512 | | +0.0019 | 0 |
| two, layers 5 and 9 | 1 | 1.3520 | | +0.0027 | 1 |
| three, layers 4, 7, 10 | 2 | 1.3562 | 1.3573, 1.3552 | +0.0069 | 0 |
| three, layers 3, 7, 11 | 1 | 1.3576 | | +0.0083 | 0 |
| one, layer 7, with streams in the last 20% | 1 | 1.3639 | | +0.0146 | 0 |

- With one TONN layer in the middle, ExeaLM is ahead of the Llama with gated attention, QK-norm, value residual and
  XSA on all three seeds, pairing seed with seed (by 0.0006, 0.0008 and 0.0013), and on 14 of 18 sets. The margin is
  small; it is consistent.
- Every TONN layer beyond one costs, and the stream phase costs 0.015 on windows read alone.
- This layout is 0.072 ahead of the Qwen3.5-style model (1.3484 against 1.4204) and 0.128 ahead of the plain Llama.

## 6. The 3B Qwen3.5-style run uses the flash-linear-attention kernel (jobs 56081770, 56081833)

Our Gated DeltaNet layer (`exealm/model/delta.py`) ran its recurrence as `chunk_gated_delta`, chunk by chunk in plain
PyTorch. At 3B that made the Qwen3.5-style run take 13.0 s a step on four H100s (job 56081618, cancelled at step 100):
165 hours, 660 GPU-hours. `--fla 1` runs the same recurrence on the Triton kernel Qwen3-Next uses
(`fla.ops.gated_delta_rule.chunk_gated_delta_rule`, flash-linear-attention 0.6.0). `fla_check.json`: one batch of
random tokens, the same weights, forward and backward, each kernel against our kernel in fp32 without autocast:

| Preset | Kernel, precision | Loss off by | Gradient cosine (median, min) |
|---|---|---|---|
| 0.1B, 8 windows | ours, bf16 | 0.00032 | 0.968, 0.937 |
| | fla, bf16 | 0.00014 | 0.968, 0.946 |
| | fla, fp32 | 0.00002 | 0.9999, 0.9998 |
| 0.5B, 2 windows | ours, bf16 | 0.00184 | 0.900, 0.781 |
| | fla, bf16 | 0.00165 | 0.897, 0.766 |
| | fla, fp32 | 0.00012 | 0.9996, 0.9991 |

- In fp32 the two kernels agree (the rest is TF32 inside the Triton kernel). Under bf16, the precision the study trains
  in, the fla kernel is as far from the fp32 result as ours is: the spread is bf16 rounding of gradients at
  initialisation, the same for both.
- Forward plus backward at 2,048 tokens a window: 3.79 times as fast at 0.1B, 2.82 at 0.5B, 3.38 at 3B (2 windows).
- Only the 3B run uses it (job 56081885). The curve's Qwen3.5-style runs had started on our kernel and keep it, so
  every curve point is trained the same way. Compute is compared in FLOPs, which the kernel does not change.

## 7. The distilled library at 0.1B (jobs 56082586-90)

`library_100m.jsonl`. The library (`exea-labs/exealm-library`): 18,545 entries (13,548 on research topics, 4,997 on
code), written by Qwen3-30B-A3B-Instruct-2507 from source excerpts, 7.9M tokens, with 55,940 question/answer pairs and
60,515 tasks that need an entry adapted on training topics and 6,253 and 6,604 on held-out topics. Three post-training
runs of 600 steps (65,536 tokens a step, half the library's training items and half the pretraining mix,
`experiments/post/library_post.sh`) from the seed-2 curve models: ExeaLM with a library slot (RG) grafted into its
TONN layer and searching the library, ExeaLM without it, and the Qwen3.5-style model. Bits per byte on the targets of
the held-out topics' items:

| Model | Questions, nothing | Questions, entry pasted | Tasks, nothing | Tasks, entry pasted |
|---|---|---|---|---|
| ExeaLM, post-trained | 1.674 | **0.511** | **0.798** | **0.638** |
| Qwen3.5-style, post-trained | **1.646** | 0.631 | 0.815 | 0.686 |
| ExeaLM, before post-training | 1.977 | 0.738 | 1.048 | 0.826 |
| Qwen3.5-style, before post-training | 1.945 | 0.897 | 1.055 | 0.904 |

- With the right entry in its context, ExeaLM's bits per byte on held-out questions fall by 69% (1.674 to 0.511) and
  on tasks by 20%. It uses the entry better than the Qwen3.5-style model, before and after post-training (0.511
  against 0.631 on questions, 0.638 against 0.686 on tasks). Without an entry the two are close: the Qwen3.5-style
  model is 0.028 ahead on questions, ExeaLM 0.017 ahead on tasks.
- ExeaLM's own retrieval does not work yet. The grafted RG slot returned a chunk of the right entry for 0.1% of
  questions and 7.8% of tasks (24% at any chunk of the target), and attaching the library changed nothing (1.6742
  against 1.6741 on questions). The entries reach the model only when something else puts them in its context.
- One seed, 0.1B, 600 steps of post-training that never showed an entry in context. `score.py --conditions bm25`
  measures the same with a keyword search choosing the entry (next section).

## 8. The library with a keyword search: ordinary RAG (job 56082719)

`library_bm25_100m.jsonl`. `score.py --conditions bm25`: for each held-out item, BM25 over the token ids of all 18,545
entries ranks them against the item's prefix (its question, or its instruction and context), and the first is pasted
before the prefix, as in "pasted". The search put the item's own entry first for 86.2% of questions (93.2% in the
first 5) and 72.9% of tasks (79.8%). Bits per byte on the held-out targets:

| Model | Questions: nothing, BM25, pasted | Text tasks: nothing, BM25 | Code tasks: nothing, BM25 |
|---|---|---|---|
| ExeaLM, post-trained | 1.674, **0.689**, 0.511 | 0.933, **0.754** | 0.438, **0.366** |
| Qwen3.5-style, post-trained | 1.646, 0.796, 0.631 | 0.947, 0.803 | 0.462, 0.406 |
| ExeaLM, before post-training | 1.977, 0.923, 0.738 | 1.221, 0.964 | 0.588, 0.479 |
| Qwen3.5-style, before post-training | 1.945, 1.065, 0.897 | 1.219, 1.050 | 0.613, 0.526 |

- With a keyword search choosing the entry, the library cuts ExeaLM's bits per byte on held-out questions by 59%
  (1.674 to 0.689), on text tasks by 19% and on code tasks by 16%.
- ExeaLM gains more from it than the Qwen3.5-style model, on every set, before and after post-training: 0.107 ahead
  on questions with BM25 (0.689 against 0.796), 0.049 on text tasks and 0.040 on code tasks.
- On code tasks the searched entry scores better than the item's own source entry (0.366 against 0.399 for ExeaLM):
  the search often finds an entry closer to the code in the task's context.
- Neither model was trained with entries in its context. One seed, 0.1B.

## 9. The curve at 0.1B (job 56082581, `text_100m.json`)

The study's 0.1B runs: 1,468 steps of 262,144 tokens (0.38B tokens, 4.0 per ExeaLM parameter), each model at the rate
its learning-rate check chose, the last 10% on the anneal mix, 3 seeds. Every row read alone, mean of the 18 held-out
sets:

| Model | Bits per byte | Per seed |
|---|---|---|
| ExeaLM, one TONN layer in the middle, no streams | 1.1496 | 1.1485, 1.1501, 1.1502 |
| **ExeaLM, one TONN layer in the middle, streams in the last 20% (the study's)** | **1.1566** | 1.1549, 1.1574, 1.1575 |
| ExeaLM, a TONN layer every fourth layer, streams (the plan before 26 November) | 1.1685 | 1.1685, 1.1689, 1.1681 |
| Qwen3.5-style | 1.1904 | 1.1917, 1.1906, 1.1890 |
| Plain Llama | 1.1995 | 1.2010, 1.1981, 1.1994 |

- The study's ExeaLM is ahead of the Qwen3.5-style model on all 54 set-and-seed pairs, by 0.024 to 0.061 (0.034 on
  the mean), and of the plain Llama on all 54, by 0.033 to 0.065.
- The lead is half what it was after 49M tokens (0.072 between the same layouts, sections 4-5). The 0.25B runs show the
  same in training: 0.072 at step 500, 0.039 at step 1,000. The long pair (111 tokens per parameter) tests whether the
  lead keeps shrinking with training.
- The new layout beats the old on all 54 pairs (by 0.006 to 0.024). Streams cost 0.007 on windows read alone here,
  against 0.015 in the shorter test.

## 10. Long documents at 0.1B (job 56082581, `stream8_100m.json`)

The curve's 0.1B models (3 seeds each) on the first 4,096 tokens of every held-out document at least that long (up to
64 a set, 18 sets), read as 8 windows of 512: "streamed" carries each window's state into the next (attention keys and
values of the window before; ExeaLM's TONN state; Gated DeltaNet's matrix state), "alone" reads every window by itself.
Bits per byte, mean of windows 1 to 7 (window 0 has nothing before it):

| Model | Streamed | Alone | Window 7, streamed |
|---|---|---|---|
| **ExeaLM, the study's (streams in its last 20%)** | **1.096** | 1.194 | **1.083** |
| ExeaLM, one TONN layer, trained without streams | 1.132 | 1.187 | 1.122 |
| ExeaLM, a TONN layer every fourth layer, streams | 1.110 | 1.206 | 1.097 |
| Qwen3.5-style | 1.157 | 1.230 | 1.144 |
| Plain Llama | 1.584 | 1.238 | 1.586 |

- Read as a stream, the study's ExeaLM is ahead of the Qwen3.5-style model on all 54 set-and-seed pairs, by 0.039 to
  0.098 (0.061 on average): nearly twice its lead on windows read alone (0.034, section 9).
- Part of that is the stream phase, which the baselines do not train. The same ExeaLM trained without streams is still
  ahead on all 54 pairs (by 0.011 to 0.056, 0.025 on average), with neither model trained to stream.
- The plain Llama cannot use the window before without training for it: its scores get worse streamed (1.584 against
  1.238 alone). The Qwen3.5-style model gains 0.073 from streaming untrained, ExeaLM without stream training 0.055,
  ExeaLM with it 0.099.

## 11. The library as trained RAG at 0.1B, 3 seeds (`library_rag_100m_s0.jsonl`, `_s1`, `_s2`)

`experiments/post/rag_post.sh`, once from each of the three seeds of the curve's 0.1B models (`TAG=100m_s0`,
`100m_s1` and `100m_s2`; runs `ragpost_<model>_100m_s<seed>`): the same post-training as section 7, except that half of the library share is the
training items with the entry a BM25 search picks placed before them (`experiments/library/rag_pack.py`), as at test
time. Bits per byte on the held-out topics, mean of 3 seeds:

| Model | Questions: nothing, BM25, pasted | Text tasks: nothing, BM25 | Code tasks: nothing, BM25 |
|---|---|---|---|
| ExeaLM, RAG post-training | 1.645, **0.590**, 0.405 | 0.936, **0.739** | 0.448, **0.364** |
| Qwen3.5-style, RAG post-training | 1.633, 0.704, 0.529 | 0.953, 0.791 | 0.473, 0.403 |
| ExeaLM, post-training without entries (section 8, seed 2) | 1.674, 0.689, 0.511 | 0.933, 0.754 | 0.438, 0.366 |

- With the library and a BM25 search, ExeaLM's bits per byte on held-out questions fall by 64% (1.645 to 0.590; seeds
  0.588, 0.589, 0.592), and 21% on text tasks and 19% on code tasks.
- ExeaLM is ahead of the Qwen3.5-style model with the library on every seed: by 0.110 to 0.116 on questions and 0.047
  to 0.050 on tasks. Without the library the two are within 0.02 of each other.
- Training with searched entries in context gained ExeaLM 0.097 on questions over post-training without them (seed 2).

## 12. How many times the steps the Qwen3.5-style model needs (in training, 26 November 04:20)

`experiments/expanse/step_multiplier.py`: for each in-run evaluation of an ExeaLM run (512 fixed held-out windows, the
same for every model), the step at which the paired Qwen3.5-style run (same size, seed 0) first reaches that score.
A step is the same tokens for both, and the two have the same parameters within 1%, so this is a token and compute
multiplier. Preliminary: one seed, runs still training.

| Pair | ExeaLM step: multiplier |
|---|---|
| 0.1B, 111 tokens per parameter | 1,000: 1.18; 1,500: 1.18; 2,000: 1.14; 2,500: 1.15; 3,000: 1.16; 3,500: 1.17; 4,000: 1.19 |
| 0.25B, 4.1 | 1,000: 1.34; 1,500: 1.29; 2,000: 1.24 |
| 0.5B, 4.1 | 1,000: 1.43; 1,500: 1.35 |

- The gap in bits per byte shrinks as both curves flatten (0.055 to 0.016 on the long pair), but the steps the
  Qwen3.5-style model needs to match ExeaLM do not: about 1.15 to 1.19 times through the first 4,000 of the long pair's
  39,735 steps. The long pair, trained as long per parameter as the 9B would be, decides whether that holds.
- Early in training the multiplier is larger for larger models. Read at the same fraction of each run, once the runs
  finish, before reading anything into the size trend.

## 13. The curve at 0.25B (job 56083853, `text_250m.json`)

The study's 0.25B runs: 3,869 steps (1.0B tokens, 4.0 per ExeaLM parameter), 3 seeds, the same recipe as 0.1B. Every
row read alone, mean of the 18 held-out sets:

| Model | Bits per byte | Per seed |
|---|---|---|
| **ExeaLM (one TONN layer in the middle, streams in the last 20%)** | **1.0141** | 1.0138, 1.0145, 1.0141 |
| Qwen3.5-style | 1.0255 | 1.0257, 1.0256, 1.0252 |
| Plain Llama | 1.0407 | 1.0398, 1.0417, 1.0406 |

- ExeaLM is ahead of the Qwen3.5-style model on all 54 set-and-seed pairs, by 0.007 to 0.026 (0.011 on the mean), and
  of the plain Llama on all 54, by 0.018 to 0.040. Its largest leads are on Japanese (0.024 on Wikipedia) and math
  reasoning (0.016).
- On windows read alone the lead over the Qwen3.5-style model shrinks with size: 0.034 at 0.1B, 0.011 at 0.25B. In
  training, it was 0.020 at step 2,500 and fell to 0.011 over the stream phase and the anneal; the stream phase costs
  ExeaLM on windows read alone (0.007 at 0.1B, section 9). The long-document scores at 0.25B come from the same job.

## 14. Agents on a board against one bigger model, at 0.1B (jobs 56083871-6, 56084006-7, `board_100m.json`)

`experiments/post/board_post.sh`: board slots (AG:2) grafted into the 0.1B ExeaLM's TONN layer, post-trained for 600
steps in swarms of 3 (agent j reads window j of 3 consecutive windows; `board_order` "doc", so an agent sees only the
agents before it); the same 0.1B ExeaLM, the 0.25B ExeaLM and the 0.25B Qwen3.5-style model post-trained the same steps
on the same mix, every row alone. `experiments/swarm_long.py` scores them on the first 1,536 tokens of the held-out
documents at least that long (64 a set, 18 sets). Bits per byte, mean of windows 1 and 2 (seed 0):

| Model | Parameters read per token | Windows 1-2 |
|---|---|---|
| 3 x 0.1B ExeaLM sharing a board | 95M (3 x 95M in all) | 1.181 |
| the same, board cut | 95M | 1.184 |
| 0.1B ExeaLM, reading the windows as a stream | 95M | **1.096** |
| 0.25B ExeaLM, as a stream | 251M | **0.956** |
| 0.25B Qwen3.5-style, as a stream | 252M | 0.992 |
| 0.25B ExeaLM, each window alone | 251M | 1.042 |

- The board does not do what the claim needs. Sharing it gains the swarm 0.0035 over the same agents with the board cut,
  and the swarm loses to one 0.1B model that carries the window before into the next (1.096) by 0.085, let alone to a
  0.25B model. Three small agents on this board do not beat one bigger model, at 0.1B and 600 steps of post-training.
- What does work is the stream: one model carrying its state from window to window. At 0.25B, read as a stream,
  ExeaLM is 0.036 ahead of the Qwen3.5-style model after this post-training (0.038 before it, the curve checkpoints),
  against 0.011 on windows read alone (section 13).
- The board here is narrow: two slots in the one TONN layer, 32 numbers each per 64 tokens. One seed.

## 15. Long documents at 0.25B (job 56083853, `stream8_250m.json`)

The curve's 0.25B models (3 seeds each), scored as in section 10: 4,096 tokens of every long held-out document (up to
64 a set, 18 sets) as 8 windows of 512. Bits per byte, mean of windows 1 to 7:

| Model | Streamed | Alone | Window 7, streamed |
|---|---|---|---|
| **ExeaLM** | **0.955** | 1.048 | **0.947** |
| Qwen3.5-style | 0.997 | 1.060 | 0.992 |
| Plain Llama | 1.185 | 1.076 | 1.181 |

- Read as a stream, ExeaLM is ahead of the Qwen3.5-style model on all 54 set-and-seed pairs, by 0.025 to 0.066
  (0.042 on average), against 0.011 on windows read alone (section 13).
- From 0.1B to 0.25B the lead on long documents goes from 0.061 to 0.042, and on windows read alone from 0.034 to 0.011.

| Lead over the Qwen3.5-style model | 0.1B | 0.25B |
|---|---|---|
| Windows read alone (18 sets) | 0.034 | 0.011 |
| Long documents read as a stream, windows 1-7 | 0.061 | 0.042 |

## 16. The 9B model on one node (job 56084410, `nine_b.json`; speed only, unaffected by section 17)

`experiments/expanse/nine_b.sb`: ExeaLM at 9B as the study now defines it (`exealm_mid_9b.json`: 9,276.8M parameters,
42 layers, one TONN layer at layer 21), sharded over 4 H100s with FSDP2, activation checkpointing on, 512-token rows,
bf16. Speed of the full training step (forward, backward, AdamW), effective TFLOP/s = 6 x parameters x tokens/s:

| Rows per GPU | Seconds a step | Tokens/s per GPU | Effective TFLOP/s per GPU | Peak memory |
|---|---|---|---|---|
| 2 | 0.50 | 2,053 | 114 | 36.8 GB |
| 4 | 0.70 | 2,938 | 164 | 37.8 GB |
| 8 | 1.11 | 3,705 | **206** | 39.8 GB |

- At 206 effective TFLOP/s per GPU, a 9B ExeaLM on 1T tokens takes about 75,000 H100 GPU-hours on one node's worth of
  scaling (6 x 9.28e9 x 1e12 / 206e12 / 3,600). The plan's estimate from ExeaLM's earlier 3B rate (150) was 103,000.
- Memory is at 40 of 80 GB, and speed still rises with the batch: 16 and 32 rows per GPU, and 2 and 4 nodes, are queued
  (jobs 56084546-8). Those rows replace this estimate.

## 17. A bug in multi-GPU training: the first 0.5B and 3B runs are void (fixed 26 November, 08:30)

In `exealm/train/pretrain.py` the gradient-accumulation loop's variable (`for k in range(a.accum)`) was overwritten on
its first line by the stream factor (`k = a.stream if ... else 1`), and `no_sync()` tested that overwritten `k`. On more
than one process with `--accum` above 2, every forward and backward pass ran under `no_sync()`: DDP never averaged the
processes' gradients. Each process trained on its own quarter of every batch; the in-run check saw the processes
drift apart (`processes_in_sync: false` at 13 to 16 of 16 evaluations) and copied process 0's weights to the others.
With `--zero`, each process updated its part of the weights from its own rows alone, and the check could not see it.
ExeaLM's stream phase changed `k` to 4, which switched the averaging back on for ExeaLM's last 20% at 0.5B only.

- Void: the six 0.5B curve runs (the Qwen3.5-style and Llama 0.5B models scored worse than their 0.25B models: 1.053
  and 1.078 against 1.026 and 1.041), the 0.5B and 3B learning-rate checks, and the 3B pair's first 1,100 steps. They
  are in `runs/_broken_ddp/` on Expanse. About 190 GPU-hours.
- Not affected: everything on one GPU, which is every result in sections 1 to 15 except the 0.5B rows: the 0.1B and
  0.25B curve, the long pairs, the layout search, and the library, RAG and board tests. The 9B speed measurement
  (`nine_b.py`, its own loop) measures speed only.
- Fix: the loop variable is `micro`. `tests/test_ddp.py::test_accumulation_averages_gradients_over_processes` runs
  two processes with `--accum 3` (plain and `--zero`) and checks they stay in sync and train the model one process
  trains; it fails on the old code.
- Rerun: the 3B and 0.5B learning-rate checks (jobs 56084938-50), then the 3B pair and the 0.5B curve.
- The checks with gradients averaged pick different rates. Last validation bits per byte (500 steps at 3B, 1,000 at
  0.5B):

| Model | 3e-4 | 6e-4 | 1e-3 | Picked (before the fix) |
|---|---|---|---|---|
| ExeaLM 3B | 1.193 | **1.150** | | 6e-4 (3e-4) |
| Qwen3.5-style 3B | **1.297** | 1.323 | | 3e-4 (3e-4) |
| ExeaLM 0.5B | 1.115 | 1.060 | **1.044** | 1e-3 (1e-3) |
| Qwen3.5-style 0.5B | 1.159 | 1.098 | **1.078** | 1e-3 (1e-3) |
| Llama 0.5B | 1.161 | **1.113** | 1.114 | 6e-4 (6e-4) |

  The 3B pair restarted from scratch at these rates (jobs 56085824-5, 26 November 09:48), the 0.5B curve at 11:31
  (jobs 56086585-90). The 3B pair's checkpoints now take about a minute (each process
  writes its own part of AdamW's state), against 14.5 before.

## 18. The bar: the released Qwen3.5-9B-Base on the same text (job 56087492, `external_qwen35_9b_base.json`)

`experiments/external/`: the 512 windows of each held-out set and the library's held-out items, as text, scored by
`Qwen/Qwen3.5-9B-Base` (Apache-2.0; 32 layers in Qwen3.5's 3:1 Gated DeltaNet / gated attention pattern; trained by
Alibaba on far more tokens than any model here) through vLLM, and by our seed-0 curve models on the same text. Every
number is bits over the bytes a model predicts (the window's text but its first token), so tokenizers do not matter.
Our models' means here (exact bytes, seed 0) differ from sections 9 and 13 (pack-average bytes, 3 seeds) by under 0.004.

| Held-out set | Qwen3.5-9B-Base | ExeaLM 0.25B | Qwen3.5-style 0.25B | ExeaLM 0.1B | ExeaLM 0.25B / Qwen3.5-9B |
|---|---|---|---|---|---|
| research_python | 0.467 | 0.659 | 0.672 | 0.761 | 1.41 |
| code_reasoning | 0.465 | 0.550 | 0.560 | 0.686 | 1.18 |
| algebraic_stack | 0.685 | 0.897 | 0.905 | 1.015 | 1.31 |
| pretrain_behaviors | 0.585 | 0.949 | 0.959 | 1.143 | 1.62 |
| starcoder | 0.565 | 0.871 | 0.882 | 1.014 | 1.54 |
| stackexchange | 0.608 | 0.952 | 0.965 | 1.082 | 1.57 |
| pes2o | 0.682 | 0.965 | 0.974 | 1.083 | 1.42 |
| arxiv | 0.738 | 1.077 | 1.085 | 1.210 | 1.46 |
| wikipedia_en | 0.727 | 1.199 | 1.208 | 1.356 | 1.65 |
| megamath | 0.725 | 1.043 | 1.053 | 1.174 | 1.44 |
| open_web_math | 0.787 | 1.131 | 1.139 | 1.268 | 1.44 |
| math_reasoning | 0.484 | 0.784 | 0.799 | 0.938 | 1.62 |
| dclm | 0.891 | 1.322 | 1.331 | 1.456 | 1.48 |
| txt360_web | 0.841 | 1.265 | 1.275 | 1.404 | 1.50 |
| fineweb2_ja | 0.845 | 1.204 | 1.223 | 1.343 | 1.43 |
| wikipedia_ja | 0.815 | 1.294 | 1.319 | 1.454 | 1.59 |
| txt360_qa | 0.702 | 1.091 | 1.100 | 1.223 | 1.56 |
| sft_pretrain | 0.517 | 0.939 | 0.955 | 1.115 | 1.82 |
| **Mean of the 18 sets** | **0.674** | 1.011 | 1.022 | 1.151 | 1.50 |

The library's held-out items (bits per byte on the targets; ExeaLM 0.1B from section 11, 3 seeds, RAG post-training):

| Items | Qwen3.5-9B: nothing, BM25, pasted | ExeaLM 0.1B: nothing, BM25, pasted |
|---|---|---|
| Questions | 1.031, 0.313, 0.169 | 1.645, 0.590, 0.405 |
| Text tasks | 0.498, 0.321, 0.290 | 0.936, 0.739, 0.715 |
| Code tasks | 0.157, 0.121, 0.131 | 0.448, 0.364, 0.401 |

- This is the bar a 9B ExeaLM is measured against. Our largest finished model (0.25B, 1B tokens) is 1.5 times Qwen3.5-9B's
  bits per byte on the mean; its nearest set is worked code solutions (0.550 against 0.465, 1.18 times).
- With the library and a keyword search, the 0.1B ExeaLM is ahead of Qwen3.5-9B without the library on the held-out
  research questions (0.590 against 1.031). That is a system against a model: given the same library, Qwen3.5-9B
  scores 0.313.

## 19. 16,384-token documents, and the TONN layer against attention (jobs 56088276, 56088365)

`stream32_100m.json`: the curve's models on the first 16,384 tokens of every held-out document at least that long (up
to 40 a set; 15 of the 18 sets have such documents), read as 32 windows of 512, eight times the four windows ExeaLM
streamed in training. "Streamed" and "alone" as in section 10. Bits per byte, means of window ranges:

| Model | Seeds | Streamed, windows 1-31 | Windows 1-7 | Windows 24-31 | Alone |
|---|---|---|---|---|---|
| **ExeaLM 0.1B** | 3 | **1.0726** | **1.0869** | **1.0644** | **1.1885** |
| ExeaLM 0.1B with attention in place of its TONN layer | 3 | 1.0737 | 1.0882 | 1.0654 | 1.1893 |
| Qwen3.5-style 0.1B | 3 | 1.1339 | 1.1459 | 1.1259 | 1.2223 |
| **ExeaLM 0.25B** | 1 | **0.9471** | **0.9527** | **0.9438** | **1.0538** |
| Qwen3.5-style 0.25B | 1 | 0.9956 | 0.9958 | 0.9921 | 1.0656 |

- Streams of four windows in training carry to 32: every model keeps improving deeper into the document.
- ExeaLM's lead over the Qwen3.5-style model is 0.061 at 0.1B (5.4%, 3 seeds) and 0.049 at 0.25B (4.9%, 1 seed). At
  0.25B it grows along the document, from 0.043 in windows 1-7 to 0.048 in windows 24-31.

The TONN layer against attention (`text_100m_strong.json`, `stream8_100m_strong.json` and the table above). The
control is ExeaLM with attention in the place of its TONN layer (`experiments/expanse/diag/llama_two.json`), trained
with the same recipe, streams included; 3 seeds each at 0.1B:

| Reading | ExeaLM | Attention in its place | Seeds where ExeaLM is ahead |
|---|---|---|---|
| Windows of 512 alone (section 9's) | **1.1566** | 1.1574 | 2 of 3 |
| 4,096-token documents as a stream | **1.0955** | 1.0968 | 3 of 3 |
| 16,384-token documents as a stream | **1.0726** | 1.0737 | 3 of 3 |

The TONN layer is ahead by about 0.1%. What puts ExeaLM ahead of the Qwen3.5 design is the rest of it: the attention
changes (gate, QK-norm, value residual, XSA) and the streams. At 111 tokens per parameter the control is ahead of
ExeaLM on every reading (section 24). The 0.25B control (3 seeds) is being scored (eval job 56104720).

## 20. The 0.5B pair, seed 0 (job 56090409, `text_500m.json`, `stream8_500m.json`)

The fixed-code 0.5B runs (section 17), same data, tokens and recipe; bits per byte on the 18 held-out sets:

| | ExeaLM 0.5B | Qwen3.5-style 0.5B |
|---|---|---|
| Windows of 512 read alone (section 9's reading) | **0.933** | 0.939 |
| 4,096-token documents read as a stream, windows 1-7 | **0.873** | 0.904 |
| The same documents, every window alone | **0.965** | 0.971 |

- On windows alone ExeaLM's lead shrinks with size: 0.034 at 0.1B, 0.012 at 0.25B, 0.006 at 0.5B (seed 0). It
  shrank most in the stream phase: 0.011 at step 6,000 of 7,825 in the training log, 0.004 at the end (section 21).
- Read as a stream it is 0.031 ahead (3.4%), against 0.061 at 0.1B and 0.042 at 0.25B.

**Both seeds and the plain Llama** (eval job 56094554, `text_500m_all.json`, `stream8_500m_all.json`; seed 1 of every
model trained with the study's recipe, like seed 0):

| 0.5B, 2 seeds | ExeaLM | Qwen3.5-style | Llama |
|---|---|---|---|
| Windows alone, 18 sets | **0.9334** (0.9332, 0.9335) | 0.9388 (0.9390, 0.9387) | 0.9576 (0.9575, 0.9577) |
| 4,096-token documents read as a stream | **0.8730** (0.8731, 0.8729) | 0.9057 (0.9041, 0.9072) | 1.0590 (1.0413, 1.0768) |
| The same documents, every window alone | **0.9647** | 0.9711 | 0.9901 |

- ExeaLM is ahead of the Qwen3.5-style model on all 36 set-and-seed pairs (by 0.002 to 0.013) and of the Llama on all
  36 (by 0.017 to 0.039). The seeds agree to 0.0003 on windows alone.
- Read as a stream, ExeaLM is 0.033 ahead of the Qwen3.5-style model (3.6%). The plain Llama, which never trained on
  streams, does worse with the window before carried in than without it (1.059 against 0.990).

Downstream benchmarks (lm-eval harness, zero-shot; `bench_exealm_500m_4.1x_s0.json`, `bench_qwen_500m_4.1x_s0.json`;
accuracy, normalized where the task has it):

| | HellaSwag | ARC-Easy | ARC-Challenge | PIQA | WinoGrande | SciQ | LAMBADA | Mean |
|---|---|---|---|---|---|---|---|---|
| ExeaLM 0.5B | **0.309** | 0.396 | 0.233 | 0.585 | **0.489** | **0.685** | **0.269** | **0.424** |
| Qwen3.5-style 0.5B | 0.303 | **0.409** | 0.233 | **0.596** | 0.484 | 0.677 | 0.249 | 0.421 |

At 2B training tokens both models are near chance on ARC-Challenge and WinoGrande; the means differ by 0.002, within
what one seed can tell apart. Benchmarks start to separate models later in the curve (the 3B pair).

Both seeds, with the plain Llama (`bench_<model>_500m_4.1x_s0.json` and `_s1.json`; mean of the two seeds):

| | HellaSwag | ARC-Easy | ARC-Challenge | PIQA | WinoGrande | SciQ | LAMBADA | Mean (seed 0, seed 1) |
|---|---|---|---|---|---|---|---|---|
| ExeaLM 0.5B | **0.309** | 0.402 | **0.236** | 0.584 | 0.495 | **0.699** | **0.274** | **0.429** (0.424, 0.433) |
| Qwen3.5-style 0.5B | 0.304 | **0.408** | 0.234 | **0.591** | 0.493 | 0.667 | 0.254 | 0.421 (0.422, 0.421) |
| Llama 0.5B | 0.298 | 0.393 | 0.233 | 0.584 | **0.508** | 0.647 | 0.255 | 0.417 (0.416, 0.417) |

ExeaLM's mean is ahead of the Qwen3.5-style model's in both seeds (by 0.002 and 0.012) and ahead on 5 of 7 tasks. The
largest gaps are SciQ (0.032; one seed's standard error on its 1,000 questions is about 0.015) and LAMBADA (0.020, about
0.006), which asks for the last word of a passage and so rewards reading the passage before it. ARC-Challenge and
WinoGrande stay near chance for all three models.

## 21. Keeping the streams' gain without their cost on windows alone (jobs 56090436-41, 56091069)

The stream phase (rows of four windows of one document in the last 20% of steps, at the same tokens per step) gives
ExeaLM most of its long-document lead, but each step then reads a quarter as many documents and three of every four
windows have the window before in memory. On windows read alone it costs 0.007 at 0.1B. Three ways to keep the gain
without the cost, at 0.1B, scored as sections 9 and 10 (`text_mix100m.json`, `stream8_mix100m.json`,
`text_merge100m.json`, `stream8_merge100m.json`):

| ExeaLM 0.1B | Seeds | Windows alone, 18 sets | 4,096-token documents, streamed | Same, windows alone |
|---|---|---|---|---|
| No stream phase | 3 | **1.1496** | 1.1321 | **1.1874** |
| **Stream phase, half of every step's micro-batches streams (`--stream_mix 0.5`), last 20%** | 3 | 1.1532 | 1.0969 | 1.1910 |
| The same over the last 40% (as many stream tokens as the study's recipe) | 3 | 1.1546 | 1.0956 | 1.1922 |
| The study's stream phase, last 20% | 3 | 1.1566 | 1.0955 | 1.1941 |
| The study's run and the no-stream run, weights averaged 50/50 | 3 | 1.2085 | 1.1523 | 1.2457 |
| Qwen3.5-style, for scale | 3 | 1.1904 | 1.1566 | 1.2297 |

- Mixing over the last 20% (3 seeds: 1.1516, 1.1539, 1.1541 on windows alone; 1.0946, 1.0989, 1.0973 streamed)
  gives back 0.0034 of the 0.0070 the stream phase costs on windows alone and gives up 0.0014 of its 0.037 gain on long
  documents: a better balance, not a free one. The first seed alone suggested 0.005 and no loss.
- Averaging the weights of the two runs, which part only in their last 20%, is worse than either: they are not in one
  basin after the stream phase.
- The recipe from here mixes (docs/decisions.md, 26 November, 7 p.m.): the 3B ExeaLM from its next job, before its
  stream phase, and the second long 0.1B seed at 111 tokens per parameter from step 28,400. The finished runs and the
  first long seed keep the old recipe.

## 22. Distillation into the weights at 0.1B: no gain (jobs 56088613-4, `text_mix20distill.json`)

`experiments/distill`: Qwen3.5-9B-Base continued 200,000 openings of 128 tokens from the training mix, and the 0.1B
ExeaLM was pretrained with this teacher text as 30% of its tokens (same steps, seed 0). Against the same seed trained
normally: 1.1592 against 1.1549 on windows alone, 1.0986 against 1.0939 on 4,096-token documents streamed. Teacher text
in place of a third of the data does not help a 0.1B model on held-out human text at 0.4B tokens. What a frontier model
knows goes in the library instead (section 23 when it is built).

## 23. The 3B pair in training (seed 0; the training log's held-out windows)

`runs/exealm_3b_4.1x_s0/log.jsonl` and `runs/qwen_3b_4.1x_s0/log.jsonl`: the fixed-code 3B pair (section 17), 45,688
steps each, scored every 500 steps on the same held-out windows (the training-time check, not section 9's 18-set
reading, which runs on the finished models). Bits per byte:

| Step | ExeaLM 3B | Qwen3.5-style 3B | Lead |
|---|---|---|---|
| 1,000 | **1.1465** | 1.3324 | 0.186 |
| 2,000 | **1.0194** | 1.0741 | 0.055 |
| 3,000 | **0.9482** | 0.9819 | 0.034 |
| 4,000 | **0.9104** | 0.9367 | 0.026 |
| 5,000 | **0.8876** | 0.9100 | 0.022 |
| 6,000 | **0.8693** | 0.8893 | 0.020 |
| 7,000 | **0.8548** | 0.8725 | 0.018 |
| 7,500 | **0.8498** | 0.8659 | 0.016 |
| 8,000 | **0.8446** | 0.8601 | 0.016 |
| 10,000 | **0.8288** | 0.8412 | 0.012 |
| 12,000 | **0.8175** | 0.8268 | 0.009 |
| 14,000 | **0.8080** | 0.8158 | 0.008 |
| 14,500 (27 November, 1:40 p.m.) | **0.8063** | 0.8133 | 0.007 |

The lead narrows as training goes on, by about 0.001 every 1,000 steps since step 8,000. Early leads shrink like this
at every size in the study, and the finished models settle it: at 0.5B the in-run lead ended at 0.004 and the 18-set
lead at 0.005 (section 20). If the 3B lead keeps shrinking at this rate, ExeaLM and the Qwen3.5-style model will be
about even on windows read alone before the stream phase. Long documents, where the lead held at every finished size
(3.6% to 5.4%), are read on the finished models. ExeaLM's stream phase starts near step 36,550 (early on 29 November) with the
mixed recipe of section 21; both runs end near the evening of 29 November.

## 24. The long pair at 111 tokens per parameter, finished (2 seeds; eval job 56094555, `text_111x.json`, `stream8_111x.json`)

The 0.1B pair of section 12, trained for 39,735 steps (10.4B tokens, 111 tokens per parameter, the 9B plan's ratio),
two seeds of each. ExeaLM seed 0 has the study's stream phase; seed 1 switched to the mixed stream phase of section 21
at step 28,400, before its stream phase began at step 31,788. Bits per byte, read as sections 9 and 10:

| | ExeaLM s0 (study's stream phase) | ExeaLM s1 (mixed) | Qwen3.5-style s0 | Qwen3.5-style s1 |
|---|---|---|---|---|
| Windows of 512 read alone, 18 sets | 0.9599 | **0.9559** | 0.9629 | 0.9640 |
| 4,096-token documents read as a stream, windows 1-7 | 0.9020 | **0.9019** | 0.9438 | 0.9487 |
| The same documents, every window alone | 0.9925 | **0.9889** | 0.9962 | 0.9975 |

- ExeaLM is ahead in both seeds on all three readings. On windows alone the mean lead is 0.0055 (0.958 against 0.963):
  0.003 with the study's stream phase, 0.008 with the mixed one, one seed each.
- On long documents read as a stream it is 0.044 ahead (0.902 against 0.946, 4.7%), in both seeds.
- Against section 9 at the same size and 4.1 tokens per parameter (1.157 against 1.190), the lead on windows alone
  falls from 0.034 to 0.006 when the same model trains 27 times longer, as it falls with size (section 20). The
  long-document lead falls less, from 0.061 to 0.044.

The training log's 512 held-out windows (the in-run check; lead = Qwen3.5-style minus ExeaLM) and section 12's step
multiplier (how many times ExeaLM's steps the Qwen3.5-style run needs to reach the same score):

| Step | Lead, s0 | Lead, s1 | Multiplier, s0 | Multiplier, s1 |
|---|---|---|---|---|
| 5,000 | 0.015 | 0.016 | 1.18 | 1.21 |
| 10,000 | 0.009 | 0.012 | 1.21 | 1.31 |
| 15,000 | 0.010 | 0.011 | 1.30 | 1.30 |
| 20,000 | 0.008 | 0.011 | 1.34 | 1.41 |
| 31,500 (stream phase and learning-rate decay begin) | 0.009 | 0.011 | | |
| 33,000 | 0.001 | 0.008 | | |
| 36,000 | 0.003 | 0.006 | | |
| 39,735 (end) | 0.002 | 0.007 | not reached | not reached |

- Through step 20,000 the multiplier rises, from 1.18 to 1.34 and 1.41: the Qwen3.5-style model needs a third more
  steps, and more as training goes on. After step 20,000 the matching step falls in the Qwen3.5-style run's own
  learning-rate decay (from step 31,788), where its score drops fast, so the multiplier no longer measures the same
  thing and is left out. At the end, the finished Qwen3.5-style runs never reach ExeaLM's final score.
- The study's stream phase cost seed 0 most of its lead within 1,500 steps (0.009 to 0.001). The mixed phase kept seed 1
  at 0.006 to 0.008 through the decay. This is the result behind section 21's recipe change, at the 9B's ratio.
**The attention-only control at the same length** (job 56087569, eval job 56104635, `text_strong_111x.json`,
`stream8_strong_111x.json`). ExeaLM with its TONN layer replaced by attention with the same four changes, trained the
same 39,735 steps with the study's stream phase, like ExeaLM seed 0. One seed each:

| 0.1B, 111 tokens per parameter, seed 0 | ExeaLM | Attention in its place | Qwen3.5-style |
|---|---|---|---|
| Windows of 512 read alone, 18 sets | 0.9599 | **0.9580** | 0.9629 |
| 4,096-token documents read as a stream, windows 1-7 | 0.9020 | **0.8982** | 0.9438 |
| The same documents, every window alone | 0.9925 | **0.9907** | 0.9962 |

- Trained as long per parameter as the 9B would be, the control is ahead of ExeaLM on all three readings: on 17 of
  18 sets read alone (by 0.0019 on the mean) and on all 18 read as a stream (by 0.0017 to 0.0070, 0.0038 on the mean).
  At 4.1 tokens per parameter ExeaLM was ahead by about 0.1% (section 19).
- What this control measures: the TONN layer as a plain text layer. The study's models train without the memory
  circuits (library search, board, weight edits), which are TONN circuits; an attention layer cannot hold them, so
  the control has no counterpart to them and this test says nothing about them.
- Both stay well ahead of the Qwen3.5-style model on long documents (0.898 and 0.902 against 0.944). That lead comes
  from the four attention changes and the stream training, which the control shares.
- One seed each, and ExeaLM's seed 1 with the mixed stream phase (0.9559 alone, 0.9019 streamed) shows the stream
  recipe moves the numbers by as much as the gap. The 0.5B three-way run at 20 tokens per parameter (Saturday
  morning) and the 0.25B control (3 seeds, being scored) add two more readings. The 9B keeps its TONN layer, for its
  memory circuits (`docs/decisions.md`, 27 November).

## 25. Benchmarks with the library at 0.1B, closed and open book (jobs 56094538, 56095280; `openbook/`)

The lm-eval harness, zero-shot, on four 0.1B models (seed 0): the curve's ExeaLM and Qwen3.5-style models (section 9)
and the same two after RAG post-training (section 11). Closed book is the question alone. Open book places the top
passage of each library tier before the question (384 tokens each: the written tier, `data/library/v1/bm25`, and the
public human tier, `bm25_pool_p100000`), read as a stream of 512-token windows as in section 10. The search leaves out
the 10 written entries that share text with a benchmark question (`--rag_exclude`). Both models search the same
library with the same query. Accuracy, normalized where the task has it:

| 0.1B, seed 0 | SciQ | ARC-Easy | ARC-Challenge | MMLU | HellaSwag | Mean without MMLU |
|---|---|---|---|---|---|---|
| ExeaLM, closed | 0.591 | 0.330 | 0.205 | 0.232 | 0.278 | 0.351 |
| ExeaLM, open | 0.592 | 0.337 | 0.206 | 0.231 | 0.281 | 0.354 (+0.003) |
| Qwen3.5-style, closed | 0.547 | 0.341 | 0.226 | 0.231 | 0.275 | 0.347 |
| Qwen3.5-style, open | 0.528 | 0.340 | 0.228 | 0.234 | 0.274 | 0.343 (-0.005) |
| ExeaLM with RAG post-training, closed | 0.697 | 0.349 | 0.212 | 0.231 | 0.279 | 0.384 |
| ExeaLM with RAG post-training, open | **0.700** | **0.361** | **0.229** | 0.231 | **0.281** | **0.393** (+0.008) |
| Qwen3.5-style with RAG post-training, closed | 0.654 | 0.357 | 0.205 | 0.231 | 0.274 | 0.372 |
| Qwen3.5-style with RAG post-training, open | 0.630 | 0.358 | 0.192 | 0.231 | 0.276 | 0.364 (-0.008) |

- Given the library, ExeaLM gains on 4 of the 5 tasks, more after RAG post-training (+0.012 on ARC-Easy, +0.017 on
  ARC-Challenge). The Qwen3.5-style model loses with it, most on SciQ (-0.019, and -0.024 after RAG post-training).
  One seed, at 0.1B: ARC-Challenge has 1,172 questions, so one standard error on an accuracy is about 0.012, and no
  single gain here is beyond two standard errors. It is a lead, consistent with section 11's held-out questions.
- MMLU stays at chance (0.23 for every model, 4 choices), open or closed. A 0.1B model cannot use the passages for it.
- The largest change is closed-book SciQ after RAG post-training: +0.106 for ExeaLM and +0.107 for the Qwen3.5-style
  model, with no library at test time. It comes from post-training on the library's questions and answers. Their overlap
  with the benchmarks has not been checked yet (only the search's has), so we do not claim it.
- The test that counts is the same reading at 0.5B and 3B after post-training on library v2 (section 26).

## 26. Next

The 0.5B three-way run at 20 tokens per parameter (Saturday morning), the 3B pair (Sunday evening), library v2's check
against the benchmarks and RAG post-training on it at 0.5B (jobs 56104634-9, running) and on a mid-run 3B checkpoint,
then a scaling fit and the 9B projection against section 18's bar.
