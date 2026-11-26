# Compute budget

What the planned runs cost, from measured throughput where we have it and stated assumptions where we don't.
Update this page when a GPU benchmark replaces an assumption.

## The main run: 0.1B parameters on 100B tokens

Training compute is about 6 × parameters × tokens = 6 × 0.1e9 × 100e9 = **6 × 10¹⁹ FLOPs**. Retrieval on adds
about half again (measured on the Mac: the 2:1 hybrid costs 1.01× a same-size Llama per token with the memory circuits
idle, 1.49× with retrieval on).

| Where | Throughput | Time for 100B tokens | Source |
|---|---|---|---|
| Mac Studio (M3 Ultra, MPS, fp32) | ~12,000 tokens/s, retrieval off; ~8,000 on | ~97 days; ~144 days | measured, `results/speed_hybrid21/` |
| One H100 or MI300X at ~300 TFLOPs sustained | ~500,000 tokens/s | ~56 GPU-hours; ~85 with retrieval | assumed, DATA.md |
| 8 of the same | | ~7 hours; ~11 hours | |

The 300 TFLOPs figure is an assumption. Attention runs as fused kernels on NVIDIA and AMD GPUs and TONN's circuits do
not yet, so real throughput could be up to 2× lower. **A one-hour benchmark on the 0.1B preset should come first**
(about $3–4 on one GPU).

## The scaling study: 2,000 GPU-hours on SDSC Expanse

The NAIRR Pilot approved 2,000 GPU-hours for ExeaLM on the SDSC Expanse AI Resource (nodes of four NVIDIA H100s,
80 GB each). The plan changed five times before any of it ran (18 November, twice on 24 November, twice on 25
November). This is the plan of 25 November; the earlier ones are at the end of this section. The decisions behind it
are in [decisions.md](decisions.md).

### What the runs are for

Numbers that make funders (grant programs and investors) confident in a 9B ExeaLM on 1T tokens: ExeaLM against a base Qwen3.5-style model (Gated DeltaNet
and gated attention, 3 to 1) of the same size at four sizes from 0.1B to 3B, on one curve, trained on the same tokens,
with a plain Llama beside them from 0.1B to 0.5B as the reference everyone knows. Every hour goes to that. The board,
the library and weight-alter are post-training work and run elsewhere (the MI300X).

What the application will cite: the gap at each size in bits per byte, every row read alone and on long documents read
as a stream; the gap on downstream benchmarks at 0.5B and 3B; how many GPU-hours the Qwen3.5-style model needs to
match ExeaLM (the compute multiplier, read off the curve); whether the gap holds when the 0.1B pair is trained to the
9B run's 111 tokens per parameter; and the 9B model's measured speed and memory on H100 nodes, which is the price in
the application. `experiments/expanse/report.py` writes all of it from the scores.

What we have so far, all at 0.1B. On the MI300X with ExeaLM as the study trains it, on older data and an older 32k
tokenizer ([results/finalize/final_text.json](../results/finalize/final_text.json), 3,000 steps, 2 seeds, mean of four
held-out sets), ExeaLM reached 1.387 bits per byte against 1.460 for the base Qwen3.5-style model and 1.446 for a gated Llama.
With value residual and XSA added to the baselines, the Qwen3.5-style model reached 1.426 (one seed), still behind
ExeaLM, while the gated Llama reached 1.369, ahead of it. On older data without value residual or XSA
([results/funding](../results/funding/README.md), section 3, 3 seeds), the memory-free hybrid reached 1.413 against
1.457 for the Qwen3.5-style model and 1.509 for the plain Llama.

### What a funder will ask, and what answers it

Every run is in the plan because it answers one of these. A run that answers none does not belong in the 2,000 hours.

| Question | What answers it | Where |
|---|---|---|
| Is ExeaLM better than a modern design, under the same conditions? | ExeaLM against the Qwen3.5-style model at every size: the same data, tokens, steps, optimizer and schedule; only ExeaLM reads streams in its last 20% | the 3B pair, the curve |
| Does the lead hold as the model grows? | The gap at 0.1B, 0.25B, 0.5B and 3B, and whether it shrinks | the curve, the 3B pair |
| Could the lead be noise? | Three seeds at 0.1B and 0.25B, two at 0.5B; on how many seeds ExeaLM leads | the curve, `report.py` |
| Does it hold when trained as long as the 9B would be? | The 0.1B pair at 111 tokens per parameter against the same pair at 4.1 | the long pair |
| Does ExeaLM reach the same quality for less? | The compute multiplier, in FLOPs and in GPU-hours at each model's measured speed | the curve, `report.py` |
| How does it compare with the model everyone knows? | The plain Llama beside them at 0.1B to 0.5B | the curve |
| Is it good at what it is for, research and code? | Bits per byte on each held-out set (research code, code, papers, math, ...); benchmarks at 0.5B and 3B | `eval.sb`, `report.py` |
| Does it use long documents? | Long documents read as a stream, 8 windows of 512 tokens | `eval.sb` |
| Can the team train a 9B, and what will it cost? | The 9B model's speed and memory on 1, 2 and 4 nodes; the 9B run's price at that speed | `nine_b.sb`, `report.py` |
| Can someone else check it? | Public data and tokenizer, the code on GitHub, every run's log and model on Hugging Face (`exea-labs/exealm-nairr-runs`, private until we publish it) | `train.sb`, `eval.sb` |

### The three models

All three: the 32k BPE tokenizer trained on the study's mix, tied embeddings, RMSNorm before every sublayer, a SwiGLU
MLP 8/3 of the width after every layer, attention with 64-wide heads and RoPE. At 3B: width 2,560, 34 layers, 40
heads, MLP 6,824.

| | ExeaLM | Base Qwen3.5-style | Plain Llama |
|---|---|---|---|
| Attention layers | 3 of every 4 (25 of 34 at 3B), with QK-norm, a sigmoid output gate, value residual ([ResFormer](https://arxiv.org/abs/2410.17897)) and exclusive self-attention ([XSA](https://arxiv.org/abs/2603.09078)) | 1 of every 4 (8 of 34), with QK-norm and the output gate | every layer, without QK-norm, gate, value residual or XSA |
| The other layers | TONN, every fourth layer (9 of 34): d / 32 circuits of 32 channels (80 at 3B) from the frozen starting mix: short convolutions (weight 4), delays and gated units (0.25), products, running sums, gated running sums, running sums of products, softmax pooling, the Mamba-like selective state space and oscillators (1 each); lags 0 to 128; no rotary-attention circuits | Gated DeltaNet, 3 of every 4 (26 of 34): 64-wide heads, a short convolution of 4, the gated delta rule in chunks of 64, a gated, normalized output | none |
| Rows in training | one 512-token window for the first 80% of steps, then 2,048 tokens read as four streamed windows (TONN state and the previous window's keys and values carried, the gradient through all four) | one 512-token window throughout | one 512-token window throughout |
| Parameters at 0.1B, 0.25B, 0.5B, 3B | 93.8M, 247.4M, 500.3M, 2,921.2M | 95.7M, 252.1M, 511.3M, 2,986.2M | 89.7M, 234.8M, 474.1M (not trained at 3B) |

Not in any of them: board, library or weight-alter slots, swarms, circuit search during training.

Training, the same for all three: steps of 262,144 tokens; AdamW (weight decay 0.1, gradient clipping 1.0); bf16; a
learning rate that warms up, holds and decays linearly over the last 20% to 10% of its peak, the peak from each
model's own short checks; the anneal mix for the last 10% of steps.

### The runs

| Part | Run | Parameters | Tokens | Tokens per parameter | Steps | GPUs | GPU-hours |
|---|---|---|---|---|---|---|---|
| 3B | `exealm_3b_4.1x_s0` | 2,973.5M | 12.0B | 4.0 | 45,688 | 4 | 733 |
| 3B | `qwen_3b_4.1x_s0` | 2,986.2M | 12.0B | 4.0 | 45,688 | 4 | 736 |
| curve | `exealm_100m_4.1x_s0` | 95.1M | 0.4B | 4.0 | 1,468 | 1 | 1 |
| curve | `llama_100m_4.1x_s0` | 89.7M | 0.4B | 4.3 | 1,468 | 1 | 1 |
| curve | `qwen_100m_4.1x_s0` | 95.7M | 0.4B | 4.0 | 1,468 | 1 | 1 |
| curve | `exealm_100m_4.1x_s1` | 95.1M | 0.4B | 4.0 | 1,468 | 1 | 1 |
| curve | `llama_100m_4.1x_s1` | 89.7M | 0.4B | 4.3 | 1,468 | 1 | 1 |
| curve | `qwen_100m_4.1x_s1` | 95.7M | 0.4B | 4.0 | 1,468 | 1 | 1 |
| curve | `exealm_100m_4.1x_s2` | 95.1M | 0.4B | 4.0 | 1,468 | 1 | 1 |
| curve | `llama_100m_4.1x_s2` | 89.7M | 0.4B | 4.3 | 1,468 | 1 | 1 |
| curve | `qwen_100m_4.1x_s2` | 95.7M | 0.4B | 4.0 | 1,468 | 1 | 1 |
| curve | `exealm_250m_4.1x_s0` | 250.5M | 1.0B | 4.0 | 3,869 | 1 | 8 |
| curve | `llama_250m_4.1x_s0` | 234.8M | 1.0B | 4.3 | 3,869 | 1 | 5 |
| curve | `qwen_250m_4.1x_s0` | 252.1M | 1.0B | 4.0 | 3,869 | 1 | 8 |
| curve | `exealm_250m_4.1x_s1` | 250.5M | 1.0B | 4.0 | 3,869 | 1 | 8 |
| curve | `llama_250m_4.1x_s1` | 234.8M | 1.0B | 4.3 | 3,869 | 1 | 5 |
| curve | `qwen_250m_4.1x_s1` | 252.1M | 1.0B | 4.0 | 3,869 | 1 | 8 |
| curve | `exealm_250m_4.1x_s2` | 250.5M | 1.0B | 4.0 | 3,869 | 1 | 8 |
| curve | `llama_250m_4.1x_s2` | 234.8M | 1.0B | 4.3 | 3,869 | 1 | 5 |
| curve | `qwen_250m_4.1x_s2` | 252.1M | 1.0B | 4.0 | 3,869 | 1 | 8 |
| curve | `exealm_500m_4.1x_s0` | 508.5M | 2.1B | 4.0 | 7,825 | 4 | 21 |
| curve | `llama_500m_4.1x_s0` | 474.1M | 2.1B | 4.3 | 7,825 | 4 | 13 |
| curve | `qwen_500m_4.1x_s0` | 511.3M | 2.1B | 4.0 | 7,825 | 4 | 22 |
| curve | `exealm_500m_4.1x_s1` | 508.5M | 2.1B | 4.0 | 7,825 | 4 | 21 |
| curve | `llama_500m_4.1x_s1` | 474.1M | 2.1B | 4.3 | 7,825 | 4 | 13 |
| curve | `qwen_500m_4.1x_s1` | 511.3M | 2.1B | 4.0 | 7,825 | 4 | 22 |
| long | `exealm_100m_111x_s0` | 95.1M | 10.4B | 109.6 | 39,735 | 1 | 29 |
| long | `qwen_100m_111x_s0` | 95.7M | 10.4B | 108.9 | 39,735 | 1 | 30 |
| | **The 28 runs** | | | | | | **1,708** |
| | First job on Expanse | | | | | | 6 |
| | The 9B model: speed and memory on 1, 2 and 4 nodes | | | | | | 36 |
| | Learning-rate checks | | | | | | 75 |
| | Evaluation: held-out text, long documents, benchmarks | | | | | | 100 |
| | **Reserve of the 2,000** | | | | | | **76** |

`python experiments/expanse/plan.py` prints this table and the lines that submit each run
([experiments/expanse](../experiments/expanse/README.md)).

- **3B.** The headline: ExeaLM and the Qwen3.5-style model, 12.0B tokens each, 4.1 tokens per ExeaLM parameter,
  1,456 GPU-hours.
- **The curve.** The same recipe at 0.1B, 0.25B and 0.5B with all three models, at the same 4.1 tokens per parameter,
  so all four sizes lie on one curve and the gap can be read as a trend: three seeds at 0.1B and 0.25B, two at 0.5B.
  It costs 179 GPU-hours, the plain Llama 42 of them.
- **The long pair.** ExeaLM and the Qwen3.5-style model at 0.1B trained on 10.4B tokens, 111 tokens per parameter, the
  ratio of the 9B run (1T tokens). Every other run stops at 4.1, so this is the one place the study can show that the
  lead does not close when the models train as long as the funded run would. 59 GPU-hours, about 30 hours on one GPU
  each.
- **Rates.** One AMD MI300X in bf16 ([results/funding](../results/funding/README.md), section 6): 56 effective TFLOP/s
  for ExeaLM at 0.1B and 81 at 0.5B; 85 and 125 for the plain Llama. GPU-hours = 6 × parameters × tokens / (TFLOP/s
  × 3600 × 10¹²). The Qwen3.5-style model is assumed to run at ExeaLM's rate. Its Gated DeltaNet is plain PyTorch
  here; if it is much slower on the H100, it moves to the flash-linear-attention kernels, as a fair baseline should.
  The first job measures all three.
- **Four GPUs.** A node is charged 4 GPU-hours per hour. A run on four GPUs that is S times as fast as on one costs
  4 / S times the hours above. The runs on four GPUs are 1,567 of the 1,693: at S = 3.5 they cost about 224 more, 134
  more than the reserve, so the 3B pair would lose about 1.1B of its 12B tokens.
- **The 9B model** (`9b` as the study trains it: 9.28B parameters, width 4,096, 42 layers, one TONN layer at layer 21): short runs on one, two
  and four nodes, for memory per GPU, tokens per second and scaling across nodes (`experiments/expanse/nine_b.py`,
  sharded with PyTorch's FSDP2; `go.sh nine_b`). At most 33 GPU-hours of the 36. Training the 9B for real will need
  `pretrain.py` to shard the same way, and to average the TONN layers' running output sizes across GPUs, which DDP
  copies from process 0 and FSDP does not.

### The data

The 30B-token slice of data recipe v2.1 ([data_recipe.md](data_recipe.md)), research code first, then general code,
then research papers, with math, web text, Japanese and QA: `slice-30b-r2` of the public Hugging Face dataset
`exea-labs/exealm-pretrain`, 28.5B tokens after deduplication and decontamination, 7.3B of them research code, with its
32k tokenizer and an anneal mix (published 25 November). The first build, `slice-30b` (23.5B tokens, 3.8B research
code), is there too; the study does not read it.
Each 3B model reads 12B of its tokens and nothing twice. The cluster pulls it with `experiments/data_mix/fetch.sh`
(`go.sh data`). Every run of the study uses the same build: each build has its own tokenizer. Post-training data and
the library are `exea-labs/exealm-post` and `exea-labs/exealm-library`.

### Before the long runs

1. **The first job** ([experiments/expanse](../experiments/expanse/README.md)): correctness on the H100, the three
   models' real speeds at every size, the speed-up of four GPUs, and the lead check at 0.1B.
2. **The learning-rate checks**, size by size, every model.
3. **FSDP**, before the 9B measurement. No other run needs it.

Then the curve, smallest first, then the 3B pair. [runbook.md](runbook.md) has every command.

### Reading the results

- **Equal tokens, every row alone.** `python experiments/eval_text.py runs/qwen_3b_4.1x_s0 runs/exealm_3b_4.1x_s0`.
- **Long documents, read as a stream.** `python experiments/stream_eval.py ... --windows 8`. Only ExeaLM was trained
  on streams, so the baselines' streamed numbers show what they get from the state of the window before without that
  training.
- **The compute multiplier.** At each size, the GPU-hours the Qwen3.5-style model would need to reach ExeaLM's loss
  (from a power law fitted to its four points), divided by ExeaLM's measured GPU-hours.
- **Benchmarks** at 0.5B and 3B.
- **All of it at once.** `python experiments/expanse/report.py` (`go.sh report`; `eval.sb` runs it) writes
  `results/expanse/report.md`, `report.json` and `scaling.png`: the curve with every seed, the gap and on how many seeds
  ExeaLM leads, the compute multiplier in FLOPs and in GPU-hours, the long pair, every held-out set, long documents,
  benchmarks, and the 9B run's price at the measured speed.

What would count: ExeaLM ahead of the Qwen3.5-style model at every size, on every seed, at equal tokens; the gap not
shrinking from 0.1B to 3B; the gap of the long pair no smaller than at 4.1 tokens per parameter; ExeaLM ahead of the plain
Llama from 0.1B to 0.5B; the multiplier above 1. It is evidence
for the 9B run, not proof: 9B is three times the largest model trained.

### If fewer hours fit

Shorten the 3B pair (each billion tokens fewer saves about 120 GPU-hours for the pair). Then drop the second seed at
0.5B (about 56). Never drop the Qwen3.5-style run of a pair or the long pair. The first use of the reserve, if it is
not needed: the plain Llama at 0.1B to the same 111 tokens per parameter (about 18 GPU-hours).

### The plans of 24 November and 18 November

For the record. Nothing of them ran.

| Plan | Runs | GPU-hours |
|---|---|---|
| 25 Nov morning | ExeaLM against a plain Llama at 0.1B to 3B, with the base Qwen3.5-style model beside them to 1B, 29 runs | 1,549 |
| 24 Nov evening | ExeaLM against a Qwen3.5-style model with value residual and XSA, both streamed: a 3B pair at 4.1 tokens per parameter and a 0.1B to 0.5B curve, 18 runs | 1,593 |
| 24 Nov morning | ExeaLM against a gated Llama at 0.1B, 0.25B, 0.5B and 1B at 20 tokens per parameter, the 0.1B pair at 50 and 111, a short 3B pair | 1,588 |
| 18 Nov | ExeaLM against a gated Llama: 0.1B at 110, 0.5B at 60, 1B at 15.5, 3B at 1.4 tokens per parameter | 1,503 |

### The plan in the proposal

For the record, at the same MI300X rates. Nothing of it ran.

| Run | Width, layers | Training tokens | GPU-hours |
|---|---|---|---|
| ExeaLM 0.5B (`500m`) | 1280, 22 | 30B | 307 |
| Baseline 0.5B (plain Llama) | 1280, 22 | 30B | 190 |
| ExeaLM 1B (`1b`) | 1664, 26 | 20B | 397 |
| Baseline 1B (plain Llama) | 1664, 26 | 20B | 248 |
| ExeaLM 3B (`3b`) | 2560, 34 | 8B | 477 |
| Evaluation and short post-training of all five | | | 150 |
| Library builds, restarts and reserve | | | 231 |
| **Total** | | | **2,000** |

## What it costs to rent

RunPod on-demand list prices on 2026-11-11, per GPU-hour:

| GPU | Price | GPU-hours for the main run (retrieval on) | Cost |
|---|---|---|---|
| MI300X | $2.39 | ~85 | ~$200 |
| H100 SXM | $2.69 community, $3.99 secure | ~85 | ~$230–340 |
| A100 80GB | $1.39–1.79 | ~250 | ~$350–450 |
| RTX 4090 (8, community) | $0.34 | ~500 | ~$170, about 2.6 days on 8 |

Add 30–50% for restarts, evals and data loading. If the GPU benchmark comes in 2× slower, double these.

## The other runs

- **Data-mix pilots** (DATA.md: a 0.1B model on 20B tokens, three or four mixes): each about a fifth of the main run,
  so ~$40–80 each, ~$100–300 in all.
- **Experiments at `small` size (17M)** run on the Mac Studio: about 40,000–75,000 tokens/s, so 400M tokens in 2–3 hours.
- **Post-training** (SFT on 1–2B tokens, then preference tuning and RL): small next to pretraining, roughly 1–2% of
  its compute for SFT.

## Checkpoints

- Save every 2,000 steps and at the end (`--save_every`); keep the last three and every tenth.
- Exact resume works (`--resume auto`, PR #1), so community or spot GPUs that can be interrupted are usable.
