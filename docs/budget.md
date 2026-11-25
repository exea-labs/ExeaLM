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
80 GB each). The plan changed three times before any of it ran (18 November, and twice on 24 November). This is the
plan of the evening of 24 November; the earlier ones are at the end of this section. The decisions behind it are in
[decisions.md](decisions.md).

### What the runs are for

Numbers for the next grant application, for a 9B ExeaLM: ExeaLM against the strongest same-size baseline we can
build, a Qwen3.5-style model (Gated DeltaNet and gated attention, 3 to 1), at 3B and at three smaller sizes on one
curve, trained the same way on the same tokens. Every hour goes to that. The board, the library and weight-alter are
post-training work and run elsewhere (the MI300X).

What the application will cite: the gap at each size in bits per byte, every row read alone and on long documents read
as a stream; the gap on downstream benchmarks at 0.5B and 3B; how many GPU-hours the baseline needs to match ExeaLM
(the compute multiplier, read off the curve); and the 9B model's measured speed and memory on H100 nodes, which is
the price in the application. ExeaLM already leads this baseline at 13M (by 0.011 bits per byte, 10 seeds) and at 0.1B
(by 0.044, 3 seeds; [results/funding](../results/funding/README.md) sections 1 and 3).

### The two models

Both: the 32k BPE tokenizer trained on the study's mix, tied embeddings, RMSNorm before every sublayer, a SwiGLU MLP
8/3 of the width after every layer, and attention layers with 64-wide heads, RoPE, QK-norm, a sigmoid output gate,
value residual ([ResFormer](https://arxiv.org/abs/2410.17897)) and exclusive self-attention
([XSA](https://arxiv.org/abs/2603.09078)). At 3B: width 2,560, 34 layers, 40 heads, MLP 6,824.

| | ExeaLM | Qwen3.5-style |
|---|---|---|
| Layers that are not attention | TONN, every third layer (12 of 34 at 3B) | Gated DeltaNet, 3 of every 4 (26 of 34) |
| Inside them | d / 32 circuits of 32 channels (80 at 3B), drawn from the frozen starting mix: short convolutions (weight 4), delays and gated units (0.25), products, running sums, gated running sums, running sums of products, softmax pooling, the Mamba-like selective state space and oscillators (1 each); lags 0 to 128; no rotary-attention circuits | 64-wide heads, a short convolution of 4, the gated delta rule in chunks of 64, a gated, normalized output |
| Read as a stream | TONN state and the attention keys and values of the window before | delta-rule state and convolution tail, and the attention keys and values of the window before |
| Parameters at 0.1B, 0.25B, 0.5B, 3B | 93.4M, 245.3M, 497.1M, 2,901.5M | 95.7M, 252.1M, 511.3M, 2,986.2M |

Not in either: board, library or weight-alter slots, swarms, circuit search during training.

Training, the same for both: one 512-token window per row for the first 80% of steps, then rows of 2,048 tokens read
as four streamed windows with the gradient through all four for the last 20% (`--stream 4 --stream_from 0.8`); steps of 262,144 tokens; AdamW (weight decay 0.1, gradient clipping 1.0) unless Muon wins at
0.1B first; bf16; a learning rate that warms up, holds and decays linearly over the last 20% to 10% of its peak, the
peak from each model's own short checks.

### The runs

| Part | Run | Parameters | Tokens | Tokens per parameter | Steps | GPUs | GPU-hours |
|---|---|---|---|---|---|---|---|
| 3B | `exealm_3b_4.1x_s0` | 2,901.5M | 11.9B | 4.1 | 45,381 | 4 | 710 |
| 3B | `qwen_3b_4.1x_s0` | 2,986.2M | 11.9B | 4.0 | 45,381 | 4 | 731 |
| curve | `exealm_100m_4.1x_s0` | 93.4M | 0.4B | 4.1 | 1,461 | 1 | 1 |
| curve | `qwen_100m_4.1x_s0` | 95.7M | 0.4B | 4.0 | 1,461 | 1 | 1 |
| curve | `exealm_100m_4.1x_s1` | 93.4M | 0.4B | 4.1 | 1,461 | 1 | 1 |
| curve | `qwen_100m_4.1x_s1` | 95.7M | 0.4B | 4.0 | 1,461 | 1 | 1 |
| curve | `exealm_100m_4.1x_s2` | 93.4M | 0.4B | 4.1 | 1,461 | 1 | 1 |
| curve | `qwen_100m_4.1x_s2` | 95.7M | 0.4B | 4.0 | 1,461 | 1 | 1 |
| curve | `exealm_250m_4.1x_s0` | 245.3M | 1.0B | 4.1 | 3,837 | 1 | 7 |
| curve | `qwen_250m_4.1x_s0` | 252.1M | 1.0B | 4.0 | 3,837 | 1 | 8 |
| curve | `exealm_250m_4.1x_s1` | 245.3M | 1.0B | 4.1 | 3,837 | 1 | 7 |
| curve | `qwen_250m_4.1x_s1` | 252.1M | 1.0B | 4.0 | 3,837 | 1 | 8 |
| curve | `exealm_250m_4.1x_s2` | 245.3M | 1.0B | 4.1 | 3,837 | 1 | 7 |
| curve | `qwen_250m_4.1x_s2` | 252.1M | 1.0B | 4.0 | 3,837 | 1 | 8 |
| curve | `exealm_500m_4.1x_s0` | 497.1M | 2.0B | 4.1 | 7,774 | 4 | 21 |
| curve | `qwen_500m_4.1x_s0` | 511.3M | 2.0B | 4.0 | 7,774 | 4 | 21 |
| curve | `exealm_500m_4.1x_s1` | 497.1M | 2.0B | 4.1 | 7,774 | 4 | 21 |
| curve | `qwen_500m_4.1x_s1` | 511.3M | 2.0B | 4.0 | 7,774 | 4 | 21 |
| | **The 18 runs** | | | | | | **1,577** |
| | First job on Expanse | | | | | | 6 |
| | The 9B model: speed and memory on 1, 2 and 4 nodes | | | | | | 36 |
| | Learning-rate checks | | | | | | 64 |
| | Evaluation: held-out text, long documents, benchmarks | | | | | | 100 |
| | **Reserve of the 2,000** | | | | | | **217** |

`python experiments/expanse/plan.py` prints this table and the lines that submit each run
([experiments/expanse](../experiments/expanse/README.md)).

- **3B.** The headline: 11.9B tokens each, 4.1 tokens per parameter, what the allocation can afford for two 3B
  models.
- **The curve.** The same recipe at 0.1B, 0.25B and 0.5B, at the same 4.1 tokens per parameter, so all four sizes lie
  on one curve and the gap can be read as a trend: three seeds at 0.1B and 0.25B, two at 0.5B. It costs 136 GPU-hours.
- **Rates.** One AMD MI300X in bf16 ([results/funding](../results/funding/README.md), section 6): 56 effective TFLOP/s
  for ExeaLM at 0.1B, 81 at 0.5B. GPU-hours = 6 × parameters × tokens / (TFLOP/s × 3600 × 10¹²). The Qwen3.5-style
  model is assumed to run at ExeaLM's rate. Its Gated DeltaNet is plain PyTorch here; if it is much slower on the H100,
  it moves to the flash-linear-attention kernels, as a fair baseline should. The first job measures both.
- **Four GPUs.** A node is charged 4 GPU-hours per hour. A run on four GPUs that is S times as fast as on one costs
  4 / S times the hours above. The 3B pair is 1,441 of the 1,577: at S = 3.5 it costs about 206 more, nearly the
  whole reserve.
- **The 9B model** (`9b`: 9.06B parameters, width 4,096, 42 layers): a few hundred steps on one, two and four nodes,
  for memory per GPU, tokens per second and scaling across nodes. It needs FSDP, which `exealm/train/pretrain.py`
  does not have yet.

### The data

About 23B tokens, so each 3B model reads about half of it and nothing twice: English research and reference text from
[LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4) (papers, arXiv, Wikipedia, books,
StackExchange, math), code from LLM-jp v4 and [IFM Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning),
math from [IFM Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning), web and QA text from
[IFM TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2), and Japanese from LLM-jp v4. It is published with its
tokenizer as `exea-labs/exealm-pretrain-v1` on Hugging Face, the Stack-derived code in a gated companion repo, and
the cluster pulls it with `experiments/data_mix/fetch.sh`. Post-training data and the library go to
`exea-labs/exealm-post-v1` and `exea-labs/exealm-library-v1`.

### Before the long runs

1. **The data** on Hugging Face, and the 32k tokenizer trained on it.
2. **The first job** ([experiments/expanse](../experiments/expanse/README.md)): correctness on the H100, both models'
   real speeds at every size, the speed-up of four GPUs, and the lead check at 0.1B.
3. **The learning-rate checks**, size by size.
4. **FSDP**, before the 9B measurement. No other run needs it.

Then the curve, smallest first, then the 3B pair.

### Reading the results

- **Equal tokens, every row alone.** `python experiments/eval_text.py runs/qwen_3b_4.1x_s0 runs/exealm_3b_4.1x_s0`.
- **Long documents, read as a stream.** `python experiments/stream_eval.py ... --windows 8`.
- **The compute multiplier.** At each size, the GPU-hours the Qwen3.5-style model would need to reach ExeaLM's loss
  (from a power law fitted to its four points), divided by ExeaLM's measured GPU-hours.
- **Benchmarks** at 0.5B and 3B.

What would count: ExeaLM ahead at every size, on every seed, at equal tokens; the gap not shrinking from 0.1B to 3B;
the multiplier above 1. It is evidence for the 9B run, not proof: 9B is three times the largest model trained.

### If fewer hours fit

Shorten the 3B pair (each billion tokens fewer saves about 120 GPU-hours for the pair). Then drop the second seed at
0.5B (about 42). Never drop the Qwen3.5-style run of a pair. The first use of the reserve, if it is not needed: the
0.1B pair trained to 111 tokens per parameter, the 9B run's ratio (about 60 GPU-hours).

### The plans of 24 November (morning) and 18 November

For the record. Nothing of them ran.

| Plan | Runs | GPU-hours |
|---|---|---|
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
