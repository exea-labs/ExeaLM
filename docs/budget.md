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
