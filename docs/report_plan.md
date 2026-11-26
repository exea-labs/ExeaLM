# The funding report: what it will claim, and the run behind each claim

At the end of the NAIRR hours, one report makes the case that a 9B ExeaLM, specialized for research and code, is worth
funding: for investors and for grant programs. Every number in it comes from a run in `results/expanse/`, with its
seeds and setting stated. This page tracks which runs feed which claim, so nothing the report needs is left unrun.
Status as of 25 November 2026, night.

| # | Claim | What measures it | Runs | Status |
|---|---|---|---|---|
| 1 | The code is correct and trains at scale on H100s | correctness against the CPU, exact resume, 1 and 4 GPUs, 4-GPU speed-up 3.68 | first job | **done** (results §1) |
| 2 | ExeaLM beats the strongest hybrid design (Qwen3.5-style) under the same conditions | bits per byte on 18 held-out sets, 3 seeds | lead check 0.1B | **done**: 1.393 against 1.420, all 54 set-seed pairs |
| 3 | TONN layers are the reason, not borrowed tricks | ExeaLM and the Qwen3.5-style model with and without value residual and XSA | ablations 0.1B | **done**: ExeaLM ahead both ways, all 18 sets |
| 4 | The lead holds as the model grows | the gap at 0.1B, 0.25B, 0.5B and 3B | the curve, the 3B pair | running (0.1B); 0.25B, 0.5B queued; 3B after its rate checks |
| 5 | The lead holds when trained as long as the 9B would be | the gap at 111 tokens per parameter against 4.1 | the long pair (0.1B) | queued |
| 6 | Specialized for research and code | bits per byte on research code, code, papers, math at every size; benchmarks at 0.5B and 3B | curve, 3B pair, `eval.sb` | after training |
| 7 | ExeaLM reaches the same quality for less compute | compute multiplier against the Qwen3.5-style model (FLOPs). Quote it in FLOPs only: the curve's Qwen3.5-style runs used our plain-PyTorch DeltaNet kernel, 2.8 to 3.8 times slower than the fla kernel (`results/expanse/README.md` section 6), so a GPU-hours multiplier from their measured speed would flatter ExeaLM | curve, `report.py` | after training |
| 8 | Long documents: ExeaLM uses what came before | bits per byte by window, 8 windows read as a stream | `eval.sb` (stream_eval) | after training |
| 9 | A 9B can be trained, and what it costs | 9B speed and memory on 1, 2 and 4 nodes; price for 1T tokens | `nine_b.sb`, `report.py` | to run (about 33 GPU-hours); estimate now about 101,000 GPU-hours |
| 10 | A library distilled from a strong model makes ExeaLM better at research and code | bits per byte on held-out library tasks with the library attached, removed, or the right entry pasted | library build (Qwen3-30B-A3B), `graft.py`, post-training, `experiments/library/score.py` | library to build (about 6-10 GPU-hours); post-training after the curve |
| 11 | Agents on a board beat a single bigger model | 3 agents at 0.1B sharing a board against one 0.25B model (2.6 times each agent), on tasks spread over more context than one window holds: long papers, multi-file code | `graft.py` with AG slots, post-training in swarms, a multi-window task set | to design and run after the curve |
| 12 | The layout is the best ExeaLM | fewer TONN layers against the strong attention baseline | layout test 0.1B | running |

## What the report will not claim

Results at 9B or 27B, or "3 x 9B beats 27B": the allocation cannot train those. Claims 10 and 11 are measured at small
scale, and the report says so; what they show for 9B is an extrapolation, labelled as one.

## Hours

Used: about 60 of 2,000. Planned: the curve and long pair about 240, the 3B pair about 1,090, scoring and the 9B
measurement about 136, the library and the board about 100. About 370 left in reserve.
