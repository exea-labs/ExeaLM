# The funding report: what it will claim, and the run behind each claim

At the end of the NAIRR hours, one report makes the case that a 9B ExeaLM, specialized for research and code, is worth
funding: for investors and for grant programs. Every number in it comes from a run in `results/expanse/`, with its
seeds and setting stated. This page tracks which runs feed which claim, so nothing the report needs is left unrun.
Status as of 27 November 2026, 2 a.m.

| # | Claim | What measures it | Runs | Status |
|---|---|---|---|---|
| 1 | The code is correct and trains at scale on H100s | correctness against the CPU, exact resume, 1 and 4 GPUs, 4-GPU speed-up 3.68 | first job | **done** (results §1) |
| 2 | ExeaLM beats the strongest hybrid design (Qwen3.5-style) under the same conditions | bits per byte on 18 held-out sets, 3 seeds | lead check 0.1B | **done**: 1.393 against 1.420, all 54 set-seed pairs |
| 3 | TONN layers beat Gated DeltaNet layers as a hybrid's non-attention layer | ExeaLM and the Qwen3.5-style model with and without value residual and XSA | ablations 0.1B | **done** for the every-fourth layout: ExeaLM ahead both ways, all 18 sets. With one TONN layer in the middle, most of the lead over the Qwen3.5-style model comes from the attention layers (13 of 14); a Llama with all four attention changes is 0.0009 behind it (results §5). The report must not say the TONN layer is the reason for the lead |
| 4 | The lead holds as the model grows | the gap at 0.1B, 0.25B, 0.5B and 3B | the curve, the 3B pair | **0.1B, 0.25B and 0.5B done** (results §9, §13, §20): ahead of the Qwen3.5-style model on all 54, 54 and 36 set-seed pairs; on windows read alone the lead shrinks with size, 0.034, 0.011, 0.005. 3B training (in-run 0.850 against 0.866 at step 7,500; ends 29 November). The report shows this trend as it is, with the long-document lead beside it |
| 5 | The lead holds when trained as long as the 9B would be | the gap at 111 tokens per parameter against 4.1 | the long pair (0.1B) | **done, 2 seeds** (results §24): 0.958 against 0.963 on windows alone, 0.902 against 0.946 on long documents, ahead in both seeds; the windows-alone lead shrinks with length (0.034 at 4.1 to 0.006). A 0.5B three-way run at 20 tokens per parameter is training (decisions, 27 November) |
| 6 | Specialized for research and code | bits per byte on research code, code, papers, math at every size; benchmarks at 0.5B and 3B | curve, 3B pair, `eval.sb` | after training |
| 7 | ExeaLM reaches the same quality for less compute | compute multiplier against the Qwen3.5-style model (FLOPs). Quote it in FLOPs only: the curve's Qwen3.5-style runs used our plain-PyTorch DeltaNet kernel, 2.8 to 3.8 times slower than the fla kernel (`results/expanse/README.md` section 6), so a GPU-hours multiplier from their measured speed would flatter ExeaLM | curve, `report.py`, `step_multiplier.py` | **long pair done** (results §24): 1.18-1.21 at step 5,000 rising to 1.34-1.41 at step 20,000 (2 seeds); later matches fall in the baseline's learning-rate decay. At 4.1 tokens per parameter, mid-run: about 1.25 at 0.25B, 1.18 at 0.5B |
| 8 | Long documents: ExeaLM uses what came before | bits per byte by window, 8 windows read as a stream | `eval.sb` (stream_eval) | **0.1B, 0.25B and 0.5B done** (results §10, §15, §20): 0.061, 0.042, 0.033 ahead (5.3%, 4.2%, 3.6%); 0.044 (4.7%) on the long pair; 16,384-token documents §19. 3B after its run |
| 9 | A 9B can be trained, and what it costs | 9B speed and memory on 1, 2 and 4 nodes; price for 1T tokens | `nine_b.sb`, `report.py` | **1 node done** (results §16): 206 effective TFLOP/s per GPU at 8 rows, 40 of 80 GB; about 75,000 GPU-hours for 1T tokens. 16-32 rows and 2-4 nodes queued |
| 10 | A library distilled from a strong model makes ExeaLM better at research and code | bits per byte on held-out library tasks with the library attached, removed, or the right entry pasted | library build (Qwen3-30B-A3B), `graft.py`, post-training, `experiments/library/score.py` | **measured at 0.1B** (results §7-8): with a BM25 search pasting the entry, held-out questions 1.674 to 0.689 bits per byte (-59%), code tasks -16%; ExeaLM ahead of the Qwen3.5-style model on every set with it. ExeaLM's own retrieval slot finds the entry for 0.1% of questions and adds nothing: the report claims the library with a search, not the built-in retrieval. With RAG post-training (§11, 3 seeds): 1.645 to 0.590 (-64%), 0.110 to 0.116 ahead of the Qwen3.5-style model on every seed. Next: repeat at 0.5B or 3B |
| 11 | Agents on a board beat a single bigger model | 3 agents at 0.1B sharing a board against one 0.25B model, on long documents | `board_post.sh`, `swarm_long.py` | **measured at 0.1B, negative** (results §14): the board gains 0.0035; the swarm (1.181) loses to one 0.1B model read as a stream (1.096) and to the 0.25B models. The report must not make this claim; it can report the stream instead |
| 12 | The layout is the best ExeaLM | 12 TONN layouts at 0.1B | layout tests 0.1B | **done**: one TONN layer at the middle layer, 1.348 (3 seeds) against 1.376 for every fourth layer; now the study's ExeaLM |

## What the report will not claim

Results at 9B or 27B, or "3 x 9B beats 27B": the allocation cannot train those. Claims 10 and 11 are measured at small
scale, and the report says so; what they show for 9B is an extrapolation, labelled as one.

## Hours

Used: about 60 of 2,000. Planned: the curve and long pair about 240, the 3B pair about 1,090, scoring and the 9B
measurement about 136, the library and the board about 100. About 370 left in reserve.
