# Ideas to try

Each idea needs a fair comparison before it goes in (CONTRIBUTING.md): same data, same steps, same parameter count,
against the current default. Small runs on the `tiny` preset or a 10M to 20M model are enough to start.

## Make it faster

Speed decides equal-compute comparisons. At equal steps the hybrid beats Llama; at equal training time it ties or wins
up to 27M and loses at 48M, where it takes about 1.2 times Llama's time per step (results/README.md, section 8). What
we measured (`experiments/profile_circuits.py`, Mac GPU, one layer forward and backward, width 256):
- an attention block takes about 18 ms;
- most circuit types take 15-30 ms, RP 33, SM 45;
- GI and SM took 127 and 145 ms before the parallel prefix scan in `scan.py`.

Done so far:
- the prefix scan for gated recurrences;
- one FFT for EP's two sums;
- cheaper RoPE;
- permuting weights instead of activations in the TONN layer (9% per step).

Next:
- **Fused scan kernels.** `decay_scan` and `gated_scan` (`exealm/model/scan.py`) carry most TONN circuits. A Triton
  kernel (chunked parallel scan) could close most of the 2.7× gap with the transformer.
- **Group circuits across types.** All the running-sum circuits (I, MI, MIM, EP) can share one scan call per layer.
- **Recurrent generation.** Most circuits are recurrences, so generating a token can update a small state per slot
  instead of re-reading the whole context. Then the cost per new token stays constant, instead of growing like a
  transformer's.
- **torch.compile**, FP8 matmuls, sequence packing without cross-document attention.

## Make it better
- **Muon optimizer** (Keller Jordan, 2024; shown to scale by Moonshot AI): often reaches the same loss in fewer steps
  than AdamW. Now an option (`--optimizer muon`, `exealm/train/muon.py`). At 13M it beat AdamW on two seeds
  ([results/muon](../results/muon/README.md), #21); it needs the same check at 0.1B before it can become the default.
- **Multi-token prediction** (DeepSeek-V3): predict the next 2 to 4 tokens with small extra heads; a richer training
  signal.
- **Distillation** from a larger open model whose license allows it (check LLM-jp-4 and the K2 Horizon models):
  train on its predicted probabilities, not just the next token. This is a big win for small models. A different
  tokenizer makes it harder, which is one more input to the tokenizer decision.
- **μP** (maximal update parametrization): tune learning rates on a tiny model and carry them over to 0.1B.
- **New circuit types.** Anything causal that maps three streams of h channels to h channels can join the menu. The
  search will tell us if it's worth anything.
- **Data curriculum:** the best data last (in the decay phase), harder data later.

## Post-training
- **GRPO** with verifiable rewards (DeepSeekMath, DeepSeek-R1): math and code first.
- **DPO** on Japanese and English preference data.
- **Multi-agent tasks** for the board: split-information puzzles, debate, divide-and-conquer coding.
- **Tool use** through the retrieval circuit: swap the datastore at inference for a fresh one.

## Understand it
- Which circuits does the search keep, and where? Plot the architecture across layers over training.
- What do RG circuits retrieve? Print the retrieved chunks for some queries.
- What do agents post? Train a small probe or decoder that turns messages into words.
