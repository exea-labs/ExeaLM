# Decisions

What ExeaLM is trained as, decided by Avneh Bhatia, and why. Newest first. Change an entry only with a new dated
entry that says what replaced it.

## 25 November 2026 (night): the plan stays as it is after the 0.1B ablations on Expanse

The study keeps its design: ExeaLM (a TONN layer every fourth layer, value residual and XSA, streams in the last 20%)
against the base Qwen3.5-style model at 3B and on the curve, with the plain Llama on the curve. Decided with these
results in hand (`results/expanse/README.md`, sections 1 and 3; 0.1B, 3,000 steps, the study's data, all 18 held-out
sets):

- ExeaLM beats the Qwen3.5-style model on all 18 sets on every seed: 1.393 against 1.420 bits per byte (3 seeds each),
  and also when both have value residual and XSA (1.376 against 1.403, ExeaLM without streams) or neither (1.401
  against 1.420). TONN layers beat Gated DeltaNet layers as a hybrid's non-attention layer.
- A Llama with gated attention, QK-norm, value residual and XSA beats every ExeaLM on all 18 sets: 1.349 against
  1.376 (ExeaLM without streams, 3 seeds each). It is not in the study. Avneh's decision: public posts leave it out and
  describe the Llama baseline as plain; `results/expanse/README.md` keeps its numbers.
- Streams in the last 20% cost ExeaLM 0.017 on windows read alone (1.393 against 1.376); they are kept for long
  documents read as a stream.

## 25 November 2026 (evening): every run serves the funding case for the 9B

Avneh's direction: the study exists to make funders (government grants or investors) confident in a 9B ExeaLM. Three
additions follow from the questions a reviewer of that application will ask first:

- **Does the lead hold at the 9B run's token budget?** Every curve run trains on 4.1 tokens per parameter; the 9B run
  would train on 111. The 0.1B pair (ExeaLM and the Qwen3.5-style model) now also trains to 111 tokens per parameter
  (10.4B tokens, 59 GPU-hours). The reserve falls from 149 to 90.
- **What does the 9B run cost, and can we train it?** `experiments/expanse/nine_b.py` measures the 9B model's speed
  and memory on 1, 2 and 4 nodes, sharded with FSDP2 (`go.sh nine_b`, the 36 GPU-hours already in the plan). Until it
  runs, the plan prices the 9B run at ExeaLM's 3B rate: about 187,000 GPU-hours for 1T tokens.
- **What is the headline number?** `experiments/expanse/report.py` turns the scores into the curve with every seed,
  the compute multiplier against each baseline in FLOPs and in GPU-hours, the long pair, every held-out set, long
  documents, benchmarks and the 9B price, with a figure (`report.md`, `scaling.png`).

## 25 November 2026 (later): the 3B baseline is the base Qwen3.5-style model

The headline comparison is ExeaLM against the base Qwen3.5-style model, at 3B and on the curve. The plain Llama stays
on the curve (0.1B to 0.5B) as the reference everyone knows. The 1B point is dropped to pay for it: 26 runs, 1,635
GPU-hours, a reserve of 149 ([budget.md](budget.md)). What ExeaLM is and what the baselines leave out are unchanged
from the morning entry below.

Why: on paper the Qwen3.5-style model is the stronger claim. It is a modern hybrid like ExeaLM, so beating it says
that TONN layers do better than Gated DeltaNet layers beside the same attention. It also survives the ablation a
reviewer will ask for. With value residual and XSA added to the baselines at 0.1B on the MI300X, the Qwen3.5-style
model reached 1.426 bits per byte (one seed), still behind ExeaLM's 1.387 (2 seeds), while a gated Llama reached 1.369,
ahead of it. A plain Llama without QK-norm or the attention gate gives a bigger gap but is widely read as an easy
baseline. The paper names the model "Qwen3.5-style": our implementation of the architecture, trained on our data, not
Alibaba's model.

This replaces the plain Llama at 3B and the 1B point of the morning entry.

## 25 November 2026 (morning): ExeaLM against a plain Llama, with a base Qwen3.5-style model beside them

**ExeaLM is the whole recipe:** the memory-free hybrid with a TONN layer every fourth layer, value residual and XSA in
its attention layers (with QK-norm and the output gate), and streaming in the last 20% of pretraining. None of these
is added to the baselines.

**The headline comparison is ExeaLM against a plain Llama** (no output gate, no QK-norm, no value residual, no XSA,
one window per row throughout), at every size from 0.1B to 3B. **A base Qwen3.5-style model** (Gated DeltaNet in 3 of
every 4 layers, gated attention with QK-norm, no value residual or XSA, no streaming) trains beside them from 0.1B to
1B. The study's purpose is unchanged: show that ExeaLM's lead holds as it scales, for the 9B application.

Because the plain Llama runs faster (125 TFLOP/s on the MI300X at 0.5B, against ExeaLM's 81), the 3B pair costs 1,161
GPU-hours instead of 1,456. The hours saved add a 1B point to the curve with all three models, so the curve has five
sizes. Plan and costs: [budget.md](budget.md).

This replaces the Qwen3.5-style baseline with value residual and XSA of 24 November (evening) and "value residual and
XSA in every model" and "the baselines stream the same way" of 24 and 25 November.

What the application has to say with it. At 0.1B on the MI300X (`results/finalize/final_text.json`, 3,000 steps, 2
seeds, mean of four held-out sets) ExeaLM reached 1.387 bits per byte, a gated Llama 1.446 and the base Qwen3.5-style
model 1.460. A gated Llama with value residual and XSA added reached 1.369, ahead of ExeaLM by 0.018. The two
attention changes are published work ([ResFormer](https://arxiv.org/abs/2410.17897),
[XSA](https://arxiv.org/abs/2603.09078)) that any transformer can use. A reviewer will ask how much of ExeaLM's lead
over the plain Llama they explain. The 0.1B result without them (`results/funding`, section 3, 3 seeds: the
memory-free hybrid 1.413 against the plain Llama 1.509) is the answer to cite next to the headline.

## 25 November 2026: a TONN layer every fourth layer, value residual and XSA kept

ExeaLM trains with a TONN layer every fourth layer and attention in the other three (`"tonn_every": 4` in
`experiments/expanse/exealm_core.json`; 9 TONN and 25 attention layers at 3B), not every third. At 0.1B on the MI300X,
both with value residual and XSA, every fourth beat every third on both seeds (1.414 and 1.415 against 1.421 and 1.419
bits per byte on English, 3,000 steps). The two attention changes help attention layers, so a hybrid with more of them
gains more. Value residual and XSA stay in every model: with them ExeaLM leads the Qwen3.5-style baseline by about
0.027 (1.414 against 1.441, seed 0), while a gated Llama with them is ahead of ExeaLM by about 0.011 (1.404, 2 seeds).

## 25 November 2026: stream only in the last 20% of pretraining

Pretraining reads one 512-token window per row for the first 80% of steps and rows of four streamed windows for the
last 20% (`--stream 4 --stream_from 0.8`), for ExeaLM and the baselines. On long held-out documents at 0.1B
(`results/finalize/decision_stream4.json`, 24 documents per set): streaming only at the end beat training without it on
the fourth window by 0.023 bits per byte for both the hybrid (2 seeds) and the gated Llama, and cost 0.015 to 0.02 on
the first window. Streaming the whole run was much worse for both (0.07 to 0.12 on the fourth window): at a step of
16,384 tokens it leaves a quarter as many different texts per step. ExeaLM's lead over the Llama, both streamed at the
end, grows from 0.030 on the first window to 0.041 on the fourth. This replaces "pretraining streams" of 24 November.

## 24 November 2026 (evening): the 2,000 GPU-hours and the data

**The baseline is a Qwen3.5-style model**, not a Llama: Gated DeltaNet in 3 of every 4 layers and gated attention in
the rest, with the same attention changes as ExeaLM (value residual, XSA) and read as a stream the same way. It is
the stronger, newer design, and ExeaLM already leads it at 13M and 0.1B.

**The runs:** the pair at 3B on about 12B tokens each, the headline, and the same recipe at 0.1B, 0.25B and 0.5B
(two or three seeds) so the four sizes give one curve. Every hour of the allocation goes to numbers for the next grant
application; post-training work runs on the MI300X. Plan and costs: [budget.md](budget.md).

**The data:** a 23B-token research-and-coding mix with Japanese, from LLM-jp Corpus v4 and IFM's K2 Horizon sets, with
a new 32k tokenizer trained on it, published on Hugging Face under `exea-labs` so the cluster can pull it. LLM-jp's
two Llama-scored English splits (FineWeb edu score, FineMath) are left out, as LLM-jp left them out of its own models,
to keep the trained model's license open.

## 24 November 2026: what the scaling study pretrains, and what comes after

**Pretraining reads text as a stream.** Every row is 2,048 tokens read as four windows of 512, with the TONN layers'
state and the attention layers' keys and values of the window before carried from one window to the next, and the
gradient through all four (`--stream 4`, [brain.md](brain.md)). The baselines stream the same way: a Llama keeps the
keys and values of the window before. Why: models trained one window at a time already predict later windows of long
documents 0.06 to 0.09 bits per byte better when their state is carried in, and the hybrid gains more than the Llama
(0.089 against 0.080 by the fourth window, one seed each, `results/finalize/stream_zero_shot.json`). The run of record
is `experiments/finalize`, stage 7.

**The board (AG), the library (RG) and weight-alter (WA) are post-training and inference features.** The pretrained
model has none of their slots. `python -m exealm.train.graft` adds them to a finished checkpoint, replacing each TONN
layer's weakest circuits, and they are taught afterwards (`pretrain --init`, with the library attached). Why: at 0.1B,
training in swarms put the model 0.02 to 0.03 bits per byte behind the memory-free hybrid after 3,000 steps (0.10 at
step 1,000), the library slot does nothing without a datastore, and a weight-alter slot costs 0.003 to 0.005 on text at
13M. What they are for is collaboration, knowledge and memory after pretraining:

- the library is written by a large open-weight model (text and code, [experiments/library](../experiments/library/README.md)),
  and ExeaLM is trained to find an entry and adapt it to the task rather than copy it;
- the board lets agents split a task;
- weight edits carried across windows let the model change its own weights as it reads (`wa_persist`).

**Value residual and exclusive self-attention (XSA) in every model**, ExeaLM and both baselines. Each was about 0.01 to
0.016 bits per byte ahead of the hybrid without it at the same step on the MI300X (one seed; second seeds running).
Neither costs compute. Peri-LN did not help at 0.1B and Canon layers helped the Llama by 0.004: both are out.

**No circuit search during training.** The starting circuit mix (short convolutions favoured, no rotary-attention
circuits) is the frozen result of the earlier searches ([results/funding](../results/funding/README.md), sections 1 and
3) and is the same at every size. A search at 13M gained 0.001 to 0.002 bits per byte and replaced one circuit in a
run, and on four GPUs three wait during each round.

**Still open:** the optimizer (Muon beat AdamW at 13M on both of 2 seeds; both models of a comparison always use the
same one).
