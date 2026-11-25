# Decisions

What ExeaLM is trained as, decided by Avneh Bhatia, and why. Newest first. Change an entry only with a new dated
entry that says what replaced it.

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
