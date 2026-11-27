# Post-training ExeaLM for research and code

Status (26 November, 11 p.m.): a design draft. Since it was written the library has become library v2 (a 25B-token
human tier and DeepSeek-V4-Flash entries, searched in two tiers: [library.md](library.md)), the stream phase mixes
streams with windows alone (decisions.md, 26 November, 7 p.m.), and distillation into the weights found no gain at 0.1B
(results/expanse/README.md, section 22). Read it with those changes in mind.

Design, 26 November 2026. Nothing here has run unless it cites a section of
[results/expanse/README.md](../results/expanse/README.md) (written "§N" below). Each stage names the code that exists
(file and function) and says what is missing. This document changes no training code and launches no job.

The goal is the funding case: evidence that a 9B ExeaLM trained on about 1T tokens can be the best open-weight 9B model
for research and code. The plan builds on three edges measured at small scale: a distilled library used as ordinary
retrieval-augmented generation (RAG), reading long documents as a stream, and research-and-code pretraining data. It
does not build on ExeaLM's own retrieval circuit (RG, §7) or the agent board (AG, §14).

## Summary

| Stage | What it teaches | Data | Size at 9B | Code today |
|---|---|---|---|---|
| A. Long context | papers and code longer than the 1,024 tokens attention reaches today | long papers and repository code, half the pretraining mix | about 30B tokens | `pretrain.py --seq --stream` runs. A RoPE base setting, a repository packer and sharding (FSDP) are missing |
| B. RAG mid-training | read searched library entries and ignore wrong ones | library items with their top 3 searched entries, half the pretraining mix | about 15B tokens | `rag_pack.py` and `rag_post.sh` ran at 0.1B (§11). Top-k, aligned rows and target-only loss are missing |
| C. SFT | answer research and code requests with entries in a tool turn | teacher-written conversations, execution-checked code, open chat sets | about 2.4M conversations, 3B tokens, 2 epochs | `sft.py` runs on one GPU. Accumulation, sharding, long rows and the teacher generators are missing |
| D. RL with verifiable rewards | pass unit tests, reach exact answers | code problems with tests, math with exact answers, short-answer research questions | about 60k tasks | `grpo.py` runs on one GPU with math, exact and code rewards. Fast decoding, a sandbox, micro-batches and sharding are missing |
| E. Retrieval as a tool | write its own search queries | teacher trajectories with real search results | about 200k trajectories, inside C and D | the chat format has a tool role. The tool loop is missing |

- **First release uses "always retrieve".** The search runs on the user's message and the entries go in a tool turn,
  the setting §11 measured. Learned search calls (Stage E) replace it only if they beat it at 3B.
- **Validation now** costs about 150 GPU-hours of experiments at 0.5B and 3B plus about 45 GPU-hours of teacher time
  for SFT and RL data, out of the roughly 700 spare. Library v2 (`experiments/expanse/library_v2.sb`) adds about 140
  if its hours are not already in the plan (Section 5).
- **The 9B recipe** costs about 5,500 GPU-hours of training and 2,700 of teacher time, about a tenth of the 75,000 for
  pretraining (Section 2.6).
- **Open decisions** are in Section 7.

## 1. What the plan starts from

### 1.1 Results it builds on

| Edge | Measured | Where | What it does not show |
|---|---|---|---|
| Library with a keyword search (BM25) | ExeaLM 0.1B after RAG post-training: held-out research questions 1.645 to 0.590 bits per byte (64% lower), text tasks 21% lower, code tasks 19% lower. 0.110 to 0.116 ahead of the Qwen3.5-style model on every seed | §11, 3 seeds | One size (0.1B). Bits per byte on target text, not answer accuracy. The questions come from the same model that wrote the entries, and their answers are copied from the entries, so they suit retrieval better than real questions do |
| The same, against the released model | ExeaLM 0.1B with the library 0.590; Qwen3.5-9B-Base without it 1.031 | §18 | A system against a model. Given the same library, Qwen3.5-9B-Base scores 0.313 |
| Streaming | Long documents, windows 1 to 7: ExeaLM ahead of the Qwen3.5-style model by 0.061 at 0.1B and 0.042 at 0.25B. On windows read alone: 0.034 and 0.011 | §10, §15 | Our Qwen3.5-style model, not Qwen3.5-9B, which reads a whole document in one context. The lead shrinks with size |
| Research and code data | 7.3B of 28.5B tokens are research code (recipe v2.1). ExeaLM 0.25B is closest to Qwen3.5-9B-Base on code: 1.18 times its bits per byte on worked code solutions and 1.31 on algebraic code, against 1.50 on the mean of the 18 sets | §18, [data_recipe.md](data_recipe.md) | No direct test of the data against another mix at 9B |
| Not used | RG found the right entry for 0.1% of questions (§7). Three agents on a board lost to one model read as a stream (§14) | | |

### 1.2 The code today

| File: function | What it does today | Missing for this plan |
|---|---|---|
| `exealm/train/sft.py`: `format_chat`, `sft_loss`, `load_conversations`, `encode`, `batches` | Chat format `<|im_start|><|role|>...<|im_end|>`, loss on assistant tokens only (tool and user turns are masked), one conversation per row cut to `--max_len` (1,024), rows of similar length batched | `--accum`, more than one GPU (DDP or FSDP), rows longer than a window read as a stream, packing short conversations |
| `exealm/train/dpo.py`: `dpo_loss`, `answer_logp` | DPO against a frozen copy of the starting model | More than one GPU. Preference data is not chosen |
| `exealm/train/grpo.py`: `group_advantages`, `grpo_loss`, `REWARDS`, `sample_group` | Section 2.4 lists it in full | Fast sampling, a sandbox, micro-batches, F1 reward, multi-turn tool rollouts, sharding |
| `exealm/generate.py`: `generate` | Lockstep sampling, temperature and top-k, left padding | A cache: it re-reads the whole context for every new token. Left padding is read as text, so callers group prompts by length |
| `exealm/chat.py`: `CHAT_TEMPLATE`, `chat_prompt` | The template as Jinja and as token ids | The tool loop (Stage E) |
| `exealm/train/pretrain.py` | `--init` (weights, fresh optimizer), `--data` mix JSON, `--seq`, `--stream k --stream_from f`, `--config` for fields that do not change weights | A per-source loss mask; rows that start at document starts (`TokenShards.batch` draws random offsets) |
| `exealm/model/baseline.py`: `Attention.forward`; `exealm/model/common.py`: `rotate` | In a stream, keys and values of the window before are carried | Keeps only one window back. RoPE base is fixed at 10,000 |
| `experiments/library/build.py`: `cmd_topics`, `cmd_write`, `cmd_morph`, `cmd_pack`, `check_task`, `parse_output`, `VLLMBackend` | Topics from training packs, entries written from a source excerpt (`--source_tokens`, prompt `v2-grounded`), question/answer pairs kept only if the answer is in the entry (`grounded`), code that must parse (`python_error`), adaptation tasks that may not copy (`--max_copy 0.5`, `--max_leak 24`), hash split by topic (`is_heldout`) | A check that a grounded entry's facts are in its source, a run check for code, an OpenAI-API backend for a shared vLLM server, generators for SFT conversations and code problems with tests |
| `experiments/library/score.py`: `bm25_pick`, `layout`, `score` | Bits per byte on targets with nothing, the right entry pasted, or the BM25 first pick pasted; BM25 over token ids | Top-k entries, entries placed in earlier stream windows, answer accuracy from generation |
| `experiments/expanse/library_v2.sb` | Library v2: DeepSeek-V4-Flash (FP8) on a whole node writes entries for 100,000 research topics and 30,000 code topics from 1,200-token source excerpts, then adaptation tasks, packed to `data/library_v2/v2` | Nothing for the library itself. SFT data is a separate step |
| `experiments/library/rag_pack.py` | Each training item with its BM25 first pick before it, one document | Top-k, a marker token, target spans |
| `experiments/post/rag_post.sh`, `library_post.sh` | 600 steps of 65,536 tokens: half the pretraining mix, half library items (a quarter with the searched entry), 1 GPU, then `score.py` | `LIB` (hard-coded to `data/library/v1`), GPU count and batch as settings, for 3B and library v2 |
| `exealm/data/post.py` | Open SFT, preference and RLVR sets to the three loops' JSONL formats, with licenses and revisions | Teacher data |
| `exealm/data/decontam.py`, `benchmarks.py`, `dedup.py`; `experiments/data_mix/bench_extra.py` | 13-gram exact decontamination of packs; benchmark export (HellaSwag, ARC, MMLU, GSM8K, HumanEval, JCommonsenseQA; and in `bench_extra.py` MBPP, APPS, CodeContests, LiveCodeBench, MATH, SciQ, PubMedQA); MinHash near-duplicates | Exports for HumanEval+ and MBPP+ extra tests, BigCodeBench, DS-1000, CRUXEval, SciCode, GPQA, QASPER, MMLU-Pro, RepoBench, CrossCodeEval, LitQA2 |
| `exealm/eval/harness.py` | lm-evaluation-harness: likelihoods and greedy generation, no chat template, 1,024 tokens | Chat template, sampling, longer inputs, retrieval, streams, code execution |
| `experiments/external/` | Scores a released model through vLLM on the same text as our models | Generation benchmarks |

### 1.3 Five limits that shape the plan

1. **Attention reaches one window back.** In a stream, `Attention.forward` keeps only the last window's keys and values
   (`stream[key] = (k, v)`). With 512-token windows, exact attention covers at most 1,024 tokens. Anything older
   reaches the model only through the one TONN layer's state.
2. **Positions.** `rotate` uses a fixed RoPE base of 10,000. Training has seen distances up to 1,023.
3. **No decoding cache.** `generate` re-reads the whole context for each token. Sampling for RL and for code
   benchmarks costs the square of the length.
4. **One GPU per post-training loop.** `sft.py`, `dpo.py` and `grpo.py` have no DDP or FSDP. `nine_b.py` shards the 9B
   with FSDP2 for speed tests only. A 3B model fits one H100 in `sft.py` only with small batches.
5. **Untrained chat tokens.** The tokenizer has `<|im_start|>`, `<|tool|>` and `<|retrieved|>`
   (`exealm/data/tokenizer.py`, `SPECIAL`), but pretraining text never contains them.

Library entries average about 430 tokens (7.9M tokens over 18,545 entries). One entry and a question fit in 1,024
tokens. Three entries do not. Section 2.1 is the response to limits 1 and 2. Limit 3 needs a decoder before the
generation experiments V3 and V4 (Section 5.4).

## 2. The 9B recipe

Order: pretraining (1T tokens, streams in the last 20%), then A, B, C, D. E is part of C and D.

### 2.1 Stage A: long context and streaming

**Why it is needed.** Approximate lengths of what the model must read and write:

| Input | Tokens |
|---|---|
| HumanEval+ or MBPP+ prompt and answer | 200 to 600 |
| LiveCodeBench problem and answer with short reasoning | 1,000 to 3,000 |
| SciCode problem with background and answer | 2,000 to 5,000 |
| Question with 3 library entries | about 1,500 |
| Research paper (QASPER) | 4,000 to 10,000 |
| Repository-level completion | 8,000 to 32,000 |

Only the first row fits in 1,024 tokens of exact attention.

**Two ways to extend it.**

- **A1, longer windows.** Train with `--seq 4096 --stream 4`: full attention inside a 4,096-token window, the window
  before carried, rows of 16,384 tokens. This runs today. It needs a `rope_base` field in `ModelConfig`, passed to
  `rotate` from `Attention.forward` (missing, small). The field does not change the weights, so `--config` can set it
  at `--init`.
- **A2, more windows carried.** Keep 512-token windows and keep the keys and values of the last N windows (a
  `stream_kv_windows` setting in `Attention.forward`, missing). With N = 8 that is sliding-window attention over 4,096
  tokens at a fixed cost per token. It keeps ExeaLM's window-by-window design.

Validation V1 (Section 5) picks between them at 0.5B. The default is A1.

**Data.** Half long documents (at least 8,192 tokens) from `arxiv`, `pes2o` and `research_python`, half the anneal
mix, so held-out text does not get worse. Repository-level code needs the files of one repository joined in import
order. The packs hold one file per document. The source records keep each file's repository and path (`meta`, see
`experiments/data_mix/cards.py`), so a packer can group them; that packer is missing.

**Size and place.** About 30B tokens at 9B (3% of 1T). It can replace part of pretraining's stream phase or follow
the anneal (open decision 1).

**Gate.** `experiments/stream_eval.py --seq 2048 --windows 8` and `--seq 512 --windows 32` improve; the 18 held-out
sets read alone (`experiments/eval_text.py`) lose no more than 0.01 bits per byte; a passkey test (a number hidden at
a random depth of 2k to 16k tokens) passes (missing, small).

### 2.2 Stage B: RAG mid-training

**What ran.** `experiments/post/rag_post.sh` post-trains with `pretrain.py --init` on a mix: half the pretraining
mix, a quarter the library's training items, a quarter the same items with the BM25 first pick before them
(`rag_pack.py`). Loss is on every token. Rows are 512 tokens drawn at random offsets (`TokenShards.batch`), so many rows separate a question from
its entry or start mid-item. §11 got its result despite this.

**Changes for 9B.**

1. **Top 3 entries, in the search's order.** The right entry is there only when the search finds it, as at test
   time. About one question row in ten and one task row in five will hold no right entry (§8: the right entry was in
   the first 5 for 93% of questions and 80% of tasks). Those rows teach the model to fall back on what it knows.
   (`rag_pack.py`: a `--topk` option; `score.py`: the same layout as a condition.)
2. **A marker.** Each entry starts with `<|retrieved|>` and its topic line, so the token used in SFT is trained here.
3. **Aligned rows and target-only loss.** Item rows start at the item's start and carry loss on the target only.
   `qa/` and `tasks/` packs already store target spans (`targets.npy`). Needed: `rag_pack.py` writes spans, and the
   loader sets targets outside them to -100 for sources marked `"targets_only": true` (missing, medium).
4. **A better search.** `bm25_pick` treats token ids as words, with a binary query. Replace it with word-level BM25
   fused with a dense retriever (reciprocal rank fusion). Candidates: `bm25s` (MIT) and an Apache-2.0 embedding model
   such as Qwen3-Embedding-0.6B. V6 picks one by first-hit rate on held-out items at 18k, 100k and 1M entries. BM25
   found the right entry first for 86% of questions and 73% of tasks at 18,545 entries (§8). That rate will fall as
   the library grows.
5. **Entries in the stream.** Place the entries in the windows before the question and read them as a stream. This
   tests whether ExeaLM's state carries them, which would make many entries cheap. Score it first (`score.py`, a
   `streamed` layout, missing); train on it only if it scores close to entries in the same window.
6. **Library v2 from the teacher** (`library_v2.sb`, Section 3). Library v1 stays for comparison with §7 to §11.

**Mix and size.** Half the anneal mix, half library items (half of those with entries). About 15B tokens at 9B. It
could instead be part of the anneal. [decisions.md](decisions.md) (24 November) makes the library a post-training
feature. That entry is about the RG, AG and WA slots; whether library items may enter the anneal as plain data is not
decided. Open decision 2.

### 2.3 Stage C: SFT on teacher-written data

**Format.** One JSON conversation per line, as `sft.py` reads it. Retrieved entries go in a `tool` turn, so they are
masked from the loss:

```json
{"messages": [
  {"role": "system", "content": "You answer research and coding questions. References may be given; use them when they are relevant and say when they are not."},
  {"role": "user", "content": "How does the method in the gated delta rule paper decide what to forget?"},
  {"role": "tool", "content": "<|retrieved|>[1] Gated DeltaNet\n...\n<|retrieved|>[2] Mamba-2\n...\n<|retrieved|>[3] DeltaNet\n..."},
  {"role": "assistant", "content": "..."}],
 "source": "teacher:deepseek-v4-flash:rqa-v1", "topic": "Gated DeltaNet", "split": "train", "entries": [101, 77, 2045]}
```

Code records also carry `"tests"` and `"entry_point"`, so the same problem can be checked, held out or reused for RL.

**Buckets.**

| Bucket | What the teacher writes | Check before keeping | Validation | 9B |
|---|---|---|---|---|
| Research questions with retrieval | A question about a topic, the answer from the top 3 searched entries, citing them; "the references do not cover this" when the search missed | The answer's facts are in the cited entries (string check as `grounded`, then an optional teacher check) | 15k | 300k |
| Paper reading | Questions on a full paper (CC-BY or CC0 where released) and answers with a quoted line of evidence | The quote is in the paper | 3k | 100k |
| Code | A problem, a solution and tests, seeded from library code topics and permissively licensed functions in `research_python` and `starcoder` | The solution passes its tests in the sandbox, and the tests fail a mutated solution | 12k | 500k |
| Research code | NumPy, SciPy, pandas and PyTorch tasks with checks on computed values | As for code | 5k | 200k |
| Library adaptation | The existing morph tasks (`build.py morph`) as chat, the entry in a tool turn | `check_task`, as today | 10k | 300k |
| Debugging | Code with a failing test and its traceback; the fix | The fix passes | 3k | 100k |
| Math | Problems with exact answers, worked solutions | The answer matches (`reward_math`) | 5k | 200k |
| Search trajectories (Stage E) | Section 2.5 | The final answer is right | 2k | 200k |
| General instruction following | Open sets through `exealm/data/post.py` (`dolci-sft`, `tulu3-sft`, `dolly-ja`) | License per subset (Section 3.5) | 15k | 400k |
| Japanese | Research questions and code in Japanese | As above | 0 | 50k |

That is about 70k conversations and 70M tokens for validation, about 2.4M conversations and 3B tokens at 9B, trained
for 2 epochs.

**Reasoning traces.** [DATA.md](../DATA.md) says a small model may gain little from long traces. Start with short
reasoning before the answer (at most 1,000 tokens) for code and math. Long traces need Stage A. At 9B, test traces up
to 4,000 tokens on 10% of the code and math buckets (open decision 4).

**Settings to start.** At 0.5B, try learning rates 3e-5, 1e-4 and 3e-4 (`sft.py` defaults to 3e-4). Use the best
at 3B. At 9B, 1e-5 to 2e-5 with 2 epochs, a common range at this size; a short sweep at 3B narrows it.

**Missing.** `--accum` in `sft.py` (small). DDP for 3B, FSDP for 9B (large; reuse the FSDP2 wrapping in
`experiments/expanse/nine_b.py`). Rows longer than `--max_len` read as streams (medium) unless Stage A makes 4,096-token
windows normal. Generators for each bucket (Section 3).

### 2.4 Stage D: preference and RL

**What `grpo.py` supports today.**

- Tasks: `{"prompt" or "messages", "answer", "reward"}` per line (`load_tasks`). Messages may include a `tool` turn,
  so retrieval in context works for RL now.
- Rewards (`REWARDS`): `math` (last `\boxed{}` or last number, `reward_math`), `exact` (string equality), `code`
  (`reward_code`: the first Python block plus the test code, run with `python -I` in an empty folder with no
  environment, a 5-second limit and on Linux a 2 GB memory limit; refused without `--allow_code_exec`). The docstring
  says this is not a sandbox.
- Loop: `--prompts` tasks a step, `--group` samples each (`sample_group`, one `generate` call per task), advantages
  normalized within the group (`group_advantages`), PPO-style clipped ratio plus a KL term to a frozen copy of the
  start (`grpo_loss`, `--beta 0.04`), `--epochs 2` passes on each batch. Greedy evaluation on held-out tasks.
- One GPU, all rows in one forward pass, 32-bit weights with bf16 autocast, `--max_new_tokens 256`.

**What is missing for code RL at 3B and 9B.**

1. **Fast sampling.** Without a cache, a 1,000-token answer to a 500-token prompt reads about a million tokens,
   against 1,500 with one. Three changes, cheapest first. (a) Sample every prompt of a step in one call with left
   padding masked: `generate` reads padding as text, which is why `sample_group` makes one call per task of 8 rows.
   (b) A per-token cache built from the stream machinery: TONN's stream state read with one-token windows, and
   attention keeping all earlier keys and values (the A2 change with no limit). A test must show that this gives the
   same outputs as a full forward pass. (c) Failing (b), a stream decoder that re-reads only the current window from a
   copy of the state; it bounds the cost per token by one window, which helps long inputs but not short ones.
2. **A sandbox.** Run `reward_code` inside an Apptainer container on Expanse with no network, a read-only image and a
   scratch folder (check what Expanse allows). The same runner checks teacher solutions in Section 3.
3. **Micro-batches** in the loss passes, and **no reference copy when `--beta 0`** (memory at 3B: 32-bit weights,
   gradients, Adam and a reference copy are about 60 GB before activations).
4. **Skip groups whose rewards are all equal.** Their advantage is zero, so they cost a forward and backward pass for
   the KL term alone.
5. **An F1 reward** for short research answers (normalized token F1 against the answer). `exact` is too strict for
   text.
6. **Multi-turn rollouts** for Stage E (the search tool runs mid-answer).
7. **Sharding** for 9B.

**Tasks.**

| Set | Source | Reward | 9B |
|---|---|---|---|
| Code | Teacher problems with tests, never used in SFT, kept if the SFT model passes 1 to 7 of 8 samples | `code` | 30k |
| Math | `rlvr` in `post.py` (ODC-By) and teacher problems with checked answers | `math` | 20k |
| Research | Held-out-split library questions are excluded. Train-split questions with short answers, top 3 entries in a tool turn | F1 (missing) | 10k |

**Settings to start.** At 0.5B: the defaults, `--max_new_tokens 512`, `--group 8`. At 9B: learning rate 1e-6 to 2e-6,
group 8 or 16, `--beta` 0 to 0.001, answers up to 2,048 tokens.

**DPO.** `dpo.py` works and `post.py` has three preference sets (`tulu3-pref`, `dolci-pref`, `ultrafeedback`). It
targets chat style, not research or code correctness. Leave it out of the validation. At 9B, a short DPO pass on
on-policy pairs judged by the teacher is optional (open decision 8). Qwen3.5-9B-Base is a base model, so it is a poor
preference judge. Use it to rate difficulty (few-shot pass rate on code problems) and to mark which research questions
it answers without an entry. That splits the evaluation into closed-book and retrieval-needed questions.

### 2.5 Stage E: retrieval as a tool

**Format.** The assistant writes one line, `SEARCH: <query>`, and ends its turn. The runtime runs the search and adds a
`tool` turn with the top 3 entries. The assistant then answers or searches again, at most 3 times. `format_chat`
already supervises the query (an assistant turn) and masks the results (a tool turn). Missing: the loop in
`exealm/chat.py` and in the evaluation path that stops at `<|im_end|>`, reads the line, searches and continues.

**Data.** The teacher gets the question and a description of the search. Each query it writes is run against our
library with our retriever, and the real results go back to it. Keep trajectories whose final answer is right
(library questions with known answers, code with tests). Mix in questions that need no search, with no search call.

**RL.** Needs multi-turn rollouts in `grpo.py` (item 6 above). Reward the final answer only.

**Is it feasible?** For SFT at 0.5B to 3B, yes: it is a format and a loop. Whether it helps is the question. The test
(V5) compares the first-hit rate of the model's own queries with the raw question used as the query, and answer
accuracy against always-retrieve. If the model's queries do not beat the raw question at 3B, the first release keeps
always-retrieve. At 9B, the use that always-retrieve cannot cover is a search mid-answer, for example an API's
signature halfway through writing code.

### 2.6 Compute at 9B

At the measured 206 effective TFLOP/s per GPU (§16), a 9B trains on about 13.3M tokens per GPU-hour (75 GPU-hours per
billion tokens).

| Item | Tokens or count | GPU-hours | How it was estimated |
|---|---|---|---|
| A. Long context | 30B | about 2,700 | 1.2 times the 512-token cost for 4,096-token windows |
| B. RAG mid-training | 15B | about 1,100 | |
| C. SFT | 3B, 2 epochs | about 600 | 1.3 times for padding |
| D. GRPO | about 1B sampled tokens | 500 to 1,000 | depends on the decoder; with a vLLM-class engine at the low end |
| Evaluation | | about 300 | |
| **Training and evaluation** | | **about 5,500** | 7% of 75,000 |
| Teacher data | about 4.9B output tokens | about 2,700 | at the planning rate of Section 3.1 |

The 9B post-training loops need FSDP in `pretrain.py`, `sft.py` and `grpo.py`, and the TONN layer's running output
sizes averaged across GPUs (the [budget.md](budget.md) note on the 9B run).

## 3. Teacher data

### 3.1 The teacher and how the builder reaches it

DeepSeek-V4-Flash (284B mixture of experts, 13B active, MIT, FP8 weights) served with vLLM 0.34 on one node of 4
H100s. `experiments/expanse/library_v2.sb` already runs it for the library.

- `build.py`'s `VLLMBackend` runs vLLM in the same process (`vllm.LLM`). `library_v2.sb` passes `--tensor_parallel 4`
  and `--max_model_len 8192`. Paper-reading SFT data (Section 2.3) needs 16,384 or more.
- `VLLMBackend` passes `dtype="bfloat16"`. vLLM takes the weight format from an FP8 checkpoint's config, so this
  should only set the activations; the trial job confirms it.
- `chat_text` passes `enable_thinking=False`, Qwen3's switch. Check how DeepSeek-V4-Flash's template turns thinking
  on and off, and record it in each record's `prompt_version`. Entries and questions need no thinking. Verified code
  solutions and math may use it.
- If the teacher runs as a vLLM server so that several jobs share it, `build.py` needs a backend for the
  OpenAI-compatible API (missing, about 40 lines).
- **Throughput is unknown until measured.** The plan uses 2,000 output tokens a second for the node, about 1.8M
  output tokens per GPU-hour. The trial job of `library_v2.sb` (`TEXT_TOPICS=200`) reports the rate, and the sizes
  below scale to it.

### 3.2 What it writes

Output tokens are estimates: about 700 per research entry with its questions, 500 per code entry, 800 per reply of
adaptation tasks.

| Product | Seeds | Now (count, output tokens) | 9B (count, output tokens) |
|---|---|---|---|
| Research entries with 3 to 5 questions, grounded (`library_v2.sb`) | First 1,200 tokens of `pes2o`, `arxiv` and `wikipedia_en` training documents | 100k, 70M | 1M, 700M |
| Code entries (`library_v2.sb`) | Code topics from `research_python`, `starcoder` and `algebraic_stack` | 30k, 15M | 200k, 100M |
| Adaptation tasks (`library_v2.sb`) | `build.py morph` on the new entries | 130k entries, 104M | 600k entries, 480M |
| **Library v2, with 30% rejected** | | **about 250M, 140 GPU-hours** (two 24-hour jobs) | |
| SFT conversations (Section 2.3), new generators | Entries, papers, code topics, permissive functions | 55k, 50M | 2M, 2B |
| Code problems with tests (SFT and RL), new generators | Code topics and permissive functions | 10k, 12M | 300k, 360M |
| Search trajectories, new generators | Library questions, code problems | 2k, 2M | 200k, 160M |
| **SFT and RL data, with 30% rejected** | | **about 80M, 45 GPU-hours** | |
| **All** | | **about 330M, 185 GPU-hours** | **about 4.9B with 30% rejected, 2,700 GPU-hours** |

For the 9B library, consider longer excerpts than 1,200 tokens for papers (more than the abstract and introduction),
which the larger context of the teacher allows.

The 18,545-entry library v1 (Qwen3-30B-A3B-Instruct-2507, `exea-labs/exealm-library`) stays as it is. Its held-out
topics stay held out in v2, because the split hashes the topic with the same salt (`is_heldout`, salt
`exealm-library`).

### 3.3 Filters

Existing, kept: wrong format, entry under 100 words or "UNKNOWN" (`parse_output`); fewer than 3 answers stated in the
entry (`grounded`); code that does not parse (`python_error`); a task target that copies more than half its length
from the entry, or a prefix that copies more than 24 tokens (`check_task`).

New:

1. **Facts in the source.** Every number, date and capitalized name in a grounded entry must appear in its source
   excerpt (the `grounded` test applied per fact). Today only the prompt asks for this; nothing checks it.
2. **Code runs.** A code entry's code runs in the sandbox with the scientific Python stack installed. A code problem
   is kept only if its solution passes its tests and the tests fail a mutated solution (an operator or constant
   changed), so tests that accept anything are dropped.
3. **Answers supported.** For research questions with retrieval, the answer's facts appear in the cited entries. An
   optional second pass asks the teacher whether the answer follows from the entries (about 20% more teacher tokens).
4. **Near-duplicates.** `exealm.data.dedup` over all generated text, by bucket.
5. **Length.** Drop conversations whose answer is cut by the teacher's length limit.

### 3.4 Splits, held-out sets and decontamination

1. **Seeds come only from training packs** (`*/train`), never from the 18 held-out sets, as `library.sb` and
   `teacher_text.py` do today.
2. **One split for everything derived from a topic.** SFT conversations, RL tasks and trajectories use the topic's
   `is_heldout` side. Held-out-topic material is never trained on. RL code problems are disjoint from SFT problems.
3. **Exact overlap.** Pack every generated set and the library, then run
   `python -m exealm.data.decontam <packs> --benchmark NAME=dir ... --ngram 13` against: every benchmark of Section 4
   (the exports in `exealm/data/benchmarks.py` and `experiments/data_mix/bench_extra.py`, plus the missing ones of
   Section 1.2), the 18 held-out sets, and the library's `qa/heldout` and `tasks/heldout`. Drop any document with a
   match.
4. **Paraphrases.** Run `exealm.data.dedup` with the benchmark exports listed first. Generated text it drops is a near
   copy of a benchmark item. The teacher may have seen the benchmarks, and asking it for "a function that..." can
   return one from memory.
5. **Canaries.** Search all generated text and the library for benchmark canary strings (GPQA and BIG-bench publish
   one).
6. **Dates.** LiveCodeBench is scored only on problems released after the data cutoff of the teacher, of our corpus
   and of every compared model.

### 3.5 Licenses

| Item | License | Action |
|---|---|---|
| DeepSeek-V4-Flash outputs | MIT | Allowed for training. Record the model, revision and the hash of its license file in every record (`build.py` records `model` and `prompt_version`) |
| Library v1 (Qwen3-30B-A3B-Instruct-2507) | Apache-2.0 | Keep |
| Qwen3.5-9B-Base | Apache-2.0 | Comparison and difficulty rating |
| arXiv full text (`allenai/olmo-mix-1124`, arxiv) | ODC-By for the collection; each paper keeps its own license (arXiv's distribution license, CC0, CC-BY, CC-BY-SA, CC-BY-NC variants) | Training as in pretraining. For a released library, either keep only CC0, CC-BY and CC-BY-SA papers (needs the per-paper license from arXiv metadata, missing) or release entries with collection-level attribution. Open decision 6 |
| peS2o v2 | ODC-By | Attribute |
| Wikipedia | CC-BY-SA | Entries grounded on it are released under CC-BY-SA, in their own folder ([data_recipe.md](data_recipe.md) keeps share-alike sources apart) |
| Code seeds (`starcoder`, `research_python`) | Per-file licenses | Seed only from MIT, BSD and Apache files; record repository and license |
| Open SFT sets (`post.py`) | ODC-By and others; mixtures carry their subsets' terms | Keep the project's rule against Llama-scored or Llama-written data: drop SmolTalk's Magpie part and any subset generated by Llama or a closed API. Check each subset card (open decision 5) |
| Benchmarks | Various | Evaluation only, never training |

## 4. Evaluation

### 4.1 Two kinds of comparison

Every table states which kind it is.

- **Model to model.** Each model alone, the same prompt, the same decoding, the same scorer. Base against base, and
  post-trained against post-trained. Long documents are cut to the same budget for every model.
- **System to system.** Each model with the same library, the same retriever, the same top 3 entries and the same
  generation budget. On long documents each model reads the full text its own way: ExeaLM as a stream, the others in
  their context window.
- **System against model** (ExeaLM with the library, another model without) is reported only labelled as such and is
  never the headline, as in §18.

### 4.2 Benchmarks

| Area | Benchmark | What it measures | Library | Stream | Usable at 0.5B to 3B |
|---|---|---|---|---|---|
| Code | HumanEval+ and MBPP+ (EvalPlus) | Function synthesis with extra tests | S2S arm | No | MBPP+ yes, as pass@1 and as bits per byte on the reference solution |
| Code | LiveCodeBench (code generation, problems after the cutoff) | Contest problems, contamination-limited by date | S2S arm | At 9B | No |
| Code | BigCodeBench | Tasks that call many libraries | Yes: this is where code entries should help | No | Bits per byte only |
| Research code | DS-1000 | NumPy, pandas, SciPy and similar tasks | Yes | No | Bits per byte only |
| Code reasoning | CRUXEval | Predict a function's output or input | No | No | Yes |
| Repository code | RepoBench and CrossCodeEval | Completion with code from other files in context | No | Yes | 3B, as bits per byte |
| Agentic code | SWE-bench Verified | Fix real issues with a scaffold | S2S | Yes | No. Report at 9B, not as the headline |
| Science | GPQA Diamond | Graduate-level science multiple choice | S2S arm | No | No (near chance) |
| Science code | SciCode | Scientific coding subproblems with tests | S2S arm | At 9B | No |
| Science | MMLU-Pro (STEM subsets) | Knowledge and reasoning | S2S arm | No | No |
| Paper reading | QASPER | Questions on full NLP papers | No | Yes: the main streaming test | 3B, as F1 and bits per byte |
| Literature | LitQA2 (LAB-Bench) | Questions that need a specific paper | Yes | No | No |
| Math | MATH-500, GSM8K | Exact answers | No | No | GSM8K at 3B |
| Long context | RULER at 4k to 32k | Retrieval and tracking in long inputs | No | Yes | 3B |
| Ours | 18 held-out sets, long documents, library held-out items | Bits per byte; for library items also answer accuracy | Yes | Yes | Yes |

**Small-scale signal.** A 3B model trained on 12B tokens scores near the floor on most of these. Below 9B, score each
benchmark by bits per byte on the reference answer given the prompt (and multiple choice by likelihood), beside the
standard metric. The trend from 0.5B to 3B in these scores is what the application can extrapolate, labelled as an
extrapolation.

### 4.3 Running them with the library and streaming

1. **A decoder.** The cached decoder of Section 2.4, batched, with temperature and top-p, in a new
   `exealm/eval/gen.py`. It applies `CHAT_TEMPLATE` for post-trained models and plain prompts for base models.
2. **Official scorers.** Write samples in each benchmark's own format (EvalPlus `samples.jsonl`, LiveCodeBench,
   BigCodeBench, SciCode) and score them with the benchmark's own code in the sandbox container. The harness adapter
   (`exealm/eval/harness.py`) stays for likelihood tasks.
3. **The retrieval hook.** Query = the user message. Top 3 entries from library v2 with the retriever V6 picks. For
   ExeaLM they go in a `tool` turn; for models without a tool role, before the question in the user turn, in the same
   words. Log every retrieved entry id, so a reviewer can check that no entry holds a benchmark solution. The library
   goes through the decontamination of Section 3.4 first.
4. **Streams.** Long inputs (QASPER papers, repository context, RULER) are read as windows with state carried, then the
   question and the answer are decoded in the last window.
5. **Other models** through vLLM, as `experiments/external/score_hf.py` does, each with its own chat template. Report
   greedy pass@1, and pass@1 from 10 samples where the benchmark's paper does. Models with a thinking mode are run at
   two output budgets (4,096 and 16,384 tokens) and both are reported.
6. **Frozen in advance.** The benchmark list, prompts, decoding and the headline metric are written down before the
   9B scores are seen.

### 4.4 Models to compare

Refresh this list on the day the comparison is frozen: it reflects releases up to mid-2026, and licenses must be read
from each model card then.

| Model | Kind | Why it is in the list |
|---|---|---|
| Qwen3.5-9B-Base and Qwen3.5-9B | Hybrid (Gated DeltaNet and gated attention), Apache-2.0 | The bar (§18), and the design our Qwen3.5-style baseline copies |
| Qwen3-8B | Transformer, Apache-2.0 | The previous Qwen generation at this size |
| GLM-4-9B-0414 | Transformer, MIT | Strong 9B with a permissive license |
| NVIDIA Nemotron Nano 9B v2 | Hybrid (Mamba-2 and attention), NVIDIA open model license | A hybrid at 9B |
| Falcon-H1 7B | Hybrid (Mamba-2 and attention), Falcon license | A hybrid near this size |
| Llama 3.1 8B | Transformer, Llama 3.1 license | The reference everyone knows |
| Gemma 2 9B | Transformer, Gemma terms | A 9B from Google |
| Granite 3.3 8B | Transformer, Apache-2.0 | Permissive 8B |
| OLMo 3 7B | Transformer, Apache-2.0, open data and code | The fully open comparison: the class ExeaLM is in |
| Seed-Coder-8B, Qwen2.5-Coder-7B | Code models, MIT and Apache-2.0 | Code specialists at this size |
| DeepSeek-R1-Distill-Qwen-7B | Reasoning distillation, MIT | A 7B distilled from a frontier teacher, as ExeaLM's SFT is |

### 4.5 What the claim has to mean

"Best open 9B for research and code" needs a fixed suite and two separate statements:

- **System to system:** on the frozen suite, ExeaLM-9B with library v2 and the retriever scores at least as well as
  each listed model with the same library and retriever. This is where §11's lead (ExeaLM uses entries better than
  the Qwen3.5-style model) would show.
- **Model to model:** reported in full, whatever it shows. Qwen3.5-9B was trained on far more than 1T tokens. Our
  0.25B model is at 1.50 times its bits per byte on the 18 held-out sets (§18). Post-training will not close a
  pretraining gap, so the application should not promise a model-to-model win.

A narrower statement may be easier to support: best fully open 9B (weights, data and code), against OLMo-class models
(open decision 7).

## 5. Small-scale validation now

### 5.1 Checkpoints and order

- **0.5B:** `exealm_500m_4.1x_s0` and `s1`, `qwen_500m_4.1x_s0` and `s1`, restarted 26 November at 11:31 with the DDP
  fix (§17).
- **3B:** `exealm_3b_4.1x_s0` and `qwen_3b_4.1x_s0` on Sunday 29 November. If they are still training, use both runs'
  `ckpt.pt` at the same step. A checkpoint from before the anneal understates a finished model; say so beside every
  number from it.
- **Order:** V0 and V6 now; V1, V4 and V5 on the 0.5B models; V2 then V3 on Sunday's 3B models.

Every experiment has a control. 0.5B runs use 1 GPU of the shared partition; 3B runs use one node with `--zero 1`.
Planning rates: about 140M tokens per GPU-hour at 0.5B and 30M at 3B (ExeaLM's measured eager rates, §1; the
one-layer layout should be faster). SFT on one GPU is counted at half of that, for padding.

### 5.2 Experiments

**V0. Teacher data (library v2 about 140 GPU-hours, SFT and RL data about 45).** The library is
`library_v2.sb`, as written. First its trial: 200 research and 100 code entries, read by a person, and the measured
rate.

```bash
sbatch -A fli100 --export=ALL,TEXT_TOPICS=200,CODE_TOPICS=100,CODE_DOCS=2000,PACK=0,OUT=data/library_v2_trial \
    experiments/expanse/library_v2.sb                                          # read the entries, note the rate
sbatch -A fli100 experiments/expanse/library_v2.sb                             # library v2: data/library_v2/v2
python -m exealm.data.post dolci-sft --max_rows 15000 --out data/post          # the open general bucket
```

The SFT, code-problem and trajectory generators are missing (`experiments/post/teacher.py`, reusing `build.py`'s
`make_backend`, `run_batches`, `append_jsonl` and reject files). So are the new filters of Section 3.3. They can run
in the same node allocation after the library, or against a shared vLLM server (the missing OpenAI-API backend).

**V1. Long context at 0.5B (about 25 GPU-hours).** From `exealm_500m_4.1x_s0`, 1,200 steps of 262,144 tokens (315M),
learning rate 3e-4, on a mix of half long documents and half the anneal mix (`data/mix_long.json`, missing: a filter
that packs documents of at least 8,192 tokens). Each arm is about 4.5 GPU-hours: streams read at about half the speed
of single windows (§2).

| Arm | Settings | Role |
|---|---|---|
| a | `--seq 512 --stream 4 --batch 8 --accum 16` | Control: today's stream phase, same data |
| b | `--seq 2048 --stream 4 --batch 2 --accum 16` | A1 |
| c | b with `rope_base` 500,000 (field missing) | A1 with longer positions |
| d | `--seq 512 --stream 16`, last 4 windows of keys and values carried (missing) | A2, if the code is ready |
| e | b on `qwen_500m_4.1x_s0` | Model control |

```bash
python -m exealm.train.pretrain --init runs/exealm_500m_4.1x_s0/model.pt --search 0 --data data/mix_long.json \
    --val data/mix_pretrain_val.json --seq 2048 --stream 4 --batch 2 --accum 16 --steps 1200 --warmup 120 \
    --lr 3e-4 --out runs/long_b_500m
python experiments/stream_eval.py runs/long_*_500m --mix data/mix_pretrain_val.json --seq 2048 --windows 8 --docs 64
python experiments/stream_eval.py runs/long_*_500m --mix data/mix_pretrain_val.json --seq 512 --windows 32 --docs 64
python experiments/eval_text.py runs/long_*_500m --mix data/mix_pretrain_val.json         # forgetting check
```

Also scored: the passkey test and library questions with the top 3 entries (`score.py`, top-k missing). It decides
open decision 1.

**V2. RAG mid-training at 0.5B and 3B (about 33 GPU-hours).** Library v2. Same tokens per step as §11 (65,536).

| Arm | Size | Model | Recipe | Role |
|---|---|---|---|---|
| a | 0.5B, 2 seeds | ExeaLM | `rag_post.sh` as it is, 600 steps | Links to §11 |
| b | 0.5B, 2 seeds | ExeaLM | top 3, marker, aligned rows, target-only loss | The Stage B changes |
| c | 0.5B, 2 seeds | Qwen3.5-style | as b | Model control |
| d | 3B | ExeaLM | as b (or a, if b is not better at 0.5B), 3,000 steps (197M tokens) | Main |
| e | 3B | Qwen3.5-style | as d | Model control |
| f | 3B | ExeaLM | pretraining mix only, same steps | Data control: what the library items add |

```bash
python experiments/library/rag_pack.py --library data/library_v2/v2 --out data/library_v2/v2/rag/train
LIB=$PWD/data/library_v2/v2 TAG=500m_s0 bash experiments/post/rag_post.sh runs/exealm_500m_4.1x_s0 runs/qwen_500m_4.1x_s0
python experiments/library/score.py --library data/library_v2/v2 runs/ragpost_exealm_3b runs/ragpost_qwen_3b \
    runs/plainpost_exealm_3b --conditions none,pasted,bm25 --out results/expanse/library_rag_3b.jsonl
```

`rag_post.sh` hard-codes `LIB=$PWD/data/library/v1` and one GPU, so the second line works only after a small edit
that makes `LIB`, the GPU count and `--batch`/`--accum` settings. Also scored: answer accuracy from generation on
1,000 held-out questions (`score.py` addition), the `streamed` layout, and Qwen3.5-9B-Base on the same v2 items
through `experiments/external/` (system-to-system bar: both with the library). Hours: about 2 for the six 0.5B runs,
about 7 for each 3B run, about 10 for scoring.

What it answers: does ExeaLM's lead in using entries (§11, 0.110 at 0.1B) hold at 0.5B and 3B? This is the trend the
application needs most.

**V3. SFT at 0.5B and 3B (about 53 GPU-hours).** The validation mix of Section 2.3 (about 70k conversations and 70M
tokens, 2 epochs). `sft.py` needs `--accum` first. Hours: about 2 for each 0.5B run, about 9 for each 3B run, about 10
for generation-based scoring.

| Arm | Start | SFT data | Role |
|---|---|---|---|
| 0.5B a, b, c | V2 arm b | teacher mix at learning rates 3e-5, 1e-4, 3e-4 | Picks the rate |
| 3B a | V2 arm d | teacher mix | Main |
| 3B b | V2 arm d | open sets only (`dolci-sft`, `tulu3-sft`), same tokens | Data control: what the teacher adds |
| 3B c | V2 arm e | teacher mix | Model control |
| 3B d | the 3B checkpoint without Stage B | teacher mix | Stage control: what Stage B adds |

```bash
python -m exealm.train.sft --ckpt runs/ragpost_exealm_3b/model.pt --tokenizer $TOK --data data/post/sft_v1.jsonl.gz \
    --val data/post/sft_v1_heldout.jsonl.gz --batch 8 --accum 4 --steps 4400 --max_len 1024 --lr 1e-4 \
    --out runs/sft_exealm_3b                                    # --accum is the flag still to add
```

Scored: MBPP+ and HumanEval+ pass@1 and bits per byte on reference solutions; CRUXEval; GSM8K; QASPER F1; held-out
library questions with retrieval (accuracy); SFT held-out loss.

**V4. GRPO on code at 0.5B (about 30 GPU-hours).** From the best 0.5B SFT model. Tasks: teacher code problems with
tests, not used in SFT, kept where the SFT model passes 1 to 7 of 8 samples. Without a cache, sampling dominates:
about 15 GPU-hours an arm for 300 steps, a few with the batched sampling of Section 2.4.

| Arm | What | Role |
|---|---|---|
| a | `grpo.py`, 300 steps, `--prompts 8 --group 8 --max_new_tokens 512` | Main |
| b | `sft.py` on the teacher's solutions to the same tasks, same number of passes | Control: is RL better than more SFT? |
| c | a with `--beta 0` | The KL term; dropped first if hours are short |

```bash
apptainer exec --containall --no-home <sandbox image> \
  python -m exealm.train.grpo --ckpt runs/sft_exealm_500m/model.pt --tokenizer $TOK --data data/post/code_rl.jsonl.gz \
    --val data/post/code_rl_heldout.jsonl.gz --steps 300 --prompts 8 --group 8 --max_new_tokens 512 \
    --allow_code_exec --out runs/grpo_exealm_500m
```

Scored: held-out task pass@1 and MBPP+. If fewer than about 2,000 tasks have a nonzero pass rate at 0.5B, move V4 to
3B after the cached decoder exists (about 40 GPU-hours more).

**V5. Retrieval as a tool at 0.5B (about 3 GPU-hours).** SFT with 2k search trajectories added, against the same SFT
with always-retrieve. Scored: first-hit rate of the model's queries against the raw question as the query; answer
accuracy on held-out library questions; searches per answer. Repeat at 3B (about 9 GPU-hours) only if 0.5B shows a
gain.

**V6. The retriever (about 2 GPU-hours, mostly CPU).** Token-id BM25 (`bm25_pick`), word-level BM25, a dense
retriever, and the two fused, on library v2 held-out items: first-hit and top-5 rates at 18k entries, at v2's size,
and with 1M padding entries (a v1 plus v2 plus random text stand-in). Picks the retriever for V2 onward.

### 5.3 Budget

| Item | GPU-hours |
|---|---|
| V1 long context, 0.5B | about 25 |
| V2 RAG mid-training, 0.5B and 3B | about 33 |
| V3 SFT, 0.5B and 3B | about 53 |
| V4 GRPO, 0.5B | about 30 |
| V5 tool use, 0.5B | about 3 |
| V6 retriever | about 2 |
| **Experiments** | **about 146** |
| V0 teacher: SFT conversations, code problems, trajectories | about 45 |
| **This plan's total** | **about 191** |
| V0 teacher: library v2 (`library_v2.sb`), if not already counted in the planned runs | about 140 |
| Optional: V4 at 3B, V5 at 3B | about 50 |

The teacher figures assume 1.8M output tokens per GPU-hour. The trial job replaces that rate.

### 5.4 Code to write, in the order the experiments need it

| # | What | Where | Size | Needed by |
|---|---|---|---|---|
| 1 | SFT and code-problem generators, new filters, OpenAI-API backend | `experiments/post/teacher.py`, `build.py` | medium | V0 |
| 2 | Sandbox runner (Apptainer, no network) used by filters and `reward_code` | new script, `grpo.py` | small | V0, V4 |
| 3 | `rope_base`; optional `stream_kv_windows`; long-document pack filter; passkey test | `config.py`, `baseline.py`, `common.py`, `experiments/` | small | V1 |
| 4 | `rag_pack.py --topk`, marker, target spans; loader target-only loss and aligned rows; `score.py` top-k, `streamed` layout, answer accuracy; `rag_post.sh` settings | `experiments/library/`, `exealm/data/loader.py` | medium | V2 |
| 5 | `sft.py --accum` | `sft.py` | small | V3 |
| 6 | Batched sampling with padding masked; a per-token cache from the stream state (or the stream decoder); `exealm/eval/gen.py` writing samples for official scorers; the missing benchmark exports | `generate.py`, `exealm/eval/`, `exealm/data/benchmarks.py` | medium | V3, V4 |
| 7 | `grpo.py`: micro-batches, skip equal-reward groups, no reference at `--beta 0`, F1 reward | `grpo.py` | small | V4 |
| 8 | Tool loop in chat and evaluation; trajectory generator | `chat.py`, `exealm/eval/gen.py`, `teacher.py` | medium | V5 |
| 9 | Word-level BM25, dense retriever, fusion | `experiments/library/search.py` | medium | V6 |
| 10 | FSDP in `pretrain.py`, `sft.py`, `grpo.py`; repository packer | | large | 9B only |

## 6. Risks and what would change the plan

| Risk | What would show it | What changes |
|---|---|---|
| The model-to-model gap to Qwen3.5-9B is too large to close (0.25B at 1.50 times its bits per byte) | The 3B pair and its trend | The application leads with system-to-system and fully open claims, and says plainly what the model alone does |
| ExeaLM's lead in using entries fades with size, as its text lead did (0.034 to 0.011) | V2: the 3B lead over the Qwen3.5-style model with BM25 | The library is presented as a data asset that any model can use, not as an architecture result |
| Library items flatter retrieval (same writer, copied answers) | V3 gains on library questions without gains on QASPER, BigCodeBench or DS-1000 | Weight the external benchmarks; write library questions with a different model or from papers directly |
| The search finds the right entry less often in a large library | V6 at 1M entries | Hybrid retrieval, top 3 or 5, entries in the stream; Stage E matters more |
| Longer windows cost the stream advantage, or a 4,096 window costs too much at 9B | V1 arms b to d against a | Choose A2, or keep 512-token windows and accept a shorter exact reach |
| Teacher throughput is low or the model does not fit 4 H100s with room for the cache | The trial job of `library_v2.sb` | Fewer library v2 topics; Qwen3-30B-A3B for bulk entries; DeepSeek-V4-Flash only for code solutions and checks |
| The teacher reproduces benchmark items | Decontamination and the paraphrase pass of Section 3.4 drop many items | Drop the seed families that produced them; report what was dropped |
| RL gets no signal at 0.5B (pass rates near zero) | V4 task filter | Easier generated tasks, or V4 at 3B after the decoder exists |
| Model-written code escapes the runner | | Code execution only inside the container; no execution on login nodes |
| arXiv per-paper licenses block releasing the library | License audit | Release only CC0, CC-BY and CC-BY-SA entries; keep the rest for training only |
| Generation without a cache makes evaluation and RL too slow | V3 and V4 timing | The cached decoder moves ahead of V3; export to a vLLM model class is the longer route |

## 7. Open decisions

1. **Long context:** longer windows (A1) or more windows carried (A2); in pretraining's stream phase or after the
   anneal. After V1.
2. **Stage B's place:** after pretraining (today's rule for library features) or inside the anneal as data.
3. **Teacher hours:** whether library v2 (about 140 GPU-hours at the planning rate) and the SFT and RL data (about
   45) come out of the 700 spare, and whether library v2 stays at 130,000 topics if the measured rate is low.
4. **Reasoning traces:** short (up to 1,000 tokens) for validation; whether to test longer traces at 3B.
5. **Open SFT sets:** which subsets of Tulu 3 and Dolci pass the rule against Llama-written or closed-API data;
   whether SmolTalk is out.
6. **Released library license:** CC0, CC-BY and CC-BY-SA papers only, or collection-level attribution.
7. **The headline claim:** "best open 9B" or "best fully open 9B", and the frozen benchmark suite and metric.
8. **DPO at 9B:** none, or a short pass on teacher-judged pairs before GRPO.
9. **Japanese:** keep a small Japanese share (the data and held-out sets have it) or leave it to a later release.
10. **First release retrieval:** always-retrieve, unless V5 at 3B shows learned search calls ahead.
