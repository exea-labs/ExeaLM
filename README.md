# ExeaLM

An open language model built by high schoolers, run by [Exea Labs](https://exealabs.org). Code and weights are public under Apache 2.0.

**The idea:** ExeaLM is a 0.1B-parameter (about 100 million) language model, pretrained from scratch on about 100 billion tokens and then post-trained into an assistant, like any other LLM. What's different is the architecture, **TONN**, which builds two new abilities into every layer: **its own RAG**, where the model learns its own embeddings, decides what to search for, and pulls text from a read-only datastore, and **agent-to-agent communication**, where copies of the model post messages to a shared board and read each other's. Both are circuits inside the network, trained end to end like everything else, not tools bolted on afterwards.

**The bet:** a 0.1B model can take on problems far bigger than its size, because it doesn't have to do everything alone. Its knowledge doesn't all have to fit in 100 million weights, because the model looks things up through its own RAG. A large problem doesn't have to fit in one copy's context or skill, because a swarm of copies splits the work and shares what they find through the board. And because both abilities are part of the architecture from the first step of pretraining, the model learns to search and collaborate the way it learns language, not through tools added at the end. This is a hypothesis we are testing, not a result yet. The headline experiment is a swarm of 0.1B agents with retrieval against a single larger model on the same tasks, at the same inference compute.

**Status:** skeleton. No weights are released yet. The model code (TONN, the retrieval circuit, the agent-board circuit, the architecture search) runs and is tested. The data scripts (download, filter, tokenizer, packing) work on small samples. Training at scale, post-training and evals are partly written: those are open tasks.

## Join

Fill out the sign-up form so we can list you as a contributor: https://docs.google.com/forms/d/e/1FAIpQLSdPHVZhuXxBDUf-cJM2zDbj1JZPsduu95Y_kTJ_g4gKAXdDVw/viewform

The repo is public, so you can open an issue or pull request without signing up. The form gives us a record of who worked on the project. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to start. Pick something from the [to-do list](#to-do-list): almost all of it runs on a laptop or a free Colab GPU.

## Start in five minutes

```bash
git clone https://github.com/exea-labs/ExeaLM && cd ExeaLM
pip install -e ".[dev,data]"
pytest                                                           # 17 tests, a few seconds, CPU is fine

# train a tiny model on made-up data (no download, about 10 seconds on a laptop CPU)
python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --datastore_chunks 2000

# the fair comparison: same run with the plain transformer baseline
python -m exealm.train.pretrain --preset tiny --data synthetic --steps 300 --model llama --out runs/llama

# real data: one file of Japanese Wikipedia from the LLM-jp corpus (~320 MB)
python -m exealm.data.download llmjp --subset ja/ja_wiki --max_files 1 --out data/raw
python -m exealm.data.sources                                    # print the two candidate 100B-token mixes
```

Full data walkthrough (filter, tokenizer, pack, then train the 17M `small` preset, which fits on a laptop or a free Colab GPU): [docs/data.md](docs/data.md).

## The plan

### 1. Architecture: TONN

TONN is a sequence model that builds its own architecture from data. A normal transformer layer is attention plus an MLP. A TONN layer instead holds 20 small **circuits**, each one of 14 types:
- memories: delays, leaky running sums, gated memories, a Mamba-like selective state space, an oscillating memory
- local and global mixing: short convolutions, rotary attention
- the two new memory circuits, described in sections 2 and 3

After each TONN layer comes a gated MLP, as in a transformer.

**ExeaLM is a hybrid.** Every third layer is a TONN layer and the others are full attention layers (5 TONN and 9
attention layers in the 0.1B model), and TONN's starting circuits favour short convolutions. Pure TONN lost to a
same-size Llama on real tokens, mostly on code. The hybrid beats it, with both seeds, on every kind of text (bits per
byte, lower is better):

| Model (13M parameters unless marked, 3,000 steps) | English | Japanese wiki | Japanese web | Code |
|---|---|---|---|---|
| Llama-style transformer (2 seeds) | 1.467 | 1.540 | 1.420 | 1.789 |
| Pure TONN (2 seeds) | 1.480 | 1.568 | 1.422 | 1.979 |
| **ExeaLM hybrid (2 seeds)** | **1.452** | **1.521** | **1.404** | **1.746** |
| *27M parameters:* Llama (2 seeds) | 1.446 | 1.523 | 1.402 | 1.778 |
| *27M parameters:* **ExeaLM hybrid (2 seeds)** | **1.412** | **1.483** | **1.371** | **1.702** |
| *48M, same training time:* Llama (2 seeds) | 1.369 | 1.439 | 1.333 | 1.667 |
| *48M, same training time:* **ExeaLM hybrid, 2:1 layout (2 seeds)** | **1.366** | **1.436** | **1.327** | **1.653** |
| *0.1B (90M), same training time:* Llama (1 seed, lr 1e-3) | 1.392 | 1.468 | 1.352 | 1.713 |
| *0.1B (90M), same training time:* **ExeaLM hybrid (1 seed, lr 1e-3)** | **1.373** | **1.451** | **1.339** | **1.692** |

That's small scale and short training. At 6,000 steps the hybrid is still ahead on every kind of text, but by less
(0.006). After the speed work the hybrid costs 1.13 times a Llama per token at 0.1B. Given the same training time on
the same GPU, it ties Llama at 13M and wins at 27M. At 48M the half-attention layout lost (0.014), but the 2:1 layout
(two attention layers per TONN layer) **wins at equal compute at 48M: 1.445 vs 1.452, both seeds, every kind of
text.** That layout is now the default. At 0.1B, with the same training time and the same learning rate, it leads
Llama by 0.017 on every kind of text (one seed so far; seed 1 is running). Details:
[results/README.md](results/README.md#8-overnight-sweep-making-tonn-beat-llama-experimentsrun_queuesh-resultssweep).

Nobody picks which circuit goes where. While the model trains, TONN switches each kind of circuit off on text it has not trained on and keeps it only if the error goes up. Useless circuits are replaced by new ones, drawn more often from the kinds that have proven useful. Early in training the architecture changes; later it is frozen.

Paper: [TONN: Networks That Build Their Own Architecture from Data](https://www.preprints.org/manuscript/202611.0994) (Avneh Singh Bhatia, Preprints.org, November 2026). It is a concept paper and has not been peer reviewed. Its larger-scale results are mixed, so we test TONN against a plain transformer at equal compute on small models before any large run.

The language-model evidence so far comes from small tests only: 4.85M parameters, byte-level English web text ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)), and the same 25 MB of training text for every model.

| Model | Bits per byte (lower is better) |
|---|---|
| Llama-style transformer, same size | 1.813 |
| TONN, seed 0 | **1.738** |
| TONN, seed 1 | 1.766 |
| Transformer, 3× more data (82 MB) | 1.690 |
| TONN, 3× more data (82 MB) | **1.622** |

Caveats:
- TONN currently trains about 2.7 times slower per step, because its kernels aren't optimized yet.
- The seed-to-seed spread is real.
- 4.85M parameters is a long way from 0.1B.

The project's first job is to check that the lead holds at larger scale. Details: [docs/architecture.md](docs/architecture.md).

### 2. Its own RAG (the RG circuit, read-only)

Every layer has at least one **RG** circuit. Once per 64 tokens it:
1. turns what it has read so far into a **query** (each RG circuit learns its own);
2. searches a **read-only datastore** of training-text chunks, using **embeddings made by the model's own encoder** (no outside embedding model);
3. weighs the top results by how well they match;
4. lets each position read the retrieved text with attention.

All of it is learned from the next-token loss. The datastore never returns chunks from the document being read, so the model can't cheat by looking up the answer.

### 3. Agents that talk to each other (the AG circuit, read and write)

Every layer has at least one **AG** circuit, which connects to the **agent board**. Once per 64 tokens, each copy of the model posts a message: what it is about, what it says, and how strongly it wants it read, so it can stay quiet. At every position it reads the earlier messages of the other agents in its swarm with attention, and it can choose to read nothing. The datastore is read-only; the board is read *and* write. At inference, many running agents can share a persistent board.

**Guaranteed, then chosen:** every layer starts with one RG and one AG circuit, and the search can never remove the last one, because both need training before they become useful. The search is free to add more wherever they earn it. Design and open questions: [docs/memory.md](docs/memory.md).

### 4. Swarms: agents that work together

A **swarm** is a few agents (4 by default) sharing one board. Training runs many swarms at once: each batch is split into swarms, each with a private board, and swarms never hear each other.

- **Pretraining: swarms pull data together.** Each agent reads ordinary training text, so swarms cost no extra data. The swarm is chosen so that what one agent reads helps the others: parts of one long document (built), related documents, or the same topic in Japanese and English (to do). Some swarms are random, so agents also learn when to ignore the board. Each agent also searches the datastore on its own and can share what it finds.
- **Post-training: swarms solve tasks together.** Each agent gets part of the information and one must answer. Because a swarm runs in one forward pass, the answer's loss trains every agent's messages directly. After that, RL rewards the whole swarm for solving split-information questions, divide-and-conquer math, and write-and-test code.

To our knowledge, ExeaLM is the first language model with agent-to-agent communication built into its architecture. Until an experiment shows agents learning to say something useful, that is a design, not a result. Design, rules and open questions: [docs/swarms.md](docs/swarms.md).

### 5. Data: about 100 billion tokens

Selected from open corpora, including:
- **[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)** and **[DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)**: filtered English web text, the largest part of the mix.
- **[NII LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4):** 19.5T tokens in total, including 688B Japanese and 17.8T English, plus code, Chinese and Korean. It downloads without an account.
- **The data released with [K2 Horizon](https://ifm.ai/blog/k2)** (Institute of Foundation Models, [collection](https://huggingface.co/collections/IFM/k2-horizon)): [TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) (web and QA), [Pretrain-Behaviors](https://huggingface.co/datasets/IFM/Pretrain-Behaviors), [Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning), [Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning), [SFT-Reasoning](https://huggingface.co/datasets/IFM/SFT-Reasoning).

There are two candidate 100B-token mixes: DATA.md's (English-heavier) and mix proposal v0 (about 21% Japanese). Small ablations decide between them. Every dataset link and license is in [DATA.md](DATA.md) and [docs/data.md](docs/data.md), and both mixes are in [`exealm/data/sources.py`](exealm/data/sources.py).

**Why 100 billion tokens:** extra tokens help less and less. 100B tokens on a 0.1B model is 1,000 tokens per parameter, about 50 times the compute-optimal ratio, on purpose. By the Chinchilla loss fit, predicted loss drops about 0.19 going from 20B to 100B tokens, about 0.06 from 100B to 200B, and about 0.05 from 200B to 400B. Past roughly 200B tokens, extra data buys little. The math is in [DATA.md](DATA.md).

### 6. Training

- **Pretraining:**
  - next-token prediction with a warmup-stable-decay learning rate;
  - the architecture search runs during a pilot run and is then frozen;
  - the retrieval index is refreshed as the model's own encoder learns;
  - the best data is saved for the final decay phase.

  Compute: about 6 × 10¹⁹ FLOPs. That is about 56 MI300X GPU-hours at transformer speed; the default ExeaLM trains at that speed, and retrieval adds about half again (about 85 GPU-hours), so well under a day on 8 GPUs. [docs/training.md](docs/training.md)
- **Post-training,** in this order:
  1. supervised fine-tuning on conversations ([`sft.py`](exealm/train/sft.py));
  2. preference tuning ([`dpo.py`](exealm/train/dpo.py));
  3. reinforcement learning with **GRPO** ([`grpo.py`](exealm/train/grpo.py), the method behind DeepSeek-R1);
  4. **swarm SFT and swarm RL**: swarms share a board and are trained and rewarded together. This is where the AG circuits really learn to talk.
- **Compute:** one central run on cloud compute arranged by Exea. You don't need a GPU to contribute. Weights will be released at https://huggingface.co/exea-labs/ExeaLM.

## Results so far

Small-scale tests on a Mac Studio, all in [results/README.md](results/README.md):
- **Agents learn to talk.** Facts held by one agent reach another only through the board: 99.8% correct with the board,
  1.5% (chance) without, 3 seeds. With 4 agents, one gathers facts from the other three.
- **The model builds its own RAG.** On facts that change every step, it finds the right one among 1,024 it has never
  seen: 100% correct with its datastore, 1.5% (chance) without. It learns its own embeddings, queries and reading.
  The first design never learned; the report says what broke and why. On real text retrieval is not yet a net win:
  with a perfect search it would gain 0.040 bits per token on a Wikipedia knowledge task, but the model's own search
  finds the right article only about 30% of the time.
- **On real text the board helps:** on the hybrid, swarms beat a matched control by 0.014 bits per byte (0.8%, both seeds; 0.006 on pure TONN), and training as a swarm makes each agent better even with its board cut (0.010).
- **Tokenizer: 32k.** It wins or ties 49k and 64k on English, Japanese and code at a fixed token budget.
- **Speed:** the default 0.1B model (2:1 hybrid) trains at the same speed as a same-size Llama (1.01x per token), 1.49x with retrieval on.
- **Pure TONN lost to Llama on real tokens; the hybrid wins.** Pure TONN trailed a same-size Llama, mostly on code. A
  hybrid (TONN layers among attention layers, convolution-heavy start) beats Llama on English, Japanese and code
  with both seeds: at equal size and steps (13M, 27M, 48M) and, with two attention layers per TONN layer, at equal
  training time at 48M (0.007 bits per byte). That is now the default architecture.

## To-do list

Starting points. Almost everything here runs on a laptop or a free Google Colab GPU, using the `tiny` preset or the 17M-parameter `small` preset. Items marked **(starter)** are good first tasks. Items marked **(GPU)** need bigger GPUs: maintainers run those on Exea's compute, and you can still write and test the code on a small model. To claim an item, say so in #exealm or open a draft pull request. Anything that changes the model or the training has to win a fair small-scale comparison first ([CONTRIBUTING.md](CONTRIBUTING.md)).

### Swarms and agents (`exealm/model/board.py`, `exealm/data/loader.py`, [docs/swarms.md](docs/swarms.md))
- [x] Proof of mechanism: agents learn to pass facts over the board during training, 99.8% vs 1.5% with the board cut ([results](results/README.md#1-agents-learn-to-talk-through-the-board-during-training-experimentsswarm_relaypy))
- [x] First real-text experiment: split swarms beat a matched control by 0.006 bits per byte at 5M parameters; split batches cost 0.021 ([results](results/README.md#2-swarms-on-real-text-experimentsswarm_textpy))
- [ ] Swarm batches that keep as many independent stretches of text as solo batches (the 0.021 cost above)
- [x] (starter) Measure board use during training: AG worth in the search log, how often agents read nothing, post strength (`board` in eval records, `board_worth` in search records)
- [ ] Related swarms: group documents that are near each other in the model's own retrieval space
- [ ] Cross-lingual swarms: pair Japanese and English Wikipedia articles on the same topic (interlanguage links)
- [ ] Swarm sizes that vary within a run (2 to 8)
- [ ] Agent identity: a role or id signal on posts so agents can specialize
- [ ] Retrieval division of labor: do agents in a swarm pull different chunks? Try rewarding it
- [ ] Swarm-aware multi-GPU training, keeping each swarm on one GPU (testable on a CPU with two processes)
- [ ] Swarm SFT data: split-information tasks built from multi-hop QA. This also fills the "tool use and agentic" gap in [DATA.md](DATA.md)
- [ ] Swarm SFT loop: loss on the answering agent, trained through the board (test on the `tiny` preset)
- [ ] Swarm RL (GRPO): whole-swarm attempts with a shared reward (test on the `tiny` preset)
- [ ] Swarm task suite: split-information QA, divide-and-conquer math, write-and-test code, propose-and-check, cross-lingual QA
- [ ] Swarm generation: all agents generate in lockstep in one batch
- [ ] Board server so agents on different machines share one board (HTTP or Redis)
- [ ] Message decoder: turn board messages into text people can read
- [ ] Safety: agents treat other agents' messages as information, not instructions (data and an eval)
- [ ] Swarm eval: the same tasks solo vs swarms of 2, 4 and 8
- [ ] Headline eval: a swarm of 0.1B agents with retrieval vs a single larger open model (for example 1B) on the same large tasks at the same inference compute. Build and test the harness now on `tiny` checkpoints and small open models

### Retrieval (`exealm/model/retrieval.py`, `exealm/memory/datastore.py`)
- [ ] FAISS (IVF-PQ) index for the datastore (FAISS runs on a CPU, so start there)
- [ ] Choose what goes in the datastore: a curated few billion tokens of reference text
- [ ] Re-indexing schedule: how stale can the stored keys get? (`small` preset)
- [ ] Cheaper retrieval: cache encodings, share retrievals across layers, tune top-k and chunk length
- [ ] Key size: 32 vs 64 vs 128 numbers (`small` preset)
- [x] (starter) Script that prints what RG retrieved for a prompt (`python -m exealm.eval.show_retrieval`)
- [ ] Swap the datastore at inference: can the model use new facts without retraining?
- [ ] Make retrieval pay off on real text. A perfect search would gain 0.040 bits per token on a Wikipedia knowledge task. Rare-word codes in the keys raise the search's hit rate from 12% to 30%, but the net gain over a no-retrieval model is not yet established (+0.026 and -0.002 on two seeds; results §6). Next: better pooling, hard negatives, a bigger encoder, larger models

### Architecture (`exealm/model/`)
- [x] TONN vs Llama with the `small` preset on real tokens, two seeds each: Llama ahead, most of all on code ([results](results/README.md#7-tonn-vs-llama-on-real-tokens-experimentstokenizer_lmpy---models-))
- [x] Close the gap: TONN layers alternating with attention layers, with a convolution-heavy start, beat Llama at equal size and steps (13M, 27M) and at equal compute at 27M ([results](results/README.md#8-overnight-sweep-making-tonn-beat-llama-experimentsrun_queuesh-resultssweep))
- [ ] Confirm the hybrid's equal-compute lead at 48M (running), then at 0.1B **(GPU)**
- [ ] Hybrid layout: which layers get attention, and how many (layout variants queued)
- [ ] Re-run the swarm and retrieval experiments on the hybrid (they ran on pure TONN)
- [ ] Ablations at `small` size: with and without RG, AG and swarms
- [ ] New circuit types, for example a gated delta-rule memory or sliding-window attention
- [ ] Shape sweep at fixed parameters: depth, width, slots per layer, channels per slot
- [ ] Search schedule: how long to search, how much to replace, how long new circuits are protected (on the hybrid with a good start it adds about 0.001 today)
- [ ] Which circuit types survive the search, and does that change from `tiny` to `small`?
- [ ] (starter) Plot the architecture the search chose, layer by layer, over training
- [ ] (GPU) Confirm the winners at the full 0.1B size

### Speed (`exealm/model/scan.py`)
- [ ] Fused Triton kernels for `decay_scan` and `gated_scan` (any NVIDIA GPU works, including a free Colab T4; must also run on AMD MI300X)
- [ ] A compile-friendly TONN layer: one code path for every circuit type, so `torch.compile` can fuse it (today it gains 8% for the hybrid and 20% for Llama)
- [ ] One scan call for all running-sum circuits in a layer
- [ ] Recurrent state cache, so generating a token doesn't re-read the whole context
- [ ] `torch.compile` support
- [x] Training-speed benchmark: TONN 1.5x a same-size Llama per token, 2.7x with retrieval on ([results](results/README.md#5-speed-of-the-real-model-experimentsspeedpy), `experiments/speed.py`)
- [x] (starter) Generation speed: `python -m exealm.eval.speed` times training and generation tokens per second against a same-size Llama
- [ ] (GPU) Run the tests and a short training run on an MI300X and report what breaks

### Data ([DATA.md](DATA.md), `exealm/data/`)
- [ ] Download and stage every DATA.md source: parallel, resumable, checksummed (build and test it on samples)
- [ ] (starter) Count tokens in each K2 Horizon subset (streams, no big download)
- [ ] (starter) Write a formatter for each K2 Horizon subset, since their fields differ
- [x] Remove near-duplicates across all sources (MinHash LSH, runs on a CPU): `python -m exealm.data.dedup`
- [ ] Decontaminate against every benchmark we report
- [ ] Filter personal data and toxicity
- [ ] Quality classifier for Japanese web text (small classifier, Colab-sized)
- [ ] Mix ablations: three or four versions of the mix with the `small` preset; the best two get confirmed at 0.1B **(GPU)**
- [ ] Streaming loader with per-source weights over many shard files (`MultiShards` in `loader.py` mixes one file per source)
- [ ] (starter) Data card: every source, license, filter and token count

### Tokenizer (`exealm/data/tokenizer.py`)
- [x] Train 32k, 49k and 64k candidates and compare them by bytes per token and small-model loss: 32k wins or ties ([results](results/README.md#4-tokenizer-size-experimentstokenizer_comparepy-experimentstokenizer_lmpy))
- [ ] Re-check 32k against 49k on a bigger model, then publish the tokenizer on Hugging Face
- [ ] Final special tokens: chat roles, agent roles, board

### Training (`exealm/train/pretrain.py`)
- [ ] Multi-GPU: DDP, then FSDP (DDP is testable on a CPU with two processes)
- [x] Exact resume from checkpoints (`--save_every`, `--stop_at`, `--resume auto`)
- [ ] Weights & Biases logging
- [ ] Context extension near the end of training
- [ ] Optimizer: Muon vs AdamW (`small` preset)
- [ ] Multi-token prediction heads (`small` preset)
- [ ] Learning-rate transfer from small models (μP): tune on `tiny`/`small`, carry over to 0.1B
- [ ] (starter) Budget sheet: GPU-hours per run, checkpoint cadence, what we can afford

### Post-training (`exealm/train/`)
- [ ] SFT loop and data loader (pack conversations so they can't see each other; test on `tiny`)
- [ ] DPO loop
- [ ] GRPO loop with math and code rewards
- [ ] Chat template
- [ ] Do long reasoning traces help a model this small? Test on an existing small open model such as SmolLM2-135M (Colab)

### Evals (`exealm/eval/`)
- [ ] (starter) lm-evaluation-harness wrapper (test it on a `tiny` checkpoint)
- [ ] Japanese evals (llm-jp-eval)
- [ ] Retrieval eval: questions answered in the datastore but not in the prompt
- [ ] Run evals automatically on every checkpoint
- [ ] Results table in this README

### Release, docs and community
- [ ] Hugging Face model card and a loader, so the model loads with one line
- [ ] Demo: chat with one agent or a swarm (build it on a `tiny` checkpoint now)
- [ ] (starter) Diagrams: a TONN layer, RG, AG, a swarm
- [ ] (starter) Japanese translation of this README
- [ ] (starter) Glossary of terms
- [ ] (starter) Beginner notebook: train the `tiny` model and look inside it
- [ ] Technical report

## Repo layout

```
exealm/
  config.py              model sizes: "100m" (the real one, 101M parameters), "small" (17M, laptop/Colab), "tiny" (tests)
  model/
    tonn.py              the TONN layer and the full model
    circuits.py          the circuit menu
    retrieval.py         RG: the model's own RAG engine
    board.py             AG: the agent-board circuit
    search.py            worth measurement and circuit replacement
    scan.py              the running-sum recurrences (needs a fast kernel)
    baseline.py          Llama-style transformer to compare against
  memory/
    datastore.py         read-only retrieval datastore
    board.py             persistent read-write agent board
  data/                  sources + licenses, download, quality filters, tokenizer, packing, swarm batches
  train/                 pretrain, sft, dpo, grpo
  eval/                  evaluation plan
docs/                    architecture, memory (RAG + agents), swarms, data, training, ideas
experiments/             one script per experiment (swarm relay, swarms on text, RAG lookup, RAG on text,
                         tokenizer, TONN vs Llama, speed)
results/                 what they found: results/README.md, plus the raw JSON and logs
tests/                   run with pytest
```

## Roadmap

1. [x] Skeleton: TONN model with RG and AG circuits, architecture search, training script, tests
2. [ ] Data pipeline: download, dedupe, filter, and mix the 100B tokens
3. [ ] Tokenizer
4. [ ] Small-scale architecture experiments: TONN vs Llama from 10M to 100M parameters, with and without RG, AG and swarms
5. [ ] Fast kernels and multi-GPU training
6. [ ] Evaluation harness (English and Japanese, retrieval, agents)
7. [ ] Full training run
8. [ ] Post-training (SFT, DPO, GRPO, swarm training) and release

Each phase gets an owner. Open tasks are in the [to-do list](#to-do-list).

## Community

Read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md). Questions go in #exealm on the Exea Discord.

## License

Apache 2.0. See [LICENSE](LICENSE). Training data keeps its own licenses, listed per source in [`exealm/data/sources.py`](exealm/data/sources.py).
