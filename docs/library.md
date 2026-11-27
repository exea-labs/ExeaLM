# The library: what it holds, how it is searched, how it ships

Plan as of 26 November 2026, 11 p.m. The library is a knowledge base the model searches before it answers: a small
model with it should answer research and code questions like a far larger one. Results: results/expanse/README.md
(sections 7, 8, 11 and 18 for v1; library v2 numbers go in a new section when its tests run).

## What it holds (library v2)

| Tier | What | Size | State |
|---|---|---|---|
| Written | entries by DeepSeek-V4-Flash (284B MoE, MIT): 150,000 research topics drawn from the human tier's own documents (arXiv 50k with machine learning first, papers 30k, English Wikipedia 35k, Japanese Wikipedia 10k, KAKEN 10k, Stack Exchange 15k) and 50,000 code topics, each with questions and adaptation tasks; prompt v3-frontier (the writer's own expert entry, the source only saying what the topic is) | about 200M tokens | writing on Expanse (job 56091937): research entries by about 2 a.m. on 27 November, code entries and tasks after |
| Written | v1: 18,545 entries by Qwen3-30B-A3B | 8M tokens | done, public |
| Human | passages of real text no ExeaLM model trained on (the 90% of each recipe source outside the training slice, plus all of English and Japanese Wikipedia, KAKEN and Stack Exchange), checked against every held-out set and 14 benchmarks (experiments/library/human_tier.py) | 25.0B tokens, 36.8M passages | done: 44% research (arXiv, papers and abstracts, KAKEN), 28% Wikipedia, 21% code (about 2M repositories), 4% Q&A, 2% math (shares of 25.0B tokens, rounded) |

The written tier gives clean, dense answers; the human tier gives breadth. Licenses: only redistributable sources;
the open questions (KAKEN's mirror, pes2o papers' own licenses, Stack Exchange attribution) are in
experiments/library/README.md and must be settled before the human tier is published.

## How it is searched

BM25 over token ids (`experiments/library/index.py`), the same search the library tests use, in two tiers: the
written entries and the human passages are searched apart and the model reads the top passage of each. In one ranking
the 36.8M human passages crowd the entries out (the answer in the top passage fell from 0.74 to 0.44 for v1's held-out
questions); in two tiers the written tier keeps 0.72.

Measured on Expanse (H100), the whole library:

| Search | Index size | Latency per query | Throughput |
|---|---|---|---|
| CPU, full index | 76 GB | 426 ms median | |
| CPU, compact index (common tokens dropped, 8-bit weights, 3-byte numbers) | 24.6 GB | 349 ms | |
| GPU, compact index | 24.6 GB of GPU memory | 3.5 to 7.5 ms | about 525 a second batched |
| Tantivy (Rust engine, block-max WAND) on CPU, v1 | 0.01 GB | 0.42 ms, same top passage 99.2% of the time | |

Being measured: static pruning (each token keeps its 20,000 or 100,000 best passages), which should cut both the
index and the per-query work several times over; and Tantivy at 2.4M passages, for CPU servers and laptops.

## How it ships

- **One repository** with the weights, the library text as token ids (zstd: 50 GB to about 21 GB to download,
  unpacked once and memory-mapped), the passage index (titles, sources, licenses, URLs) and the search index (Tantivy
  for CPU, the compact index for GPU).
- **Two editions**: core (the written tier, Wikipedia, machine-learning arXiv: about 10 GB, fits a laptop) and full
  (25B tokens).
- **One call**: `from_pretrained()` downloads and loads the model and its library; `ask(question)` searches both tiers
  and answers.

## How it is hosted (the 9B behind a router)

- A retrieval service on one GPU holds the compact index and serves every model replica over the cluster network:
  milliseconds a query against about 50 to 100 ms of the 9B's own prefill, so the library adds a few percent.
- The search starts as the request arrives, alongside tokenization; repeated questions come from a cache.
- The passage text sits uncompressed on local NVMe (50 GB) for random access.

## How the model learns to use it

- RAG post-training (experiments/post/rag_post.sh) on library v2 at 0.5B (ExeaLM and the Qwen3.5-style model) and on a
  mid-run 3B checkpoint, with the top written entry and the top human passage before each item. Long contexts are
  what ExeaLM's streams are for: the passages are read as earlier windows.
- Tests: v1 and v2 held-out questions (topics never trained on), and a new test the written tier cannot answer alone:
  held-out documents and questions no entry was written from, scored with and without the human passages.

## Later

Knowledge in the weights themselves: memory layers (Berges et al., ICML 2025; key-value tables of up to 128B
parameters read sparsely; in their abstract, models with them beat dense models with twice the compute, most of all
on factual tasks), tested at 0.1B against the same
model with library search. Plain weights cannot hold the library (about 2 bits of knowledge per parameter, Allen-Zhu
and Li, Physics of Language Models 3.3), and putting teacher text into pretraining did not help at 0.1B (section 22).
