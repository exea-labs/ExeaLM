# The library: what it holds, how it is searched, how it ships

Plan as of 26 November 2026, 11 p.m. The library is a knowledge base the model searches before it answers: a small
model with it should answer research and code questions like a far larger one. Results: results/expanse/README.md
(sections 7, 8, 11 and 18 for v1; library v2 numbers go in a new section when its tests run).

## What it holds (library v2)

| Tier | What | Size | State |
|---|---|---|---|
| Written | entries by DeepSeek-V4-Flash (284B MoE, MIT): 150,000 research topics drawn from the human tier's own documents (arXiv 50k with machine learning first, papers 30k, English Wikipedia 35k, Japanese Wikipedia 10k, KAKEN 10k, Stack Exchange 15k) and 50,000 code topics, each with questions and adaptation tasks; prompt v3-frontier (the writer's own expert entry, the source only saying what the topic is) | 99M tokens: 145,376 research and 48,936 code entries, 733,756 questions and tasks | done 27 November, 7 a.m. (job 56091937). 268 entries share a 13-token window with a benchmark (PubMedQA 209, MMLU 31, SciQ 8, APPS 8, HumanEval 4, MATH 3, ARC-Easy 3, HellaSwag 2; `v2/bench_hits.json`). Searched with v1 as one written tier (212,857 passages), the answer to a held-out v2 question is in the top passage for 0.703 of 2,000 queries, 0.058 from the human tier, 6.9 ms a query for both tiers (job 56104634) |
| Written | v1: 18,545 entries by Qwen3-30B-A3B | 8M tokens | done, public |
| Human | passages of real text no ExeaLM model trained on (the 90% of each recipe source outside the training slice, plus all of English and Japanese Wikipedia, KAKEN and Stack Exchange), checked against every held-out set and 14 benchmarks (experiments/library/human_tier.py) | 25.0B tokens, 36.8M passages | done: 44% research (arXiv, papers and abstracts, KAKEN), 28% Wikipedia, 21% code (about 2M repositories), 4% Q&A, 2% math (shares of 25.0B tokens, rounded) |

The written tier gives clean, dense answers; the human tier gives breadth. Published 27 November: 18.8B of the human
tier's 25.0B tokens, with a license and URL on every passage
([exea-labs/exealm-library](https://huggingface.co/datasets/exea-labs/exealm-library), `v2/human`). Left out: arXiv
papers under arXiv's default license (5.2B tokens, not redistributable) and Stack Exchange (1.0B, CC-BY-SA needs each
post's author, which the source files lack). The library that ships with the model is the published part and the
written tier; the study's tests search all 25.0B.

## How it is searched

BM25 over token ids (`experiments/library/index.py`), the same search the library tests use, in two tiers: the
written entries and the human passages are searched apart and the model reads the top passage of each. In one ranking
the 36.8M human passages crowd the entries out (the answer in the top passage fell from 0.74 to 0.44 for v1's held-out
questions); in two tiers the written tier keeps 0.72.

Measured on Expanse (H100), the whole library (36.8M human passages and v1), v1's held-out questions as queries:

| Search | Index size | Latency per query (median) | Same top passage as the full index |
|---|---|---|---|
| CPU, full index | 76 GB | 426 ms | 1.00 |
| CPU, compact index (common tokens dropped, 8-bit weights, 3-byte numbers) | 24.6 GB | 349 ms | 1.00 |
| GPU, compact index, part by part | 24.6 GB of GPU memory | 3.9 ms (batched: about 525 a second) | 1.00 |
| GPU, each token's 100,000 best passages (static pruning) | 20.0 GB | 6.2 ms (batched: about 714 a second) | 0.86 (top-5 overlap 0.89) |
| GPU, each token's 20,000 best passages | 10.0 GB | 5.4 ms (batched: about 1,206 a second) | 0.60 |
| **GPU, 100,000 best, all parts merged into one index** | **25.0 GB** | **2.0 ms** (p95 2.8) | 0.86 |
| Tantivy (Rust engine, block-max WAND) on CPU, one 2.4M-passage part | 1.09 GB | 26.6 ms | 0.93 |
| Tantivy on CPU, v1 | 0.01 GB | 0.42 ms | 0.99 |

For a hosted model the merged GPU index is the path: a lookup costs about as much as generating a tenth of a token on
the 9B. Tantivy's index is the smallest by far (about 17 GB for the whole pool, extrapolated from one part) and serves
laptops and CPU servers.

## How it ships

- **One repository** with the weights, the library text as token ids (zstd -19: 50 GB to 22.9 GB to download, measured
  on a 200M-token sample; as UTF-8 text it would be 23.7 GB, `results/expanse/library_compress.json`;
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

- The v2 post-training (jobs 56104636-9, started 27 November, 1:41 p.m.) packs every entry, the 268 that overlap a
  benchmark included. Held-out questions are unaffected. Before any benchmark is read open-book after it, those
  entries and their items come out of the pack (or the benchmark questions they overlap are scored apart).

## Later

A library stored as learned codes the model reads directly, smaller to ship and cheaper to read:
`docs/library_codes_proposal.md` (proposal, not approved).

Knowledge in the weights themselves: memory layers (Berges et al., ICML 2025; key-value tables of up to 128B
parameters read sparsely; in their abstract, models with them beat dense models with twice the compute, most of all
on factual tasks), tested at 0.1B against the same
model with library search. Plain weights cannot hold the library (about 2 bits of knowledge per parameter, Allen-Zhu
and Li, Physics of Language Models 3.3), and putting teacher text into pretraining did not help at 0.1B (section 22).
