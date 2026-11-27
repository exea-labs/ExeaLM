# ExeaLM data plan for 1T tokens

Draft of 27 November 2026, for the project lead. Not approved. It answers the review comment that there is no plan
for 1T tokens. It extends data recipe v2.1 ([exealm_recipe_v2.json](../experiments/recipe/exealm_recipe_v2.json),
300B tokens) under the rules in [DATA.md](../DATA.md) and these constraints from the project lead:

- No Llama-scored or Llama-rewritten sources: nothing filtered by a classifier trained on Llama labels, nothing
  generated or rewritten by a Llama model.
- Research and coding first.
- Every license must allow training. Redistributable sources are preferred, and the ones that are not are named.
- Japanese stays in the mix.
- About 4 epochs or fewer per source.

**How the numbers were read.** Token counts come from each dataset's card or paper, read on 27 November 2026, and each
one is linked. Cards count with different tokenizers (GPT-2, Llama 2, OLMo's, StarCoder's, Comma's). Our 32k
tokenizer reads 4.38 bytes per token on English, 2.67 on code and 5.36 on Japanese web
([results/tokenizer/compare.json](../results/tokenizer/compare.json)). On code it makes more tokens than the cards
count: StarCoderData's own count is about 3.1 bytes per token (783 GB, about 250B tokens, [card][starcoderdata]). So
code epochs computed from card counts are upper bounds. Where a source has two counts, epochs use the lower one. The
build measures exact counts in our tokenizer before training.

## 1. Summary

ExeaLM-9B pretrains on 1,000B tokens: 900B in the main phase and 100B in the anneal over the last 10% of steps. They
come from about 750B unique tokens in 18 sources, 1.33 passes on average, so this is a larger build than v2.1, not
v2.1 read 3.3 times. Code is 36.5% (research code 7.6%, general code 28.9%), Stack Exchange questions and answers 5.6%,
research text 27.0%, math 4.0% (6.6% with AlgebraicStack's math code, which is counted as research code), general web
and books 20.5%, and Japanese 6.4%. The most repeated sources of known size are OpenWebMath and Japanese Wikipedia, at
3.3 epochs each. The research code files and AlgebraicStack overlap, and together they see 1.9 to 3.8 epochs,
depending on sizes measured at the start of the build. No source is planned above 4 epochs. Every source's license
allows training. All are redistributable except arXiv papers under arXiv's default license. No source in the plan was
scored, filtered or rewritten by a Llama model, as far as its card and paper say. Five candidate sources are held
because their cards do not settle that check (section 2.2), and the 1T total does not depend on them. The check also found that 28B of
v2.1's 300B tokens (9.3%) come from sources that fail the no-Llama rule (section 2.3).

## 2. Sources

### 2.1 The mix

"Available" is the card's count. "Used" counts both phases. Epochs are used divided by available. Licenses were read
from the cards on 27 November 2026; confirm them again before the build.

| Source | Tokens available (citation) | Used | Epochs | Share of 1T | License | Redistributable | Notes, including the no-Llama check |
|---|---|---|---|---|---|---|---|
| **Research code** | | **76B** | | **7.6%** | | | |
| Research code files: Python files that import a scientific or ML library (v2.1's list of 21), plus all R, Julia, Fortran and MATLAB files, from [common-pile/stackv2][cp-stackv2] | 8 to 16B, our estimate: StarCoderData has 60.40 GB of Python and 3.39 GB of R, Julia and Fortran ([StarCoder paper][starcoder-paper]); v2.1 keeps at most 1 in 8 Python files ([download_slice.py](../experiments/recipe/download_slice.py) fetches them 8 times over); the open Stack v2 holds 2 to 4 times StarCoderData's code after near-dedup (4,774.7 GB raw, [card][cp-stackv2], times The Stack v2's dedup ratio of 32.1/67.5 TB, [card][stack-v2]) | 24B | 1.5 to 3.0 | 2.4% | per-file open licenses (Blue Oak Council certified), [card][cp-stackv2] | yes | No Llama: a license filter and our import filter. Measured first; see the shortfall rule below |
| Jupyter notebooks from [common-pile/stackv2][cp-stackv2], code and markdown cells, outputs cut | 10 to 20B, our estimate: 13.12 GB of notebooks in StarCoderData ([StarCoder paper][starcoder-paper]) times 2 to 4, at 2.67 bytes per token | 26B | 1.3 to 2.6 | 2.6% | per-file open licenses | yes | No Llama. That the subset keeps `.ipynb` files is assumed; check on the first shard |
| AlgebraicStack ([olmo-mix-1124][olmo2] `algebraic-stack`) | 11.0B ([Proof-Pile-2][pp2], Llama 2 tokenizer; 6.1B of it Python); 11.8B ([OLMo 2 mix][olmo2]) | 26B | 2.4 | 2.6% | ODC-BY; code keeps its own licenses | yes | No Llama: hand-written heuristics ([card][pp2]). Its Python overlaps the research code files: the two together use 50B of an estimated 13 to 27B unique, 1.9 to 3.8 epochs |
| **General code** | | **289B** | | **28.9%** | | | |
| StarCoder ([olmo-mix-1124][olmo2] `starcoder`), minus the research code files | 83.0B ([OLMo 2 mix][olmo2]) | 80B | 0.96 | 8.0% | ODC-BY; per-file permissive licenses from The Stack v1.2 | yes | No Llama: StarCoder's heuristic filters. Remove repositories on The Stack's current opt-out list |
| Open Stack v2 ([common-pile/stackv2][cp-stackv2], the openly licensed part of The Stack v2), minus everything above | 4,774.7 GB of UTF-8 code in 218,364,133 files ([card][cp-stackv2]), about 1.8T tokens before near-dedup; we assume at least 500B after dedup and file filters | 209B | at most 0.42 | 20.9% | per-file open licenses (Blue Oak Council certified) | yes | No Llama: license filter only ([Common Pile paper][cp-paper]). Replaces v2.1's gated The Stack v1 part. The Stack v2 itself also holds unlicensed files ([StarCoder2 paper][starcoder2-paper]); this subset drops them |
| **Questions and answers** | | **56B** | | **5.6%** | | | |
| Stack Exchange ([common-pile/stackexchange][cp-se]) | 23.9B ([Comma training set][comma]); 33,415,400 threads, 103.7 GB ([card][cp-se]) | 56B | 2.3 | 5.6% | CC BY-SA 2.5, 3.0 or 4.0 per post | yes, in the share-alike folder | No Llama. Authors are in the metadata, so attribution is possible; v2.1's copy (1.5B, Dolmino) lacks them. v2.1 counted Stack Exchange as code |
| **Research text** | | **270B** | | **27.0%** | | | |
| arXiv full text ([olmo-mix-1124][olmo2] `arxiv`, from RedPajama via Proof-Pile-2) | 20.8B ([OLMo 2 mix][olmo2]); 29B ([Proof-Pile-2][pp2], Llama 2 tokenizer) | 60B | 2.9 | 6.0% | ODC-BY as published by AI2; each paper keeps its own license, most of them arXiv's non-exclusive license | partly: only CC BY, CC BY-SA and CC0 papers | No Llama. Training on the non-CC papers is open decision 2. Common Pile's CC-licensed arXiv is 6.0B ([Comma training set][comma]) |
| peS2o v2 ([allenai/peS2o][pes2o]) | 58.6B ([OLMo 2 mix][olmo2]); 38.97M papers, 42.01B whitespace-separated words ([card][pes2o]) | 139B | 2.4 | 13.9% | ODC-BY | yes, as AI2 publishes it | No Llama: no model-based scoring on the card; cutoff 3 January 2023. Deduped after arXiv, which removes arXiv copies: at a fifth removed it sees 3.0 epochs |
| PubMed Central articles under CC BY, CC BY-SA or CC0 ([Common Pile][cp-paper] `pubmed`) | 36.6B ([Comma training set][comma]) | 30B | 0.8 | 3.0% | CC BY, CC BY-SA or CC0 per article | yes | No Llama. Overlaps peS2o and is deduped after it: at half removed, 1.6 epochs |
| English Wikimedia wikis (Wikipedia, Wikibooks, Wikisource, Wiktionary and others, March 2025, talk pages included; [Common Pile][cp-paper] `wikimedia`) | 15.8B ([Comma training set][comma]) | 32B | 2.0 | 3.2% | CC BY-SA | yes, in the share-alike folder | No Llama. Replaces v2.1's 4B of Wikipedia |
| Open-access academic books ([Common Pile][cp-paper] `doab`) | 3.0B ([Comma training set][comma]) | 9B | 3.0 | 0.9% | open licenses per book, as Common Pile filters them | yes | No Llama. New source |
| **Math** | | **40B** | | **4.0%** | | | |
| OpenWebMath ([olmo-mix-1124][olmo2] `open-web-math`) | 12.2B ([OLMo 2 mix][olmo2]); 14.7B ([card][owm], Llama tokenizer per the [paper][owm-paper]) | 40B | 3.3 | 4.0% | ODC-By 1.0, plus Common Crawl's terms of use | yes | No Llama: its MathScore classifier learned labels set by a rule (does the page hold one of the 100 most common LaTeX commands), not by an LLM ([paper][owm-paper]). Deduped after Stack Exchange, which raises its epochs; measured and capped |
| **General web and books** | | **205B** | | **20.5%** | | | |
| DCLM-Baseline ([mlfoundations/dclm-baseline-1.0][dclm]), highest fastText scores | 4T tokens, 3B documents ([card][dclm]); 3.70T ([OLMo 2 mix][olmo2]) | 200B | 0.05 | 20.0% | CC-BY-4.0, plus Common Crawl's terms of use | yes | No Llama: the fastText classifier was trained on OpenHermes 2.5 and r/ExplainLikeImFive ([card][dclm]); no Llama labels |
| Project Gutenberg ([Common Pile][cp-paper] `project_gutenberg`) | 5.7B ([Comma training set][comma]) | 5B | 0.9 | 0.5% | public domain in the US | yes | No Llama. Replaces v2.1's 3B of Dolma books |
| **Japanese** | | **64B** | | **6.4%** | | | |
| FineWeb-2 Japanese ([HuggingFaceFW/fineweb-2][fineweb2] `jpn_Jpan`) | 1.50 TB of UTF-8 text, 400,138,563 documents, 331B words ([card][fineweb2]); about 280B tokens at our 5.36 bytes per token | 56B | 0.2 | 5.6% | ODC-By 1.0, plus Common Crawl's terms of use | yes | No Llama: heuristic filters and GlotLID language ID; the authors avoided model-based quality filtering ([card][fineweb2]) |
| Japanese Wikipedia ([wikimedia/wikipedia][wiki-ja] `20231101.ja`) | about 1.5B (v2.1 takes all of it); 1,389,467 articles, 3.94 GB of Parquet ([dataset size API][wiki-ja]) | 5B | 3.3 | 0.5% | CC BY-SA 3.0 | yes, in the share-alike folder | No Llama |
| KAKEN research-grant summaries (LLM-jp Corpus v4 `ja_kaken`) | about 0.9B (LLM-jp corpus v3 count in the [Drop-Upcycling paper][dropupc]; v4 not counted) | 2.5B | 2.8 | 0.25% | CC BY 4.0 | yes | No Llama. LLM-jp's GitLab failed every download on 24 November ([data_recipe.md](data_recipe.md)); find a mirror or retry |
| Aozora Bunko (clean) and e-Gov law ([globis-university/aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean), [nlp-waseda/e_gov](https://huggingface.co/datasets/nlp-waseda/e_gov)) | about 0.1B each ([data_recipe.md](data_recipe.md), recipe v1) | 0.5B | 2.5 | 0.05% | CC BY 4.0 | yes | No Llama. Back from recipe v1 |
| **Total** | about 750B unique used | **1,000B** | 1.33 on average | **100%** | | all but non-CC arXiv | |

**Shortfall rule.** The research code files, the notebooks and the open Stack v2 are estimates. The build counts them
first. If a source's unique tokens after dedup are fewer than a quarter of its planned use, it is read 4 times and its
remaining tokens go to the source with the most headroom in the same group: the open Stack v2 for code, PubMed and
then DCLM for research text and math, FineWeb-2 Japanese for Japanese. The total stays 1,000B and no source goes above
4 epochs.

**By bucket, against v2.1.**

| Bucket | Main phase | Anneal | Total | Share of 1T | v2.1 share of 300B |
|---|---|---|---|---|---|
| Research code | 63B | 13B | 76B | 7.6% | 24.3% |
| General code | 270B | 19B | 289B | 28.9% | 28.3% |
| Stack Exchange | 44B | 12B | 56B | 5.6% | 1.0% |
| Research text (papers, reference) | 232B | 38B | 270B | 27.0% | 22.3% |
| Math | 34.5B | 5.5B | 40B | 4.0% | 12.3% |
| General web and books | 197B | 8B | 205B | 20.5% | 7.0% |
| Japanese | 59.5B | 4.5B | 64B | 6.4% | 3.3% |
| QA and instruction-style text | 0 | 0 | 0 | 0% | 1.3% |
| **Total** | **900B** | **100B** | **1,000B** | **100%** | 100% (rounded) |

Research code and math fall well below v2.1's shares. Most of v2.1's research code (45B of 73B) and part of its math
(24B of 37B) came from sources this plan excludes or holds (sections 2.2 and 2.3), and what remains is capped at about
4 epochs. Section 5 lists ways to raise both.

### 2.2 Held until their makers confirm how they were made

These pass the license check, but their cards do not say enough to pass the no-Llama check. None of them counts toward
the 1T total.

| Source | Size | Why it is held | If cleared |
|---|---|---|---|
| [IFM/Code-Reasoning][ifm-code] (`code-no-thinking`, `code-no-thinking-hard`, `code-task-synthesis`, `codeio`) | not counted ([docs/data.md](data.md)); v2.1 uses 35B of it | The card names no generating model and says some subsets went through synthetic-data generation. The parts of the [K2-V2 report][k2v2] we read do not cover these subsets | Mostly into the anneal (v2.1 weight 3.0), in place of open Stack v2 and DCLM tokens |
| [IFM/Math-Reasoning][ifm-math] (`math-rewrite`, `math-dialogue`) | not counted; v2.1 uses 6B | The card names no model and no source text. If the rewrites start from MegaMath or FineMath pages, the text was selected by a Llama-trained classifier | Into the anneal, in place of DCLM tokens |
| [IFM/Pretrain-Behaviors][ifm-beh] (`data-science`, `planning`, `reasoning`) | not counted; v2.1 uses 10B | The [TxT360-Midas card][midas] says its reasoning and other behaviors were generated with Qwen2.5-32B-Instruct and Qwen3-32B. The IFM card does not say which of its subsets those are | Research code, at v2.1's 10B |
| FinePDFs Japanese ([HuggingFaceFW/finepdfs][finepdfs] `jpn_Jpan`) | 116.31B tokens, 31,393,277 documents ([card][finepdfs]) | No model-based quality filter outside English. The card says a vision-language model discards pages where extraction hallucinated, and does not name it | Japanese research and government PDFs, in place of FineWeb-2 Japanese tokens |
| olmOCR science PDFs ([Dolma 3 pool][dolma3]) | 972B tokens, 101M documents ([card][dolma3]) | The [Olmo 3 report][olmo3] partitions them by WebOrganizer topic, a classifier trained to imitate Llama-3.1-405B-Instruct ([WebOrganizer paper][weborganizer]). The pool holds 101M documents against 108M after filtering in the report, and the gap is not explained. The papers keep their publishers' copyright | Not planned even if cleared, because of the copyright question |

### 2.3 Excluded under the no-Llama rule

| Source | Size | Reason | In v2.1? |
|---|---|---|---|
| [FineWeb-Edu][fineweb-edu] | 1.3T tokens (score 3 and up); 5.4T (score 2 and up) | Its classifier was trained on annotations from Llama3-70B-Instruct ([card][fineweb-edu]) | no |
| [FineMath][finemath] | 34B (3+), 9.6B (4+) | Classifier trained on Llama-3.1-70B-Instruct annotations ([card][finemath]) | no |
| [Stack-Edu][stack-edu], and Common Pile's `stackv2_edu` (67.8B, [Comma training set][comma]) | 125B ([card][stack-edu]) | Stack-Edu's classifiers were trained on Llama3-70B-Instruct annotations ([card][stack-edu]); Common Pile says it follows SmolLM2's process for `stackv2_edu` ([paper][cp-paper]) | no |
| [MegaMath][megamath], all parts | 371.6B: Web 263.9B, Web-Pro 15.1B, Code 28.1B, synthetic 64.5B ([card][megamath]) | Web: the [paper][megamath-paper] says Llama-3.1-70B-Instruct annotated math relevance for its fastText seed data (its introduction describes the seeds differently, so we read it as Llama-scored until LLM360 says otherwise). Web-Pro: rewritten by Llama-3.3-70B-Instruct. Code: recalled with Llama-3.1-70B-Instruct annotations. Synthetic QA: partly Llama-3.3-70B-Instruct | yes: `megamath` (MegaMath-Web), 18B |
| [TxT360-v2][ifm-txt] `web-high-medium`, `txt360-qa`, `web-high-nltk-qa` | not counted | The [K2-V2 report][k2v2] builds its BestOfWeb tier with the ProX document filter, and ProX's models were trained on annotations from Llama-3 models ([ProX paper][prox]). TxT360-QA is generated (by Mistral-7B-Instruct-v0.3) from BestOfWeb documents. [data_recipe.md](data_recipe.md) also records the FineWeb-Edu score as one input to the web buckets | yes: `txt360_web` 6B, `txt360_qa` 3B |
| IFM/SFT-Reasoning `3efforts-pretrain` | not counted | It appears to come from [TxT360-3efforts][3efforts], which takes STEM items from Llama-Nemotron-Post-Training-Dataset-v1 and says most, not all, answers were regenerated with GPT-OSS-120B | yes: `sft_pretrain`, 1B |
| Dolma 3 mix (6T) | 5.93T ([Olmo 3 report][olmo3]) | Contains Stack-Edu and FineMath, and weights web text by WebOrganizer topic | no |

Also left out, for license reasons: The Stack v2's unlicensed files, and Nemotron-CC (gated under an "other" license,
as [DATA.md](../DATA.md) already says).

### 2.4 Reserves

Clean sources not needed for 1T, kept as headroom: the rest of DCLM-Baseline (about 3.8T); [FineWeb][fineweb]
(18.5T GPT-2 tokens, ODC-By; heuristic filters only, and its authors say they avoided machine-learning quality
filters); the rest of the open Stack v2 (at least 290B by our estimate); the rest of FineWeb-2 Japanese (about 220B);
USPTO patents (157.4B, public domain, [Comma training set][comma]); Common Pile's openly licensed peS2o (43.3B), the
strict alternative to peS2o v2 in open decision 2.

## 3. Order and annealing

v2.1 sets the priority as research code, then general code, then research papers, then math, then everything else.
This plan keeps it in three places.

1. **Dedup precedence.** Near-duplicates keep the copy from the first source listed
   ([exealm/data/dedup.py](../exealm/data/dedup.py)). The order is: research code files, notebooks, AlgebraicStack,
   StarCoder, open Stack v2, Stack Exchange, arXiv, peS2o, PubMed, Wikimedia, DOAB, OpenWebMath, Gutenberg, DCLM; then
   Japanese Wikipedia, KAKEN, Aozora and e-Gov, FineWeb-2 Japanese.
2. **Build order.** Research code is built and counted first, because its size is the least certain and the shortfall
   rule depends on it.
3. **The anneal.** The last 10% of steps reweights toward the best material, in v2.1's order.

The main phase (the first 900B) reads every source at its main share, in shards in a fixed shuffled order, as the
scaling study does. The mixed stream phase covers the last 20% of steps (`--stream_mix 0.5`,
[decisions.md](decisions.md), 26 November, 7 p.m.), so the anneal falls inside it. Capped sources reuse their own documents in the anneal, and those tokens are
counted in their epochs above. Sources with spare supply draw fresh documents for the anneal: the 8B of DCLM with the
highest fastText scores and 13B of open Stack v2 files, both held out of the main phase.

| Source | Main phase | Anneal (last 10%) | Anneal share over main share | v2.1 anneal_weight |
|---|---|---|---|---|
| Research code files | 19B | 5B | 2.4 | 2.0 |
| Notebooks | 22B | 4B | 1.6 | (new) |
| AlgebraicStack | 22B | 4B | 1.6 | 1.5 |
| Stack Exchange | 44B | 12B | 2.5 | 3.0 |
| DOAB | 7B | 2B | 2.6 | (new) |
| Wikimedia (English) | 27B | 5B | 1.7 | 2.0 |
| peS2o | 120B | 19B | 1.4 | 2.0 |
| arXiv | 52B | 8B | 1.4 | 2.0 |
| PubMed | 26B | 4B | 1.4 | (new) |
| OpenWebMath | 34.5B | 5.5B | 1.4 | 1.5 |
| Japanese Wikipedia | 4B | 1B | 2.3 | 1.5 |
| KAKEN | 2B | 0.5B | 2.3 | 1.5 |
| StarCoder | 74B | 6B | 0.7 | 1.0 |
| Open Stack v2 | 196B | 13B | 0.6 | 1.0 (The Stack v1) |
| FineWeb-2 Japanese | 53B | 3B | 0.5 | 0.7 |
| DCLM | 192B | 8B | 0.4 | 0.5 |
| Gutenberg | 5B | 0 | 0 | 1.0 |
| Aozora and e-Gov | 0.5B | 0 | 0 | (not in v2.1) |
| **Total** | **900B** | **100B** | | |

v2.1's weights are applied before normalizing, so its ratios are its weights divided by about 1.65. Research code,
Stack Exchange and papers get more of the anneal here than in v2.1 because v2.1 gave much of its anneal to the
synthetic sets held above (weight 3.0). If IFM's sets are cleared, Code-Reasoning and Math-Reasoning go first into the
anneal, in place of open Stack v2 and DCLM tokens.

## 4. Processing

The pipeline is the one that built `slice-30b-r2` ([experiments/data_mix/README.md](../experiments/data_mix/README.md)),
run per file and streamed, plus a near-dedup step sized for this build.

### 4.1 Steps

1. **Download.** Pin every dataset to a commit and record it in `sources.json`, as the benchmark exporter already does.
   Take files in a fixed shuffled order (seed 0), as [download_slice.py](../experiments/recipe/download_slice.py) does.
   Mirror the raw files on Expanse once, so a failed host (LLM-jp's GitLab on 24 November) does not stall the build.
2. **Extract and filter.** The existing extract step (empty texts and special-token strings dropped), the research code
   filter, StarCoder-style file filters for the open Stack v2, notebook conversion, and removal of repositories on The
   Stack v2's current opt-out list (v2.2.0 removed opt-outs made before 29 July 2026, [card][stack-v2]).
3. **Exact dedup** across every source, whitespace and case ignored (the existing `dedup` step).
4. **Near-dedup** across every source with MinHash LSH, the FineWeb setup in
   [exealm/data/dedup.py](../exealm/data/dedup.py): word 5-grams (character 5-grams for Japanese), 112 hashes in 14
   bands of 8. That script keeps its index in memory and is meant for tens of millions of documents. This build has
   about 810M documents (218M code files, about 330M DCLM documents fetched, 80M FineWeb-2 Japanese, the rest smaller),
   so the index has to be sharded by band, or the step run with datatrove's MinHash, which uses the same parameters.
   Signatures take about 0.45 TB (810M documents times 560 bytes).
5. **Validation split.** 0.1% of every source's documents by hash, plus long documents (4,096 tokens and up) kept
   whole, as in v2.1.
6. **Tokenize and decontaminate** (section 4.3), then **assemble** packed parts per source, **verify**, and publish.

### 4.2 Decontamination

- **Rule.** A document is dropped if it shares any 13-token window with a benchmark, counted in our tokenizer's tokens,
  not words ([docs/data.md](data.md)). The slice's refinements stay
  ([experiments/data_mix/README.md](../experiments/data_mix/README.md)): for code benchmarks only the problem statement
  is checked; a window found in 2 or more benchmark items is treated as boilerplate; a window needs at least 3 distinct
  words of 3 or more letters (a CJK character counts as half a word). Without them, plain matching dropped 76% of
  `code_reasoning` on generic code lines; with them, 0.8%.
- **Benchmarks.** The 14 sets the slice build uses: HumanEval, MBPP (test), APPS (test), CodeContests (test),
  LiveCodeBench, GSM8K (test), MATH (test), ARC-Easy and ARC-Challenge (test), MMLU (test), SciQ (test), PubMedQA
  (labeled), HellaSwag (validation) and JCommonsenseQA (validation) ([exealm/data/benchmarks.py](../exealm/data/benchmarks.py),
  [experiments/data_mix/bench_extra.py](../experiments/data_mix/bench_extra.py)). DATA.md's rule is every benchmark we
  report, so any benchmark the 9B report adds is exported and indexed before the build starts.
- **Report.** Documents dropped per source and per benchmark (`decontam.json`). PubMedQA is built from PubMed
  abstracts, so expect drops in PubMed and peS2o; the library's written tier had 209 PubMedQA hits of 268 benchmark
  hits ([docs/library.md](library.md)).
- **Limits.** Exact token overlap only. Paraphrases, translations and benchmark solutions reworded in code repositories
  are not caught.

### 4.3 Tokenization and packing

- Retrain the 32,768-token byte-level BPE on a 1.5 GB sample of this mix (15 to 30 minutes,
  [experiments/data_mix/README.md](../experiments/data_mix/README.md)). The current tokenizer's sample included
  MegaMath and TxT360-v2 text, which this plan excludes. The vocabulary size stays at 32,768 because the 9B's measured
  speed uses it ([experiments/expanse/nine_b.py](../experiments/expanse/nine_b.py)).
- Pack as `uint16` with document boundaries kept, one directory of parts per source, as the slice is packed.
- Share-alike sources (Wikimedia, Japanese Wikipedia, Stack Exchange) stay in their own folder under their own license.
  arXiv papers under arXiv's default license are not published as text or tokens; the data card lists their arXiv IDs
  so others can rebuild them.

### 4.4 Storage and cost

Sizes follow from about 750B unique tokens and measured rates. Assumed rates are marked.

| Item | Size or time | Basis |
|---|---|---|
| Downloads (compressed) | about 2.1 to 2.4 TB | 1.1 to 1.4 TB for all of the open Stack v2 (4,774.7 GB of text, gzip of 3.5 to 4.5 times assumed); the rest at 1.2 compressed bytes per token with v2.1's oversampling ([download_slice.py](../experiments/recipe/download_slice.py)) |
| Kept text, UTF-8 | about 2.8 TB | 324B code tokens at 2.67 bytes, 372B English at 4.38, 59B Japanese at 5.36 |
| Packed tokens | 1.5 TB as `uint16`; 0.69 TB with zstd -19, 0.90 TB with zstd -3 | 2 bytes per token; 0.91 and 1.21 bytes per token measured on library text ([library_compress.json](../results/expanse/library_compress.json)) |
| Peak disk | about 6.5 TB if nothing is deleted on the way; about 4 TB if raw files are deleted after extraction | sum of the rows above plus 0.45 TB of MinHash signatures and the tokenized intermediates |
| Extract | about 60 process-hours | 8.4 TB of text read at 39.5 MB/s per process (measured on a 12-core Mac, [data_mix README](../experiments/data_mix/README.md)) |
| Near-dedup | about 230 to 450 core-hours | 810M documents at an assumed 500 to 1,000 documents per second per core; not measured at this scale |
| Tokenize and decontaminate | about 410 to 520 process-hours | about 825B tokens (750B plus top-ups) at 0.44 to 0.56M tokens/s per process (measured); decontamination adds about 3% |
| Assemble and verify | about 15 to 25 process-hours | 10 to 15M tokens/s per source (measured) |
| CPU in total | about 1,000 to 2,500 core-hours, or 8 to 20 node-hours on Expanse's 128-core nodes | the rows above, with a factor of 1.5 to 2 assumed for Expanse's EPYC 7742 cores against the Mac's, plus margin; node size from the [Expanse user guide][expanse] |
| Cost on a commercial cloud | a few hundred dollars | at an assumed \$0.03 to \$0.05 per core-hour and \$0.02 to \$0.03 per GB-month for one month of 6.5 TB |

The cost is small next to the pretraining run itself: \$202,000 to \$299,000 for 75,000 H100 GPU-hours at list prices
(funding report, first page). The constraint is disk and where the build runs: the MI300X box has about 420 GB free
([data_mix README](../experiments/data_mix/README.md)), so it cannot hold the 1.5 TB of packed output. Expanse's Lustre
file system (12 PB, [SDSC][hpc]; scratch capped at 2 million files per user and not backed up,
[ACCESS documentation][access-expanse]) can, if the build writes large shards.
Whether CPU hours on Expanse are available beside the GPU allocation is open (section 5).

### 4.5 Timeline

The funding report puts the 1T data build in month 1. Four weeks:

| Week | Work | Done when |
|---|---|---|
| 1 | Ask IFM and Hugging Face the provenance questions in section 2.2. Write the per-file streaming loop (not written yet, [data_mix README](../experiments/data_mix/README.md)) and the band-sharded near-dedup. Freeze the benchmark list. Retrain the tokenizer. Run the whole pipeline on 1% of every source | measured: research code and notebook sizes, the open Stack v2's dedup ratio, MinHash speed; shortfall rule applied |
| 2 | Download and extract every source; exact dedup | raw text on Expanse, per-source counts |
| 3 | Near-dedup, validation split, tokenize and decontaminate | per-source tokens, `decontam.json` |
| 4 | Assemble, verify, write the data card (license, revision, unique tokens and epochs per source), publish to `exea-labs/exealm-pretrain`; spare days for reruns | the 1T build, readable with `open_data` |

## 5. Risks and open decisions

### 5.1 Open decisions for the project lead

1. **Should 9B pretraining exclude the library's documents?** Open. The library's human tier is 25.0B tokens: the 90%
   of each recipe source outside the training slice, plus all of English and Japanese Wikipedia, KAKEN and Stack
   Exchange; 44% research, 28% Wikipedia, 21% code, 4% Q&A, 2% math ([docs/library.md](library.md)). This plan, as
   written, includes them, because it reads most of each capped source in full.
   - Include: the 9B holds its best research text in its weights. Tests of what the library adds can no longer use the
     human tier as text the model never saw.
   - Exclude: those tests stay clean. Pretraining loses Japanese Wikipedia and KAKEN entirely (7.5B used, moved to
     FineWeb-2 Japanese), English Wikipedia's articles (part of the Wikimedia row), and about 11B of paper text. The
     plan still fits under 4 epochs. If Wikipedia's articles are about 5B of the Wikimedia row's 15.8B (our estimate,
     not counted), that row rises to about 3 epochs on what remains.
2. **Papers without an open license.** arXiv papers under arXiv's default license, and any peS2o papers outside CC BY,
   CC BY-SA and CC0, are trained on as v2.1, OLMo 2 and RedPajama do, on the strength of AI2's ODC-BY release; the
   papers' own licenses do not grant that. The strict version uses Common Pile's open subsets: peS2o 43.3B, arXiv 6.0B,
   PubMed 36.6B ([Comma training set][comma]). Paper supply falls from about 116B to about 86B before overlaps, and at
   the same epochs about 75B used tokens move to other sources.
3. **Raising research code and math.** Both are below v2.1's shares (7.6% against 24.3%, 4.0% against 12.3%). Options,
   none in this plan yet:
   - clearing IFM's sets (section 2.2);
   - our own research-code and math classifiers, trained on labels from a model that is not Llama (DeepSeek-V4-Flash,
     MIT, already writes the library), run over the open Stack v2 and DCLM;
   - OpenWebMath's rule-based method (labels from LaTeX commands, no LLM) run again over FineWeb or DCLM;
   - research code written by DeepSeek-V4-Flash: at the funding report's teacher-data rate (2,700 GPU-hours for 4.9B
     tokens), 10B tokens would cost about 5,500 GPU-hours.
4. **Japanese share.** 6.4% here, against 3.3% in v2.1, 10% in recipe v1 and 21% in candidate B
   ([docs/data.md](data.md)). DATA.md asks for small ablations before the mix is fixed.
5. **Where the build runs.** Expanse CPU nodes (if an allocation for them exists), a rented CPU machine, or the MI300X
   box (too little disk without streaming uploads).
6. **The study's own data.** `slice-30b-r2`, which every study run trained on, follows v2.1, so about 9% of it comes
   from the sources excluded in section 2.3, and its tokenizer was trained on a sample that included them. The funding
   report describes the 9B's data as recipe v2.1 (Table "The 9B pretraining run"). Whether to restate that, and how
   the paper describes the study's data, is for the lead.

### 5.2 Risks

- **Counts.** Card counts use different tokenizers. The research code, notebook and open Stack v2 sizes are estimates.
  Overlaps (peS2o with arXiv and PubMed, AlgebraicStack with research code, OpenWebMath with Stack Exchange) are not
  measured. Week 1 measures all of them; the shortfall rule keeps every source at or under 4 epochs.
- **Provenance.** The no-Llama check reads cards and papers. A pipeline step a card does not mention would not be
  caught. Every source is pinned to a commit, and the check is repeated before release. The MegaMath paper is
  inconsistent about its seed data, and we excluded it on the stricter reading.
- **Repetition.** [Muennighoff et al.][muennighoff] found that up to 4 epochs of repeated data changed loss
  negligibly compared with unique data. That result repeats a whole dataset; repeating some
  sources inside a mix is common practice (Comma v0.1-1T read peS2o, arXiv, Stack Exchange and Wikimedia 6 times,
  [Comma training set][comma]) but is not the same experiment.
- **Mix quality.** General web is 20.5%. Comma v0.1-1T, also trained on a mix heavy in papers and code, lagged
  comparable models on HellaSwag and PIQA ([Common Pile paper][cp-paper]). FineWeb-2 Japanese has no quality
  classifier. DATA.md asks for small ablations first: train the 0.5B on this mix and on v2.1 for the same tokens and
  compare on the 18 held-out sets.
- **Licenses and terms.** Common Crawl text (DCLM, OpenWebMath, FineWeb-2) relies on the publishers' dataset licenses
  and Common Crawl's terms of use. In 2024 Stack Exchange added download terms against LLM training to its official
  dumps; critics say this conflicts with CC BY-SA, which forbids added restrictions ([report][se-terms]; secondary
  source, the agreement itself not read). Common Pile built its copy from community dumps of December 2024 and the
  official dumps of July 2024 ([Common Pile paper][cp-paper]). The Stack's opt-outs keep growing; the build filters
  against the latest list.
- **Engineering.** The streaming loop and the band-sharded near-dedup are not written. MinHash speed at 810M documents
  is not measured. Hugging Face rate limits or outages can stretch week 2.

## What changes from v2.1

- Out (no-Llama rule): `megamath` (18B), `txt360_web` (6B), `txt360_qa` (3B), `sft_pretrain` (1B).
- Held (provenance unclear): `code_reasoning` (35B), `math_reasoning` (6B), `pretrain_behaviors` (10B).
- Replaced: the gated The Stack v1 part with Common Pile's openly licensed Stack v2; Dolmino's Stack Exchange (no
  authors) and Wikipedia with Common Pile's versions; Dolma books with Common Pile's Gutenberg.
- New: notebooks, PubMed (open licenses only), open-access academic books (DOAB), and Aozora Bunko and e-Gov from
  recipe v1.
- The machine-readable recipe (a v3 JSON next to v2) is not written yet; it follows once the lead approves the shares.

[olmo2]: https://huggingface.co/datasets/allenai/olmo-mix-1124
[pp2]: https://huggingface.co/datasets/EleutherAI/proof-pile-2
[owm]: https://huggingface.co/datasets/open-web-math/open-web-math
[owm-paper]: https://arxiv.org/abs/2310.06786
[pes2o]: https://huggingface.co/datasets/allenai/peS2o
[comma]: https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset
[cp-paper]: https://arxiv.org/abs/2506.05209
[cp-stackv2]: https://huggingface.co/datasets/common-pile/stackv2
[cp-se]: https://huggingface.co/datasets/common-pile/stackexchange
[starcoderdata]: https://huggingface.co/datasets/bigcode/starcoderdata
[starcoder-paper]: https://arxiv.org/abs/2305.06161
[stack-v2]: https://huggingface.co/datasets/bigcode/the-stack-v2
[starcoder2-paper]: https://arxiv.org/abs/2402.19173
[dclm]: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
[fineweb]: https://huggingface.co/datasets/HuggingFaceFW/fineweb
[fineweb-edu]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
[finemath]: https://huggingface.co/datasets/HuggingFaceTB/finemath
[stack-edu]: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
[megamath]: https://huggingface.co/datasets/LLM360/MegaMath
[megamath-paper]: https://arxiv.org/abs/2504.02807
[fineweb2]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
[finepdfs]: https://huggingface.co/datasets/HuggingFaceFW/finepdfs
[dolma3]: https://huggingface.co/datasets/allenai/dolma3_pool
[olmo3]: https://arxiv.org/abs/2512.13961
[weborganizer]: https://arxiv.org/abs/2502.10341
[k2v2]: https://arxiv.org/abs/2512.06201
[prox]: https://arxiv.org/abs/2409.17115
[midas]: https://huggingface.co/datasets/LLM360/TxT360-Midas
[3efforts]: https://huggingface.co/datasets/LLM360/TxT360-3efforts
[ifm-code]: https://huggingface.co/datasets/IFM/Code-Reasoning
[ifm-math]: https://huggingface.co/datasets/IFM/Math-Reasoning
[ifm-beh]: https://huggingface.co/datasets/IFM/Pretrain-Behaviors
[ifm-txt]: https://huggingface.co/datasets/IFM/TxT360-v2
[wiki-ja]: https://datasets-server.huggingface.co/size?dataset=wikimedia/wikipedia&config=20231101.ja
[dropupc]: https://arxiv.org/abs/2502.19261
[muennighoff]: https://arxiv.org/abs/2305.16264
[se-terms]: https://www.devclass.com/development/2024/07/30/stack-exchange-restricts-access-to-dump-of-user-contributed-data-critics-complain-this-contradicts-license/1625192
[expanse]: https://www.sdsc.edu/support/user_guides/expanse.html
[hpc]: https://sdsc.edu/services/hpc/hpc_systems.html
[access-expanse]: https://access-ci.atlassian.net/wiki/spaces/ACCESSdocumentation/pages/467111296/Expanse+-+SDSC
