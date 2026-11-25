# ExeaLM data recipe v1: research and code

One recipe for pretraining and post-training, weighted toward research papers, code and math, with English and Japanese. Every source can be redistributed, and the built dataset is public on Hugging Face under `exea-labs`. The machine-readable version, which the pipeline reads, is [experiments/recipe/exealm_recipe_v1.json](../experiments/recipe/exealm_recipe_v1.json).

## Size

The targets below add up to **200B pretraining tokens**. Multiply every target by one factor between 0.5 and 1.5 for 100B to 300B; the shares stay the same. Post-training is about 8B tokens. The scaling study ([budget.md](budget.md)) trains on a uniform slice of this recipe: each source at its share, shards in a fixed shuffled order, so the 3B runs' 12B tokens are the first slice of the same data.

| Bucket | Tokens at 200B | Share |
|---|---|---|
| Code | 60B | 30% |
| Research and reference | 50B | 25% |
| Math | 30B | 15% |
| Educational web | 36B | 18% |
| Japanese | 20B | 10% |
| QA and instruction-style text | 4B | 2% |
| **Total** | **200B** | **100%** |

Research, code and math together are 70%.

## Pretraining sources

| Bucket | Source | Subset | License | Tokens at 200B | Note |
|---|---|---|---|---|---|
| Code | [allenai/olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) | starcoder | ODC-BY | 44B | GitHub code in many languages, as in OLMo 2 |
| Code | [IFM/Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning) | code-no-thinking, code-no-thinking-hard, code-task-synthesis, codeio | Apache-2.0 | 8B | problems with worked solutions |
| Code | bigcode/the-stack (via LLM-jp v4 code_stack) | permissive repositories | per-file licenses; The Stack terms | 4B | gated companion repo exea-labs/exealm-pretrain-stack |
| Code | [IFM/Pretrain-Behaviors](https://huggingface.co/datasets/IFM/Pretrain-Behaviors) | data-science, planning, reasoning | Apache-2.0 | 2.5B | data science and planning |
| Code | [allenai/dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) | stackexchange | CC-BY-SA | 1.5B | programming and science Q&A; share-alike folder |
| Research and reference | [allenai/peS2o](https://huggingface.co/datasets/allenai/peS2o) | v2 | ODC-BY | 30B | open-access academic papers |
| Research and reference | [allenai/olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) | arxiv | ODC-BY | 15B | arXiv full text |
| Research and reference | [allenai/dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) | wiki | CC-BY-SA | 4B | share-alike folder |
| Research and reference | [allenai/dolma](https://huggingface.co/datasets/allenai/dolma) | books (Project Gutenberg) | ODC-BY | 1B | public-domain books |
| Math | [IFM/MegaMath](https://huggingface.co/datasets/IFM/MegaMath) | web, code and synthetic | ODC-BY | 12B |  |
| Math | [allenai/olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) | open-web-math | ODC-BY | 10B |  |
| Math | [allenai/olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) | algebraic-stack | ODC-BY | 4B | math code |
| Math | [IFM/Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning) | math-rewrite, math-dialogue | Apache-2.0 | 4B |  |
| Educational web | [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) |  | CC-BY-4.0 | 26B | web filtered by a fastText classifier (no Llama scores) |
| Educational web | [IFM/TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) | web-high-medium | CC-BY-4.0 | 10B | its quality buckets use the FineWeb-Edu score as one input |
| Japanese | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | jpn_Jpan | ODC-BY | 17.4B |  |
| Japanese | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | 20231101.ja | CC-BY-SA-3.0 | 1.5B | share-alike folder |
| Japanese | LLM-jp Corpus v4 (GitLab) | ja_kaken | CC-BY-4.0 | 0.9B | Japanese research-grant summaries |
| Japanese | [globis-university/aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean) |  | CC-BY-4.0 | 0.1B |  |
| Japanese | [nlp-waseda/e_gov](https://huggingface.co/datasets/nlp-waseda/e_gov) |  | CC-BY-4.0 | 0.1B | Japanese law |
| QA and instruction-style text | [IFM/TxT360-v2](https://huggingface.co/datasets/IFM/TxT360-v2) | txt360-qa, web-high-nltk-qa | CC-BY-4.0 | 3B |  |
| QA and instruction-style text | [IFM/SFT-Reasoning](https://huggingface.co/datasets/IFM/SFT-Reasoning) | 3efforts-pretrain | Apache-2.0 | 1B |  |

**Annealing.** The last 10% of training steps reweight the same sources toward the best material: worked code and math solutions, papers, StackExchange and Wikipedia count up to three times their base share, web text half (`anneal_weight` in the JSON).

## Post-training sources (about 8B tokens)

| Bucket | Source | Subset | License | Tokens | Note |
|---|---|---|---|---|---|
| code | [IFM/Code-Reasoning](https://huggingface.co/datasets/IFM/Code-Reasoning) | code-thinking-v1, code-thinking-hard | Apache-2.0 | 1.2B |  |
| code | [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | split_0, split_1 | CC-BY-4.0 | 0.8B |  |
| code+math+science | [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) |  | Apache-2.0 | 1.2B |  |
| math | [IFM/Math-Reasoning](https://huggingface.co/datasets/IFM/Math-Reasoning) | math-thinking-qwen, socratic-math-dialogue | Apache-2.0 | 1B |  |
| math | [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) |  | CC-BY-4.0 | 1B |  |
| research | [allenai/SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) | 4096 | ODC-BY | 0.3B | scientific literature tasks |
| research+code | [exea-labs/exealm-library](https://huggingface.co/datasets/exea-labs/exealm-library) | qa, morph | Apache-2.0 (written by Qwen3) | 0.5B | find a library entry and adapt it |
| general | [IFM/SFT-Reasoning](https://huggingface.co/datasets/IFM/SFT-Reasoning) | instruction-following | Apache-2.0 | 1B |  |
| general | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |  | ODC-BY (subsets keep their own terms) | 0.6B |  |
| japanese | to be chosen |  | must allow redistribution | 0.4B | open slot |

## What is left out, and why

- FineWeb-Edu, FineMath and LLM-jp's en_fineweb/en_finemath: scored with Llama 3 models, which LLM-jp judged would block an open model license.
- Stack v2, StarCoderData and Stack-Edu as public files: they are gated by BigCode's terms. StarCoder code comes from AI2's ODC-BY release in the OLMo 2 mix instead, and The Stack v1 goes in a separate gated repo (`exea-labs/exealm-pretrain-stack`) that asks users to accept The Stack's terms.
- TxT360-v2's web buckets use the FineWeb-Edu classifier as one of several inputs. They stay in, at 5% of the mix; drop them if the release needs to be free of any Llama-derived filtering.

## Licenses and credit

Mostly ODC-BY, CC-BY-4.0 and Apache-2.0. Share-alike sources (Wikipedia, StackExchange; CC-BY-SA) stay in their own folders under their own license. Common Crawl-derived sources (DCLM, TxT360-v2 web, FineWeb-2, MegaMath's web part) also fall under Common Crawl's terms of use. Each dataset card credits every source with its license. LLM-jp Corpus v4 subsets are taken from their primary distributions on Hugging Face (its GitLab server failed every download on 24 November); only KAKEN comes from its GitLab.

## Repositories

- `exea-labs/exealm-pretrain`: raw text by source (parquet), packed tokens, the 32k tokenizer, the mix file.
- `exea-labs/exealm-pretrain-stack`: The Stack v1 part, gated.
- `exea-labs/exealm-post`: post-training sets.
- `exea-labs/exealm-library`: the library written by Qwen3, with its QA and morph tasks.
