# Data mix: the recipe, built into packed tokens

This folder turns the data recipe, [experiments/recipe/exealm_recipe_v2.json](../recipe/exealm_recipe_v2.json)
(human version: [docs/data_recipe.md](../../docs/data_recipe.md)), into the files training reads: one packed
directory per source and split (`exealm.data.pack`'s format), `mix.json` with weights equal to the targets,
`mix_anneal.json` for the last 10% of steps (weights = target x `anneal_weight`), and the 32k tokenizer. The recipe is
the only place sources, licenses and targets are set; `config.py` adds only how to read each source.

| Repo | What |
|---|---|
| `exea-labs/exealm-pretrain` | pretraining packs and tokenizer; the first slice goes under `slice-30b/` |
| `exea-labs/exealm-pretrain-stack` | The Stack v1 (gated); not in the first slice (download_slice.py skips gated sources) |
| `exea-labs/exealm-post` | post-training sets (not built yet; the pipeline supports them) |
| `exea-labs/exealm-library` | the library (made by experiments/library) |

The repos exist already. `upload.py` never creates or reconfigures one.

## The first slice, on the box (tonight)

`experiments/recipe/download_slice.py` puts the slice's raw files in `/work/data_raw/<repo name>/<path>` and lists them
in `/work/data_raw/manifest_slice.json`. The pipeline reads those files in place and downloads nothing for them.

```bash
cd ExeaLM
pip install -e '.[data]' pyarrow zstandard huggingface_hub          # datasets: needed by the benchmarks step
python -m pytest -q tests/test_data_mix.py                         # 25 s: every step on tiny local fixtures

export WORK=/work/data_mix                                         # about 200 GB free needed (see below)
M="python -m experiments.data_mix.pipeline"

# 1. extract the slice's files (filters, text hashes); stops at any file still downloading: rerun when it is done
$M fetch --work $WORK --slice_manifest /work/data_raw/manifest_slice.json --release slice-30b --tokens 30e9 \
    --download_workers 20 --part_parallel 4
# 2.-8. (the build settings above are remembered in $WORK/build.json)
$M dedup --work $WORK
$M tokenizer --work $WORK
$M benchmarks --work $WORK
nohup $M tokenize --work $WORK > $WORK/tokenize.out 2>&1; tail -3 $WORK/tokenize.out
$M assemble --work $WORK
$M cards --work $WORK
$M verify --work $WORK
# 9. upload slice-30b/ (packs, tokenizer, mix files, sources.json, README) to exea-labs/exealm-pretrain
python -m experiments.data_mix.upload --work $WORK --repos pretrain --dry-run
python -m experiments.data_mix.upload --work $WORK --repos pretrain
```

Before uploading, read `$WORK/reports/`: `fetch.json` (parts with `"enough": false` or `waiting_for`),
`decontam.json` (what was dropped, and the benchmark windows that matched), `assemble/<source>.json` (`short` > 0
means a source has less than its target; the manifest did not have enough files for it), `verify.json`.

Then, on SDSC: `bash experiments/data_mix/fetch.sh --release slice-30b --no-stack` downloads the slice and writes
`data/mix_pretrain.json`, `data/mix_pretrain_val.json` and `data/mix_pretrain_anneal.json` with local paths.

### Time and disk for the 30B slice (20 cores)

| Step | Time | Disk after | Notes |
|---|---|---|---|
| fetch | 30 to 60 min | `extract/` about 70 GB | reads the ~77 GB of downloaded files (research_python's are 8x oversampled for its filter) at about 40 MB of text per second per process |
| dedup | 5 to 10 min | 2 GB | |
| tokenizer | 15 to 30 min | 1.5 GB | 1.5 GB sample, by group: research code 20%, code 15%, research 20%, math 12%, web 8%, Japanese 20%, QA 5% |
| benchmarks | 5 to 10 min | 1 GB | downloads the 14 benchmark sets (APPS and LiveCodeBench are a few GB of jsonl) |
| tokenize | 45 to 75 min | `tok/` about 62 GB | 0.4 to 0.55M tokens/s per process measured on a Mac |
| assemble | 10 to 20 min | `repos/` about 60 GB | packed tokens only (add `--raw` for Parquet text) |
| verify | 5 min | | `--decontam` also reruns exealm.data.decontam (informational, 20 to 30 min) |
| upload | 10 to 30 min | | about 60 GB |
| total | 2.5 to 4.5 h | peak about 200 GB plus the raw files | `rm -rf $WORK/extract $WORK/tok` after upload frees about 130 GB |

## Sizes: scale and slices

`--scale s` (0.5 to 1.5) multiplies every recipe target (300B at s = 1). `--tokens T` builds the first T pretraining
tokens: every source at the same fraction T / (300B x s) of its target, whether or not this build has it (the slice
has no The Stack, books or KAKEN, so it holds about 28.6B of its 30B). Run again with a larger `--tokens` in the same
work directory to extend a build: files are read in the same fixed order, training documents are taken in a fixed
order (file order, then a hash of the text), the tokenizer is kept, and validation is 0.1% of documents by hash plus
long documents from each source's first file, so the earlier build's training and validation documents are the start
of the new ones. `python -m experiments.data_mix.config --tokens 30e9` prints the targets.

## The steps

| Step | What it does |
|---|---|
| `fetch` | Each source's files in a fixed order (the slice manifest's order, or a seeded shuffle of the host's listing), until the text kept covers (target / epochs) x (1 + 0.1%) x (1 + margin). Applies the source's filter and writes the kept documents to `extract/` with a text hash and a sampling key. Files without a `text` column (MegaMath's `megamath-code`) are recorded and skipped |
| `dedup` | Exact duplicates (whitespace and case ignored) across every source; the copy a build reaches first is kept. Then validation: 0.1% of every source's documents by hash, plus up to 256 whole long documents (at least 4,097 tokens) from its first file |
| `tokenizer` | 32,768-token byte-level BPE trained with `exealm.data.tokenizer.train` on a sample of the training side; measures bytes per token per source and checks the round trip. Kept for every later build (`--retrain` to replace it) |
| `benchmarks` | Exports the decontamination benchmarks (exealm.data.benchmarks: HumanEval, GSM8K, ARC, MMLU, HellaSwag, JCommonsenseQA; bench_extra.py: MBPP, APPS, CodeContests, LiveCodeBench, MATH, SciQ, PubMedQA) and indexes their 13-token windows |
| `tokenize` | Selects training documents in order until each source's target, tokenizes in a process pool, drops documents with a 13-token window equal to a benchmark window, and tops up sources that this leaves short (up to 4 rounds) |
| `assemble` | `packed/<source>/{train,val}`, the tokenizer, `mix.json`, `mix_val.json`, `mix_anneal.json` (and `mix_public*.json` when The Stack is in the build), `sources.json` (every source file used) |
| `cards` | README for the repo (or the release folder) from the reports |
| `verify` | Opens every pack with `TokenShards` and every mix with `open_data`, checks document ends, offsets and token ids, that no text is in two places, and the long validation documents |

Filters from the recipe: `research_python` keeps Python files of olmo-mix's StarCoder that import one of the listed
scientific or ML libraries; `starcoder` drops exactly those. TxT360-v2 web keeps English and drops its Wikipedia rows.
Every source drops empty texts and texts containing a special-token string. StackExchange has 2 epochs: half its
target is packed, and its mix weight is the full target.

## Decontamination: what changed from plain 13-token matching

On the real rows (IFM Code-Reasoning, Math-Reasoning, TxT360-v2, SFT-Reasoning, Japanese Wikipedia and FineWeb-2),
plain 13-token matching against the benchmarks' full text dropped 76% of `code_reasoning`: the windows were generic
code from benchmark solutions (`if __name__ == "__main__":`, `= sys.stdin.buffer.read`, `#include <bits/stdc++.h>`,
"1, 2, 3, 4, 5, 6, 7"). So, set in `config.py` (`decontam`):

- for HumanEval, MBPP, APPS, CodeContests and LiveCodeBench only the problem statement is checked (the prompt,
  question or description), not solutions or tests;
- a benchmark window found in 2 or more different benchmark items is boilerplate and ignored;
- a window needs at least 3 distinct words of 3+ letters (a CJK character counts as half a word).

With these, the same rows lose 0.8% of `code_reasoning` (HumanEval prompt headers such as
`from typing import List, Tuple\n\n\ndef find`) and nothing elsewhere. Setting `fields` to `{}` and `common_items` and
`min_words` to 0 gives exactly `exealm.data.decontam`'s rule.

## Measured on a 12-core Mac (2026-11-24)

| What | Rate |
|---|---|
| Extract (filter, hash, write Parquet) | 39.5 MB of text and 2,375 documents per second per process |
| Tokenizer training (32k) | about 4 MB/s on 440 MB of real text (the repo's own run, results/tokenizer/compare.json) |
| Tokenize | 0.44 to 0.56M tokens/s (1.6 to 2.8 MB/s) per process; decontamination adds 13 to 26M tokens/s of work, about 3% |
| Assemble | 10 to 15M tokens/s per source |
| Download from Hugging Face | 5.7 MB/s on one connection from this Mac (not used for the slice: the box has the files) |

## The full 300B build (next)

The box has about 420 GB free, so the full build has to stream: per source, take the next file in the fixed order,
download it, extract and tokenize it, append its documents to the source's pack, upload the new pack part, and delete
the raw and intermediate files. The pieces exist (fetch works per file and keeps its order, tokenize per chunk,
assemble writes `sources.json`); what is left is the loop that runs them per file and appends to packs in parts
(`packed/<source>/train/part-00000/`, read with a glob as a ShardSet) instead of one directory. Publish packed tokens
and `sources.json`, not raw text. Not written yet.
