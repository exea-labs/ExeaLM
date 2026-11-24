# Distilled library

A library for ExeaLM's retrieval circuit (RG): an encyclopedia of text and code written by a large open-weight LLM,
packed with our 32k tokenizer so `--datastore_data` can load it, plus tasks that teach and test using it.

The goal: ExeaLM relies on library knowledge and morphs it. It retrieves an entry with RG and adapts it to the task at
hand, instead of copying it or holding it in its weights. Today the model's own search finds the right Wikipedia
article about 30% of the time with rare-word keys, and random web chunks barely help
([results §6](../../results/README.md)). So the library is written for a small model that reads 64-token chunks:
short declarative sentences, numbers, dates and names, canonical code, and every part naming its topic so a chunk makes
sense on its own.

| Step | What it writes |
|---|---|
| `build.py topics` | A topic list. Text: titles or first lines of documents. Code (`--kind code`): modules from import lines, the functions and APIs called through them, and the names of functions defined in many files, most used first. Or your own list, one topic per line |
| `build.py write` | One entry per topic. Text: a dense encyclopedia entry and 3 to 5 question/answer pairs whose answers are in it. Code (`--kind code`): an explanation, the canonical code and notes on parameters and pitfalls; Python must parse (`ast.parse`), other languages go by the code block's tag |
| `build.py morph` | 3 to 5 tasks per entry that need the entry changed, not copied. Code: rename and retype, extend, handle an edge case, combine with a second entry, port to another style, fix a buggy variant. Text: combine two facts, restate for another reader, compute from stated numbers |
| `build.py pack` | The library, and the pairs and tasks as documents with their target spans, split into train and held-out by topic |
| `score.py` | Bits per byte on the targets of held-out pairs and tasks, with the library attached or removed, or the right entry pasted |

## Building one on the MI300X box

```bash
# 0. Once: the 32k tokenizer and the token packs (store_wiki: English Wikipedia; train_code: code)
bash experiments/gpu_session/setup.sh

# 1. Topics
python experiments/library/build.py topics data/tokpack/bpe32k/store_wiki --min_doc_tokens 256 --shuffle \
    --limit 20000 --out data/library/topics.txt
python experiments/library/build.py topics data/tokpack/bpe32k/train_code --kind code --limit 5000 \
    --out data/library/code_topics.txt                       # skim it: local project modules can slip in

# 2. Entries. MODEL and CODE_MODEL are placeholders: choose the models first (below). The first run downloads the
#    weights from Hugging Face (tens of GB), and that download needs the project lead's OK before it runs.
MODEL="<hf-model-id>"
CODE_MODEL="<hf-model-id>"
python experiments/library/build.py write --topics data/library/topics.txt --backend vllm --model "$MODEL" \
    --out data/library/entries.jsonl --limit 200            # time a small batch and read some entries first
python experiments/library/build.py write --topics data/library/topics.txt --backend vllm --model "$MODEL" \
    --out data/library/entries.jsonl                        # the rest; after a crash, run the same line again
python experiments/library/build.py write --kind code --topics data/library/code_topics.txt --backend vllm \
    --model "$CODE_MODEL" --out data/library/code_entries.jsonl

# 3. Morph tasks (one model can do text and code; run once per model if you use two)
python experiments/library/build.py morph data/library/entries.jsonl data/library/code_entries.jsonl \
    --backend vllm --model "$CODE_MODEL" --out data/library/tasks.jsonl

# 4. Pack with the 32k tokenizer (results/tokenizer/bpe32k, the default)
python experiments/library/build.py pack data/library/entries.jsonl data/library/code_entries.jsonl \
    --tasks data/library/tasks.jsonl --out data/library/v1
```

- **Backends.** `--backend vllm` needs vLLM for ROCm (AMD's `rocm/vllm` Docker image, or vLLM's ROCm install guide).
  `--backend transformers` (`pip install transformers accelerate`) works anywhere but is much slower.
  `--backend mock` writes fake entries and tasks with no model, for a dry run on a laptop.
- **Resuming.** `write` and `morph` append to their output and skip topics already in it. What they cannot use goes
  to `<out>.rejects.jsonl` with the reason, and is tried again on the next run.
- **What gets rejected.** Text entries: the wrong format, too short, fewer than 3 answers found in the entry, or
  "UNKNOWN". Code entries: no code block, no notes, or Python that does not parse. Tasks: a target whose longest
  stretch shared with the entry is more than half of it (`--max_copy 0.5`), an instruction or context that shares more
  than 24 tokens in a row with the entry (`--max_leak`), or Python that does not parse as a whole task document.
- **Models.** Use open-weight models whose licence allows training on their outputs; Qwen3 models are Apache-2.0.
  Text: `Qwen/Qwen3-32B` (about 65 GB in bf16) or `Qwen/Qwen3-30B-A3B-Instruct-2507` (about 61 GB, faster: 3B active
  parameters). Code: `Qwen/Qwen3-Coder-30B-A3B-Instruct` or `Qwen/Qwen3-30B-A3B-Instruct-2507`. Each fits one MI300X
  (192 GB) with room for the cache. Closed APIs' terms generally forbid training other models on their outputs.
- **Languages.** Text topics with kana or kanji get Japanese entries, the rest English (`--lang`). Code topics read
  "os.path.join (Python)"; the language in brackets picks the code block's language (`--code_lang` for topics
  without one). The code miner reads Python, JavaScript, Java, Go, Rust and C/C++ imports.
- **Prompts** are `SYSTEM`, `PROMPT`, `CODE_PROMPT` and the morph prompts in `build.py`. Change the matching
  `*_PROMPT_VERSION` whenever you change one: every entry and task records its model and prompt version.

## What `pack` writes

| File | What it is |
|---|---|
| `tokens.bin`, `offsets.npy`, `meta.json` | The library, packed like `exealm.data.pack` output: one entry (text or code) per document, the topic on its first line. `meta.json` also has a `source` block |
| `SOURCE.md` | Where the text came from: models, prompt versions, dates, counts, dropped tasks and a suggested `--datastore_chunks` |
| `index.jsonl` | One line per library document: `doc`, `topic`, `kind`, `lang`, `split`, `n_tokens`, `model`, `prompt_version` |
| `qa/train/`, `qa/heldout/` | Question/answer pairs, one document each: `Question: ...\nAnswer:` then the answer |
| `tasks/train/`, `tasks/heldout/` | Morph tasks, one document each: the instruction and context, then the target. Code tasks are programs: the instruction as comments, the context, the target |

Every `qa/` and `tasks/` directory is a packed directory (TokenShards reads it) with `targets.npy`, each target's
token span `[start, end)` in `tokens.bin`, and `items.jsonl`, one line per document: the item, `source_doc` (its entry
in the library), `also_doc` for a task that combines two entries, and `target_start`, `target_end` (counted from the
document's start) and `target_bytes`. The prefix and target are tokenized apart, so the span is exact. The source entry
is never in a task document: it stays in the library.

A hash of the topic decides its side (10% held out by default, `--heldout`), so all pairs and tasks of a topic are on
one side, and adding topics later never moves an old one. A task that combines two entries is dropped if they are on
different sides. Train topics come first in the library, then held-out ones.

## Training with it

```bash
python -m exealm.train.pretrain --preset 100m --data experiments/expanse/mix_tokpack.json \
    --val data/tokpack/bpe32k/val_en --datastore_data data/library/v1 --datastore_chunks N ...
```

`N` is in `SOURCE.md`: about three draws per token, since `Datastore.from_shards` samples chunks with replacement.
The library must be packed with the tokenizer the model trains with (`meta.json` `vocab_size`).

The train tasks (`tasks/train`, `qa/train`) are what teaches the model to retrieve and adapt: train on them with the
library attached and the loss on the target spans only. No trainer does that yet: `exealm.train.pretrain` puts the
loss on every token and `exealm.train.sft` has no datastore. Adding the tasks to a `--data` mix works today but trains
on the prefixes too.

## Evaluating with it

```bash
python experiments/library/score.py --library data/library/v1 runs/exealm/model.pt runs/llama/model.pt \
    --out results/library/score.jsonl
```

For each checkpoint, each set (`qa`, `tasks`) and each condition, it prints bits per byte on the target tokens alone
(and per kind: text and code tasks apart):

| Condition | What the model gets | Applies to |
|---|---|---|
| `library` | RG searches the library | models with RG slots (ExeaLM) |
| `none` | nothing: the model alone | every model |
| `pasted` | the source entry pasted before the task: ordinary RAG with the right document, an upper bound | every model |

ExeaLM `library` against `none` is what the library is worth to it; Llama `pasted` against `none` is what the right
document is worth when a perfect search hands it over. For `library` it also reports how often the model's own search
returns a chunk of the source entry in its top-k: `search_first` at the target's first chunk (the search that reads the
end of the question or instruction) and `search_any` at any chunk of the target.

Details: rows are laid out so the prefix ends on a chunk boundary (RG searches once per chunk, with the chunk before as
the query). The datastore is every chunk of every entry unless `--datastore_chunks` asks for a sample like
pretraining's. Library document numbers are shifted past the scored rows', so a scored row never hides an entry from
the search. `--split train` scores the training items, `--limit` a few items of each set.
