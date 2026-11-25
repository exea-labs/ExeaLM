"""The pipeline's config: the recipe's sources and targets, plus how to read each source.

The recipe, experiments/recipe/exealm_recipe_v1.json (human version: docs/data_recipe.md), is the single source of
truth for every source, subset, license, token target (at the 200B scale) and anneal weight. This file adds only what
the recipe does not say: where each source's files are and how to read them (ACCESS), a guess of its bytes per token
(used to decide how many files to download), and which buckets come from Common Crawl or are share-alike.

    bucket   one recipe entry: one packed directory per split (packed/<name>/train, packed/<name>/val), one weight in
             the mix .json. Its target is tokens_b x 1e9 x scale x fraction (see apply_scale)
    part     one file set inside a bucket (a subset, a language, a folder), with its share of the bucket's tokens

Buckets are listed in recipe order, which is also the dedup priority on ties. A bucket that the recipe lists but that
cannot be built yet (the library's tasks, the open Japanese SFT slot) has "enabled": False and a reason. An
"optional" bucket (KAKEN, on LLM-jp's GitLab) is skipped when its download fails.

Sources:
  hf       a Hugging Face dataset repo, files chosen by glob patterns, pinned to the commit resolved at fetch time
  urls     a list of URLs kept in a Hugging Face repo (allenai/dolma lists its files on olmo-data.org)
  llmjp    LLM-jp Corpus v4 on NII's GitLab (only KAKEN; its file server failed every download on 2026-11-24)

Formats (the "fmt" of a part): Parquet or JSON lines (.json.gz, .jsonl.gz, .jsonl.zst, .json.zst), and
  {"kind": "text", "field": "text"}                       the document is one field, kept byte for byte (default)
  {"kind": "fields", "fields": [...], "sep": "\\n\\n"}      fields joined (a question and its answer); rows whose
                                                         field equals one of "skip_values" are dropped
  {"kind": "messages", "field": "messages"}               a conversation: its turns joined by a blank line, and the
                                                         turns themselves kept in `meta` for exealm.train.sft
"""
import copy
import json
import os
import re

B = 1_000_000_000
M = 1_000_000

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.join(HERE, "..", "recipe", "exealm_recipe_v2.json")

LLMJP = dict(host="https://gitlab.llm-jp.nii.ac.jp", project="datasets/llm-jp-corpus-v4", ref="main")

REPOS = {
    "pretrain": dict(repo_id="exea-labs/exealm-pretrain", title="ExeaLM pretraining data", tokenizer=True),
    "pretrain_stack": dict(repo_id="exea-labs/exealm-pretrain-stack", title="ExeaLM pretraining data: The Stack v1",
                           gated=True),
    "post": dict(repo_id="exea-labs/exealm-post", title="ExeaLM post-training data", library_slot=True),
    "library": dict(repo_id="exea-labs/exealm-library", title="ExeaLM library", empty=True),
}

LICENSE_URLS = {
    "ODC-BY": "https://opendatacommons.org/licenses/by/1-0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-SA": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "per-file licenses; The Stack terms": "https://huggingface.co/datasets/bigcode/the-stack",
}
COMMON_CRAWL_TERMS = "https://commoncrawl.org/terms-of-use"

TXT360_WEB_FILTER = dict(
    lang_in=["en", "eng_Latn"],                    # meta.lang; rows with no language are dropped
    quality_in=["High", "Medium-High"],            # quality_category (every row checked on 2026-11-24 is Medium-High)
    drop_subsets=["wikipedia"],                    # Wikipedia is CC-BY-SA: it stays in its own buckets
    min_edu_score=None,                            # fineweb_edu_classifier_score (a Llama-3-trained classifier); off
)

JSON_META = dict(meta_field="metadata")            # AI2's JSON lines keep source metadata under "metadata"
STACK_LANGS = {"python": 0.30, "javascript": 0.10, "typescript": 0.07, "java": 0.10, "c": 0.08, "c++": 0.10,
               "c-sharp": 0.06, "go": 0.06, "rust": 0.05, "shell": 0.03, "sql": 0.02, "markdown": 0.03}
STACK_META = ["max_stars_repo_name", "max_stars_repo_path", "max_stars_repo_licenses", "max_stars_repo_head_hexsha",
              "hexsha", "lang"]


def hf(repo, patterns, share=1.0, **kw):
    return dict(source="hf", repo=repo, patterns=[patterns] if isinstance(patterns, str) else list(patterns),
                share=share, **kw)


def urls(repo, list_file, contains, share=1.0, **kw):
    return dict(source="urls", repo=repo, list_file=list_file, contains=contains, share=share, **kw)


def llmjp(path, share=1.0, **kw):
    return dict(source="llmjp", path=path, share=share, **kw)


def subsets(repo, names, shares=None, **kw):
    shares = shares or [1 / len(names)] * len(names)
    return [hf(repo, f"{n}/*.parquet", s, label=n, **kw) for n, s in zip(names, shares)]


# How to read each recipe source. Keys are the recipe's "name"s. bpt: guessed bytes of text per token (the tokenizer
# step measures the real value). cc: Common Crawl-derived (its terms of use also apply).
ACCESS = {
    # ---- pretraining -------------------------------------------------------------------------------------------
    "research_python": dict(parts=[hf("allenai/olmo-mix-1124", "data/starcoder/*/python-*.json.gz", label="python",
                                      **JSON_META)], bpt=3.0, lang="code"),
    "starcoder": dict(parts=[hf("allenai/olmo-mix-1124", "data/starcoder/*.json.gz", **JSON_META)], bpt=2.7,
                      lang="code"),
    "code_reasoning": dict(parts=subsets("IFM/Code-Reasoning", ["code-no-thinking", "code-no-thinking-hard",
                                                                "code-task-synthesis", "codeio"]), bpt=3.2,
                           lang="en+code"),
    "the_stack_v1": dict(parts=[hf("bigcode/the-stack", f"data/{lang}/*.parquet", share, label=lang,
                                   fmt=dict(kind="text", field="content"), meta_columns=STACK_META)
                                for lang, share in STACK_LANGS.items()],
                         bpt=2.6, lang="code", repo="pretrain_stack", in_tokenizer_sample=False, gated=True),
    "pretrain_behaviors": dict(parts=subsets("IFM/Pretrain-Behaviors", ["data-science", "planning", "reasoning"]),
                               bpt=3.5, lang="en+code"),
    "stackexchange": dict(parts=[hf("allenai/dolmino-mix-1124", "data/stackexchange/*.json.gz", **JSON_META)],
                          bpt=3.9, lang="en+code", share_alike=True),
    "pes2o": dict(parts=[hf("allenai/peS2o", "data/v2/*.json.gz", label="v2")], bpt=4.3, lang="en"),
    "arxiv": dict(parts=[hf("allenai/olmo-mix-1124", "data/arxiv/*.json.gz", **JSON_META)], bpt=3.6, lang="en"),
    "wikipedia_en": dict(parts=[hf("allenai/dolmino-mix-1124", "data/wiki/*.json.gz", **JSON_META)], bpt=4.4,
                         lang="en", share_alike=True),
    "books": dict(parts=[urls("allenai/dolma", "urls/v1_7.txt", "/books/")], bpt=4.3, lang="en"),
    "megamath": dict(parts=[hf("IFM/MegaMath", "megamath-web/*.parquet", 0.7, label="megamath-web",
                               meta_columns=["url", "math_score"]),
                            hf("IFM/MegaMath", "megamath-text-code-block/*.parquet", 0.15, label="text-code-block"),
                            hf("IFM/MegaMath", "megamath-translated-code/*.parquet", 0.15, label="translated-code")],
                     bpt=3.3, lang="en+code", cc=True),
    "open_web_math": dict(parts=[hf("allenai/olmo-mix-1124", "data/open-web-math/*.json.gz", **JSON_META)], bpt=3.6,
                          lang="en", cc=True),
    "algebraic_stack": dict(parts=[hf("allenai/olmo-mix-1124", "data/algebraic-stack/*.json.gz", **JSON_META)],
                            bpt=3.0, lang="en+code"),
    "math_reasoning": dict(parts=subsets("IFM/Math-Reasoning", ["math-rewrite", "math-dialogue"]), bpt=3.3, lang="en"),
    "dclm": dict(parts=[hf("mlfoundations/dclm-baseline-1.0", "global-shard_*/*.jsonl.zst", label="dclm-baseline", meta_keys=["url"])],
                 bpt=4.4, lang="en", cc=True),
    "txt360_web": dict(parts=[hf("IFM/TxT360-v2", "web-high-medium/*.parquet", filter=TXT360_WEB_FILTER,
                                 meta_columns=["subset", "quality_category", "fineweb_edu_classifier_score", "meta.lang"])],
                       bpt=4.4, lang="en", cc=True),
    "fineweb2_ja": dict(parts=[hf("HuggingFaceFW/fineweb-2", "data/jpn_Jpan/train/*.parquet", label="jpn_Jpan",
                                  meta_columns=["id", "url", "dump"])], bpt=5.3, lang="ja", cc=True),
    "wikipedia_ja": dict(parts=[hf("wikimedia/wikipedia", "20231101.ja/*.parquet", meta_columns=["title", "url"])],
                         bpt=4.3, lang="ja", share_alike=True),
    "kaken": dict(parts=[llmjp("ja/ja_kaken")], bpt=4.8, lang="ja", optional=True),
    "aozora": dict(parts=[hf("globis-university/aozorabunko-clean", "*.jsonl.gz", label="aozorabunko-clean", meta_field="meta")], bpt=4.0,
                   lang="ja"),
    "e_gov": dict(parts=[hf("nlp-waseda/e_gov", "data/*.parquet", label="e_gov", meta_columns=["metadata"])], bpt=4.0, lang="ja"),
    "txt360_qa": dict(parts=[hf("IFM/TxT360-v2", "txt360-qa/*.parquet", 0.5, label="txt360-qa",
                                meta_columns=["subset", "meta.url"]),
                             hf("IFM/TxT360-v2", "web-high-nltk-qa/*.parquet", 0.5, label="web-high-nltk-qa")],
                      bpt=4.4, lang="en", cc=True),
    "sft_pretrain": dict(parts=subsets("IFM/SFT-Reasoning", ["3efforts-pretrain"]), bpt=3.8, lang="en"),
    # ---- post-training -----------------------------------------------------------------------------------------
    "code_reasoning_sft": dict(parts=subsets("IFM/Code-Reasoning", ["code-thinking-v1", "code-thinking-hard"]),
                               bpt=3.2, lang="en+code"),
    "open_code_reasoning": dict(parts=[hf("nvidia/OpenCodeReasoning", f"{s}/*.parquet", share, label=s,
                                          fmt=dict(kind="fields", fields=["input", "output"], skip_values=dict(input=["-"])),
                                          meta_columns=["id", "source", "license", "dataset"])
                                       for s, share in (("split_0", 0.85), ("split_1", 0.15))],
                                bpt=3.2, lang="en+code"),
    "openthoughts3": dict(parts=[hf("open-thoughts/OpenThoughts3-1.2M", "data/*.parquet", label="openthoughts3",
                                    fmt=dict(kind="messages", field="conversations"), meta_columns=["source", "domain"])],
                          bpt=3.3, lang="en+code"),
    "math_reasoning_sft": dict(parts=subsets("IFM/Math-Reasoning", ["math-thinking-qwen", "socratic-math-dialogue"]),
                               bpt=3.0, lang="en"),
    "open_math_reasoning": dict(parts=[hf("nvidia/OpenMathReasoning", "data/cot-*.parquet", label="cot",
                                          fmt=dict(kind="fields", fields=["problem", "generated_solution"]),
                                          meta_columns=["problem_source", "generation_model", "expected_answer"])],
                                bpt=3.0, lang="en"),
    "sciriff": dict(parts=[hf("allenai/SciRIFF", "4096/train-*.parquet", label="4096-train",
                              fmt=dict(kind="fields", fields=["input", "output"]), meta_columns=["_instance_id"])],
                    bpt=4.2, lang="en"),
    "instruction_following": dict(parts=subsets("IFM/SFT-Reasoning", ["instruction-following"]), bpt=4.0, lang="en"),
    "tulu3_sft": dict(parts=[hf("allenai/tulu-3-sft-mixture", "data/*.parquet", label="tulu-3-sft",
                                fmt=dict(kind="messages", field="messages"), meta_columns=["id", "source"])],
                      bpt=4.0, lang="en"),
}

DISABLED = {
    "library_tasks": "made by experiments/library and uploaded with exealm-library; exealm-post has a library/ slot",
    "japanese_sft": "the recipe has not chosen a source yet",
    "exealm_generated": "our own generated tasks: not made yet",
    "benchmark_train_splits": "benchmark training splits: not built yet (the recipe's benchmark_train list)",
}

# Benchmarks removed from everything by 13-gram overlap: the recipe's decontamination list (exealm.data.benchmarks
# exports the first seven, experiments/data_mix/bench_extra.py the rest), plus HellaSwag and JCommonsenseQA, which
# the repo also reports.
DECONTAM = ["humaneval", "mbpp", "apps", "code_contests", "livecodebench", "gsm8k", "math", "arc_easy", "arc_challenge",
            "mmlu", "sciq", "pubmedqa", "hellaswag", "jcommonsenseqa"]


def credit(entry):
    src = entry["source"]
    if src.startswith("LLM-jp"):
        return "[LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4) (LLM-jp, NII)"
    repo = src.split(" ")[0]
    return f"[{repo}](https://huggingface.co/datasets/{repo})"


def bucket_from_recipe(entry, kind):
    name = entry["name"]
    acc = ACCESS.get(name)
    b = dict(name=name, group=entry.get("bucket", ""), kind=kind, source=entry["source"], subset=entry.get("subset", ""),
             license=entry["license"], anneal_weight=float(entry.get("anneal_weight", 1.0)),
             epochs=float(entry.get("epochs", 1)), recipe_filter=entry.get("filter"),
             full_target=int(round(entry["tokens_b"] * B)), target=int(round(entry["tokens_b"] * B)),
             description=entry.get("note") or entry.get("subset") or name, credit=credit(entry),
             repo="post" if kind == "post" else "pretrain", enabled=True, optional=False, share_alike=False,
             common_crawl=False, parts=[], bpt=4.0, lang="en")
    if name in DISABLED or acc is None:
        b.update(enabled=False, disabled_reason=DISABLED.get(name, "no access spec in experiments/data_mix/config.py"))
        return b
    acc = copy.deepcopy(acc)
    b["parts"] = acc.pop("parts")
    b["common_crawl"] = acc.pop("cc", False)
    b.update(acc)
    if entry.get("gated"):
        b["repo"], b["gated"] = "pretrain_stack", True
    if entry.get("optional"):
        b["optional"] = True
    return b


def resolve_filters(buckets_):
    """The recipe's per-source filters, as part filters: python_imports_any keeps Python files importing one of the
    libraries; exclude_source drops whatever that other source's filter keeps (starcoder minus research_python)."""
    by = {b["name"]: b for b in buckets_}
    for b in buckets_:
        f = b.get("recipe_filter") or {}
        extra = {}
        if "python_imports_any" in f:
            extra["python_imports_any"] = list(f["python_imports_any"])
        if "exclude_source" in f:
            other = (by[f["exclude_source"]].get("recipe_filter") or {}).get("python_imports_any")
            if other:
                extra["python_imports_none"] = list(other)
        if extra:
            for p in b["parts"]:
                p["filter"] = dict(p.get("filter") or {}, **extra)


def bucket(name, repo, target, parts, license, lang, bpt, credit, description, common_crawl=False, share_alike=False,
           enabled=True, anneal_weight=1.0, group="", optional=False, **kw):
    """A bucket written by hand (tests use these instead of the recipe)."""
    return dict(name=name, repo=repo, target=int(target), full_target=int(target), parts=parts, license=license,
                lang=lang, bpt=bpt, credit=credit, description=description, common_crawl=common_crawl,
                share_alike=share_alike, enabled=enabled, anneal_weight=anneal_weight, group=group, optional=optional,
                kind="post" if repo == "post" else "pretrain", source="", subset="", **kw)


CONFIG = dict(
    name="exealm-recipe-v1",
    seed=20261124,                 # orders the files of each part and keys every document
    llmjp=LLMJP,
    repos=REPOS,
    split=dict(
        val_fraction=0.001,        # the recipe: 0.1% of every source's documents, by a hash of the text (no target)
        val_tokens=0,              # or (when val_fraction is 0) this many tokens per bucket
        val_long_docs=256,         # per bucket: documents of at least long_tokens tokens, for the long-document test
        val_long_min=128,          # always at least this many, even past val_tokens (a book is 100k+ tokens)
        long_tokens=8 * 512 + 1,   # experiments/stream_eval.py reads 8 windows of 512 and the target of the last one
        val_files=1,               # validation comes from each part's first files, which every build has
    ),
    fetch=dict(
        margin=0.3,                # download until a part has (target + val) x (1 + margin) tokens of text, by bpt
        min_files=1,
    ),
    select=dict(margin=0.02,       # select this much more text than the target, by measured bytes per token
                max_rounds=4),     # then top up parts that decontamination left short, at most this many rounds
    tokenizer=dict(
        vocab_size=32768,          # fits uint16 packing
        measure_mb=10,             # MB of other documents per part to measure bytes per token and check round trips
        sample_total_mb=1500,      # the training sample, split over groups (more Japanese and code than in the mix)
        group_shares=dict(research_code=0.2, code=0.15, research=0.2, math=0.12, web=0.08, japanese=0.2, qa=0.05),
    ),
    # common_items: a benchmark window found in this many different benchmark items is boilerplate and ignored;
    # min_words: a window needs this many distinct words (lists of numbers and bare signatures do not count). On real
    # rows, plain 13-token matching dropped 76% of IFM Code-Reasoning (`if __name__ == "__main__":`, primes lists);
    # with fields, common_items and min_words, see README. 0 and 0 give exactly exealm.data.decontam's rule
    # fields: for the code benchmarks, only the problem statement is checked (solutions and tests are mostly generic
    # code); the other benchmarks use every exported field
    decontam=dict(ngram=13, min_matches=1, common_items=2, min_words=3, benchmarks=DECONTAM,
                  fields=dict(humaneval=["prompt"], mbpp=["text"], apps=["question"], code_contests=["description"],
                              livecodebench=["question"])),
    raw=dict(mb_per_file=1000),    # raw Parquet files of about this many MB of text
)


def load(path=None, recipe=RECIPE):
    """The config. Without `path`: CONFIG with buckets from the recipe at full (200B) scale. With a .json `path`: that
    file (same structure, buckets written out; tests use small ones)."""
    if path is not None:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = copy.deepcopy(CONFIG)
        with open(recipe, encoding="utf-8") as f:
            r = json.load(f)
        cfg["recipe"] = dict(name=r["name"], date=r.get("date"), path=os.path.relpath(recipe, os.path.join(HERE, "..", "..")),
                             anneal=r.get("anneal", {}), excluded=r.get("excluded", []))
        cfg["buckets"] = [bucket_from_recipe(e, "pretrain") for e in r["pretrain"]] + \
                         [bucket_from_recipe(e, "post") for e in r.get("post", [])]
        resolve_filters(cfg["buckets"])
        cfg["recipe"]["total_pretrain_tokens_b"] = r.get("total_pretrain_tokens_b")
        cfg["recipe"]["decontamination"] = r.get("decontamination")
        cfg["recipe"]["splits"] = r.get("splits")
    for b in cfg["buckets"]:
        b.setdefault("full_target", b["target"])
        b.setdefault("anneal_weight", 1.0)
        b.setdefault("kind", "post" if b["repo"] == "post" else "pretrain")
        b.setdefault("epochs", 1.0)
    cfg.setdefault("build", dict(scale=1.0, fraction=1.0, post_fraction=1.0))
    cfg.setdefault("release", "")
    return cfg


def unique_target(b):
    """Tokens of distinct documents a bucket needs: its target over its epochs (StackExchange is read twice). The
    mix weight stays the target, so the loader reads it that many times."""
    return int(round(b["target"] / max(1.0, float(b.get("epochs", 1.0)))))


def use_manifest(cfg, manifest_path, root):
    """Read sources from files already downloaded by experiments/recipe/download_slice.py instead of the hosts: each
    bucket becomes one part listing the manifest's files, in the manifest's order (a fixed shuffle, so a larger slice
    starts with the same files). Buckets the manifest lacks are left out of this build, with the reason."""
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    srcs = man.get("sources", {})
    for b in cfg["buckets"]:
        if not b["enabled"]:
            continue
        m = srcs.get(b["name"])
        if not m:
            why = "skipped by download_slice.py" if b["name"] in man.get("skipped", []) else "not in the slice manifest"
            b.update(enabled=False, disabled_reason=f"{why} ({os.path.basename(manifest_path)})")
            continue
        old = b["parts"]
        merged = dict(source="manifest", repo=m["repo"], root=root, share=1.0, label="slice",
                      files=[dict(path=x["path"], size=x.get("bytes")) for x in m["files"]],
                      filter=next((p.get("filter") for p in old if p.get("filter")), None),
                      fmt=next((p.get("fmt") for p in old if p.get("fmt")), None),
                      meta_field=next((p.get("meta_field") for p in old if p.get("meta_field")), "meta"),
                      meta_keys=sorted({k for p in old for k in p.get("meta_keys") or []}),
                      meta_columns=sorted({k for p in old for k in p.get("meta_columns") or []}))
        b["parts"] = [{k: v for k, v in merged.items() if v is not None}]
    cfg["manifest"] = dict(path=os.path.abspath(manifest_path), tokens=man.get("tokens"))
    return cfg


def apply_scale(cfg, scale=1.0, tokens=None, post_tokens=None):
    """Targets for this build. Every full-scale target is multiplied by `scale` (0.5 to 1.5: 100B to 300B for the
    200B recipe). `tokens` builds only the first slice: that many pretraining tokens, every source at its share (a
    fraction of each target), drawn from the start of each source's fixed file order, so a later, larger build
    extends it. Post-training takes the same fraction unless `post_tokens` is given."""
    if not 0.5 <= scale <= 1.5:
        raise SystemExit(f"--scale must be between 0.5 and 1.5, not {scale}")
    # the fraction is of the whole recipe, so every source is at the same fraction of its target whether or not this
    # build has it (a slice without The Stack is 10% of each other source, not more)
    full_pre = sum(b["full_target"] for b in cfg["buckets"] if b["kind"] == "pretrain") * scale
    full_post = sum(b["full_target"] for b in cfg["buckets"] if b["kind"] == "post") * scale
    frac = 1.0 if tokens is None else min(1.0, tokens / full_pre)
    post_frac = frac if post_tokens is None else (min(1.0, post_tokens / full_post) if full_post else 0.0)
    for b in cfg["buckets"]:
        f = post_frac if b["kind"] == "post" else frac
        b["target"] = int(round(b["full_target"] * scale * f))
    cfg["build"] = dict(scale=scale, fraction=frac, post_fraction=post_frac,
                        pretrain_tokens=int(round(full_pre * frac)), post_tokens=int(round(full_post * post_frac)))
    return cfg


def has_extracts(work, name, cfg):
    """True when fetch extracted at least one file for some part of the bucket."""
    b = next(x for x in cfg["buckets"] if x["name"] == name)
    return any(os.path.isdir(os.path.join(work, "extract", part_id(b, i))) and
               any(f.endswith(".json") for f in os.listdir(os.path.join(work, "extract", part_id(b, i))))
               for i in range(len(b["parts"])))


def buckets(cfg, names=None, repo=None, work=None):
    """Enabled buckets, in recipe order; optionally only some names, one repo, and (with `work`) without optional
    buckets that fetch could not get."""
    out = [b for b in cfg["buckets"] if b.get("enabled", True) and b["target"] > 0]
    if names:
        missing = set(names) - {b["name"] for b in cfg["buckets"]}
        if missing:
            raise SystemExit(f"unknown buckets: {sorted(missing)}")
        out = [b for b in out if b["name"] in names]
    if repo:
        out = [b for b in out if b["repo"] == repo]
    if work is not None:
        out = [b for b in out if not b.get("optional") or has_extracts(work, b["name"], cfg)]
    return out


def part_id(b, i):
    p = b["parts"][i]
    if p.get("label"):
        tail = p["label"]
    elif p["source"] == "llmjp":
        tail = p["path"].split("/")[-1]
    elif p["source"] == "urls":
        tail = p["contains"].strip("/").replace("/", "-")
    elif p["source"] == "manifest":
        tail = "slice"
    elif p.get("subset"):
        tail = p["subset"]
    else:
        tail = p["patterns"][0].split("/*")[0].split("/")[-1] or "files"
    tail = re.sub(r"[^A-Za-z0-9._-]+", "-", tail.replace("+", "p")).strip("-")
    return f"{b['name']}.{i}.{tail}"


def parts(cfg, names=None, work=None):
    """(bucket, part index, part, part id) for every part of every enabled bucket."""
    for b in buckets(cfg, names, work=work):
        total = sum(p["share"] for p in b["parts"])
        if abs(total - 1) > 1e-6:
            raise SystemExit(f"bucket {b['name']}: part shares sum to {total}, not 1")
        for i, p in enumerate(b["parts"]):
            yield b, i, p, part_id(b, i)


def check(cfg):
    names = [b["name"] for b in cfg["buckets"]]
    if len(names) != len(set(names)):
        raise SystemExit("bucket names must be unique")
    for b in cfg["buckets"]:
        if b["repo"] not in cfg["repos"]:
            raise SystemExit(f"bucket {b['name']}: unknown repo {b['repo']}")
    if cfg["tokenizer"]["vocab_size"] > 65535:
        raise SystemExit("vocab_size must fit uint16 packing (at most 65535)")
    list(parts(cfg))


def sample_mb(cfg):
    """MB of each bucket in the tokenizer's training sample: the config's sample_mb if given, else sample_total_mb split
    over the recipe's groups by group_shares, and within a group by target."""
    t = cfg["tokenizer"]
    if t.get("sample_mb"):
        return dict(t["sample_mb"])
    bs = [b for b in buckets(cfg) if b["kind"] == "pretrain" and b.get("in_tokenizer_sample", True)]
    out = {}
    for g, share in t["group_shares"].items():
        members = [b for b in bs if b["group"] == g]
        tot = sum(b["full_target"] for b in members)
        for b in members:
            out[b["name"]] = round(t["sample_total_mb"] * share * b["full_target"] / tot, 2)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="print the buckets and targets of a build")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tokens", type=float, default=None, help="pretraining tokens of this build (the first slice)")
    a = ap.parse_args()
    cfg = apply_scale(load(), a.scale, a.tokens)
    check(cfg)
    print(f"build: scale {a.scale}, {cfg['build']['fraction']:.4f} of every target: "
          f"{cfg['build']['pretrain_tokens'] / B:.1f}B pretraining, {cfg['build']['post_tokens'] / B:.2f}B post-training")
    for repo in cfg["repos"]:
        bs = buckets(cfg, repo=repo)
        if bs:
            print(f"{cfg['repos'][repo]['repo_id']}: {sum(b['target'] for b in bs) / B:.2f}B tokens")
        for b in bs:
            print(f"  {b['name']:22s} {b['target'] / B:7.3f}B  anneal x{b['anneal_weight']:<4}  {b['license']}"
                  f"{'  +Common Crawl terms' if b['common_crawl'] else ''}{'  (optional)' if b.get('optional') else ''}")
    off = [b for b in cfg["buckets"] if not b["enabled"]]
    for b in off:
        print(f"  not built: {b['name']}: {b['disabled_reason']}")
