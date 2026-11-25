"""Step 7, cards: the README dataset card of each repo, from the reports the other steps wrote.

Each card credits every source with its license, says which buckets come from Common Crawl (and so also fall under
Common Crawl's terms of use) and which are share-alike, lists every source file used (commit, path, SHA-256), the token
counts, the filters, deduplication and decontamination, and how to download the data.
"""
import glob
import json
import os
import time

from . import config
from .common import gb, human, log, read_json, write_json

LLMJP_NOTES = """\
Notes carried over from the LLM-jp Corpus v4 README:

- Copyright for each document belongs, in principle, to its original author. The corpus is meant for pretraining
  large language models. Use, including redistribution, is allowed within the scope of each sub-corpus's license.
- LLM-jp distributes the corpus from a server in Japan to comply with Japanese copyright law, and notes that Japanese
  copyright law may not apply when the data is redistributed from servers outside Japan. This copy is hosted by
  Hugging Face, outside Japan.
- Sub-corpora marked "Common Crawl" above come from Common Crawl, and their users must also comply with the
  [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).
- LLM-jp takes requests to delete documents from the corpus at llm-jp-corpus (at) nii.ac.jp. Documents removed
  upstream are removed here in the next release.
"""

TOKENIZER_USE = """\
```python
from tokenizers import Tokenizer                      # pip install tokenizers
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
ids = tok.encode("こんにちは、世界。").ids
```
"""


STACK_TERMS_FALLBACK = """\
## Terms of Use for The Stack

The official text could not be fetched when this card was made. Read and accept the Terms of Use for The Stack at
https://huggingface.co/datasets/bigcode/the-stack#terms-of-use-for-the-stack before using this dataset."""


def stack_terms(paths):
    """The Terms of Use for The Stack, word for word from bigcode/the-stack's card (its third point requires anyone
    who hosts a copy to include them and require users to agree). Fetched once and kept in the manifest; without
    network a pointer is used, and upload refuses the gated repo until the real text is in."""
    path = os.path.join(paths.manifest, "the_stack_terms.md")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read().strip()
    try:
        from huggingface_hub import HfApi
        text = HfApi().dataset_info("bigcode/the-stack").card_data.to_dict()["extra_gated_prompt"].strip()
        os.makedirs(paths.manifest, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        return text
    except Exception as e:                                       # offline (tests): a pointer, flagged at upload
        log.warning(f"could not fetch The Stack's terms of use ({type(e).__name__}); the card points to them instead")
        return STACK_TERMS_FALLBACK


def license_ids(bs):
    """Hugging Face license ids for the card's metadata ("other" for anything without one)."""
    ids = []
    for b in bs:
        lic = b["license"]
        v = ("odc-by" if lic.startswith("ODC-BY") else "cc-by-4.0" if lic.startswith("CC-BY-4.0")
             else "cc-by-sa-3.0" if lic.startswith("CC-BY-SA-3.0") else "cc-by-sa-4.0" if lic.startswith("CC-BY-SA")
             else "apache-2.0" if lic.startswith("Apache-2.0") else "other")
        if v not in ids:
            ids.append(v)
    return ids


def header(cfg, paths, repo, bs, stats):
    r = cfg["repos"][repo]
    langs = sorted({lang for b in bs for lang in b["lang"].replace("code", "").split("+") if lang})
    total = sum((stats.get(b["name"], {}).get("splits", {}).get("train", {}).get("tokens") or b["target"]) for b in bs)
    size = "10B<n<100B" if total > 10e9 else "1B<n<10B" if total > 1e9 else "100M<n<1B" if total > 1e8 else "n<100M"
    lines = ["---", f"pretty_name: {json.dumps(r['title'])}", "license:"] + [f"- {x}" for x in license_ids(bs) or ["other"]]
    lines += ["language:"] + [f"- {x}" for x in langs or ["en"]]
    lines += ["task_categories:", "- text-generation", "size_categories:", f"- {size}"]
    lines += ["tags:", "- pretraining" if repo.startswith("pretrain") else "- post-training", "- exealm"]
    if any("code" in b["lang"] for b in bs):
        lines.append("- code")
    if r.get("gated"):
        terms = stack_terms(paths)
        prompt = ("This dataset holds source code from The Stack v1 (bigcode/the-stack). Each file keeps the "
                  "license of the repository it came from. The Stack's terms of use follow; by requesting access you "
                  "accept them for this copy too.\n\n" + terms)
        lines += ["extra_gated_prompt: |-"] + [("  " + line).rstrip() for line in prompt.splitlines()]
        lines += ["extra_gated_fields:",
                  f"  {json.dumps('I have read the Terms of Use for The Stack and agree with them')}: checkbox",
                  f"  {json.dumps('I will update my copy when files are removed on request')}: checkbox"]
    if bs:
        lines.append("configs:")
        for b in bs:
            lines += [f"- config_name: {b['name']}", "  data_files:",
                      "  - split: train", f"    path: raw/{b['name']}/train/*.parquet",
                      "  - split: validation", f"    path: raw/{b['name']}/val/*.parquet"]
    lines.append("---")
    return "\n".join(lines) + "\n"


def bucket_table(bs, stats, anneal=True):
    head = "| Bucket | Group | What | Training tokens (target) | Validation tokens | Long val docs |" + \
        (" Anneal weight |" if anneal else "") + " License | Common Crawl | Share-alike |"
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for b in bs:
        s = stats.get(b["name"], {}).get("splits", {})
        tr, va = s.get("train", {}), s.get("val", {})
        rows.append(f"| `{b['name']}` | {b.get('group', '')} | {b['description']} | {human(tr.get('tokens', 0))} "
                    f"({human(b['target'])}) | {human(va.get('tokens', 0))} | {va.get('long_docs', 0)} |"
                    + (f" {b.get('anneal_weight', 1.0)} |" if anneal else "")
                    + f" {b['license']} | {'yes' if b['common_crawl'] else 'no'} | {'yes' if b['share_alike'] else 'no'} |")
    return "\n".join(rows) + "\n"


def part_label(p):
    if p["source"] == "llmjp":
        return f"LLM-jp Corpus v4 `{p['path']}`"
    if p["source"] == "urls":
        return f"`{p['repo']}` file list `{p['list_file']}`, URLs with `{p['contains']}`"
    if p["source"] == "manifest":
        return f"{len(p['files'])} files listed by the slice manifest (download_slice.py), in its fixed order"
    return ", ".join(f"`{x}`" for x in p["patterns"])


def sources_section(cfg, bs):
    out = ["## Sources and licenses", ""]
    for b in bs:
        lic_url = config.LICENSE_URLS.get(b["license"], "")
        lic = f"[{b['license']}]({lic_url})" if lic_url else b["license"]
        cc = " Built from Common Crawl: the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use) also " \
             "apply." if b["common_crawl"] else ""
        files = "; ".join(part_label(p) + (f" ({p['share']:.0%})" if len(b["parts"]) > 1 else "") for p in b["parts"])
        out.append(f"- `{b['name']}`: {b['credit']}, {files}. License: {lic}.{cc}")
    credits = sorted({b["credit"] for b in bs})
    out += ["", "Credit, with thanks, to the authors of every source: " + "; ".join(credits) + "."]
    if any(p["source"] == "llmjp" for b in bs for p in b["parts"]):
        out += ["", "LLM-jp Corpus v4 was built by the [LLM-jp](https://llm-jp.nii.ac.jp/) Corpus Building Working "
                    "Group; its `ja_kaken` sub-corpus is the outline of each research project in "
                    "[KAKEN](https://kaken.nii.ac.jp/), operated by NII.", "", LLMJP_NOTES]
    if any(b["share_alike"] for b in bs):
        sa = ", ".join(f"`{b['name']}`" for b in bs if b["share_alike"])
        out += ["", f"**Share-alike.** {sa} are under CC BY-SA licenses. They are kept in their own folders "
                    "(`raw/<bucket>/`, `packed/<bucket>/`) so their license stays clear: if you share them, or "
                    "something adapted from them, it must carry the same license."]
    return "\n".join(out) + "\n"


def files_section(cfg, paths, bs):
    out = ["## Source files used", "",
           "Every file read, pinned to a commit. `docs` is how many of its documents are in this dataset "
           "(training plus validation).", "",
           "| Bucket | Source | Commit | File | Bytes | SHA-256 | Docs used |", "|---|---|---|---|---|---|---|"]
    for b in bs:
        rep = read_json(paths.p("reports", "assemble", f"{b['name']}.json"), {})
        used = {}
        for split in ("train", "val"):
            for f in rep.get("splits", {}).get(split, {}).get("files", []):
                used[f["path"]] = used.get(f["path"], 0) + f["docs"]
        for i, p in enumerate(b["parts"]):
            pid = config.part_id(b, i)
            for j in sorted(glob.glob(os.path.join(paths.extract_dir(pid), "*.json"))):
                info = read_json(j)
                pv = info["provenance"]
                src = "llm-jp-corpus-v4" if pv["source"] == "llmjp" else pv["repo"]
                out.append(f"| `{b['name']}` | {src} | `{pv['revision'][:10]}` | [{pv['path']}]({pv.get('url', '')}) "
                           f"| {pv['size']} | `{pv['sha256']}` | {used.get(pv['path'], 0)} |")
    return "\n".join(out) + "\n"


def filters_section(cfg, paths, bs):
    out = ["## Filters", "",
           "Every document's text is kept byte for byte. A document is dropped when its text is empty or contains one "
           "of the tokenizer's special-token strings (such as `<|endoftext|>`), and by the bucket's filter below.", ""]
    for b in bs:
        for i, p in enumerate(b["parts"]):
            f = p.get("filter")
            if f:
                parts = []
                if f.get("lang_in"):
                    parts.append(f"language (`meta.lang`) in {f['lang_in']}")
                if f.get("quality_in"):
                    parts.append(f"`quality_category` in {f['quality_in']}")
                if f.get("drop_subsets"):
                    parts.append(f"`subset` not in {f['drop_subsets']}")
                if f.get("min_edu_score") is not None:
                    parts.append(f"`fineweb_edu_classifier_score` >= {f['min_edu_score']}")
                out.append(f"- `{b['name']}` ({p.get('label') or part_label(p)}): " + "; ".join(parts) + ".")
    drops = {}
    for b in bs:
        for i, p in enumerate(b["parts"]):
            for j in glob.glob(os.path.join(paths.extract_dir(config.part_id(b, i)), "*.json")):
                info = read_json(j)
                d = drops.setdefault(b["name"], dict(read=0, kept=0, reasons={}))
                d["read"] += info["rows_read"]
                d["kept"] += info["rows_kept"]
                for k, v in info["dropped"].items():
                    d["reasons"][k] = d["reasons"].get(k, 0) + v
    if drops:
        out += ["", "| Bucket | Documents read | Kept by the filters | Dropped, by reason |", "|---|---|---|---|"]
        for name, d in drops.items():
            reasons = ", ".join(f"{k}: {v}" for k, v in sorted(d["reasons"].items())) or "none"
            out.append(f"| `{name}` | {d['read']} | {d['kept']} | {reasons} |")
    return "\n".join(out) + "\n"


def dedup_section(cfg, paths, bs):
    rep = read_json(paths.report("dedup"), {})
    out = ["## Deduplication", "",
           "Exact duplicates are removed within and across every bucket of the pretraining and post-training sets "
           "together. Two documents are duplicates when their text is the same after collapsing whitespace and "
           "lowercasing (SHA-1 of the normalized text). The copy kept is the one a build reaches first: each "
           "source's files are read in a fixed shuffled order, and files from all sources are interleaved by how far "
           "into its share a build must be to read them; ties go by recipe order. So a larger build never drops a "
           "document that a smaller one kept because of a file only the larger build reads. Near-duplicates are not "
           "removed.", ""]
    rows = []
    for b in bs:
        for i, p in enumerate(b["parts"]):
            pr = rep.get("parts", {}).get(config.part_id(b, i))
            if pr:
                of = ", ".join(f"{k.split('.')[0]}: {v}" for k, v in pr["duplicates_of"].items()) or "none"
                rows.append(f"| `{b['name']}` | {p.get('label') or part_label(p)} | {pr['documents']} | "
                            f"{pr['duplicates']} | {of} |")
    if rows:
        out += ["| Bucket | Source | Documents after filters | Duplicates dropped | Kept copy was in |",
                "|---|---|---|---|---|"]
        out += rows
    return "\n".join(out) + "\n"


def decontam_section(cfg, paths, bs):
    rep = read_json(paths.report("benchmarks"), {})
    n = cfg["decontam"]["ngram"]
    out = ["## Decontamination", "",
           f"A document is dropped when any {n}-token window of it (new tokenizer's tokens, not words) equals a "
           f"window of one of these held-out benchmark sets. This is the rule of `exealm.data.decontam` with its "
           f"defaults (n = {n}, one match is enough); windows never cross a document's end. Exact token overlap "
           "only: paraphrases are not caught.", "",
           "| Benchmark | Dataset | Config / split | Revision | Windows |", "|---|---|---|---|---|"]
    for name, r in rep.get("benchmarks", {}).items():
        sample = " (sample only)" if r.get("sample") else ""
        out.append(f"| {name} | {r.get('repo')} | {r.get('config')} / {r.get('split')} | `{(r.get('revision') or '')[:10]}` "
                   f"| {r.get('windows')}{sample} |")
    dropped = {}
    for b in bs:
        st = read_json(paths.p("reports", "assemble", f"{b['name']}.json"), {})
        for split in ("train", "val"):
            for k, v in st.get("splits", {}).get(split, {}).get("contaminated_dropped", {}).items():
                dropped.setdefault(b["name"], {}).setdefault(k, 0)
                dropped[b["name"]][k] += v
    out += ["", "Documents dropped: " + ("; ".join(f"`{k}` " + ", ".join(f"{bn} {c}" for bn, c in v.items())
                                                 for k, v in dropped.items()) if dropped else "none.")]
    dec = read_json(paths.report("decontam"), {})
    names = {b["name"] for b in bs}
    top = [w for w in dec.get("top_windows", []) if w["documents"] > 1][:10]
    if top:
        out += ["", "The benchmark windows that matched most often (in documents of all buckets):", "",
                "| Benchmark | Window | Documents |", "|---|---|---|"]
        out += [f"| {w['benchmark']} | `{json.dumps(w['window'], ensure_ascii=False)[1:-1].replace('|', '\\|')}` "
                f"| {w['documents']} |" for w in top]
    shares = {k: v["share_dropped"] for k, v in dec.get("buckets", {}).items() if k in names and v["share_dropped"]}
    if shares:
        out += ["", "Share of selected documents dropped: " + ", ".join(f"`{k}` {v * 100:.2f}%" for k, v in shares.items())
                + "."]
    return "\n".join(out) + "\n"


def layout_section(cfg, repo, bs, root):
    name = cfg["repos"][repo]["repo_id"]
    short = name.split("/")[-1]
    raw = any(os.path.isdir(os.path.join(root, "raw", b["name"])) for b in bs)
    out = ["## Layout", "", "```"] + ([
           "raw/<bucket>/train/*.parquet     text, n_tokens, source, source_file, source_row, meta",
           "raw/<bucket>/val/*.parquet"] if raw else []) + [
           "packed/<bucket>/train/           tokens.bin (uint16), offsets.npy (int64 document starts), meta.json",
           "packed/<bucket>/val/",
           "sources.json                     every source file used, per bucket: repo, commit, path, size, documents"]
    if cfg["repos"][repo].get("tokenizer"):
        out.append("tokenizer/tokenizer.json         the 32k byte-level BPE these packs use")
    pre = f"{cfg['release']}/" if cfg.get("release") else ""
    if repo in ("pretrain", "post"):
        out.append("mix.json, mix_val.json           bucket weights for training (exealm.data.loader.open_data)")
    if repo == "pretrain":
        out.append("mix_anneal.json                  weights for the last 10% of steps: share x anneal_weight")
    if repo == "pretrain" and config.buckets(cfg, repo="pretrain_stack"):
        out.append("mix_public*.json                 the same three without the gated Stack bucket")
    if cfg["repos"][repo].get("library_slot"):
        out.append("library/                         the library's question-answer and morph tasks (empty for now)")
    out += ["```", "",
            ("Row *i* of a bucket's raw Parquet files, read in file-name order, is document *i* of its pack. `meta` "
             "is JSON: the source's own metadata (ids, URLs, licenses where the source has them; a conversation's "
             "turns for chat sources). " if raw else "No raw text in this release: `sources.json` names every source "
             "file, and the sources are public. ") +
            "A packed document is its tokens followed by `<|endoftext|>` (id 0).", "",
            "## Download", "", "Packed tokens, tokenizer and mix files only (what training reads):", "", "```bash",
            f"hf download {name} --repo-type dataset --include '{pre}packed/*' '{pre}tokenizer/*' '{pre}mix*.json' "
            f"--local-dir data/{short}", "```", ""] + ([] if pre else [
            "Raw text of one bucket, with the `datasets` library:", "",
            "```python", "from datasets import load_dataset",
            f"ds = load_dataset(\"{name}\", \"{bs[0]['name'] if bs else 'BUCKET'}\", split=\"train\", streaming=True)",
            "```", ""]) + [
            "In the ExeaLM repository, `bash experiments/data_mix/fetch.sh` downloads the packs and writes mix files "
            "with local paths."]
    return "\n".join(out) + "\n"


def tokenizer_section(cfg, paths):
    rep = read_json(paths.report("tokenizer"), {})
    if not rep:
        return ""
    share = ", ".join(f"{k} {v * 100:.1f}%" for k, v in rep.get("sample_share", {}).items())
    bpt = ", ".join(f"{k} {v['bytes_per_token']}" for k, v in rep.get("buckets", {}).items() if v.get("bytes_per_token"))
    return "\n".join([
        "## Tokenizer", "",
        f"`tokenizer/tokenizer.json`: byte-level BPE with {rep['vocab_size']} tokens (fits uint16), trained with "
        "`exealm.data.tokenizer.train` (the ExeaLM repo's settings: GPT-2 style byte-level pre-tokenizer without a "
        "prefix space, no normalization, byte-level decoder). Special tokens, in id order from 0: "
        + ", ".join(f"`{k}`" for k in rep["special_ids"]) + ". Decoding gives back the input exactly.", "",
        f"Trained on {rep['sample_total_bytes'] / 1e6:.1f} MB of text from the training side of the pretraining "
        f"buckets (no validation documents), by bytes: {share}. The Stack bucket is not in the sample.", "",
        f"Bytes of text per token, measured on other documents of each bucket: {bpt}.", "", TOKENIZER_USE])


def library_slot_section():
    return """\
## Library slot

`library/` is reserved for the question-answer pairs and morph tasks that `experiments/library/build.py` makes for
ExeaLM's distilled library (the library itself goes in
[exea-labs/exealm-library](https://huggingface.co/datasets/exea-labs/exealm-library)). It is empty in this
release (the recipe's `library_tasks`, 0.5B tokens at full scale). When it is filled, `build.py pack` output goes here as packed directories, with the same tokenizer:

```
library/qa/train/  library/qa/heldout/         Question: ...\\nAnswer: ... documents, targets.npy, items.jsonl
library/tasks/train/  library/tasks/heldout/   morph tasks: instruction and context, then the target
```
"""


def write_card(cfg, paths, repo):
    r = cfg["repos"][repo]
    root = paths.repo_dir(cfg, repo)
    os.makedirs(root, exist_ok=True)
    bs = config.buckets(cfg, repo=repo, work=paths.work)
    stats = {b["name"]: read_json(paths.p("reports", "assemble", f"{b['name']}.json"), {}) for b in bs}
    total = sum(s.get("splits", {}).get("train", {}).get("tokens", 0) for s in stats.values())
    when = time.strftime("%Y-%m-%d")
    if r.get("empty"):
        text = library_card(cfg, r)
        os.makedirs(os.path.join(root, "library"), exist_ok=True)
        with open(os.path.join(root, "library", "README.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write("The library goes here: `tokens.bin`, `offsets.npy`, `meta.json`, `SOURCE.md` and `index.jsonl` "
                    "from `experiments/library/build.py pack`. Empty for now.\n")
    else:
        intro = {
            "pretrain": "ExeaLM's pretraining data, built from the recipe in docs/data_recipe.md: research papers, code "
                        "and math, with English and Japanese. Each bucket is one recipe source, packed with the 32k "
                        "tokenizer that ships in `tokenizer/`, with a held-out validation split.",
            "pretrain_stack": f"The Stack v1 part of ExeaLM's pretraining data ({cfg['repos']['pretrain']['repo_id']}). "
                              "It is a separate, gated repository because each file keeps the license of the "
                              "repository it came from.",
            "post": "ExeaLM's post-training data, from the same recipe: code, math and science reasoning, instruction "
                    "following. Each document is one example as text: IFM's sets as upstream has them (prompt, "
                    "reasoning and answer in one field); question/answer fields joined by a blank line; "
                    "conversations as their turns joined by a blank line, with the turns themselves in `meta` "
                    "(`messages`) for exealm.train.sft's chat template.",
        }[repo]
        title = f"{r['title']}: {cfg['release']}" if cfg.get("release") else r["title"]
        # a release folder's README documents that folder; the repo's own card (its root README) is left alone
        parts = [header(cfg, paths, repo, bs, stats) if not cfg.get("release") else "", f"# {title}", "", intro, "",
                 f"{human(total)} training tokens in {len(bs)} bucket{'s' if len(bs) != 1 else ''}, made on {when} by "
                 "[experiments/data_mix](https://github.com/exea-labs/ExeaLM/tree/main/experiments/data_mix) in the "
                 "ExeaLM repository. Token counts include one `<|endoftext|>` per document.", "",
                 bucket_table(bs, stats, anneal=repo != "post")]
        parts += [build_section(cfg, paths, repo)]
        if repo == "pretrain_stack":
            parts += [STACK_NOTES, stack_terms(paths) + "\n"]
        if repo == "pretrain" and config.buckets(cfg, repo="pretrain_stack"):
            st = config.buckets(cfg, repo="pretrain_stack")
            names = ", ".join(f"`{b['name']}`" for b in st)
            parts += [f"The Stack v1 bucket ({names}, "
                      f"{human(sum(b['target'] for b in st))} tokens) is in the gated repository "
                      f"[{cfg['repos']['pretrain_stack']['repo_id']}](https://huggingface.co/datasets/"
                      f"{cfg['repos']['pretrain_stack']['repo_id']}), because each of its files keeps its own "
                      "license. `mix.json` includes it as one more source; `mix_public.json` leaves it out.\n"]
        if repo != "pretrain":
            parts += [f"The tokenizer is in [{cfg['repos']['pretrain']['repo_id']}](https://huggingface.co/datasets/"
                      f"{cfg['repos']['pretrain']['repo_id']}) under `tokenizer/`.\n"]
        parts += [layout_section(cfg, repo, bs, root)]
        if r.get("tokenizer"):
            parts += [tokenizer_section(cfg, paths)]
        parts += [sources_section(cfg, bs), filters_section(cfg, paths, bs), dedup_section(cfg, paths, bs),
                  decontam_section(cfg, paths, bs), validation_section(cfg)]
        if r.get("library_slot"):
            parts += [library_slot_section()]
            os.makedirs(os.path.join(root, "library"), exist_ok=True)
            with open(os.path.join(root, "library", "README.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write("Reserved for the library's question-answer pairs and morph tasks (experiments/library). "
                        "Empty in this release.\n")
        parts += [files_section(cfg, paths, bs)]
        text = "\n".join(parts)
    path = os.path.join(root, "README.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    log.info(f"card: {path}")
    return path


STACK_NOTES = """\
## License and opt-out

Each file in this dataset keeps the license of the repository it came from. The Stack v1 was collected from
permissively licensed repositories; check a file's license (in `meta`, with its repository and path) before you reuse
it. By downloading, you accept [The Stack's terms of use](https://huggingface.co/datasets/bigcode/the-stack).

**Opt-out requests are honoured.** If your code is here and you want it removed, open a discussion on this dataset's
page, or use BigCode's [Am I in The Stack?](https://huggingface.co/spaces/bigcode/in-the-stack) opt-out: files removed
from The Stack are removed here in the next release. Please keep your copy up to date and delete removed files.
"""


def build_section(cfg, paths, repo):
    """What recipe this is, at what scale, and how much of it; what the recipe lists that this build lacks."""
    b = cfg.get("build", {})
    rec = cfg.get("recipe", {})
    kind = "post" if repo == "post" else "pretrain"
    full = sum(x["full_target"] for x in cfg["buckets"] if x["kind"] == kind and x["enabled"]) * b.get("scale", 1)
    frac = b.get("post_fraction" if kind == "post" else "fraction", 1.0)
    out = ["## This build", ""]
    if rec:
        out.append(f"Recipe `{rec.get('name')}` ({rec.get('date')}), [{rec.get('path')}]"
                   f"(https://github.com/exea-labs/ExeaLM/blob/main/{rec.get('path')}); targets at scale "
                   f"{b.get('scale', 1)} (the recipe is written at 200B pretraining tokens).")
    if frac < 1:
        out.append(f"This is the first slice of that data: {frac:.1%} of every target, {human(full * frac)} of "
                   f"{human(full)} tokens. Each source's files are read in a fixed shuffled order and each source "
                   "stops at its share, so a later, larger build extends this one: its training documents start with "
                   "these, and the validation split stays the same.")
    else:
        out.append(f"Every target in full: {human(full)} tokens.")
    skipped = read_json(paths.report("fetch"), {}).get("skipped", {})
    missing = [x for x in cfg["buckets"] if x["kind"] == kind and x["repo"] == repo
               and (not x["enabled"] or x["name"] in skipped)]
    if missing:
        out += ["", "In the recipe but not in this build:"]
        out += [f"- `{x['name']}` ({human(x['full_target'])} at full scale): "
                f"{x.get('disabled_reason') or 'optional, and its download failed: ' + skipped.get(x['name'], '')}"
                for x in missing]
    if kind == "pretrain" and rec.get("excluded"):
        out += ["", "Left out on purpose: " + " ".join(rec["excluded"])]
    return "\n".join(out) + "\n"


def validation_section(cfg):
    s = cfg["split"]
    return f"""\
## Validation split

Each bucket holds out about {human(s['val_tokens'])} tokens on top of its training target, drawn at random (by a hash
of the text) from the first file of each source, which every build has, so the validation split is the same in every
build. It has at least {s.get('val_long_min', 128)} documents (up to {s['val_long_docs']}) of at least
{s['long_tokens']} tokens for the long-document test (`experiments/stream_eval.py` reads 8 windows of 512 tokens).
No text is in both splits, or in two buckets.
"""


def library_card(cfg, r):
    return f"""---
pretty_name: {json.dumps(r['title'])}
license: other
language:
- en
- ja
tags:
- exealm
- retrieval
---
# {r['title']}

The distilled library for ExeaLM's retrieval circuit: encyclopedia entries and canonical code written by an
open-weight model, packed for `--datastore_data`. **Empty for now**: `experiments/library/build.py` in the ExeaLM
repository makes the content, and it will be uploaded here with its own sources, models and licenses listed.

## Layout (planned)

```
library/            the library: tokens.bin, offsets.npy, meta.json, SOURCE.md, index.jsonl (build.py pack)
raw/entries.jsonl   the entries as text, with model and prompt version
```

The question-answer pairs and morph tasks made from it go in
[{cfg['repos']['post']['repo_id']}](https://huggingface.co/datasets/{cfg['repos']['post']['repo_id']}) under `library/`.
Both are packed with the tokenizer in
[{cfg['repos']['pretrain']['repo_id']}](https://huggingface.co/datasets/{cfg['repos']['pretrain']['repo_id']}).
"""


def run(cfg, paths, args):
    out = {}
    for repo in cfg["repos"]:
        if (cfg["repos"][repo].get("empty") and not cfg.get("release")) or \
                config.buckets(cfg, repo=repo, work=paths.work):
            out[repo] = write_card(cfg, paths, repo)
    write_json(paths.report("cards"), out)
    return out


__all__ = ["run", "write_card", "gb"]
