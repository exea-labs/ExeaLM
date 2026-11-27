"""The human tier of library v2: real human-written passages that no ExeaLM model has trained on.

Library v1 (data/library/v1) is written by a model. The human tier is cut from the recipe's own sources
(experiments/recipe/exealm_recipe_v2.json), from files that the training slice did not read: the slice
(data/exealm-pretrain/slice-30b-r2) took the first 10% of each source in a fixed shuffled file order, and its
sources.json lists every file it read, so a file not in that list was never trained on. Each document is cut into
passages of about 512 to 1024 tokens at paragraph boundaries; each passage starts with its document's title on its own
line, as v1 entries start with their topic. The result is packed like exealm.data.pack output (TokenShards reads it).

Steps (each skips what is done, so a step that dies is simply run again):

    plan        list each source's files at a pinned commit, drop the slice's files, fix a shuffled order
                (<root>/manifest/order/<source>.json). The proof-pile-2 folders of allenai/olmo-mix-1124 (arxiv/,
                algebraic-stack/, open-web-math/) each mix arXiv papers, OpenWebMath pages and code, so their files
                are sorted by what their first document is, and every document is checked again when read
    fetch       download the files the next parts need (login node: network only), and the arXiv metadata
                (titles and per-paper licenses). Plain HTTP with resume and size/SHA-256 checks
    benchmarks  export the decontamination benchmarks (login node: network only), as the slice's build did
    index       window hashes of everything that must stay clean: every val split of the slice, library v1's held-out
                qa/ and tasks/ packs, and the benchmarks; plus a 1-in-32 sample of the slice's training windows
    passages    one part of the pool: each source from where the previous part stopped, until its share of
                --part_tokens. Documents go through a light quality filter, passages through exact deduplication
                and the 13-gram checks (Slurm job: this is the CPU work)
    pack        the part's TokenShards pack: tokens.bin, offsets.npy, meta.json, index.jsonl (one line per passage,
                in pack order) and passages.jsonl.gz (the text), plus stats.json
    run         index (if missing), passages and pack for one part

Parts: <root>/human_pool/part_000, part_001, ... each its own pack; part_000 is human_pool_v0.

What is dropped, per passage (counts in each part's stats.json):
    val, heldout   any 13-token window equal to one in a val split of the slice / library v1's held-out packs
    bench:<name>   a benchmark window, with the slice's rule (experiments/data_mix: only problem statements of code
                   benchmarks; boilerplate windows found in 2+ items and windows with fewer than 3 words ignored)
    train_overlap  half or more of its sampled windows are in the slice's training data: the same text reached
                   training through another source (pes2o and arXiv share papers, StackExchange and web copies, ...)
    duplicate      the same text (whitespace and case ignored) as an earlier passage of this or an earlier part
    quality        mostly non-text, base64-like blobs, minified lines, repeated lines, special-token strings, short

Licenses: only redistributable ones. arXiv papers by their own license in the arXiv metadata (CC BY, CC BY-SA, CC0,
public domain; the default arXiv licence and NC/ND licences are dropped). Code with a recorded repository license
must be permissive (PERMISSIVE); StarCoder files without one come from The Stack v1.2's permissive set. Every passage
keeps its license and source in index.jsonl.

    python -m experiments.library.human_tier plan
    python -m experiments.library.human_tier fetch --upto_tokens 2e9
    python -m experiments.library.human_tier benchmarks
    python -m experiments.library.human_tier run --part 0 --part_tokens 2e9 --workers 16
"""
import argparse
import fnmatch
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

ROOT = "/expanse/lustre/projects/fli100/abhatia7/library_v2"
SLICE = "data/exealm-pretrain/slice-30b-r2"
LIBV1 = "data/library/v1"
NGRAM = 13
TRAIN_SAMPLE_SHIFT = 59                 # a training window is sampled when its hash's top 5 bits are 0: 1 in 32
SEED = "exealm-human-tier"

OLMO = "allenai/olmo-mix-1124"
DOLMINO = "allenai/dolmino-mix-1124"
PROOF_PILE = ["data/arxiv/*.json.gz", "data/algebraic-stack/*.json.gz", "data/open-web-math/*.json.gz"]
STARCODER = ["data/starcoder/*.json.gz"]
ARXIV_META = ("librarian-bots/arxiv-metadata-snapshot", ["data/train-*.parquet"])    # CC0, weekly mirror of Kaggle's

# StarCoder languages that are not code worth a library entry (issues and commits also have no clear license)
NOT_CODE = {"github-issues-filtered-structured", "git-commits-cleaned", "json", "yaml", "xml", "html", "css",
            "jupyter-structured-clean-dedup"}
DOCS = {"markdown", "restructuredtext"}
PYTHON = {"python", "jupyter-scripts-dedup-filtered"}
SCI_IMPORTS = {"numpy", "scipy", "torch", "tensorflow", "jax", "sklearn", "pandas", "matplotlib", "statsmodels", "sympy",
               "astropy", "Bio", "networkx", "transformers", "xarray", "numba", "cupy", "skimage", "seaborn", "pymc",
               "qiskit"}                # the recipe's research_python filter
PERMISSIVE = {"MIT", "MIT-0", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSD-3-Clause-Clear", "0BSD", "ISC",
              "Zlib", "BSL-1.0", "CC0-1.0", "Unlicense", "CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0",
              "PSF-2.0", "Python-2.0", "PostgreSQL", "NCSA", "X11", "Unicode-DFS-2016"}

# share: of each part's tokens. bpt: bytes of text per token (the slice's measured values; guesses for new sources).
# yield: share of a file's text that ends up in passages (licenses, filters, short documents), for planning downloads.
SOURCES = {
    "pes2o": dict(group="research", repo="allenai/peS2o", patterns=["data/v2/train-*.json.gz"], reader="pes2o",
                  license="ODC-BY", bpt=4.39, lang="en", kind="prose", share=0.26, yield_=0.9),
    "arxiv": dict(group="research", repo=OLMO, patterns=PROOF_PILE, file_kind="arxiv", reader="arxiv", bpt=3.67,
                  lang="en", kind="latex", share=0.10, yield_=0.15),
    "kaken": dict(group="research", repo="hpprc/llmjp-kaken", patterns=["data/train-*.parquet"], reader="titled",
                  license="CC-BY-4.0", bpt=4.8, lang="ja", kind="ja", share=0.04, yield_=0.8,
                  columns=["id", "url", "title", "text"]),
    "starcoder": dict(group="code", repo=OLMO, patterns=STARCODER, langs_not=NOT_CODE | DOCS | PYTHON, reader="code",
                      bpt=3.16, lang="code", kind="code", share=0.15, yield_=0.7),
    "research_python": dict(group="code", repo=OLMO, patterns=STARCODER, langs=PYTHON, reader="code", sci=True,
                            bpt=3.35, lang="code", kind="code", share=0.08, yield_=0.35),
    "algebraic_stack": dict(group="code", repo=OLMO, patterns=PROOF_PILE, file_kind="code", reader="code", bpt=2.68,
                            lang="code", kind="code", share=0.06, yield_=0.7),
    "code_docs": dict(group="code", repo=OLMO, patterns=STARCODER, langs=DOCS, reader="code", bpt=3.6, lang="en",
                      kind="docs", share=0.06, yield_=0.6),
    "stackexchange": dict(group="qa", repo=DOLMINO, patterns=["data/stackexchange/*.json.gz"], reader="stackexchange",
                          bpt=3.6, lang="en", kind="prose", share=0.10, yield_=0.85),
    "wikipedia_en": dict(group="reference", repo=DOLMINO, patterns=["data/wiki/*.json.gz"], reader="first_line",
                         license="CC-BY-SA", bpt=3.89, lang="en", kind="prose", share=0.07, yield_=0.75),
    "wikipedia_ja": dict(group="reference", repo="wikimedia/wikipedia", patterns=["20231101.ja/*.parquet"],
                         reader="titled", license="CC-BY-SA-3.0", bpt=4.02, lang="ja", kind="ja", share=0.03,
                         yield_=0.7, columns=["id", "url", "title", "text"]),
    "open_web_math": dict(group="math", repo=OLMO, patterns=PROOF_PILE, file_kind="web", reader="web",
                          license="ODC-BY", bpt=3.67, lang="en", kind="latex", share=0.03, yield_=0.8),
    "megamath": dict(group="math", repo="IFM/MegaMath", patterns=["megamath-web/*.parquet"], reader="web",
                     license="ODC-BY", bpt=3.81, lang="en", kind="latex", share=0.02, yield_=0.7,
                     columns=["id", "url", "text"]),
}
ORDER = list(SOURCES)
SPECIAL = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|>",
           "<|tool|>", "<|retrieved|>", "<|board|>"]            # exealm.data.tokenizer.SPECIAL (checked in the test)

# passages, in estimated tokens (bytes / bpt); the packed counts are exact
TARGET, MAX, UNIT, MERGE_MAX, MIN_TOKENS, MAX_PER_DOC = 768, 1024, 512, 1280, 128, 24
ALPHA_MIN = dict(prose=0.6, docs=0.45, latex=0.4, ja=0.5, code=0.3)    # letters among non-space characters
BLOB = re.compile(r"[A-Za-z0-9+/=_\-]{256,}")
BIB = re.compile(r"\\begin\{thebibliography\}|\\bibliography\{|\\section\*?\{\s*(References|Bibliography)\s*\}|"
                 r"\\printbibliography")
CODE_HEADER = re.compile(r"^(?:<reponame>|<filename>|<gh_stars>)[^\n]*\n")
PARA = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
SENT = re.compile(r"(?<=[.!?。！？])[ \t]+|(?<=[。！？])")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# plan and fetch (login node)

def slice_files(slice_dir):
    """{repo: {path, ...}} of every file the slice read, and {repo: revision} it read them at."""
    used, revs = {}, {}
    for b in read_json(os.path.join(slice_dir, "sources.json"))["buckets"].values():
        for f in b["files"]:
            used.setdefault(f["repo"], set()).add(f["path"])
            revs.setdefault(f["repo"], f["revision"])
    return used, revs


def hf_token():
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def hf_url(repo, rev, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def head_doc(repo, rev, path, nbytes=400_000):
    """The first JSON document of a .json.gz file on the Hub, from its first bytes."""
    tok = hf_token()
    req = urllib.request.Request(hf_url(repo, rev, path), headers={"Range": f"bytes=0-{nbytes}",
                                                                   **({"Authorization": f"Bearer {tok}"} if tok else {})})
    raw = urllib.request.urlopen(req, timeout=120).read()
    text = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw).decode("utf-8", "ignore")
    return json.loads(text.split("\n", 1)[0])


def doc_kind(rec):
    """What a proof-pile-2 document is: an arXiv paper, an OpenWebMath page, or code."""
    d = rec.get("doc") if isinstance(rec.get("doc"), dict) else {}
    if d.get("arxiv_id"):
        return "arxiv"
    if "warc_path" in d or "extraction_info" in d:
        return "web"
    return "code"


def starcoder_lang(path):
    return os.path.basename(path).rsplit("-", 1)[0]


def listing(repo, rev, patterns):
    from huggingface_hub import HfApi
    prefixes = sorted({p.split("*")[0].rsplit("/", 1)[0] if "/" in p.split("*")[0] else "" for p in patterns})
    files = {}
    for prefix in prefixes:
        kw = dict(path_in_repo=prefix) if prefix else {}
        for f in HfApi().list_repo_tree(repo, repo_type="dataset", revision=rev, recursive=True, **kw):
            if getattr(f, "size", None) is not None and any(fnmatch.fnmatchcase(f.path, p) for p in patterns):
                files[f.path] = dict(path=f.path, size=f.size, sha256=f.lfs.sha256 if getattr(f, "lfs", None) else None)
    return [files[k] for k in sorted(files)]


def revision(root, repo, slice_revs):
    path = os.path.join(root, "manifest", "revisions.json")
    revs = read_json(path, {})
    if repo not in revs:
        if repo in slice_revs:
            revs[repo] = slice_revs[repo]             # the commit the slice read: the same files
        else:
            from huggingface_hub import HfApi
            revs[repo] = HfApi().dataset_info(repo).sha
        write_json(path, revs)
    return revs[repo]


def cmd_plan(args):
    used, slice_revs = slice_files(args.slice)
    kinds_path = os.path.join(args.root, "manifest", "proofpile_kinds.json")
    kinds = read_json(kinds_path, {})
    for name in args.sources or ORDER:
        s = SOURCES[name]
        out = os.path.join(args.root, "manifest", "order", f"{name}.json")
        if os.path.exists(out) and not args.replan:
            continue
        rev = revision(args.root, s["repo"], slice_revs)
        files = [f for f in listing(s["repo"], rev, s["patterns"]) if f["path"] not in used.get(s["repo"], set())]
        if s.get("langs"):
            files = [f for f in files if starcoder_lang(f["path"]) in s["langs"]]
        if s.get("langs_not"):
            files = [f for f in files if starcoder_lang(f["path"]) not in s["langs_not"]]
        if s.get("file_kind"):
            for f in files:
                if f["path"] not in kinds:
                    kinds[f["path"]] = doc_kind(head_doc(s["repo"], rev, f["path"]))
                    write_json(kinds_path, kinds)
            files = [f for f in files if kinds[f["path"]] == s["file_kind"]]
        random.Random(f"{SEED}:{name}").shuffle(files)
        skipped = sorted(p for p in used.get(s["repo"], set()) if any(fnmatch.fnmatchcase(p, q) for q in s["patterns"]))
        write_json(out, dict(source=name, repo=s["repo"], revision=rev, patterns=s["patterns"], seed=SEED,
                             slice_files_left_out=skipped, files=files))
        log(f"plan {name}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.1f} GB "
            f"({len(skipped)} slice files left out)")


def order(root, name):
    o = read_json(os.path.join(root, "manifest", "order", f"{name}.json"))
    if o is None:
        raise SystemExit(f"no file order for {name}: run plan first")
    return o


def raw_path(root, repo, path):
    return os.path.join(root, "raw", repo, path)


def download(url, dst, size=None, sha256=None):
    from experiments.data_mix.sources import download_url, with_retries
    tok = hf_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    t = time.time()
    with_retries(lambda: download_url(url, dst, size, sha256, headers), f"download {url}", attempts=10)
    n = os.path.getsize(dst)
    log(f"downloaded {dst}: {n / 1e6:.0f} MB in {time.time() - t:.0f} s")


def text_ratio(path):
    return 2.8 if path.endswith(".parquet") else 3.2


def files_needed(root, name, upto_tokens):
    """The prefix of the source's file order that should cover its share of `upto_tokens` pool tokens."""
    s, o = SOURCES[name], order(root, name)
    need = s["share"] * upto_tokens * 1.3 / s["yield_"]
    out, have = [], 0.0
    for f in o["files"]:
        if have >= need:
            break
        out.append(f)
        have += f["size"] * text_ratio(f["path"]) / s["bpt"]
    return o, out, have


def cmd_fetch(args):
    jobs = []
    for name in args.sources or ORDER:
        o, files, have = files_needed(args.root, name, args.upto_tokens)
        log(f"fetch {name}: {len(files)} of {len(o['files'])} files, {sum(f['size'] for f in files) / 1e9:.1f} GB "
            f"(about {have / 1e9:.2f}B tokens of documents)")
        for f in files:
            dst = raw_path(args.root, o["repo"], f["path"])
            if not (os.path.exists(dst) and os.path.getsize(dst) == f["size"]):
                jobs.append((hf_url(o["repo"], o["revision"], f["path"]), dst, f["size"], f["sha256"]))
    if "arxiv" in (args.sources or ORDER):
        jobs += arxiv_meta_jobs(args.root)
    log(f"fetch: {len(jobs)} files to download, {sum(j[2] or 0 for j in jobs) / 1e9:.1f} GB")
    with ThreadPoolExecutor(args.threads) as ex:
        for fut in [ex.submit(download, *j) for j in jobs]:
            fut.result()
    if "arxiv" in (args.sources or ORDER):
        build_arxiv_meta(args.root)
    log("fetch: done")


def arxiv_meta_jobs(root):
    repo, pats = ARXIV_META
    path = os.path.join(root, "manifest", "arxiv_meta_files.json")
    meta = read_json(path)
    if meta is None:
        from huggingface_hub import HfApi
        rev = HfApi().dataset_info(repo).sha
        meta = dict(repo=repo, revision=rev, files=listing(repo, rev, pats))
        write_json(path, meta)
    return [(hf_url(repo, meta["revision"], f["path"]), raw_path(root, repo, f["path"]), f["size"], f["sha256"])
            for f in meta["files"] if not (os.path.exists(raw_path(root, repo, f["path"]))
                                           and os.path.getsize(raw_path(root, repo, f["path"])) == f["size"])]


def arxiv_license(url):
    """SPDX-style name of an arXiv paper's license if it allows redistribution with attribution, else None."""
    u = (url or "").lower()
    m = re.search(r"creativecommons\.org/licenses/(by|by-sa)/(\d\.\d)", u)
    if m:
        return f"CC-{m.group(1).upper()}-{m.group(2)}"
    if "publicdomain/zero" in u:
        return "CC0-1.0"
    if "creativecommons.org/licenses/publicdomain" in u or "publicdomain/mark" in u:
        return "public domain"
    return None


def arxiv_id_key(aid):
    return (aid or "").strip().replace("/", "").lower()


def build_arxiv_meta(root):
    """manifest/arxiv_meta.parquet: id, title and license of every arXiv paper whose license allows redistribution."""
    out = os.path.join(root, "manifest", "arxiv_meta.parquet")
    if os.path.exists(out):
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    meta = read_json(os.path.join(root, "manifest", "arxiv_meta_files.json"))
    ids, titles, lics, counts = [], [], [], Counter()
    for f in meta["files"]:
        t = pq.read_table(raw_path(root, meta["repo"], f["path"]), columns=["id", "title", "license"])
        for aid, title, lic in zip(*(t.column(c).to_pylist() for c in ("id", "title", "license"))):
            name = arxiv_license(lic)
            counts[name or (lic or "none")] += 1
            if name and title:
                ids.append(arxiv_id_key(aid))
                titles.append(" ".join(title.split()))
                lics.append(name)
    pq.write_table(pa.table(dict(id=ids, title=titles, license=lics)), out + ".tmp", compression="zstd")
    os.replace(out + ".tmp", out)
    write_json(os.path.join(root, "manifest", "arxiv_licenses.json"),
               dict(revision=meta["revision"], kept=len(ids), counts=dict(counts.most_common())))
    log(f"arXiv metadata: {len(ids)} of {sum(counts.values())} papers have a redistributable license")


def bench_work(root):
    return os.path.join(root, "bench_work")


def cmd_benchmarks(args):
    """Export the decontamination benchmarks into <root>/bench_work/benchmarks (network: login node)."""
    for k, d in (("HF_HUB_CACHE", "hub"), ("HF_DATASETS_CACHE", "datasets")):      # home has a 100 GB cap
        os.environ.setdefault(k, os.path.join(args.root, "hf_cache", d))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")              # Xet's downloader was killed for memory on login
    from experiments.data_mix import config
    from experiments.data_mix.bench_extra import EXTRA, export as export_extra
    from exealm.data.benchmarks import BENCHMARKS
    out = os.path.join(bench_work(args.root), "benchmarks")
    os.makedirs(out, exist_ok=True)
    for name in config.DECONTAM:
        dst = os.path.join(out, name)
        if os.path.exists(os.path.join(dst, "text.jsonl.gz")):
            continue
        log(f"benchmark export {name}")
        if name in BENCHMARKS:
            subprocess.run([sys.executable, "-m", "exealm.data.benchmarks", "--benchmark", name, "--out", dst],
                           check=True)
        else:
            export_extra(name, dst)
    log("benchmarks: exported")


# ---------------------------------------------------------------------------------------------------------------------
# index (Slurm job): what must stay clean, and a sample of what was trained on

def window_hashes(tokens):
    from exealm.data.decontam import window_hashes as wh
    return wh(tokens, NGRAM)


def clean_windows(tokens, eot=0):
    """Hashes of the windows of `tokens` that do not contain an end-of-text token (they never cross documents)."""
    t = np.asarray(tokens)
    if len(t) < NGRAM:
        return np.empty(0, np.uint64)
    h = window_hashes(t)
    z = np.concatenate([[0], np.cumsum(t == eot, dtype=np.int64)])
    return h[(z[NGRAM:] - z[:-NGRAM]) == 0]


def pack_tokens(path):
    meta = read_json(os.path.join(path, "meta.json"))
    return np.memmap(os.path.join(path, "tokens.bin"), dtype=meta["dtype"], mode="r")


def _train_sample(job):
    path, a, b = job
    t = np.asarray(pack_tokens(path)[a:b])
    h = clean_windows(t)
    return np.unique(h[(h >> np.uint64(TRAIN_SAMPLE_SHIFT)) == 0])


def cmd_index(args):
    idx = os.path.join(args.root, "index")
    os.makedirs(idx, exist_ok=True)
    meta = read_json(os.path.join(idx, "meta.json"), {})
    packed = os.path.join(args.slice, "packed")
    buckets = sorted(os.listdir(packed))
    if not os.path.exists(os.path.join(idx, "clean_hash.npy")) or args.reindex:
        hs, ks, counts = [], [], {}
        for b in buckets:                                   # kind 1: the slice's val splits
            h = np.unique(clean_windows(pack_tokens(os.path.join(packed, b, "val"))))
            hs.append(h)
            ks.append(np.full(len(h), 1, np.int16))
            counts[f"val/{b}"] = int(len(h))
        for sub in ("qa/heldout", "tasks/heldout"):          # kind 2: library v1's held-out packs
            h = np.unique(clean_windows(pack_tokens(os.path.join(args.libv1, sub))))
            hs.append(h)
            ks.append(np.full(len(h), 2, np.int16))
            counts[f"libv1/{sub}"] = int(len(h))
        bench = bench_index(args)                             # kind 10 + i: benchmark i
        names = []
        if bench is not None:
            names = [str(x) for x in bench["names"]]
            hs.append(bench["hash"])
            ks.append((10 + bench["bench"]).astype(np.int16))
            for i, n in enumerate(names):
                counts[f"bench/{n}"] = int((bench["bench"] == i).sum())
        H = np.concatenate(hs)
        K = np.concatenate(ks)
        o = np.argsort(H, kind="stable")
        np.save(os.path.join(idx, "clean_hash.npy"), H[o])
        np.save(os.path.join(idx, "clean_kind.npy"), K[o])
        meta.update(clean_windows=counts, benchmarks=names, ngram=NGRAM, bench_report=read_json(
            os.path.join(bench_work(args.root), "reports", "benchmarks.json"), {}).get("failed"))
        write_json(os.path.join(idx, "meta.json"), meta)
        log(f"index: {len(H)} windows that must stay clean ({len(names)} benchmarks)")
    if not os.path.exists(os.path.join(idx, "train_s32.npy")) or args.reindex:
        jobs, chunk = [], 32_000_000
        for b in buckets:
            p = os.path.join(packed, b, "train")
            n = len(pack_tokens(p))
            jobs += [(p, a, min(n, a + chunk + NGRAM - 1)) for a in range(0, n, chunk)]
        log(f"index: sampling training windows from {len(jobs)} chunks")
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            parts = list(ex.map(_train_sample, jobs))
        S = np.unique(np.concatenate(parts))
        np.save(os.path.join(idx, "train_s32.npy"), S)
        meta.update(train_sample=dict(windows=int(len(S)), rate=f"1/{2 ** (64 - TRAIN_SAMPLE_SHIFT)}",
                                      buckets=buckets, seconds=round(time.time() - t0)))
        write_json(os.path.join(idx, "meta.json"), meta)
        log(f"index: {len(S)} sampled training windows in {time.time() - t0:.0f} s")


def bench_index(args):
    """The slice's benchmark index (experiments/data_mix tok.run_benchmarks), built from the exports, or None."""
    work = bench_work(args.root)
    if not os.path.isdir(os.path.join(work, "benchmarks")):
        log("index: NO benchmark exports (run the benchmarks step on the login node): benchmarks not checked")
        return None
    from experiments.data_mix import config
    from experiments.data_mix.common import Paths
    from experiments.data_mix.tok import run_benchmarks
    tok_dir = os.path.join(work, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    shutil.copy(os.path.join(args.slice, "tokenizer", "tokenizer.json"), os.path.join(tok_dir, "tokenizer.json"))
    cfg = config.load()
    run_benchmarks(cfg, Paths(work), argparse.Namespace(bench_limit=None))
    z = np.load(os.path.join(work, "benchmarks", "index.npz"))
    return {k: z[k] for k in ("hash", "bench", "names")}


# ---------------------------------------------------------------------------------------------------------------------
# reading documents: each reader turns one raw record into (title, body, info) or raises Skip

class Skip(Exception):
    pass


def first_line(text):
    title, _, body = text.partition("\n")
    return title.strip(), body


def clean_title(t):
    t = " ".join((t or "").split()).lstrip("#").strip()
    return t[:300]


def read_pes2o(rec, s, ctx):
    title, body = first_line(rec["text"])
    pid = str(rec.get("id"))
    return title, body, dict(doc_key=pid, url=f"https://api.semanticscholar.org/CorpusID:{pid}",
                             license=s["license"], subset=rec.get("source"))


def read_first_line(rec, s, ctx):
    title, body = first_line(rec["text"])
    md = rec.get("metadata") or {}
    return title, body, dict(doc_key=str(rec.get("id")), url=md.get("url"), license=s["license"])


def read_titled(rec, s, ctx):
    title, body = rec.get("title") or "", rec.get("text") or ""
    if title and body.startswith(title + "\n"):                 # a title line of its own (not the first sentence)
        body = body[len(title) + 1:]
    return title, body, dict(doc_key=str(rec.get("id")), url=rec.get("url"), license=s["license"])


def se_license(md):
    def norm(x):
        m = re.match(r"CC BY-SA (\d\.\d)", x or "")
        return f"CC-BY-SA-{m.group(1)}" if m else None
    lic = sorted({v for v in (norm(md.get("question_content_license")), norm(md.get("answer_content_license"))) if v})
    if not lic:
        raise Skip("stackexchange: no CC BY-SA license recorded")
    return "+".join(lic)


def read_stackexchange(rec, s, ctx):
    md = rec.get("metadata") or {}
    title, body = first_line(rec["text"])
    site = (md.get("forum") or "").replace("_", ".")
    url = f"https://{site}/questions/{md.get('question_id')}" if site and md.get("question_id") else None
    return title, body, dict(doc_key=str(rec.get("id")), url=url, license=se_license(md), site=site)


def read_arxiv(rec, s, ctx):
    d = rec.get("doc") if isinstance(rec.get("doc"), dict) else {}
    if doc_kind(rec) != "arxiv":
        raise Skip("not an arXiv paper")
    aid = d.get("arxiv_id")
    hit = ctx["arxiv"].get(arxiv_id_key(aid))
    if hit is None:
        raise Skip("arXiv license not CC BY/BY-SA/CC0/public domain")
    title, lic = hit
    body = rec["text"]
    m = BIB.search(body)
    if m:
        body = body[:m.start()]
    return title, body, dict(doc_key=aid, url=d.get("url") or f"https://arxiv.org/abs/{aid}", license=lic)


def read_web(rec, s, ctx):
    if "doc" in rec and doc_kind(rec) != "web":
        raise Skip("not an OpenWebMath page")
    url = (rec.get("doc") or {}).get("url") if isinstance(rec.get("doc"), dict) else rec.get("url")
    text = rec["text"]
    title, body = first_line(text)
    if len(title) > 200 or not title.strip("# "):
        title, body = re.sub(r"^https?://(www\.)?", "", url or "")[:200], text
    return title, body, dict(doc_key=str(rec.get("id")), url=url, license=s["license"])


def licenses_of(md):
    v = md.get("max_stars_repo_licenses")
    if v is None:
        return None
    if isinstance(v, str):
        v = re.findall(r"[A-Za-z0-9.\-+]+", v)
    return [str(x) for x in v]


def read_code(rec, s, ctx):
    md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    d = rec.get("doc") if isinstance(rec.get("doc"), dict) else {}
    if s.get("file_kind") and doc_kind(rec) != "code":
        raise Skip("not code")
    info = {**d, **{k: v for k, v in md.items() if v is not None}}
    text = rec["text"]
    m = CODE_HEADER.match(text)
    if m:
        hdr = m.group(0)
        text = text[m.end():]
        r = re.search(r"<reponame>([^<\n]*)", hdr)
        f = re.search(r"<filename>([^<\n]*)", hdr)
        info.setdefault("max_stars_repo_name", r.group(1) if r else None)
        info.setdefault("max_stars_repo_path", f.group(1) if f else None)
    repo = info.get("max_stars_repo_name") or info.get("repo")
    path = info.get("max_stars_repo_path") or info.get("path") or ""
    lic = licenses_of(info)
    if lic is not None:
        if not lic or any(x not in PERMISSIVE for x in lic):
            raise Skip("repository license not permissive")
        license_ = "+".join(lic)
    elif s.get("file_kind"):
        license_ = "not recorded (proof-pile-2 algebraic-stack; collection ODC-BY)"
    else:
        license_ = "not recorded (StarCoder: The Stack v1.2 permissive set; collection ODC-BY)"
    if s.get("sci"):
        from experiments.data_mix.extract import imported
        if not imported(text) & SCI_IMPORTS:
            raise Skip("Python not importing a scientific or ML library")
    title = f"{repo}/{path}" if repo else (path or "untitled")
    return title, text, dict(doc_key=str(info.get("hexsha") or rec.get("id")), license=license_,
                             url=f"https://github.com/{repo}" if repo else None,
                             code_lang=info.get("lang") or info.get("extension") or info.get("ext"))


READERS = dict(pes2o=read_pes2o, first_line=read_first_line, titled=read_titled, stackexchange=read_stackexchange,
               arxiv=read_arxiv, web=read_web, code=read_code)


# ---------------------------------------------------------------------------------------------------------------------
# passages

def units(body, bpt):
    """(start, end) spans of `body` no longer than UNIT estimated tokens: paragraphs, or for a longer paragraph its
    lines, sentences, or (last resort) cuts at a space."""
    limit = int(UNIT * bpt)
    bounds = [0] + [m.end() for m in PARA.finditer(body)] + [len(body)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out += split_span(body, a, b, limit)
    return [(a, b) for a, b in out if body[a:b].strip()]


def nbytes(s):
    return len(s.encode("utf-8"))


def split_span(text, a, b, limit, level=0):
    if nbytes(text[a:b]) <= limit:
        return [(a, b)]
    if level == 0:
        cuts = [m.end() + a for m in re.finditer(r"\n", text[a:b])]
    elif level == 1:
        cuts = [m.end() + a for m in SENT.finditer(text[a:b])]
    else:                                     # hard cut near every `limit` bytes, at a space when there is one
        out, i = [], a
        step = max(1, int(limit * (b - a) / max(1, nbytes(text[a:b]))))          # characters per `limit` bytes
        while i < b:
            j = min(b, i + step)
            if j < b:
                k = text.rfind(" ", i + step // 2, j)
                j = k + 1 if k > i else j
            out.append((i, j))
            i = j
        return out
    cuts = [c for c in cuts if a < c < b]
    if not cuts:
        return split_span(text, a, b, limit, level + 1)
    out, i = [], a
    for c in cuts + [b]:
        out += split_span(text, i, c, limit, level + 1) if nbytes(text[i:c]) > limit else [(i, c)]
        i = c
    # join neighbours back up to the limit, so a long paragraph is not cut into single lines
    merged = []
    for s, e in out:
        if merged and nbytes(text[merged[-1][0]:e]) <= limit:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def chunk(body, title, bpt, max_per_doc=MAX_PER_DOC):
    """Passage bodies of about TARGET estimated tokens (at most MAX), cut between units; the last one is merged into
    the one before when it is short and the two fit in MERGE_MAX."""
    head = nbytes(title) / bpt + 1
    us = units(body, bpt)
    sizes = [nbytes(body[a:b]) / bpt for a, b in us]
    spans, cur, size = [], None, 0.0
    for (a, b), n in zip(us, sizes):
        if cur is not None and head + size + n > MAX:
            spans.append((cur, size))
            cur, size = None, 0.0
        if cur is None:
            cur = (a, b)
        else:
            cur = (cur[0], b)
        size += n
        if head + size >= TARGET:
            spans.append((cur, size))
            cur, size = None, 0.0
    if cur is not None:
        if spans and size < TARGET / 2 and head + spans[-1][1] + size <= MERGE_MAX:
            (a0, _), n0 = spans.pop()
            spans.append(((a0, cur[1]), n0 + size))
        else:
            spans.append((cur, size))
    return [body[a:b].strip("\n").rstrip() for (a, b), _ in spans[:max_per_doc]]


def quality(text, kind):
    """Why a passage body is dropped, or None."""
    if any(x in text for x in SPECIAL):
        return "special-token string"
    chars = "".join(text.split())
    if len(chars) < 64:
        return "near-empty"
    if sum(c.isalpha() for c in chars) / len(chars) < ALPHA_MIN[kind]:
        return "mostly non-text"
    if BLOB.search(text):
        return "blob"
    lines = text.split("\n")
    if kind == "code" and max(len(x) for x in lines) > 1000:
        return "minified"
    long_lines = [x.strip() for x in lines if len(x.strip()) >= 20]
    if len(long_lines) >= 6 and len(set(long_lines)) < 0.5 * len(long_lines):
        return "repeated lines"
    return None


def norm_hash(text):
    return int.from_bytes(hashlib.blake2b(" ".join(text.split()).lower().encode("utf-8"), digest_size=8).digest(),
                          "little", signed=True)


_G = {}


def _init(tok_path, index_dir, name, root, train_threshold):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    from tokenizers import Tokenizer
    _G["tok"] = Tokenizer.from_file(tok_path)
    _G["src"] = SOURCES[name]
    _G["name"] = name
    _G["threshold"] = train_threshold
    _G["clean"] = np.load(os.path.join(index_dir, "clean_hash.npy"), mmap_mode="r")
    _G["kind"] = np.load(os.path.join(index_dir, "clean_kind.npy"), mmap_mode="r")
    _G["train"] = np.load(os.path.join(index_dir, "train_s32.npy"), mmap_mode="r")
    meta = read_json(os.path.join(index_dir, "meta.json"), {})
    _G["bench_names"] = meta.get("benchmarks") or []
    ctx = {}
    if SOURCES[name]["reader"] == "arxiv":
        import pyarrow.parquet as pq
        t = pq.read_table(os.path.join(root, "manifest", "arxiv_meta.parquet"))
        ctx["arxiv"] = dict(zip(t.column("id").to_pylist(), zip(t.column("title").to_pylist(),
                                                                 t.column("license").to_pylist())))
    _G["ctx"] = ctx


def member(sorted_arr, q):
    """For each query, the index in sorted_arr where it is found, or -1."""
    if not len(sorted_arr) or not len(q):
        return np.full(len(q), -1, np.int64)
    i = np.searchsorted(sorted_arr, q)
    i = np.minimum(i, len(sorted_arr) - 1)
    return np.where(np.asarray(sorted_arr[i]) == q, i, -1)


def kind_name(k):
    if k == 1:
        return "val"
    if k == 2:
        return "heldout"
    names = _G["bench_names"]
    return f"bench:{names[k - 10]}" if 0 <= k - 10 < len(names) else f"bench:{k - 10}"


def check(ids_list):
    """For each passage (token lists): the reason it must go (val, heldout, bench:<name>, train_overlap) or None,
    and the estimated share of its windows that are in the slice's training data."""
    n = len(ids_list)
    lens = np.array([len(x) for x in ids_list], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens + 1)[:-1]])
    flat = np.zeros(int(lens.sum() + n), dtype=np.uint16)                # each passage followed by end-of-text (0)
    for s, x in zip(starts, ids_list):
        flat[s:s + len(x)] = x
    reasons, overlap = [None] * n, np.zeros(n)
    if len(flat) < NGRAM:
        return reasons, overlap
    h = window_hashes(flat)
    z = np.concatenate([[0], np.cumsum(flat == 0, dtype=np.int64)])
    ok = np.flatnonzero((z[NGRAM:] - z[:-NGRAM]) == 0)
    hv = h[ok]
    pid = np.searchsorted(starts, ok, side="right") - 1
    hit = member(_G["clean"], hv)
    for w in np.flatnonzero(hit >= 0):
        p = pid[w]
        if reasons[p] is None:
            reasons[p] = kind_name(int(_G["kind"][hit[w]]))
    sampled = (hv >> np.uint64(TRAIN_SAMPLE_SHIFT)) == 0
    sp = pid[sampled]
    found = member(_G["train"], hv[sampled]) >= 0
    tot = np.bincount(sp, minlength=n)
    got = np.bincount(sp[found], minlength=n)
    overlap = np.where(tot > 0, got / np.maximum(tot, 1), 0.0)
    for p in range(n):
        if reasons[p] is None and tot[p] >= 3 and overlap[p] >= _G["threshold"]:
            reasons[p] = "train_overlap"
    return reasons, overlap


def process(batch):
    """One batch of raw records of one file -> kept passages and counts. batch: (rank, path, first_row, records)."""
    rank, path, row0, records = batch
    s, name, ctx = _G["src"], _G["name"], _G["ctx"]
    reader = READERS[s["reader"]]
    stats = Counter()
    cands = []                                  # (row, k, title, body, info)
    for i, raw in enumerate(records):
        stats["docs"] += 1
        try:
            rec = json.loads(raw) if isinstance(raw, (bytes, str)) else raw
            title, body, info = reader(rec, s, ctx)
        except Skip as e:
            stats[f"doc skipped: {e}"] += 1
            continue
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            stats[f"doc unreadable: {type(e).__name__}"] += 1
            continue
        title = clean_title(title)
        body = (body or "").replace("\r\n", "\n").replace("\r", "\n")
        if not title or not body.strip():
            stats["doc skipped: no title or empty"] += 1
            continue
        stats["docs used"] += 1
        for k, part in enumerate(chunk(body, title, s["bpt"])):
            why = quality(part, s["kind"])
            if why:
                stats[f"quality: {why}"] += 1
                continue
            cands.append((row0 + i, k, title, part, info))
    if not cands:
        return [], stats
    texts = [f"{t}\n{b}" for _, _, t, b, _ in cands]
    ids = [e.ids for e in _G["tok"].encode_batch(texts)]
    keep = [j for j, x in enumerate(ids) if len(x) >= MIN_TOKENS]
    stats["quality: short"] += len(ids) - len(keep)
    reasons, overlap = check([ids[j] for j in keep])
    out = []
    for j, why, ov in zip(keep, reasons, overlap):
        if why:
            stats[f"dropped: {why}"] += 1
            stats[f"dropped tokens: {why.split(':')[0]}"] += len(ids[j])
            continue
        row, k, title, body, info = cands[j]
        out.append(dict(ids=np.asarray(ids[j], dtype=np.uint16), text=texts[j], hash=norm_hash(body), row=row,
                        chunk=k, title=title, info=info, overlap=round(float(ov), 3)))
    return out, stats


def iter_records(path, start_row, columns=None, batch_bytes=4 << 20):
    """(first_row, [records]) batches of a .json.gz (raw lines) or .parquet (dicts) file, from start_row on."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        row = 0
        for rb in pf.iter_batches(batch_size=512, columns=columns):
            n = rb.num_rows
            if row + n <= start_row:
                row += n
                continue
            recs = rb.to_pylist()[max(0, start_row - row):]
            first = max(row, start_row)
            # split into batches of about batch_bytes of text
            buf, size, b0 = [], 0, first
            for k, r in enumerate(recs):
                buf.append(r)
                size += len(r.get("text") or "")
                if size >= batch_bytes:
                    yield b0, buf
                    b0, buf, size = first + k + 1, [], 0
            if buf:
                yield b0, buf
            row += n
        return
    unz = shutil.which("pigz") or "gzip"
    p = subprocess.Popen([unz, "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 22)
    try:
        buf, size, row, b0 = [], 0, 0, start_row
        for line in p.stdout:
            if row >= start_row:
                buf.append(line)
                size += len(line)
                if size >= batch_bytes:
                    yield b0, buf
                    b0, buf, size = row + 1, [], 0
            row += 1
        if buf:
            yield b0, buf
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def part_dir(root, part):
    return os.path.join(root, "work", f"part_{part:03d}")


def start_cursor(root, part, name):
    """Where this part reads `name` from: where the previous part stopped."""
    if part == 0:
        return dict(file=0, row=0)
    prev = read_json(os.path.join(part_dir(root, part - 1), name, "done.json"))
    if prev is None:
        raise SystemExit(f"part {part - 1} has not finished {name}: run it first")
    return prev["end"]


def seen_hashes(root, part):
    seen = set()
    for p in range(part):
        for name in ORDER:
            f = os.path.join(part_dir(root, p), name, "hashes.npy")
            if os.path.exists(f):
                seen.update(np.load(f).tolist())
    return seen


def run_source(args, name, target, seen, index_dir):
    s = SOURCES[name]
    o = order(args.root, name)
    out = os.path.join(part_dir(args.root, args.part), name)
    if os.path.exists(os.path.join(out, "done.json")):
        d = read_json(os.path.join(out, "done.json"))
        seen.update(np.load(os.path.join(out, "hashes.npy")).tolist())
        log(f"{name}: done already ({d['tokens'] / 1e6:.0f}M tokens)")
        return d
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    cur = start_cursor(args.root, args.part, name)
    tok_path = os.path.join(args.slice, "tokenizer", "tokenizer.json")
    stats, lens, hashes = Counter(), [], []
    tokens = passages = 0
    end = dict(cur)
    files_used = []
    t0 = time.time()
    f_tok = open(os.path.join(out, "tokens.bin"), "wb")
    f_idx = open(os.path.join(out, "index.jsonl"), "w", encoding="utf-8")
    f_txt = gzip.open(os.path.join(out, "passages.jsonl.gz"), "wt", encoding="utf-8", compresslevel=3)

    def batches():
        for rank in range(cur["file"], len(o["files"])):
            f = o["files"][rank]
            path = raw_path(args.root, o["repo"], f["path"])
            if not (os.path.exists(path) and os.path.getsize(path) == f["size"]):
                raise FileNotFoundError(f"{name}: file {rank} ({f['path']}) is not downloaded: run fetch with a "
                                        f"larger --upto_tokens, then run this part again")
            for row0, recs in iter_records(path, cur["row"] if rank == cur["file"] else 0, s.get("columns")):
                yield rank, path, row0, recs
            yield rank, path, None, None                    # end of file marker

    stopped = False
    missing = None
    last_log = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(tok_path, index_dir, name, args.root, args.train_threshold)) as ex:
        inflight = []
        gen = batches()
        exhausted = False

        def feed():
            nonlocal exhausted, missing
            while not exhausted and len(inflight) < 3 * args.workers:
                try:
                    b = next(gen)
                except StopIteration:
                    exhausted = True
                    return
                except FileNotFoundError as e:
                    exhausted, missing = True, str(e)
                    return
                rank, path, row0, recs = b
                if recs is None:
                    inflight.append((b, None))
                else:
                    inflight.append((b, ex.submit(process, b)))

        feed()
        while inflight:
            (rank, path, row0, recs), fut = inflight.pop(0)
            if fut is None:                                  # a file finished: the cursor moves to the next one
                end = dict(file=rank + 1, row=0)
                feed()
                continue
            res, st = fut.result()
            stats.update(st)
            rel = os.path.relpath(path, os.path.join(args.root, "raw", o["repo"]))
            if not files_used or files_used[-1]["path"] != rel:
                files_used.append(dict(path=rel, rank=rank, first_row=row0, docs=0))
            files_used[-1]["docs"] += len(recs)
            for p in res:
                if p["hash"] in seen:
                    stats["dropped: duplicate"] += 1
                    continue
                seen.add(p["hash"])
                ids = p["ids"]
                f_tok.write(ids.tobytes())
                f_tok.write(b"\x00\x00")
                lens.append(len(ids) + 1)
                hashes.append(p["hash"])
                pid = f"p{args.part:03d}.{name}.{passages:07d}"
                info = p["info"]
                rec = dict(id=pid, source=name, group=s["group"], file=f"{o['repo']}/{rel}", src_doc=p["row"],
                           doc_key=info.get("doc_key"), chunk=p["chunk"], title=p["title"], license=info["license"],
                           url=info.get("url"), lang=s["lang"], n_tokens=len(ids) + 1,
                           n_bytes=nbytes(p["text"]), train_overlap=p["overlap"])
                for k in ("site", "subset", "code_lang"):
                    if info.get(k):
                        rec[k] = info[k]
                f_idx.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_txt.write(json.dumps(dict(id=pid, source=name, title=p["title"], license=info["license"],
                                            url=info.get("url"), text=p["text"]), ensure_ascii=False) + "\n")
                passages += 1
                tokens += len(ids) + 1
            end = dict(file=rank, row=row0 + len(recs))
            if tokens >= target:
                stopped = True
                for _, fu in inflight:
                    if fu is not None:
                        fu.cancel()
                inflight.clear()
                break
            feed()
            if time.time() - last_log > 60:
                last_log = time.time()
                log(f"{name}: {tokens / 1e6:.0f}M of {target / 1e6:.0f}M tokens, {passages} passages, "
                    f"{tokens / (time.time() - t0) / 1e6:.2f}M tokens/s")
        gen.close()
    f_tok.close()
    f_idx.close()
    f_txt.close()
    if missing and not stopped:
        log(f"{name}: STOPPED SHORT at {tokens / 1e6:.0f}M of {target / 1e6:.0f}M tokens: {missing}")
        if not args.allow_short:
            raise SystemExit(missing)
    np.save(os.path.join(out, "lengths.npy"), np.array(lens, dtype=np.int64))
    np.save(os.path.join(out, "hashes.npy"), np.array(hashes, dtype=np.int64))
    done = dict(source=name, part=args.part, target=int(target), tokens=int(tokens), passages=passages,
                start=cur, end=end, short=not stopped, files=files_used, repo=o["repo"], revision=o["revision"],
                stats=dict(sorted(stats.items())), seconds=round(time.time() - t0))
    write_json(os.path.join(out, "done.json"), done)
    log(f"{name}: {tokens / 1e6:.0f}M tokens in {passages} passages ({time.time() - t0:.0f} s); "
        f"dropped {', '.join(f'{k[9:]} {v}' for k, v in sorted(stats.items()) if k.startswith('dropped: '))}")
    return done


def cmd_passages(args):
    index_dir = args.index_dir or os.path.join(args.root, "index")
    for f in ("clean_hash.npy", "clean_kind.npy", "train_s32.npy"):
        if not os.path.exists(os.path.join(index_dir, f)):
            raise SystemExit(f"no {f} in {index_dir}: run index first")
    seen = seen_hashes(args.root, args.part)
    names = args.sources or ORDER
    for name in names:
        target = SOURCES[name]["share"] * args.part_tokens
        run_source(args, name, target, seen, index_dir)


def cmd_pack(args):
    """Concatenate the part's sources (in ORDER) into one TokenShards pack with index.jsonl and passages.jsonl.gz."""
    work = part_dir(args.root, args.part)
    out = args.out or os.path.join(args.root, "human_pool", f"part_{args.part:03d}")
    dones = {n: read_json(os.path.join(work, n, "done.json")) for n in ORDER}
    missing = [n for n, d in dones.items() if d is None]
    if missing and not args.allow_short:
        raise SystemExit(f"sources not done in part {args.part}: {missing}")
    names = [n for n in ORDER if dones[n] is not None]
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    offsets, n_tok, n_bytes, doc = [], 0, 0, 0
    per = {}
    with open(os.path.join(tmp, "tokens.bin"), "wb") as ft, open(os.path.join(tmp, "index.jsonl"), "w",
                                                                  encoding="utf-8") as fi, \
            open(os.path.join(tmp, "passages.jsonl.gz"), "wb") as fx:
        for n in names:
            lens = np.load(os.path.join(work, n, "lengths.npy"))
            offsets.append(n_tok + np.concatenate([[0], np.cumsum(lens)[:-1]]) if len(lens) else np.empty(0, np.int64))
            with open(os.path.join(work, n, "tokens.bin"), "rb") as f:
                shutil.copyfileobj(f, ft, 1 << 24)
            src_bytes, licenses = 0, Counter()
            with open(os.path.join(work, n, "index.jsonl"), encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    src_bytes += rec["n_bytes"]
                    licenses[rec["license"]] += 1
                    fi.write(json.dumps(dict(doc=doc, **rec), ensure_ascii=False) + "\n")
                    doc += 1
            with open(os.path.join(work, n, "passages.jsonl.gz"), "rb") as f:     # gzip members concatenate
                shutil.copyfileobj(f, fx, 1 << 24)
            tok = int(lens.sum())
            if tok != dones[n]["tokens"] or os.path.getsize(os.path.join(work, n, "tokens.bin")) != 2 * tok:
                raise SystemExit(f"{n}: lengths add up to {tok}, done.json says {dones[n]['tokens']}")
            n_tok += tok
            n_bytes += src_bytes
            d = dones[n]
            per[n] = dict(group=SOURCES[n]["group"], passages=d["passages"], tokens=d["tokens"], target=d["target"],
                          short=d["short"], bytes=src_bytes, repo=d["repo"], revision=d["revision"],
                          start=d["start"], end=d["end"], files=[f["path"] for f in d["files"]],
                          licenses=dict(licenses.most_common()), stats=d["stats"])
    offsets = np.concatenate(offsets).astype(np.int64)
    np.save(os.path.join(tmp, "offsets.npy"), offsets)
    tok_file = os.path.join(args.slice, "tokenizer", "tokenizer.json")
    index_meta = read_json(os.path.join(args.index_dir or os.path.join(args.root, "index"), "meta.json"), {})
    meta = dict(dtype="uint16", vocab_size=32768, n_tokens=int(n_tok), n_docs=int(len(offsets)),
                bytes_per_token=n_bytes / max(1, n_tok), tokenizer=os.path.join(SLICE, "tokenizer"),
                tokenizer_sha256=file_sha256(tok_file), eot_id=0,
                source=dict(kind="human tier", built_by="experiments/library/human_tier.py", part=args.part,
                            packed=time.strftime("%Y-%m-%d"), title_line=True,
                            passage_tokens=dict(target=TARGET, max=MAX, merge_max=MERGE_MAX, min=MIN_TOKENS,
                                                max_per_doc=MAX_PER_DOC),
                            decontamination=dict(ngram=NGRAM, clean_windows=index_meta.get("clean_windows"),
                                                 benchmarks=index_meta.get("benchmarks"),
                                                 train_overlap_threshold=args.train_threshold,
                                                 train_sample=index_meta.get("train_sample")),
                            sources={n: {k: v for k, v in p.items() if k not in ("stats", "files")}
                                     for n, p in per.items()}))
    write_json(os.path.join(tmp, "meta.json"), meta)
    write_json(os.path.join(tmp, "stats.json"), per)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.replace(tmp, out)
    log(f"pack: {out}: {len(offsets)} passages, {n_tok / 1e9:.3f}B tokens")
    if args.part == 0 and not args.out:
        link = os.path.join(args.root, "human_pool_v0")
        if not os.path.lexists(link):
            os.symlink(os.path.join("human_pool", "part_000"), link)
    return meta


def cmd_run(args):
    cmd_index(args)
    cmd_passages(args)
    cmd_pack(args)


def cmd_stats(args):
    rows, tot = [], Counter()
    for d in sorted(os.listdir(os.path.join(args.root, "human_pool"))):
        st = read_json(os.path.join(args.root, "human_pool", d, "stats.json"))
        if not st:
            continue
        for n, p in st.items():
            drops = {k[9:]: v for k, v in p["stats"].items() if k.startswith("dropped: ")}
            rows.append((d, n, p["tokens"], p["passages"], drops))
            tot[n] += p["tokens"]
    for r in rows:
        print(f"{r[0]}  {r[1]:16s} {r[2] / 1e6:8.1f}M tokens {r[3]:9d} passages  dropped {r[4]}")
    print("total", {k: f"{v / 1e6:.0f}M" for k, v in tot.items()}, f"{sum(tot.values()) / 1e9:.3f}B")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["plan", "fetch", "benchmarks", "index", "passages", "pack", "run", "stats"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--slice", default=SLICE)
    ap.add_argument("--libv1", default=LIBV1)
    ap.add_argument("--sources", nargs="*", choices=ORDER)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--part_tokens", type=float, default=2e9)
    ap.add_argument("--upto_tokens", type=float, default=2e9, help="fetch: files for this many pool tokens in all")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--index_dir", default=None, help="a copy of <root>/index (e.g. on node-local disk)")
    ap.add_argument("--train_threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replan", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--allow_short", action="store_true", help="finish a part with a source short of its share")
    args = ap.parse_args(argv)
    dict(plan=cmd_plan, fetch=cmd_fetch, benchmarks=cmd_benchmarks, index=cmd_index, passages=cmd_passages,
         pack=cmd_pack, run=cmd_run, stats=cmd_stats)[args.step](args)


if __name__ == "__main__":
    main()
