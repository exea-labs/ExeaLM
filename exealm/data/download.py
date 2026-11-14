"""Download a sample of the training data to work with locally.

LLM-jp Corpus v4 (no account needed; files are 100 MB to 3 GB of jsonl.gz each):
    python -m exealm.data.download llmjp --subset ja/ja_wiki --max_files 1 --out data/raw
    python -m exealm.data.download llmjp --subset code/code_stack --max_mb 100 --out data/raw   # first 100 MB of text
    python -m exealm.data.download llmjp --list ja/ja_cc            # just list the files

Any Hugging Face dataset in DATA.md, for example K2 Horizon (needs `pip install -e ".[data]"`; streams, so no full
download):
    python -m exealm.data.download hf --repo IFM/TxT360-v2 --config web-high-medium --max_rows 20000 --out data/raw

Everything ends up as jsonl.gz files with one {"text": ..., "source": ...} object per line, which tokenizer.py and
pack.py read. The full 100B-token download is a job for the training cluster, not a laptop.
"""
import argparse
import gzip
import json
import os
import urllib.request

from .sources import K2, LLMJP, LLMJP_API, LLMJP_RAW


def llmjp_files(path):
    """Files under one LLM-jp sub-corpus directory, through GitLab's API (recursive, paged)."""
    out, page = [], 1
    while True:
        url = f"{LLMJP_API}?path={path}&recursive=true&per_page=100&page={page}"
        with urllib.request.urlopen(url, timeout=60) as r:
            items = json.load(r)
        if not items:
            return sorted(out)
        out += [it["path"] for it in items if it["type"] == "blob" and it["path"].endswith(".jsonl.gz")]
        page += 1


def download_llmjp(subset, max_files, out, max_mb=None, skip=0):
    """Whole files, or with max_mb a sample: stream the first file and keep complete documents up to max_mb MB of
    text (the rest of the file is never downloaded)."""
    files = llmjp_files(subset)[skip:skip + max_files]
    os.makedirs(out, exist_ok=True)
    for f in files:
        dst = os.path.join(out, "llmjp__" + f.replace("/", "__"))
        if max_mb:
            dst = dst.replace(".jsonl.gz", f".first{int(max_mb)}MB.jsonl.gz")
        if os.path.exists(dst):
            print("have", dst)
            continue
        print("get", f)
        if max_mb:
            n = 0
            with urllib.request.urlopen(LLMJP_RAW + f, timeout=120) as r, gzip.GzipFile(fileobj=r) as gz, \
                    gzip.open(dst + ".part", "wt", encoding="utf-8") as fo:
                for line in gz:
                    text = json.loads(line)["text"]
                    fo.write(json.dumps({"text": text, "source": "llm-jp-v4/" + subset}, ensure_ascii=False) + "\n")
                    n += len(text.encode("utf-8"))
                    if n >= max_mb * 1e6:
                        break
        else:
            urllib.request.urlretrieve(LLMJP_RAW + f, dst + ".part")
        os.replace(dst + ".part", dst)
        if max_mb:
            return


def extract_text(rec):
    """One training document from a record whose fields we do not know in advance (K2 Horizon subsets differ).
    Known K2 subsets use the verified formatters in formatters.py instead of this heuristic."""
    if isinstance(rec.get("text"), str):
        return rec["text"]
    for key in ("messages", "conversations", "conversation"):
        if isinstance(rec.get(key), list):
            return "\n\n".join(str(m.get("content", m.get("value", ""))) if isinstance(m, dict) else str(m)
                               for m in rec[key])
    pairs = [("question", "answer"), ("prompt", "response"), ("problem", "solution"), ("instruction", "output")]
    for a, b in pairs:
        if a in rec and b in rec:
            return f"{rec[a]}\n\n{rec[b]}"
    return "\n\n".join(v for v in rec.values() if isinstance(v, str) and len(v) > 20)


def download_hf(repo, config, max_rows, out):
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    from .formatters import get_k2_formatter
    formatter = get_k2_formatter(repo, config) if repo in K2 else extract_text
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train", streaming=True)
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, f"hf__{repo.replace('/', '__')}__{config}.jsonl.gz")
    n = 0
    with gzip.open(dst, "wt", encoding="utf-8") as f:
        for rec in ds:
            text = formatter(rec)
            if text:
                f.write(json.dumps({"text": text, "source": f"{repo}/{config}"}, ensure_ascii=False) + "\n")
                n += 1
            if n >= max_rows:
                break
    print(f"wrote {n} documents to {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["llmjp", "hf"])
    ap.add_argument("--subset", default="ja/ja_wiki", help=f"LLM-jp path, one of: {', '.join(LLMJP)}")
    ap.add_argument("--list", default=None, help="list the files of an LLM-jp path and exit")
    ap.add_argument("--max_files", type=int, default=1)
    ap.add_argument("--max_mb", type=float, default=None, help="LLM-jp: stream a sample of this many MB of text")
    ap.add_argument("--skip_files", type=int, default=0, help="LLM-jp: start at this file (e.g. 1 for a held-out file)")
    ap.add_argument("--repo", default="IFM/TxT360-v2")
    ap.add_argument("--config", default="web-high-medium")
    ap.add_argument("--max_rows", type=int, default=20000)
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    if a.list:
        print("\n".join(llmjp_files(a.list)))
    elif a.source == "llmjp":
        download_llmjp(a.subset, a.max_files, a.out, a.max_mb, a.skip_files)
    else:
        download_hf(a.repo, a.config, a.max_rows, a.out)


if __name__ == "__main__":
    main()
