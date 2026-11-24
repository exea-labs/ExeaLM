"""Build the token packs the 13M experiments read, from public text, when the private packs are not available.

Reads <raw>/{en,ja_wiki,ja_web,code}.jsonl.gz and writes
<out>/{train_en,train_ja,train_code,val_en,val_ja_wiki,val_ja_web,val_code}.
The last --val_mb megabytes of each held-out source become its validation split, so no document is in both.

--fetch downloads the raw files that are missing (`pip install datasets`): FineWeb-Edu sample-10BT (English),
Wikipedia 20231101.ja and FineWeb-2 jpn_Jpan (Japanese), codeparrot-clean (Python code). If --tokenizer does not
exist, it also downloads the project's 32k tokenizer from https://huggingface.co/exea-labs/ExeaLM (`pip install
huggingface_hub`). These are stand-ins for the private packs behind results/funding, so compare runs built from
them with each other, not with those numbers.

python experiments/leave_one_out_data.py --fetch --raw data/raw --out data/tokpack/bpe32k
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exealm.data.pack import pack  # noqa: E402


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def split(docs, val_mb):
    n, size = len(docs), 0
    while n > 0 and size < val_mb * 1e6:
        n -= 1
        size += len(docs[n].encode("utf-8"))
    return docs[:n], docs[n:]


def write(docs, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")


SOURCES = {"en": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 200),
           "ja_wiki": ("wikimedia/wikipedia", "20231101.ja", "text", 80),
           "ja_web": ("HuggingFaceFW/fineweb-2", "jpn_Jpan", "text", 80),
           "code": ("codeparrot/codeparrot-clean", None, "content", 60)}


def fetch(name, path):
    from datasets import load_dataset
    repo, cfg, field, mb = SOURCES[name]
    ds = load_dataset(repo, cfg, split="train", streaming=True) if cfg else load_dataset(repo, split="train", streaming=True)
    n = 0
    with gzip.open(path + ".part", "wt", encoding="utf-8") as f:
        for rec in ds:
            f.write(json.dumps({"text": rec[field]}, ensure_ascii=False) + "\n")
            n += len(rec[field].encode("utf-8"))
            if n >= mb * 1e6:
                break
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--fetch", action="store_true", help="download the four sources that are not in --raw yet")
    ap.add_argument("--tokenizer", default="results/tokenizer/bpe32k")
    ap.add_argument("--out", default="data/tokpack/bpe32k")
    ap.add_argument("--val_mb", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(a.raw, exist_ok=True)
    raw = {k: os.path.join(a.raw, k + ".jsonl.gz") for k in SOURCES}
    tokenizer = a.tokenizer
    if a.fetch:
        for k, path in raw.items():
            if not os.path.exists(path):
                fetch(k, path)
        if not os.path.exists(tokenizer):
            from exealm.hub import TOKENIZER, _download
            tokenizer = _download(TOKENIZER)
    tmp = os.path.join(a.out, "_split")
    os.makedirs(tmp, exist_ok=True)
    en_tr, en_va = split(read(raw["en"]), a.val_mb)
    wiki_tr, wiki_va = split(read(raw["ja_wiki"]), a.val_mb)
    web_tr, web_va = split(read(raw["ja_web"]), a.val_mb)
    code_tr, code_va = split(read(raw["code"]), a.val_mb)
    parts = dict(train_en=en_tr, train_ja=wiki_tr + web_tr, train_code=code_tr,
                 val_en=en_va, val_ja_wiki=wiki_va, val_ja_web=web_va, val_code=code_va)
    for name, docs in parts.items():
        f = os.path.join(tmp, name + ".jsonl.gz")
        write(docs, f)
        pack([f], tokenizer, os.path.join(a.out, name))


if __name__ == "__main__":
    main()
