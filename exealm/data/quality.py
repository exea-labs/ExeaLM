"""Picking the best 100B tokens (DATA.md): filters, deduplication and quality scores.

What is here works but is simple: rule-based filters (in the spirit of the Gopher and FineWeb rules) with separate
rules for Japanese, and exact deduplication by hash. Near-duplicates across sources are removed in the next step,
exealm/data/dedup.py (MinHash LSH). What should replace or extend this file:

  TODO(data): a learned quality score, like FineWeb-Edu's educational-value classifier, trained for Japanese too.
  TODO(data): decontamination: drop documents that contain our evaluation benchmarks' questions.
  TODO(data): measure each filter by training two small models (with and without it) on the same token budget.

python -m exealm.data.quality data/raw/*.jsonl.gz --out data/filtered
"""
import argparse
import gzip
import hashlib
import json
import os
import re

from .files import expand

KANA = re.compile(r"[぀-ヿ]")
CJK = re.compile(r"[぀-ヿ一-鿿]")


def is_japanese(text):
    """Japanese if a noticeable share of characters are kana (Chinese text has kanji but no kana)."""
    sample = text[:2000]
    return len(KANA.findall(sample)) > 0.05 * max(1, len(sample))


def check(text):
    """Return a list of reasons to drop the document (empty: keep)."""
    why = []
    n = len(text)
    if n < 200:
        why.append("too short")
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.7:
        why.append("repeated lines")
    if n and sum(c.isalnum() or CJK.match(c) is not None for c in text) / n < 0.6:
        why.append("too many symbols")
    if is_japanese(text):
        if len(CJK.findall(text)) / max(1, n) < 0.3:
            why.append("too little Japanese text")
    else:
        words = text.split()
        if len(words) < 50:
            why.append("too few words")
        elif not 3 <= sum(map(len, words)) / len(words) <= 10:
            why.append("odd word length")
        if lines and sum(l.rstrip().endswith("...") for l in lines) / len(lines) > 0.3:
            why.append("truncated lines")
    return why


def doc_hash(text):
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="data/filtered")
    a = ap.parse_args()
    a.files = expand(a.files)
    os.makedirs(a.out, exist_ok=True)
    seen, stats = set(), {}
    for path in a.files:
        dst = os.path.join(a.out, os.path.basename(path))
        kept = 0
        with gzip.open(path, "rt", encoding="utf-8") as fi, gzip.open(dst, "wt", encoding="utf-8") as fo:
            for line in fi:
                rec = json.loads(line)
                text = rec.get("text", "")
                why = check(text)
                h = doc_hash(text)
                if h in seen:
                    why.append("exact duplicate")
                seen.add(h)
                for w in why:
                    stats[w] = stats.get(w, 0) + 1
                if not why:
                    fo.write(json.dumps({"text": text, "source": rec.get("source", path)}, ensure_ascii=False) + "\n")
                    kept += 1
        print(f"{path}: kept {kept}")
    print("dropped:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
