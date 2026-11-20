"""Near-duplicate removal across every source with MinHash LSH, on a CPU, with numpy only.

DATA.md's rule is to dedupe across every source. quality.py drops exact copies; this drops documents that are mostly
the same text (mirrors, boilerplate-heavy reposts, the same article in two corpora). It matters twice for ExeaLM:
duplicates waste tokens, and near-duplicates let the RG circuit look up the text it is reading in another document
(docs/memory.md), which the datastore's same-document exclusion cannot catch.

How it works (the FineWeb setup: 5-grams, 112 hashes in 14 bands of 8):
  1. Shingles: word 5-grams of the lowercased text; Japanese has no spaces, so Japanese documents use character
     5-grams instead (whitespace removed). Each shingle is hashed with crc32.
  2. MinHash: 112 hash functions h(x) = (a x + b) mod p; a document's signature is each function's minimum over its
     shingles. Two signatures agree in a given position with probability equal to the documents' Jaccard similarity.
  3. LSH: the signature is cut into 14 bands of 8. Documents that agree on a whole band land in the same bucket and
     become candidates. A pair with Jaccard similarity s is caught with probability 1 - (1 - s^8)^14: about 0.05 at
     s = 0.5, 0.65 at s = 0.72 and 0.92 at s = 0.8.
  4. Verify (default on): a candidate counts as a duplicate only if its signature agrees with the earlier document's
     on at least --threshold of the 112 positions, which removes most false matches from step 3.

The first copy seen is kept, so list sources in order of preference (the best-filtered source first). The output has
the same files and fields as the input, minus the duplicates, and a report of which sources the duplicates matched.

    python -m exealm.data.dedup data/filtered/*.jsonl.gz --out data/dedup

Scale: the index keeps 14 band keys per document (and the 112-number signature when verifying) in memory, which is
fine for tens of millions of documents. The full 100B-token mix needs it sharded by band, or datatrove's MinHash
pipeline, which uses the same parameters.
"""
import argparse
import gzip
import json
import os
import re
import zlib
from collections import Counter

import numpy as np

from .files import expand
from .quality import is_japanese

P = np.uint64(4294967311)                 # a prime just above 2^32, so (a x + b) mod P stays inside uint64
WORD = re.compile(r"\w+", re.UNICODE)


def shingles(text, n=5):
    """Set of crc32 hashes of the text's n-grams: words, or characters for Japanese."""
    if is_japanese(text):
        s = "".join(text.split())
        grams = [s[i:i + n] for i in range(max(1, len(s) - n + 1))]
    else:
        w = WORD.findall(text.lower())
        grams = [" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))]
    return {zlib.crc32(g.encode("utf-8")) for g in grams if g}


class MinHasher:
    def __init__(self, num_perm=112, seed=0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2 ** 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 2 ** 32, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, hashes):
        """MinHash signature (num_perm,) uint32 of a set of 32-bit shingle hashes."""
        if not hashes:
            return np.full(self.num_perm, 0xFFFFFFFF, dtype=np.uint32)
        x = np.fromiter(hashes, dtype=np.uint64, count=len(hashes))
        return ((np.outer(x, self.a) + self.b) % P).min(0).astype(np.uint32)


class NearDupIndex:
    """Streaming LSH index: add() each document's signature in order; it returns the id of an earlier near-duplicate,
    or None (and then remembers the document)."""

    def __init__(self, num_perm=112, bands=14, threshold=0.7, verify=True):
        if num_perm % bands:
            raise ValueError("num_perm must be a multiple of bands")
        self.bands, self.rows, self.threshold, self.verify = bands, num_perm // bands, threshold, verify
        self.buckets = [dict() for _ in range(bands)]
        self.sigs = {}

    def add(self, doc_id, sig):
        keys = [sig[i * self.rows:(i + 1) * self.rows].tobytes() for i in range(self.bands)]
        seen = set()
        for band, k in zip(self.buckets, keys):
            other = band.get(k)
            if other is None or other in seen:
                continue
            seen.add(other)
            if not self.verify or float((self.sigs[other] == sig).mean()) >= self.threshold:
                return other
        for band, k in zip(self.buckets, keys):
            band.setdefault(k, doc_id)
        if self.verify:
            self.sigs[doc_id] = sig
        return None


def dedup_files(files, out, num_perm=112, bands=14, threshold=0.7, verify=True, ngram=5, seed=0):
    """Write every file's documents that are not near-duplicates of an earlier document (in this or an earlier file)
    to out/<same name>. Returns a report: per file, documents read and kept, and which files its duplicates matched."""
    os.makedirs(out, exist_ok=True)
    mh = MinHasher(num_perm, seed)
    index = NearDupIndex(num_perm, bands, threshold, verify)
    source_of = []                                                       # doc id -> file index
    report = {}
    for fi, path in enumerate(files):
        name = os.path.basename(path)
        dst = os.path.join(out, name)
        if os.path.abspath(dst) == os.path.abspath(path):
            raise ValueError(f"--out would overwrite the input {path}")
        read = kept = 0
        matched = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as fin, gzip.open(dst, "wt", encoding="utf-8") as fo:
            for line in fin:
                rec = json.loads(line)
                read += 1
                doc_id = len(source_of)
                source_of.append(fi)
                other = index.add(doc_id, mh.signature(shingles(rec.get("text", ""), ngram)))
                if other is None:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    matched[os.path.basename(files[source_of[other]])] += 1
        report[name] = dict(read=read, kept=kept, dropped=read - kept, duplicates_of=dict(matched))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help=".jsonl.gz files with a 'text' field, best source first")
    ap.add_argument("--out", default="data/dedup")
    ap.add_argument("--num_perm", type=int, default=112)
    ap.add_argument("--bands", type=int, default=14)
    ap.add_argument("--threshold", type=float, default=0.7, help="signature agreement needed to call a duplicate")
    ap.add_argument("--no_verify", action="store_true", help="drop every LSH candidate (faster, more false drops)")
    ap.add_argument("--ngram", type=int, default=5)
    a = ap.parse_args(argv)
    a.files = expand(a.files)
    report = dedup_files(a.files, a.out, a.num_perm, a.bands, a.threshold, not a.no_verify, a.ngram)
    for name, r in report.items():
        print(f"{name}: kept {r['kept']} of {r['read']}", json.dumps(r["duplicates_of"], ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
