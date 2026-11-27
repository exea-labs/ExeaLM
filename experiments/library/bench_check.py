"""Which library entries share text with a benchmark: every 13-token window of every entry against the benchmark
index the human tier was checked with (experiments/library/human_tier.py: the slice's 14 decontamination benchmarks,
bench_work/benchmarks/index.npz). An entry that shares a window with a benchmark question or answer could hand the
model the answer in an open-book evaluation, so it goes before any benchmark is run with the library.

    python experiments/library/bench_check.py data/library/v1 \\
        --bench /expanse/lustre/projects/fli100/abhatia7/library_v2/bench_work/benchmarks/index.npz \\
        --out data/library/v1/bench_hits.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.decontam import window_hashes  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402

NGRAM = 13


def entry_hits(lib, H, B, names, chunk=20000, eot=0):
    """{entry number: benchmark name} for every entry with a 13-token window in the benchmark index."""
    order = np.argsort(H)
    Hs, Bs = H[order], B[order]
    starts = np.asarray(lib.offsets, dtype=np.int64)
    ends = np.append(starts[1:], len(lib.tokens))
    hits = {}
    for d0 in range(0, len(starts), chunk):
        d1 = min(len(starts), d0 + chunk)
        flat = np.asarray(lib.tokens[int(starts[d0]):int(ends[d1 - 1])], dtype=np.uint16)
        if len(flat) < NGRAM:
            continue
        h = window_hashes(flat, NGRAM)
        z = np.concatenate([[0], np.cumsum(flat == eot, dtype=np.int64)])
        ok = np.flatnonzero((z[NGRAM:] - z[:-NGRAM]) == 0)               # windows inside one entry
        hv = h[ok]
        i = np.searchsorted(Hs, hv)
        i[i >= len(Hs)] = 0
        found = Hs[i] == hv
        doc = np.searchsorted(starts, ok + int(starts[d0]), side="right") - 1
        for w in np.flatnonzero(found):
            hits.setdefault(int(doc[w]), names[int(Bs[i[w]])] if int(Bs[i[w]]) < len(names) else str(int(Bs[i[w]])))
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("library", help="a packed library (TokenShards)")
    ap.add_argument("--bench", required=True, help="bench_work/benchmarks/index.npz (hash, bench, names)")
    ap.add_argument("--out", default=None, help="a JSON file: the entries to drop and the benchmark each shares")
    a = ap.parse_args(argv)
    z = np.load(a.bench, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    lib = TokenShards(a.library)
    hits = entry_hits(lib, np.asarray(z["hash"], dtype=np.uint64), np.asarray(z["bench"]), names)
    by = Counter(hits.values())
    print(f"{a.library}: {len(hits):,} of {len(lib.offsets):,} entries share a 13-token window with a benchmark"
          + (": " + ", ".join(f"{k} {v}" for k, v in by.most_common()) if by else ""), flush=True)
    if a.out:
        json.dump(dict(library=a.library, n_entries=len(lib.offsets), drop={str(k): v for k, v in sorted(hits.items())},
                       by_benchmark=dict(by)), open(a.out, "w"), indent=1)
    return hits


if __name__ == "__main__":
    main()
