"""Which benchmark questions share text with a training set: for each question of SciQ, ARC-Easy and ARC-Challenge
(and any other packed benchmark named), whether any 13-token window of its question, choices, answer or support
passage appears anywhere in the given packed training data.

bench_check.py asks the other way round (which library entries to keep out of the search). This one names the
questions, so a benchmark can be scored again with them left out: the library post-training data (data/library_v2/
v2/rag2/train: every training question and task with its BM25 entry and the human tier's top passage before it)
was never checked against the benchmarks, and post-training raises closed-book SciQ by 0.13.

    python experiments/library/bench_questions.py data/library_v2/v2/rag2/train \\
        --bench /expanse/lustre/projects/fli100/abhatia7/library_v2/bench_work/benchmarks \\
        --out results/expanse/bench_questions_v2.json
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.decontam import PackedDocuments, window_hashes  # noqa: E402

NGRAM = 13


def bench_windows(bench_dir, name):
    """hash -> set of (benchmark_id, field) for every 13-token window of every segment of one packed benchmark, and
    each benchmark_id's question text."""
    rows = [json.loads(line) for line in gzip.open(os.path.join(bench_dir, name, "text.jsonl.gz"), "rt",
                                                     encoding="utf-8")]
    docs = PackedDocuments(os.path.join(bench_dir, name + "-packed"))
    assert len(docs.offsets) == len(rows), (name, len(docs.offsets), len(rows))
    ends = np.append(docs.offsets[1:], len(docs.tokens))
    by_hash, question = defaultdict(set), {}
    for i, r in enumerate(rows):
        if r["field"] == "question":
            question[r["benchmark_id"]] = r["text"]
        for h in window_hashes(np.asarray(docs.tokens[int(docs.offsets[i]):int(ends[i])]), NGRAM).tolist():
            by_hash[h].add((r["benchmark_id"], r["field"]))
    return by_hash, question, len({r["benchmark_id"] for r in rows})


def scan(train, hashes, block=1 << 26):
    """The benchmark hashes that some window inside one training document matches."""
    docs = PackedDocuments(train)
    starts = np.asarray(docs.offsets, dtype=np.int64)
    n = len(docs.tokens)
    found = set()
    t0 = time.time()
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block + NGRAM - 1)
        h = window_hashes(np.asarray(docs.tokens[b0:b1]), NGRAM)
        pos = np.arange(b0, b0 + len(h), dtype=np.int64)
        nxt = np.searchsorted(starts, pos, side="right")             # the next document's start after each window's
        ok = (nxt >= len(starts)) | (starts[np.minimum(nxt, len(starts) - 1)] >= pos + NGRAM)
        hit = np.isin(h[ok], hashes)
        found.update(h[ok][hit].tolist())
        print(f"  {b1 / n:.0%} of {n:,} tokens, {len(found)} matching windows ({time.time() - t0:.0f}s)", flush=True)
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("train", nargs="+", help="packed training directories (tokens.bin, offsets.npy, meta.json)")
    ap.add_argument("--bench", required=True, help="bench_work/benchmarks: <name>/text.jsonl.gz and <name>-packed")
    ap.add_argument("--names", default="sciq,arc_easy,arc_challenge")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    names = a.names.split(",")
    table, questions, totals = {}, {}, {}
    for name in names:
        by_hash, q, n = bench_windows(a.bench, name)
        table[name], questions[name], totals[name] = by_hash, q, n
        print(f"{name}: {n} questions, {len(by_hash):,} distinct windows", flush=True)
    allh = np.unique(np.fromiter((h for t in table.values() for h in t), dtype=np.uint64))
    found = set()
    for t in a.train:
        print(f"scanning {t}", flush=True)
        found |= scan(t, allh)
    res = dict(train=a.train, ngram=NGRAM, rule="a question is flagged when any 13-token window of any of its fields "
               "(question, choices, answer, support passage) appears inside one training document", benchmarks={})
    for name in names:
        hit = defaultdict(set)
        for h in found:
            for bid, field in table[name].get(h, ()):
                hit[bid].add(field)
        res["benchmarks"][name] = dict(
            questions=totals[name], flagged=len(hit),
            flagged_by_field={f: sum(f in v for v in hit.values()) for f in sorted({f for v in hit.values() for f in v})},
            ids=sorted(hit, key=lambda x: int(x) if str(x).isdigit() else x),
            fields={k: sorted(v) for k, v in hit.items()},
            question_text={k: questions[name].get(k, "") for k in hit})
        print(f"{name}: {len(hit)} of {totals[name]} questions share a window "
              f"{res['benchmarks'][name]['flagged_by_field']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
