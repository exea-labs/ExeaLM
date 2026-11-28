"""Summarize the library tests on questions we did not write (indep_build.py, qasper_paper.py, qasper_lib2.py, and
score_hf.py --rows_out for released models): bits per byte of the reference answers for every model and condition, on
the items every model was scored on, with a 95% bootstrap interval (items resampled, 2,000 draws) for the comparisons
asked for.

    python experiments/external/indep_summary.py /expanse/.../external/indep_*.jsonl --out results/expanse/indep_tests.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rows", nargs="+", help="files of item rows with bits[<model>][<condition>]")
    ap.add_argument("--compare", default="ragpost_exealm_500m_v2:bm25/qwen35_9b_base:none,"
                    "ragpost_exealm_3b_v2:bm25/qwen35_9b_base:none,"
                    "ragpost_exealm_500m_v2:pasted/qwen35_9b_base:none,"
                    "ragpost_exealm_3b_v2:pasted/qwen35_9b_base:none,"
                    "ragpost_exealm_3b_v2:bm25/qwen35_9b_base:bm25,"
                    "ragpost_exealm_500m_v2:bm25/ragpost_qwen_500m_v2:bm25",
                    help="'model:cond/model:cond,...': first over second, per set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows_out", default=None, help="also write every item's bits and search notes, without the "
                    "texts (the benchmarks' text stays out of the repository)")
    a = ap.parse_args(argv)
    items = {}
    for path in a.rows:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            key = (r["set"], r["id"])
            it = items.setdefault(key, dict(set=r["set"], id=r["id"], bytes=r["target_bytes"], bits={},
                                            meta=r.get("meta", {})))
            for m, row in r.get("bits", {}).items():
                it["bits"].setdefault(m, {}).update(row)
    by_set = defaultdict(list)
    for it in items.values():
        by_set[it["set"]].append(it)
    rng = np.random.default_rng(0)
    res = dict(sets={}, compare={})
    for s, its in sorted(by_set.items()):
        cells = sorted({(m, c) for it in its for m, row in it["bits"].items() for c in row})
        table = {}
        for m, c in cells:
            have = [it for it in its if c in it["bits"].get(m, {})]
            table.setdefault(m, {})[c] = dict(n=len(have), bpb=round(sum(it["bits"][m][c] for it in have) /
                                                                     sum(it["bytes"] for it in have), 4))
        res["sets"][s] = table
        print(f"== {s} ({len(its)} items)")
        for m, row in table.items():
            print(f"   {m:28s} " + "  ".join(f"{c} {v['bpb']:.4f} (n {v['n']})" for c, v in sorted(row.items())))
        for spec in filter(None, a.compare.split(",")):
            (ma, ca), (mb, cb) = (x.split(":") for x in spec.split("/"))
            both = [it for it in its if ca in it["bits"].get(ma, {}) and cb in it["bits"].get(mb, {})]
            if not both:
                continue
            A = np.array([it["bits"][ma][ca] for it in both])
            B = np.array([it["bits"][mb][cb] for it in both])
            n = np.array([it["bytes"] for it in both], dtype=float)
            ratio = A.sum() / B.sum()
            draws = rng.integers(0, len(both), size=(2000, len(both)))
            rs = A[draws].sum(1) / B[draws].sum(1)
            lo, hi = np.percentile(rs, [2.5, 97.5])
            rec = dict(n=len(both), a=round(A.sum() / n.sum(), 4), b=round(B.sum() / n.sum(), 4),
                       ratio=round(float(ratio), 4), ci95=[round(float(lo), 4), round(float(hi), 4)],
                       items_a_better=int((A < B).sum()))
            res["compare"][f"{s}: {spec}"] = rec
            print(f"   {spec}: {rec['a']} against {rec['b']}, ratio {rec['ratio']} [{rec['ci95'][0]}, "
                  f"{rec['ci95'][1]}], first better on {rec['items_a_better']} of {rec['n']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.rows_out:
        with open(a.rows_out, "w", encoding="utf-8") as f:
            for it in items.values():
                f.write(json.dumps(dict(set=it["set"], id=it["id"], target_bytes=it["bytes"], meta=it["meta"],
                                        bits={m: {c: round(v, 3) for c, v in row.items()}
                                              for m, row in it["bits"].items()})) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
