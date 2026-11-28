"""Paired benchmark comparisons from per-question results: our runs closed and open book (openbook.sh with SAMPLES=1)
and released models (experiments/external/public_bench.sb, lm-eval's samples files).

For every pair asked for, on SciQ, ARC-Easy and ARC-Challenge and on the three pooled: each side's accuracy (normalized
where the task has it, as in results/expanse/README.md section 25), the questions only one side gets, and an exact
McNemar test. With --flags (bench_questions.py), every comparison is also given without the questions that share a
13-token window with the library post-training data.

    python experiments/library/openbook_paired.py --ours results/expanse/openbook_samples \\
        --public results/expanse/public_bench --flags results/expanse/bench_questions_v2.json \\
        --out results/expanse/openbook_paired.json
"""
import argparse
import glob
import json
import math
import os

TASKS = ("sciq", "arc_easy", "arc_challenge")


def load_ours(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["task"] in TASKS:
            out[(r["task"], int(r["doc_id"]))] = (r.get("acc_norm", r.get("acc")), r.get("question", ""))
    return out


def load_public(folder):
    out = {}
    for t in TASKS:
        files = sorted(glob.glob(os.path.join(folder, "**", f"samples_{t}_*.jsonl"), recursive=True))
        if not files:
            continue
        for line in open(files[-1], encoding="utf-8"):
            r = json.loads(line)
            out[(t, int(r["doc_id"]))] = (r.get("acc_norm", r.get("acc")), r.get("doc", {}).get("question", ""))
    return out


def mcnemar(b, c):
    """Exact two-sided McNemar p: b questions only the first side gets, c only the second."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(A, B, keep=None):
    rows = {}
    for t in TASKS + ("pooled",):
        keys = [k for k in A if k in B and (t == "pooled" or k[0] == t) and (keep is None or k in keep)]
        if not keys:
            continue
        a = [A[k][0] for k in keys]
        b = [B[k][0] for k in keys]
        only_a = sum(x > 0.5 and y < 0.5 for x, y in zip(a, b))
        only_b = sum(x < 0.5 and y > 0.5 for x, y in zip(a, b))
        rows[t] = dict(n=len(keys), a=round(sum(a) / len(keys), 4), b=round(sum(b) / len(keys), 4),
                       diff=round((sum(a) - sum(b)) / len(keys), 4), only_a=only_a, only_b=only_b,
                       p=float("%.3g" % mcnemar(only_a, only_b)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ours", required=True, help="folder of <run>_<closed|open>.jsonl")
    ap.add_argument("--public", default=None, help="folder of <model>/ lm-eval output folders")
    ap.add_argument("--flags", default=None, help="bench_questions.py output")
    ap.add_argument("--pairs", default=None, help="'A:B,...' of side names; default: the comparisons below")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sides = {}
    for f in sorted(glob.glob(os.path.join(a.ours, "*.jsonl"))):
        sides[os.path.basename(f)[:-6]] = load_ours(f)
    if a.public and os.path.isdir(a.public):
        for d in sorted(os.listdir(a.public)):
            got = load_public(os.path.join(a.public, d))
            if got:
                sides["public:" + d] = got
    print("sides:", {k: len(v) for k, v in sides.items()})
    flagged, mismatch = set(), 0
    if a.flags:
        fl = json.load(open(a.flags))["benchmarks"]
        for t in TASKS:
            for i in fl.get(t, {}).get("ids", []):
                flagged.add((t, int(i)))
        ref = next(iter(sides.values()))                      # the doc ids are row numbers of the same test splits
        for t in TASKS:
            for i, q in fl.get(t, {}).get("question_text", {}).items():
                got = ref.get((t, int(i)), (None, None))[1]
                if got is not None and " ".join(q.split()).lower() != " ".join(str(got).split()).lower():
                    mismatch += 1
        print(f"flagged questions: {len(flagged)}; text mismatches against the doc ids: {mismatch}")
    if a.pairs:
        pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    else:
        E, Q = "ragpost_exealm_500m_v2", "ragpost_qwen_500m_v2"
        e, q = "exealm_500m_4.1x_s0", "qwen_500m_4.1x_s0"
        pairs = [(f"{E}_open", f"{E}_closed"), (f"{Q}_open", f"{Q}_closed"), (f"{e}_open", f"{e}_closed"),
                 (f"{q}_open", f"{q}_closed"), (f"{E}_closed", f"{e}_closed"), (f"{Q}_closed", f"{q}_closed"),
                 (f"{E}_open", f"{Q}_open"), (f"{E}_closed", f"{Q}_closed"), (f"{e}_closed", f"{q}_closed")]
        pairs += [(f"{E}_open", s) for s in sides if s.startswith("public:")]
        pairs += [(f"{E}_closed", s) for s in sides if s.startswith("public:")]
    res = dict(metric="acc_norm (SciQ, ARC)", flagged=len(flagged), text_mismatches=mismatch, sides={}, pairs=[])
    for name, s in sides.items():
        res["sides"][name] = {t: round(sum(v[0] for k, v in s.items() if k[0] == t) /
                                       max(1, sum(k[0] == t for k in s)), 4) for t in TASKS}
    for A, B in pairs:
        if A not in sides or B not in sides:
            continue
        rec = dict(a=A, b=B, all=compare(sides[A], sides[B]))
        if flagged:
            keep = {k for k in sides[A] if k not in flagged}
            rec["unflagged"] = compare(sides[A], sides[B], keep)
        res["pairs"].append(rec)
        p = rec["all"].get("pooled", {})
        u = rec.get("unflagged", {}).get("pooled", {})
        print(f"{A} vs {B}: pooled {p.get('a')} vs {p.get('b')} ({p.get('diff'):+.4f}, {p.get('only_a')}/"
              f"{p.get('only_b')}, p={p.get('p')})" + (f"; unflagged {u.get('diff'):+.4f} p={u.get('p')}" if u else ""))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
