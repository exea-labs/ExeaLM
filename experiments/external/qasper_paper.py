"""QASPER with the paper in hand, as a research assistant has it: a keyword search (BM25 over words) inside the paper
for the question, and its top paragraphs before the question. QASPER's own setting gives the whole paper; this gives
the parts a search picks, so every model reads the same text in a short context.

Writes items in indep_build.py's format (set "qasper_paper"): "bm25" holds the paragraphs the in-paper search picks
(the top 3, at most 2,500 characters, best first), "pasted" the annotator's evidence (an upper bound). The best paragraph comes last, next to the question: our models read
a long input as 512-token windows and keep little from two or more windows back (results/expanse/README.md, 26-27). Score our
checkpoints with indep_build.py --texts and released models with score_hf.py --texts.

    python experiments/external/qasper_paper.py --out /expanse/.../external/indep_qasper_paper.jsonl
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from indep_build import qasper_items  # noqa: E402


def words(s):
    return re.findall(r"\w+", s.lower())


def bm25_top(query, paras, k=3, k1=1.2, b=0.75):
    docs = [words(p) for p in paras]
    n, avg = len(docs), sum(map(len, docs)) / max(1, len(docs))
    df = Counter(w for d in docs for w in set(d))
    q = set(words(query))
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1) /
                (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in q if w in tf)
        scores.append((s, i))
    return [paras[i] for s, i in sorted(scores, reverse=True)[:k] if s > 0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--chars", type=int, default=2500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = qasper_items()
    hit = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            paras = [p for p in it.pop("source").split("\n")[1:] if len(p) > 40]     # abstract and paragraphs
            q = it["prefix"].split("\n")[1][len("Question: "):]
            top = bm25_top(q, paras, a.k)
            kept, used = [], 0
            for p in top:                                       # best first, within the budget
                p = p[:max(0, a.chars - used)]
                if not p:
                    break
                kept.append(p)
                used += len(p) + 2
            text = "".join(p + "\n\n" for p in reversed(kept))  # best last
            ev = it.get("pasted") or ""
            hit += bool(ev) and any(e[:200] in text for e in ev.split("\n") if e)
            it.update(set="qasper_paper", type="qasper_paper", bm25=text, kind="item", sep="",
                      target_bytes=len(it["target"].encode("utf-8")), bits={}, meta={})
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items; the search's paragraphs hold an evidence paragraph for {hit / len(items):.3f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
