"""Topics for the library's written entries, taken from the human tier's own documents (experiments/library/human_tier.py):
one topic per document, its title, with the document's first passage as the source that says what the topic is.

The written tier and the human tier then cover the same ground: a search can find the frontier model's expert entry on
a topic and the human text it came from. Quotas set how many topics each source gives (research and reference sources;
code topics come from build.py's own mining of code). Within a source, documents are drawn at random with --seed;
machine-learning arXiv papers come first when the index records categories.

    python experiments/library/pool_topics.py /path/human_pool/part_* --out data/library_v2/topics.jsonl \\
        --quota arxiv:50000,pes2o:30000,wikipedia_en:35000,wikipedia_ja:10000,kaken:10000,stackexchange:15000
"""
import argparse
import gzip
import json
import os
import random

ML = ("cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.NE", "stat.ML", "cs.IR", "cs.RO")


def good_title(t):
    t = " ".join((t or "").split())
    return t if 2 <= len(t) <= 160 else None


def is_ml(r):
    cats = r.get("categories") or r.get("category") or ""
    cats = " ".join(cats) if isinstance(cats, list) else str(cats)
    return any(c in cats for c in ML)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parts", nargs="+", help="human pool parts (each with index.jsonl and passages.jsonl.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quota", required=True, help="source:count,... (sources as index.jsonl names them)")
    ap.add_argument("--source_chars", type=int, default=4000, help="the source text kept for each topic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    quota = {k: int(v) for k, v in (p.split(":") for p in a.quota.split(","))}
    parts = [p for p in a.parts if os.path.exists(os.path.join(p, "index.jsonl"))
             and os.path.exists(os.path.join(p, "passages.jsonl.gz"))]

    # Pass 1: every document's first passage, by source (part, line), ML arXiv flagged; titles deduplicated.
    cand = {s: ([], []) for s in quota}                     # (first choice, the rest)
    seen = set()
    for pi, p in enumerate(parts):
        with open(os.path.join(p, "index.jsonl"), encoding="utf-8") as f:
            for li, line in enumerate(f):
                r = json.loads(line)
                s = r.get("source")
                if s not in quota or r.get("chunk", 0) != 0:
                    continue
                t = good_title(r.get("title"))
                if not t or t.lower() in seen:
                    continue
                seen.add(t.lower())
                cand[s][0 if (s == "arxiv" and is_ml(r)) else 1].append((pi, li, t))
    rng = random.Random(a.seed)
    chosen = {}
    for s, (first, rest) in cand.items():
        rng.shuffle(first)
        rng.shuffle(rest)
        for pi, li, t in (first + rest)[:quota[s]]:
            chosen.setdefault(pi, {})[li] = (s, t)

    # Pass 2: the chosen passages' text.
    rows = []
    for pi, want in chosen.items():
        with gzip.open(os.path.join(parts[pi], "passages.jsonl.gz"), "rt", encoding="utf-8") as f:
            for li, line in enumerate(f):
                if li in want:
                    s, t = want[li]
                    text = json.loads(line)["text"]
                    rows.append(dict(topic=t, source=text[:a.source_chars], pool_source=s))
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    got = {s: sum(1 for r in rows if r["pool_source"] == s) for s in quota}
    print(f"{len(rows)} topics -> {a.out}: " + ", ".join(f"{s} {n:,} of {quota[s]:,}" for s, n in got.items()))


if __name__ == "__main__":
    main()
