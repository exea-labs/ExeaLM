"""Prepare library v2's written tier for publication (exea-labs/exealm-library, v2/written).

Reads the writer's raw output (data/library_v2: entries.jsonl, code_entries.jsonl, tasks.jsonl), the packed library's
index (data/library_v2/v2/index.jsonl: each entry's split) and its benchmark check (v2/bench_hits.json), and writes:

    entries.jsonl       research entries with their question/answer pairs: topic, origin (where the topic came from),
                        lang, split, entry, qa, model, prompt_version
    code_entries.jsonl  code entries, the same fields
    tasks.jsonl         adaptation tasks, with split
    stats.json          counts, what was left out and why, and a check that entries do not copy their sources

Left out: every entry that shares a 13-token window with one of the 14 checked benchmarks (bench_hits.json), and every
task and question of those topics. The topic's source excerpt (topics.jsonl) is not published: it is text from arXiv,
papers, Wikipedia, KAKEN and Stack Exchange, some of it under licences that do not allow redistribution.

    python experiments/library/publish_written_v2.py --out /expanse/lustre/projects/fli100/abhatia7/library_v2/public_written
"""
import argparse
import json
import os
import random
import re
from collections import Counter

MODEL = "DeepSeek-V4-Flash"


def words(text, lang):
    """Word units for the copy check: whitespace words, or characters for Japanese."""
    if lang == "ja":
        return [c for c in text if not c.isspace()]
    return re.findall(r"\w+", text.lower())


def shared_ngrams(a, b, n):
    sb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    hits = [i for i in range(len(a) - n + 1) if tuple(a[i:i + n]) in sb]
    covered = set()
    for i in hits:
        covered.update(range(i, i + n))
    return len(hits), len(covered) / max(1, len(a))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default="data/library_v2")
    ap.add_argument("--pack", default="data/library_v2/v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy_sample", type=int, default=3000)
    ap.add_argument("--max_copied", type=float, default=0.25,
                    help="leave out a research entry when more than this share of it is in 13-word windows shared "
                         "with its topic's source excerpt (20 characters for Japanese)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    split, drop = {}, set()
    by_doc = {}
    with open(os.path.join(a.pack, "index.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["topic"], r["kind"])
            split[key] = r["split"]
            by_doc[r["doc"]] = key
    hits = json.load(open(os.path.join(a.pack, "bench_hits.json")))["drop"]
    drop_bench = Counter()
    for d, bench in hits.items():
        drop.add(by_doc[int(d)])
        drop_bench[bench] += 1
    drop_topics = {t for t, _ in drop}
    origin, source = {}, {}
    with open(os.path.join(a.raw, "topics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            origin[r["topic"]] = r.get("pool_source")
            source[r["topic"]] = r.get("source", "")

    stats = dict(model=MODEL, left_out_for_benchmark_overlap=dict(entries=len(drop), by_benchmark=dict(drop_bench)))
    rng = random.Random(0)
    copy_rows = []
    copied = Counter()
    for name, kind in (("entries.jsonl", "text"), ("code_entries.jsonl", "code")):
        kept = Counter()
        with open(os.path.join(a.raw, name), encoding="utf-8") as f, \
                open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as g:
            for line in f:
                r = json.loads(line)
                key = (r["topic"], kind)
                if key in drop:
                    kept["dropped"] += 1
                    continue
                sp = split.get(key)
                if sp is None:
                    kept["not_packed"] += 1
                    continue
                if kind == "text" and source.get(r["topic"]):
                    lang = r.get("lang", "en")
                    _, cov = shared_ngrams(words(r["entry"], lang), words(source[r["topic"]], lang),
                                           20 if lang == "ja" else 13)
                    if cov > a.max_copied:
                        kept["dropped_copied"] += 1
                        copied[origin.get(r["topic"])] += 1
                        drop_topics.add(r["topic"])          # its tasks go too
                        continue
                out = dict(topic=r["topic"], origin=origin.get(r["topic"]) if kind == "text" else "code topic list",
                           lang=r.get("lang"), split=sp, entry=r["entry"], qa=r.get("qa", []), model=MODEL,
                           prompt_version=r.get("prompt_version"))
                g.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept[sp] += 1
                if kind == "text" and source.get(r["topic"]) and rng.random() < a.copy_sample / 145000:
                    lang = r.get("lang", "en")
                    n = 20 if lang == "ja" else 13
                    k, cov = shared_ngrams(words(r["entry"], lang), words(source[r["topic"]], lang), n)
                    copy_rows.append(dict(origin=origin.get(r["topic"]), lang=lang, windows=k, covered=cov))
        stats[name] = dict(kept)
    kept = Counter()
    with open(os.path.join(a.raw, "tasks.jsonl"), encoding="utf-8") as f, \
            open(os.path.join(a.out, "tasks.jsonl"), "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            r = json.loads(line)
            key = (r["topic"], r.get("kind", "text"))
            if r["topic"] in drop_topics:
                kept["dropped"] += 1
                continue
            sp = split.get(key)
            if sp is None:
                kept["not_packed"] += 1
                continue
            r["split"] = sp
            r["model"] = MODEL
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept[sp] += 1
    stats["tasks.jsonl"] = dict(kept)
    stats["left_out_as_copied"] = dict(threshold=a.max_copied, by_origin=dict(copied))
    n = len(copy_rows)
    stats["copy_check"] = dict(
        sample=n, of_kept_entries=True, rule="13-word windows (20 characters for Japanese) an entry shares with its topic's source excerpt",
        entries_with_any_shared_window=round(sum(r["windows"] > 0 for r in copy_rows) / max(1, n), 4),
        mean_share_of_entry_covered=round(sum(r["covered"] for r in copy_rows) / max(1, n), 4),
        entries_over_10_percent_covered=round(sum(r["covered"] > 0.1 for r in copy_rows) / max(1, n), 4),
        by_origin={o: round(sum(r["covered"] for r in copy_rows if r["origin"] == o) /
                            max(1, sum(r["origin"] == o for r in copy_rows)), 4)
                   for o in sorted({r["origin"] for r in copy_rows if r["origin"]})})
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
