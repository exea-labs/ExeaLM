"""The library's keyword search in Tantivy (a Rust search engine, `pip install tantivy`): the same BM25 over the same
token ids as experiments/library/index.py, for fast search on CPUs.

index.py adds up every posting of every query token, so its time grows with the library (about 350 ms a query on CPU
over 36.8M passages). A production engine skips most postings: Tantivy keeps compressed posting blocks with the best
score in each, and its top-k search (block-max WAND) passes over blocks that cannot reach the current top k. Each
passage is indexed as its token ids written as words ("t17 t9035 ..."), so the search does not depend on the language
and ranks as index.py does (BM25, k1 1.2, b 0.75; Tantivy stores passage lengths in one byte, a small difference).
No positions and no stored text: the passages stay in their TokenShards pack, found by number.

    python experiments/library/tantivy_index.py build /path/human_pool/part_003 --out /path/tantivy/part_003
    python experiments/library/tantivy_index.py bench /path/tantivy/part_003 --compare /path/bm25_pool_c/part_003 \\
        --questions data/library/v1
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.loader import TokenShards  # noqa: E402


def words(ids):
    return " ".join(f"t{int(t)}" for t in ids)


def schema():
    import tantivy
    b = tantivy.SchemaBuilder()
    b.add_text_field("t", stored=False, tokenizer_name="whitespace", index_option="freq")
    b.add_unsigned_field("id", stored=True, indexed=False, fast=True)
    return b.build()


def cmd_build(a):
    import tantivy
    lib = TokenShards(a.library)
    os.makedirs(a.out, exist_ok=True)
    index = tantivy.Index(schema(), path=a.out)                  # build on a local disk: Lustre made it ~50x slower
    writer = index.writer(heap_size=a.heap_mb * 1024 * 1024, num_threads=a.threads)
    n = len(lib.offsets)
    ends = np.append(np.asarray(lib.offsets[1:]), len(lib.tokens))
    t0 = time.time()
    for d in range(n):
        a0, e = int(lib.offsets[d]), int(ends[d]) - 1                  # without the final <|endoftext|>
        doc = tantivy.Document()
        doc.add_text("t", words(lib.tokens[a0:e]))
        doc.add_unsigned("id", d + a.base)                  # explicitly unsigned: a Python int alone becomes i64
        writer.add_document(doc)
        if d and d % 500_000 == 0:
            print(f"{d:,} of {n:,} passages, {time.time() - t0:.0f}s", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"{a.library}: {n:,} passages indexed in {time.time() - t0:.0f}s, {size / 1e9:.2f} GB -> {a.out}", flush=True)


def query_for(index, ids, skip):
    import tantivy
    terms = sorted({int(t) for t in ids} - set(skip))
    return tantivy.Query.boolean_query(
        [(tantivy.Occur.Should, tantivy.Query.term_query(index.schema, "t", f"t{t}")) for t in terms])


def cmd_bench(a):
    import tantivy
    from score import load_items
    index = tantivy.Index.open(a.index)
    index.reload()
    searcher = index.searcher()
    items, eot, sep = load_items(a.questions, ["qa"], "heldout", a.limit)
    skip = (eot, *sep)
    other = None
    if a.compare:
        from index import Index
        other = Index(a.compare)
    times, agree = [], 0
    for it in items:
        q = query_for(index, it["prefix"], skip)
        t0 = time.perf_counter()
        hits = searcher.search(q, a.k).hits
        times.append(time.perf_counter() - t0)
        top = [searcher.doc(addr)["id"][0] for _, addr in hits]
        if other is not None and top:
            d, _ = other.search(it["prefix"], k=1, max_df=1.0, skip=skip)
            agree += int(len(d) and int(d[0]) == top[0])
    t = np.array(times) * 1000
    print(f"{a.index}: {searcher.num_docs:,} passages, {len(items):,} queries, top {a.k}; ms per query: median "
          f"{np.median(t):.2f}, p95 {np.percentile(t, 95):.2f}"
          + (f"; same top passage as index.py for {agree / len(items):.3f}" if other is not None else ""), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("library", help="a TokenShards pack (one human pool part, or a packed library)")
    b.add_argument("--out", required=True)
    b.add_argument("--base", type=int, default=0, help="the first passage number, as index.py build --base")
    b.add_argument("--threads", type=int, default=8)
    b.add_argument("--heap_mb", type=int, default=4000)
    q = sub.add_parser("bench")
    q.add_argument("index")
    q.add_argument("--questions", required=True, help="a packed library whose qa/heldout questions are the queries")
    q.add_argument("--compare", default=None, help="an index.py index of the same passages: how often the top agrees")
    q.add_argument("--limit", type=int, default=1000)
    q.add_argument("--k", type=int, default=5)
    a = ap.parse_args(argv)
    {"build": cmd_build, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
