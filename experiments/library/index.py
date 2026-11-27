"""A fast keyword search over a library of any size: BM25 over token ids, as an inverted index.

score.bm25_pick multiplies a sparse (entries x vocabulary) matrix by the queries, which touches every entry of the
library for every batch: fine for v1's 18,545 entries, too slow for millions of passages. Here every token keeps the
list of the passages it appears in, with its BM25 weight there, sorted by token; a query adds up only the lists of its
own tokens and takes the top k. Tokens in more than --max_df of all passages (the, of, a newline) carry almost no
weight and have the longest lists, so a query can skip them; with max_df 1 the ranking is score.bm25_pick's exactly.

The index is three arrays saved as .npy (read with mmap, so a large index opens at once): ptr (vocabulary + 1, where
each token's list starts), doc (int32 passage numbers) and w (float32 weights), plus df and meta.json. A library too
big for one index is indexed in parts (one per pack) and searched together: Index.search merges each part's top k.

    python experiments/library/index.py build data/library/v1 --out data/library/v1/bm25
    python experiments/library/index.py bench data/library/v1/bm25 --library data/library/v1     # speed and hit rate
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.loader import TokenShards  # noqa: E402


def build(lib, k1=1.2, b=0.75, chunk_docs=200_000, eot=0):
    """(ptr, doc, w, df) for one packed library (TokenShards). A document's tokens are its own, without the
    <|endoftext|> that ends it, as score.entry_ids reads them."""
    n = len(lib.offsets)
    V = int(lib.meta.get("vocab_size") or int(np.max(lib.tokens)) + 1)
    starts = np.asarray(lib.offsets, dtype=np.int64)
    ends = np.append(starts[1:], len(lib.tokens)) - 1                   # drop each document's final <|endoftext|>
    docs, toks, tfs = [], [], []
    lens = np.zeros(n, dtype=np.float64)
    for d0 in range(0, n, chunk_docs):
        d1 = min(n, d0 + chunk_docs)
        a, e = int(starts[d0]), int(ends[d1 - 1])
        t = np.asarray(lib.tokens[a:e + 1], dtype=np.int64)
        length = ends[d0:d1] - starts[d0:d1]
        owner = np.repeat(np.arange(d0, d1, dtype=np.int64), length + 1)        # +1: the final <|endoftext|>
        keep = np.ones(len(t), dtype=bool)
        keep[np.cumsum(length + 1) - 1] = False                                 # ... which is not the document's
        key = owner[keep] * V + t[keep]
        u, c = np.unique(key, return_counts=True)
        docs.append((u // V).astype(np.int32))
        toks.append((u % V).astype(np.int32))
        tfs.append(c.astype(np.float32))
        lens[d0:d1] = length
    doc, tok, tf = np.concatenate(docs), np.concatenate(toks), np.concatenate(tfs)
    del docs, toks, tfs
    df = np.bincount(tok, minlength=V).astype(np.int64)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = (k1 * (1 - b + b * lens / max(lens.mean(), 1e-9))).astype(np.float32)
    w = idf[tok] * tf * (k1 + 1) / (tf + norm[doc])
    order = np.argsort(tok, kind="stable")
    ptr = np.zeros(V + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(df)
    return ptr, doc[order], w[order].astype(np.float32), df


class Part:
    """One index part: the passages of one pack, numbered from `base` in the whole library."""

    def __init__(self, folder, mmap=True):
        mode = "r" if mmap else None
        self.ptr = np.load(os.path.join(folder, "ptr.npy"), mmap_mode=mode)
        self.doc = np.load(os.path.join(folder, "doc.npy"), mmap_mode=mode)
        self.w = np.load(os.path.join(folder, "w.npy"), mmap_mode=mode)
        self.df = np.load(os.path.join(folder, "df.npy"))
        self.meta = json.load(open(os.path.join(folder, "meta.json")))
        self.n, self.base = self.meta["n_docs"], self.meta.get("base", 0)
        self._lib = None

    def passage(self, d):
        """The token ids of this part's passage d (its own numbering), without its <|endoftext|>."""
        if self._lib is None:
            self._lib = TokenShards(self.meta["library"])
        a = int(self._lib.offsets[d])
        e = int(self._lib.offsets[d + 1]) if d + 1 < len(self._lib.offsets) else len(self._lib.tokens)
        return [int(t) for t in self._lib.tokens[a:e - 1]]

    def scores(self, q, max_df, max_terms=0):
        """BM25 of every passage of this part for one query's unique token ids. max_terms > 0: only that many of the
        query's rarest tokens, which carry most of the weight and have the shortest lists."""
        q = [t for t in q if 0 <= t < len(self.df) and 0 < self.df[t] <= max_df * self.n]
        if max_terms and len(q) > max_terms:
            q = sorted(q, key=lambda t: self.df[t])[:max_terms]
        if not q:
            return None
        lists = [(int(self.ptr[t]), int(self.ptr[t + 1])) for t in q]
        d = np.concatenate([self.doc[a:e] for a, e in lists])
        w = np.concatenate([self.w[a:e] for a, e in lists])
        return np.bincount(d, weights=w, minlength=self.n)


class Index:
    """A library's index: one or more parts, searched together."""

    def __init__(self, folders, mmap=True):
        folders = [folders] if isinstance(folders, str) else list(folders)
        self.parts = []
        for f in folders:
            sub = sorted(os.path.join(f, p) for p in os.listdir(f) if p.startswith("part_")) \
                if not os.path.exists(os.path.join(f, "ptr.npy")) else [f]
            self.parts += [Part(s, mmap) for s in sub]
        self.n = sum(p.n for p in self.parts)

    def search(self, query_ids, k=5, max_df=1.0, skip=(0,), max_terms=0):
        """(passage numbers, scores) of the top k passages for one query (a list of token ids), best first."""
        q = sorted(set(int(t) for t in query_ids) - set(skip))
        best_d, best_s = [], []
        for p in self.parts:
            s = p.scores(q, max_df, max_terms)
            if s is None:
                continue
            top = np.argpartition(-s, min(k, len(s) - 1))[:k] if len(s) > k else np.arange(len(s))
            best_d.append(top + p.base)
            best_s.append(s[top])
        if not best_d:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        d, s = np.concatenate(best_d), np.concatenate(best_s)
        o = np.argsort(-s, kind="stable")[:k]
        return d[o], s[o]

    def passage(self, d):
        """The token ids of passage d (the library's numbering)."""
        for p in self.parts:
            if p.base <= d < p.base + p.n:
                return p.passage(int(d) - p.base)
        raise IndexError(d)


def contains(seq, sub):
    """Whether the token list sub appears in seq as a run."""
    if not sub:
        return False
    first, n = sub[0], len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1) if seq[i] == first)


def answer_tokens(target):
    """The answer's tokens to look for in a passage: without its first token, whose leading space can merge
    differently inside a passage, when the answer has more than one."""
    return target[1:] if len(target) > 1 else target


def save(folder, ptr, doc, w, df, meta):
    os.makedirs(folder, exist_ok=True)
    for name, arr in (("ptr", ptr), ("doc", doc), ("w", w), ("df", df)):
        np.save(os.path.join(folder, f"{name}.npy"), arr)
    json.dump(meta, open(os.path.join(folder, "meta.json"), "w"), indent=1)


def cmd_build(a):
    base = a.base
    for i, path in enumerate(a.libraries):
        lib = TokenShards(path)
        t0 = time.time()
        ptr, doc, w, df = build(lib, a.k1, a.b, a.chunk_docs)
        out = a.out if len(a.libraries) == 1 else os.path.join(a.out, f"part_{i:03d}")
        save(out, ptr, doc, w, df, dict(library=os.path.abspath(path), n_docs=len(lib.offsets), base=base,
                                        n_postings=int(len(doc)), k1=a.k1, b=a.b, vocab_size=len(df)))
        print(f"{path}: {len(lib.offsets):,} passages, {len(doc):,} postings, {time.time() - t0:.1f}s -> {out}",
              flush=True)
        base += len(lib.offsets)


def cmd_tiers(a):
    """Two-tier search: the written entries (v1, v2) and the human passages searched apart, the top passage of each
    given to the model. A large human tier crowds the written entries out of a single ranking; searched apart, the
    written tier keeps its precision and the human tier adds what the entries lack."""
    from score import load_items
    tiers = {name: Index(path.split("+")) for name, path in (t.split("=", 1) for t in a.tiers.split(";"))}
    items, eot, sep = load_items(a.library, ["qa"], "heldout", a.limit)
    skip = (eot, *sep)
    names = list(tiers)
    hits = {n: 0 for n in names}
    either, times = 0, []
    for it in items:
        want = answer_tokens(it["target"])
        t0 = time.perf_counter()
        tops = {n: idx.search(it["prefix"], k=1, max_df=a.max_df, skip=skip)[0] for n, idx in tiers.items()}
        times.append(time.perf_counter() - t0)
        got = {n: bool(len(d)) and contains(tiers[n].passage(int(d[0])), want) for n, d in tops.items()}
        for n in names:
            hits[n] += int(got[n])
        either += int(any(got.values()))
    n, t = len(items), np.array(times) * 1000
    print(f"{n:,} queries, max_df {a.max_df}; the answer in the top passage of: "
          + ", ".join(f"{k} ({tiers[k].n:,} passages) {v / n:.3f}" for k, v in hits.items())
          + f"; in either {either / n:.3f}; ms per query (both tiers): median {np.median(t):.1f}, p95 "
          f"{np.percentile(t, 95):.1f}", flush=True)


def cmd_bench(a):
    from score import load_items
    idx = Index(a.index)
    items, eot, sep = load_items(a.library, ["qa"], "heldout", a.limit)
    skip = (eot, *sep)
    for max_df, max_terms in [(float(x), int(m)) for x in a.max_df.split(",") for m in a.max_terms.split(",")]:
        hit1 = hit5 = ans1 = ans5 = 0
        times = []
        for it in items:
            t0 = time.perf_counter()
            d, _ = idx.search(it["prefix"], k=5, max_df=max_df, skip=skip, max_terms=max_terms)
            times.append(time.perf_counter() - t0)
            hit1 += int(len(d) and d[0] == it["source_doc"])
            hit5 += int(it["source_doc"] in set(d.tolist()))
            want = answer_tokens(it["target"])                       # does a found passage hold the answer itself?
            found = [contains(idx.passage(int(x)), want) for x in d]
            ans1 += int(bool(found) and found[0])
            ans5 += int(any(found))
        t = np.array(times) * 1000
        n = len(items)
        print(f"max_df {max_df}, max_terms {max_terms or 'all'}: {n:,} queries over {idx.n:,} passages; the question's own entry first "
              f"{hit1 / n:.3f}, in the top 5 {hit5 / n:.3f}; the answer in the first passage {ans1 / n:.3f}, in the top "
              f"5 {ans5 / n:.3f}; ms per query: median {np.median(t):.2f}, p95 {np.percentile(t, 95):.2f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="index one or more packed libraries (several: one part each)")
    b.add_argument("libraries", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--k1", type=float, default=1.2)
    b.add_argument("--b", type=float, default=0.75)
    b.add_argument("--chunk_docs", type=int, default=200_000)
    b.add_argument("--base", type=int, default=0, help="the first passage number: an index built apart from another "
                                                       "(v1 first, then the human pool) continues its numbering")
    q = sub.add_parser("bench", help="speed and hit rate on the library's held-out questions")
    q.add_argument("index", nargs="+")
    q.add_argument("--library", required=True, help="the packed library the questions come from (qa/heldout)")
    q.add_argument("--limit", type=int, default=2000)
    q.add_argument("--max_df", default="1.0,0.1,0.02")
    q.add_argument("--max_terms", default="0", help="comma list; 0: every token of the query")
    r = sub.add_parser("tiers", help="two tiers searched apart: how often each tier's top passage holds the answer")
    r.add_argument("--tiers", required=True, help="name=index[+index];name=index..., e.g. "
                                                  "written=data/library/v1/bm25;human=/path/bm25_pool")
    r.add_argument("--library", required=True)
    r.add_argument("--limit", type=int, default=2000)
    r.add_argument("--max_df", type=float, default=0.1)
    a = ap.parse_args(argv)
    {"build": cmd_build, "bench": cmd_bench, "tiers": cmd_tiers}[a.cmd](a)


if __name__ == "__main__":
    main()
