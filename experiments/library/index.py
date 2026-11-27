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


def compact(src, dst, max_df=0.1, max_postings=0):
    """A smaller copy of one index part, for shipping: the lists of tokens in more than max_df of the passages are
    dropped (searches skip them anyway: their weight is near zero and their lists are the longest, 35% of all postings
    at 0.1), weights are 8-bit (each part's largest weight / 255 per step, as search engines store impacts) and passage
    numbers 3 bytes (a part has fewer than 16.7M passages). 3 + 1 bytes a posting instead of 4 + 4.

    max_postings N > 0 also keeps only each token's N highest-weighted passages (static pruning, Carmel et al. 2001):
    the rest cannot reach a query's top results unless several of its tokens add up there, and a query then reads at
    most N postings a token instead of up to a tenth of the library."""
    p = Part(src)
    if p.n >= 1 << 24:
        raise SystemExit(f"{src}: {p.n:,} passages do not fit 3-byte numbers")
    keep = (p.df > 0) & (p.df <= max_df * p.n)
    lens = np.diff(np.asarray(p.ptr)) * keep
    if max_postings:
        lens = np.minimum(lens, max_postings)
    ptr = np.zeros(len(p.ptr), dtype=np.int64)
    ptr[1:] = np.cumsum(lens)
    total = int(ptr[-1])
    scale = float(np.max(p.w)) / 255 if len(p.w) else 1.0
    os.makedirs(dst, exist_ok=True)
    doc3 = np.lib.format.open_memmap(os.path.join(dst, "doc3.npy"), mode="w+", dtype=np.uint8, shape=(total, 3))
    wq = np.lib.format.open_memmap(os.path.join(dst, "wq.npy"), mode="w+", dtype=np.uint8, shape=(total,))
    for t in np.nonzero(keep)[0]:
        a, e, o = int(p.ptr[t]), int(p.ptr[t + 1]), int(ptr[t])
        d, w = np.asarray(p.doc[a:e], dtype=np.uint32), np.asarray(p.w[a:e])
        if max_postings and e - a > max_postings:
            sel = np.sort(np.argpartition(-w, max_postings - 1)[:max_postings])    # the best, kept in passage order
            d, w = d[sel], w[sel]
        m = len(d)
        doc3[o:o + m, 0], doc3[o:o + m, 1], doc3[o:o + m, 2] = d & 255, (d >> 8) & 255, d >> 16
        wq[o:o + m] = np.clip(np.rint(w / scale), 1, 255).astype(np.uint8)
    doc3.flush()
    wq.flush()
    np.save(os.path.join(dst, "ptr.npy"), ptr)
    np.save(os.path.join(dst, "df.npy"), np.asarray(p.df))
    json.dump(dict(p.meta, format="compact", w_scale=scale, max_df_kept=max_df, max_postings=max_postings,
                   n_postings=total),
              open(os.path.join(dst, "meta.json"), "w"), indent=1)
    return total


class Part:
    """One index part: the passages of one pack, numbered from `base` in the whole library. Reads both formats: full
    (doc.npy int32, w.npy float32) and compact (doc3.npy 3-byte, wq.npy 8-bit, see compact())."""

    def __init__(self, folder, mmap=True):
        mode = "r" if mmap else None
        self.meta = json.load(open(os.path.join(folder, "meta.json")))
        self.compact = self.meta.get("format") == "compact"
        self.ptr = np.load(os.path.join(folder, "ptr.npy"), mmap_mode=mode)
        if self.compact:
            self.doc = np.load(os.path.join(folder, "doc3.npy"), mmap_mode=mode)
            self.w = np.load(os.path.join(folder, "wq.npy"), mmap_mode=mode)
            self.scale = self.meta["w_scale"]
        else:
            self.doc = np.load(os.path.join(folder, "doc.npy"), mmap_mode=mode)
            self.w = np.load(os.path.join(folder, "w.npy"), mmap_mode=mode)
        self.df = np.load(os.path.join(folder, "df.npy"))
        self.n, self.base = self.meta["n_docs"], self.meta.get("base", 0)
        self._lib = None
        self._gpu = None

    def lists(self, a, e):
        """Passage numbers (int64) and float weights of postings a:e."""
        if self.compact:
            b = np.asarray(self.doc[a:e], dtype=np.int64)
            return b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16), np.asarray(self.w[a:e], dtype=np.float32) * self.scale
        return np.asarray(self.doc[a:e]), np.asarray(self.w[a:e])

    def to(self, dev):
        """Postings on a GPU for search_gpu. A compact part stays compact there (3-byte numbers and 8-bit weights,
        decoded per query: 4 bytes a posting, 24.6 GB for the 25B-token pool); a full part goes as int32 and float16."""
        import torch
        if self.compact:
            self._gpu = (torch.from_numpy(np.ascontiguousarray(self.doc)).to(dev),
                         torch.from_numpy(np.ascontiguousarray(self.w)).to(dev))
        else:
            d, w = self.lists(0, int(self.ptr[-1]))
            self._gpu = (torch.from_numpy(d.astype(np.int32)).to(dev), torch.from_numpy(w.astype(np.float16)).to(dev))
        return self

    def batch_spans(self, qs, max_df):
        """(starts, lengths, query numbers) of the posting lists of every query's tokens, in numpy, without a Python loop
        over tokens: the per-token lookups were most of a GPU search's time."""
        if not qs:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z
        tok = np.concatenate([np.asarray(q, dtype=np.int64) for q in qs])
        qid = np.repeat(np.arange(len(qs), dtype=np.int64), [len(q) for q in qs])
        ok = (tok >= 0) & (tok < len(self.df))
        tok, qid = tok[ok], qid[ok]
        df = self.df[tok]
        ok = (df > 0) & (df <= max_df * self.n)
        tok, qid = tok[ok], qid[ok]
        st = np.asarray(self.ptr[tok], dtype=np.int64)
        ln = np.asarray(self.ptr[tok + 1], dtype=np.int64) - st
        ok = ln > 0
        return st[ok], ln[ok], qid[ok]

    def gather(self, idx):
        """Passage numbers (int64) and float32 weights of the postings at idx, on the part's GPU."""
        doc, w = self._gpu
        if self.compact:
            b = doc[idx].long()
            return b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16), w[idx].float() * self.scale
        return doc[idx].long(), w[idx].float()

    def top_gpu_batch(self, qs, max_df, k, budget=150_000_000):
        """(local passage numbers, scores), each (len(qs), k), for several queries at once. Every posting of every
        query is gathered with one index (repeat_interleave over the lists' starts) and added into a (queries x
        passages) score table; queries go in groups of at most `budget` postings, to bound GPU memory."""
        import torch
        dev = self._gpu[0].device
        st, ln, qid = self.batch_spans(qs, max_df)
        per_q = np.bincount(qid, weights=ln, minlength=len(qs))
        out_i = np.zeros((len(qs), min(k, self.n)), dtype=np.int64)
        out_v = np.zeros((len(qs), min(k, self.n)), dtype=np.float32)
        g0 = 0
        while g0 < len(qs):
            g1, total = g0, 0
            while g1 < len(qs) and (g1 == g0 or total + per_q[g1] <= budget):
                total += per_q[g1]
                g1 += 1
            sel = (qid >= g0) & (qid < g1)
            s = torch.zeros((g1 - g0) * self.n, device=dev, dtype=torch.float32)
            T = int(ln[sel].sum())
            if T:
                st_t, ln_t = torch.from_numpy(st[sel]).to(dev), torch.from_numpy(ln[sel]).to(dev)
                q_t = torch.from_numpy(qid[sel] - g0).to(dev)
                first = torch.cumsum(ln_t, 0) - ln_t                    # where each list starts in the gathered run
                idx = torch.repeat_interleave(st_t - first, ln_t, output_size=T) + torch.arange(T, device=dev)
                d, w = self.gather(idx)
                s.index_add_(0, torch.repeat_interleave(q_t, ln_t, output_size=T) * self.n + d, w)
            v, i = torch.topk(s.view(g1 - g0, self.n), min(k, self.n), dim=1)
            out_i[g0:g1], out_v[g0:g1] = i.cpu().numpy(), v.cpu().numpy()
            g0 = g1
        return out_i, out_v

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
        got = [self.lists(int(self.ptr[t]), int(self.ptr[t + 1])) for t in q]
        got = [g for g in got if len(g[0])]
        if not got:
            return None
        d = np.concatenate([g[0] for g in got])
        w = np.concatenate([g[1] for g in got])
        return np.bincount(d, weights=w, minlength=self.n)

    def top_gpu(self, q, max_df, k):
        """(local passage numbers, scores) of this part's top k for one query, on the GPU the part was moved to."""
        i, v = self.top_gpu_batch([q], max_df, k)
        keep = v[0] > 0
        return (i[0][keep], v[0][keep]) if keep.any() else None


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

    def to(self, dev):
        """Every part's postings onto a GPU, for search_gpu."""
        for p in self.parts:
            p.to(dev)
        return self

    def search_gpu(self, query_ids, k=5, max_df=1.0, skip=(0,)):
        """search() on the GPU the index was moved to (Index.to): the same ranking, a few milliseconds a query."""
        q = sorted(set(int(t) for t in query_ids) - set(skip))
        best_d, best_s = [], []
        for p in self.parts:
            got = p.top_gpu(q, max_df, k)
            if got is not None:
                best_d.append(got[0] + p.base)
                best_s.append(got[1])
        if not best_d:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        d, s = np.concatenate(best_d), np.concatenate(best_s)
        o = np.argsort(-s, kind="stable")[:k]
        return d[o], s[o]

    def search_gpu_batch(self, queries, k=5, max_df=1.0, skip=(0,)):
        """search_gpu for many queries in one pass (a retrieval service batching its callers): a list of (passage
        numbers, scores) per query, best first."""
        qs = [sorted(set(int(t) for t in q) - set(skip)) for q in queries]
        best_d = [[] for _ in qs]
        best_s = [[] for _ in qs]
        for p in self.parts:
            d, s = p.top_gpu_batch(qs, max_df, k)
            for i in range(len(qs)):
                keep = s[i] > 0
                best_d[i].append(d[i][keep] + p.base)
                best_s[i].append(s[i][keep])
        out = []
        for bd, bs in zip(best_d, best_s):
            d, s = np.concatenate(bd), np.concatenate(bs)
            o = np.argsort(-s, kind="stable")[:k]
            out.append((d[o], s[o]))
        return out

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


def cmd_compact(a):
    src = a.index
    parts = sorted(p for p in os.listdir(src) if p.startswith("part_")) if not os.path.exists(os.path.join(src, "ptr.npy")) else [""]
    before = after = 0
    for name in parts:
        t0 = time.time()
        s, d = os.path.join(src, name), os.path.join(a.out, name)
        n_before = Part(s).meta["n_postings"]
        n_after = compact(s, d, a.max_df, a.max_postings)
        before, after = before + n_before, after + n_after
        print(f"{s}: {n_before:,} postings -> {n_after:,}, {time.time() - t0:.0f}s -> {d}", flush=True)
    print(f"{before:,} postings ({before * 8 / 1e9:.1f} GB) -> {after:,} ({after * 4 / 1e9:.1f} GB)", flush=True)


def cmd_gpubench(a):
    """GPU memory, one-query latency and batched throughput of an index moved to a GPU."""
    import torch
    from score import load_items
    idx = Index(a.index)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    idx.to("cuda")
    torch.cuda.synchronize()
    print(f"{idx.n:,} passages on the GPU in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB",
          flush=True)
    items, eot, sep = load_items(a.questions, ["qa"], "heldout", a.limit)
    qs, skip = [it["prefix"] for it in items], (eot, *sep)
    idx.search_gpu(qs[0], k=5, max_df=a.max_df, skip=skip)                   # warm-up
    times = []
    for q in qs[:500]:
        t0 = time.perf_counter()
        idx.search_gpu(q, k=5, max_df=a.max_df, skip=skip)
        times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000
    print(f"one query at a time: median {np.median(t):.2f} ms, p95 {np.percentile(t, 95):.2f} ms", flush=True)
    for b in [int(x) for x in a.batches.split(",")]:
        t0 = time.perf_counter()
        n = 0
        for i in range(0, min(len(qs), 4 * b), b):
            idx.search_gpu_batch(qs[i:i + b], k=5, max_df=a.max_df, skip=skip)
            n += len(qs[i:i + b])
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"batches of {b}: {n / dt:,.0f} queries a second, {1000 * dt / max(1, n // b):.1f} ms a batch, peak "
              f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


def cmd_agree(a):
    """How often a smaller or faster index finds what the reference finds: the same top passage, and the overlap of the
    top 5, on the library's held-out questions; and its speed on the GPU."""
    from score import load_items
    import torch
    ref, other = Index(a.reference.split("+")).to("cuda"), Index(a.index.split("+")).to("cuda")
    items, eot, sep = load_items(a.questions, ["qa"], "heldout", a.limit)
    skip, same, overlap, times = (eot, *sep), 0, 0.0, []
    for it in items:
        r, _ = ref.search_gpu(it["prefix"], k=5, max_df=a.max_df, skip=skip)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o, _ = other.search_gpu(it["prefix"], k=5, max_df=a.max_df, skip=skip)
        times.append(time.perf_counter() - t0)
        same += int(len(r) and len(o) and r[0] == o[0])
        overlap += len(set(r.tolist()) & set(o.tolist())) / max(1, len(r))
    n, t = len(items), np.array(times) * 1000
    print(f"{a.index}: same top passage as {a.reference} for {same / n:.3f}, top-5 overlap {overlap / n:.3f}; GPU ms "
          f"per query: median {np.median(t):.2f}, p95 {np.percentile(t, 95):.2f}; "
          f"{sum(p.meta['n_postings'] for p in other.parts) * 4 / 1e9:.1f} GB", flush=True)


def cmd_tiers(a):
    """Two-tier search: the written entries (v1, v2) and the human passages searched apart, the top passage of each
    given to the model. A large human tier crowds the written entries out of a single ranking; searched apart, the
    written tier keeps its precision and the human tier adds what the entries lack."""
    from score import load_items
    tiers = {name: Index(path.split("+")) for name, path in (t.split("=", 1) for t in a.tiers.split(";"))}
    if a.device != "cpu":
        for idx in tiers.values():
            idx.to(a.device)
    items, eot, sep = load_items(a.library, ["qa"], "heldout", a.limit)
    skip = (eot, *sep)
    names = list(tiers)
    hits = {n: 0 for n in names}
    either, times = 0, []
    for it in items:
        want = answer_tokens(it["target"])
        t0 = time.perf_counter()
        find = (lambda idx: idx.search_gpu(it["prefix"], k=1, max_df=a.max_df, skip=skip)) if a.device != "cpu" \
            else (lambda idx: idx.search(it["prefix"], k=1, max_df=a.max_df, skip=skip))
        tops = {n: find(idx)[0] for n, idx in tiers.items()}
        times.append(time.perf_counter() - t0)
        got = {n: bool(len(d)) and contains(tiers[n].passage(int(d[0])), want) for n, d in tops.items()}
        for n in names:
            hits[n] += int(got[n])
        either += int(any(got.values()))
    n, t = len(items), np.array(times) * 1000
    print(f"{n:,} queries on {a.device}, max_df {a.max_df}; the answer in the top passage of: "
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
    r.add_argument("--device", default="cpu", help="cuda: the postings on the GPU (Index.to, search_gpu)")
    g = sub.add_parser("gpubench", help="GPU memory, latency and batched throughput")
    g.add_argument("index", nargs="+")
    g.add_argument("--questions", required=True)
    g.add_argument("--limit", type=int, default=2000)
    g.add_argument("--max_df", type=float, default=0.1)
    g.add_argument("--batches", default="16,64,256")
    ag = sub.add_parser("agree", help="a smaller index against a reference: same top passage, top-5 overlap, speed")
    ag.add_argument("index")
    ag.add_argument("--reference", required=True)
    ag.add_argument("--questions", required=True)
    ag.add_argument("--limit", type=int, default=1000)
    ag.add_argument("--max_df", type=float, default=0.1)
    c = sub.add_parser("compact", help="a smaller copy for shipping: common tokens dropped, 8-bit weights, 3-byte numbers")
    c.add_argument("index")
    c.add_argument("--out", required=True)
    c.add_argument("--max_df", type=float, default=0.1)
    c.add_argument("--max_postings", type=int, default=0, help="keep only each token's N best passages (0: all)")
    a = ap.parse_args(argv)
    {"build": cmd_build, "bench": cmd_bench, "tiers": cmd_tiers, "compact": cmd_compact, "gpubench": cmd_gpubench,
     "agree": cmd_agree}[a.cmd](a)


if __name__ == "__main__":
    main()
