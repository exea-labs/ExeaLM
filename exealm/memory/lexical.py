"""A library searched by keywords (BM25) instead of by the model's own embeddings.

On held-out Wikipedia, a BM25 search over the first halves of articles, queried with the 64 tokens just read, finds the
right article 62% of the time; the model's learned search finds it 26% (experiments/irl_eval.py, issue #22). With a
LexicalDatastore attached, the RG circuit takes its candidates from BM25 over the previous chunk's tokens and keeps
everything else it learned: it still scores the candidates with its own encoder and reads them with its own
attention. Token ids are the words. Very common tokens (in more than `max_df` of the chunks) are skipped: their BM25
weight is close to zero and their posting lists are long.
"""
import math

import numpy as np
import torch

from .datastore import Datastore


class BM25:
    """Okapi BM25 over fixed chunks of token ids."""

    def __init__(self, chunks, k1=1.2, b=0.75, max_df=0.1):
        chunks = np.asarray(chunks)
        self.n = len(chunks)
        lens = np.array([len(c) for c in chunks], dtype=np.float64)
        self.avg = lens.mean()
        flat = np.concatenate([np.unique(c, return_counts=True)[0] for c in chunks])
        cnts = np.concatenate([np.unique(c, return_counts=True)[1] for c in chunks])
        rows = np.repeat(np.arange(self.n), [len(np.unique(c)) for c in chunks])
        order = np.argsort(flat, kind="stable")
        flat, cnts, rows = flat[order], cnts[order], rows[order]
        toks, starts = np.unique(flat, return_index=True)
        ends = np.r_[starts[1:], len(flat)]
        self.post = {}
        for t, a, e in zip(toks.tolist(), starts.tolist(), ends.tolist()):
            df = e - a
            if df > max_df * self.n:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            tf = cnts[a:e].astype(np.float64)
            norm = 1 - b + b * lens[rows[a:e]] / self.avg
            self.post[t] = (rows[a:e], idf * tf * (k1 + 1) / (tf + k1 * norm))

    def scores(self, query):
        s = np.zeros(self.n)
        for t in set(np.asarray(query).tolist()):
            p = self.post.get(t)
            if p is not None:
                s[p[0]] += p[1]
        return s

    def top(self, query, k, exclude_rows=None):
        s = self.scores(query)
        if exclude_rows is not None and len(exclude_rows):
            s[exclude_rows] = -np.inf
        k = min(k, self.n)
        idx = np.argpartition(-s, k - 1)[:k]
        return idx[np.argsort(-s[idx])], s


class LexicalDatastore(Datastore):
    """A Datastore whose search takes token queries (the RG circuit passes the previous chunk's tokens)."""

    def __init__(self, tokens, doc_ids, **bm25):
        super().__init__(tokens, doc_ids)
        self.bm25 = BM25(self.tokens.cpu().numpy(), **bm25)
        docs = self.doc_ids.cpu().numpy()
        order = np.argsort(docs, kind="stable")
        uniq, start = np.unique(docs[order], return_index=True)
        ends = np.r_[start[1:], len(docs)]
        self._rows_of = {int(u): order[a:e] for u, a, e in zip(uniq, start, ends)}

    def search_tokens(self, queries, k, exclude=None):
        """queries (M, L) token ids; exclude (M,) a document id per query whose chunks are never returned. Returns
        indices (M, k) and a validity mask (M, k) on the queries' device; a chunk with no shared word is invalid."""
        q = queries.cpu().numpy()
        ex = None if exclude is None else exclude.cpu().numpy()
        k = min(k, len(self))
        idx = np.zeros((len(q), k), dtype=np.int64)
        ok = np.zeros((len(q), k), dtype=bool)
        for i in range(len(q)):
            rows = None if ex is None else self._rows_of.get(int(ex[i]))
            top, s = self.bm25.top(q[i], k, rows)
            idx[i, :len(top)] = top
            ok[i, :len(top)] = s[top] > 0
        dev = queries.device
        return torch.from_numpy(idx).to(dev), torch.from_numpy(ok).to(dev)

    def search(self, q, k, exclude=None, block=262144):
        raise RuntimeError("LexicalDatastore is searched with tokens (search_tokens); the RG circuit does this")
