"""Optional CPU IVF-PQ search with document exclusion inside FAISS's scanner."""
import numpy as np
import torch

from .datastore import Datastore


class FaissDatastore(Datastore):
    """Same token storage/refresh/search interface as Datastore; approximate inner products.

    Importing this module does not require FAISS. Install exealm[faiss] to build an
    index. Configure nlist/pq_m/nbits for the library size and encoder key width.
    Replacing keys or document IDs directly requires rebuild_index().
    """

    def __init__(self, tokens, doc_ids, *, nlist=128, pq_m=8, nbits=8,
                 nprobe=16, train_size=50_000, seed=0, rerank=4):
        super().__init__(tokens, doc_ids)
        if min(nlist, pq_m, nprobe, train_size, rerank) < 1 or not 1 <= nbits <= 8:
            raise ValueError('positive index settings and 1 <= nbits <= 8 required')
        self.nlist, self.pq_m, self.nbits = nlist, pq_m, nbits
        self.nprobe, self.train_size, self.seed = nprobe, train_size, seed
        self.rerank = rerank
        self.index = None
        self._documents = {}

    @torch.no_grad()
    def refresh(self, encoder, emb, batch=1024, step=None):
        self.index = None  # failed refresh/rebuild must never leave a stale searchable index
        super().refresh(encoder, emb, batch, step)
        self.rebuild_index()

    @torch.no_grad()
    def rebuild_index(self):
        """Train fresh quantizers on a deterministic sample and add all current keys."""
        self.index = None
        try:
            import faiss
        except ImportError as e:
            raise ImportError('Install exealm[faiss] for FaissDatastore') from e
        if self.keys is None or self.keys.ndim != 2 or len(self.keys) != len(self):
            raise ValueError('keys must have shape (number of chunks, key width)')
        if self.doc_ids.shape != (len(self),):
            raise ValueError('one document ID per chunk required')
        keys = np.ascontiguousarray(self.keys.detach().float().cpu().numpy())
        if not np.isfinite(keys).all() or keys.shape[1] % self.pq_m:
            raise ValueError('keys must be finite and key width divisible by pq_m')
        ntrain = min(len(self), self.train_size)
        if ntrain < max(self.nlist, 2 ** self.nbits):
            raise ValueError('too few training keys for nlist/nbits; reduce them or increase train_size')
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(keys.shape[1]), keys.shape[1],
                                self.nlist, self.pq_m, self.nbits, faiss.METRIC_INNER_PRODUCT)
        index.cp.seed = self.seed
        index.pq.cp.seed = self.seed
        rows = np.random.default_rng(self.seed).choice(len(self), ntrain, replace=False)
        index.train(keys[rows])
        index.add(keys)
        docs = self.doc_ids.detach().cpu().numpy()
        order = np.argsort(docs, kind='stable').astype(np.int64)
        unique, starts = np.unique(docs[order], return_index=True)
        self._documents = {int(doc): group for doc, group in zip(unique, np.split(order, starts[1:]))}
        self._rank_keys = keys
        self.index = index

    @torch.no_grad()
    def search(self, q, k, exclude=None, block=262144):
        """Return (indices, valid) on q.device, shape (M, min(k, N)).

        Native ID selectors exclude documents before top-k selection, so no fixed
        over-fetch factor is needed. If too few eligible hits are found, double
        nprobe until k are found or all lists have been visited. Unfilled entries
        use index 0 with valid=False, as callers gather tokens before masking.
        block is accepted for compatibility; FAISS controls its own scanning.
        """
        import faiss
        if self.index is None:
            raise RuntimeError('refresh() or rebuild_index() must run before search()')
        if q.ndim != 2 or q.shape[1] != self.index.d or k < 0:
            raise ValueError('invalid query shape or negative k')
        queries = np.ascontiguousarray(q.detach().float().cpu().numpy())
        if not np.isfinite(queries).all():
            raise ValueError('queries must be finite')
        kk = min(k, len(self))
        result = np.zeros((len(q), kk), dtype=np.int64)
        valid = np.zeros_like(result, dtype=bool)
        if kk == 0 or len(q) == 0:
            return torch.from_numpy(result).to(q.device), torch.from_numpy(valid).to(q.device)
        if exclude is None:
            groups = [(np.arange(len(q)), None)]
        else:
            if exclude.shape != (len(q),):
                raise ValueError('exclude must contain one document ID per query')
            excluded = exclude.detach().cpu().numpy()
            groups = [(np.flatnonzero(excluded == doc), int(doc)) for doc in np.unique(excluded)]
        for rows, doc in groups:
            denied = self._documents.get(doc)
            selector = None if denied is None else faiss.IDSelectorNot(faiss.IDSelectorBatch(denied))
            needed = min(kk, len(self) - (len(denied) if denied is not None else 0))
            if needed == 0:
                continue
            pending = rows
            probe = min(self.nprobe, self.nlist)
            while len(pending):
                params = faiss.IVFPQSearchParameters(nprobe=probe, sel=selector)
                scores, indices = self.index.search(queries[pending], min(len(self), kk * self.rerank), params=params)
                ok = (indices >= 0) & np.isfinite(scores)
                safe = np.where(ok, indices, 0)
                scores = np.einsum('md,mcd->mc', queries[pending], self._rank_keys[safe])
                scores = np.where(ok, scores, -np.inf)
                chosen = np.argsort(-scores, axis=1)[:, :kk]
                result[pending] = np.take_along_axis(safe, chosen, axis=1)
                valid[pending] = np.take_along_axis(ok, chosen, axis=1)
                if probe == self.nlist:
                    break
                pending = pending[ok.sum(1) < needed]
                probe = min(self.nlist, probe * 2)
        return torch.from_numpy(result).to(q.device), torch.from_numpy(valid).to(q.device)
