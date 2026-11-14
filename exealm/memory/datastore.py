"""The read-only datastore that RG circuits retrieve from.

It holds chunks of training text (token ids), the document each chunk came from, and one key vector per chunk. The
keys are made by the model's OWN chunk encoder (exealm/model/retrieval.py: ChunkEncoder), so the model learns its own
embedding space for retrieval instead of borrowing someone else's. Because that encoder keeps training, the stored keys
go stale; refresh() re-encodes everything every few thousand steps (the approach REALM used).

The stored keys are only used to FIND candidates. The chosen chunks are re-encoded live in the forward pass, so
gradients still reach the encoder.

Scale: brute-force search below is fine up to a few million chunks on one GPU. The real datastore (billions of tokens)
needs an approximate index such as FAISS IVF-PQ, sharded across GPUs. That is an open task (docs/memory.md).
"""
import numpy as np
import torch


class Datastore:
    def __init__(self, tokens, doc_ids):
        """tokens (N, chunk_len) integer ids; doc_ids (N,) the document of each chunk."""
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.doc_ids = torch.as_tensor(doc_ids, dtype=torch.long)
        self.keys = None                                  # (N, key_dim), filled by refresh()
        self.refreshed_at = None

    def __len__(self):
        return self.tokens.shape[0]

    def to(self, device):
        self.tokens, self.doc_ids = self.tokens.to(device), self.doc_ids.to(device)
        if self.keys is not None:
            self.keys = self.keys.to(device)
        return self

    @classmethod
    def from_shards(cls, shards, n_chunks, chunk_len, seed=0):
        """Sample uniformly, with replacement, from windows inside documents of a TokenShards corpus.
        TODO(data): a curated datastore (encyclopedias, textbooks, code docs) instead of a random sample."""
        if n_chunks <= 0 or chunk_len <= 0:
            raise ValueError("n_chunks and chunk_len must be positive")
        offsets = np.asarray(shards.offsets, dtype=np.int64)
        if (offsets.ndim != 1 or (len(offsets) and
                (offsets[0] != 0 or offsets[-1] >= len(shards.tokens) or np.any(np.diff(offsets) <= 0)))):
            raise ValueError("invalid document offsets")
        lengths = np.diff(np.append(offsets, len(shards.tokens))) if len(offsets) else np.array([], dtype=np.int64)
        counts = np.maximum(lengths - chunk_len + 1, 0)
        cumulative = np.cumsum(counts)
        if not len(cumulative) or cumulative[-1] == 0:
            raise ValueError("no document is long enough for chunk_len")
        rng = np.random.default_rng(seed)
        windows = rng.integers(0, cumulative[-1], size=n_chunks)
        docs = np.searchsorted(cumulative, windows, side="right")
        before = np.concatenate(([0], cumulative[:-1]))
        starts = offsets[docs] + windows - before[docs]
        toks = np.stack([shards.tokens[s:s + chunk_len] for s in starts]).astype(np.int64)
        return cls(toks, docs)

    @torch.no_grad()
    def refresh(self, encoder, emb, batch=1024, step=None):
        """Re-encode every chunk with the model's current encoder."""
        was = encoder.training
        encoder.eval()
        if getattr(encoder, "lexical", 0) and not getattr(encoder, "idf_ready", False):
            encoder.set_idf(self.tokens)                             # word rarity, from the first datastore it sees
            encoder.idf_ready = True
        keys = [encoder(self.tokens[i:i + batch], emb)[1].float() for i in range(0, len(self), batch)]
        encoder.train(was)
        self.keys = torch.cat(keys)
        self.refreshed_at = step

    @torch.no_grad()
    def search(self, q, k, exclude=None, block=262144):
        """Top-k chunks by dot product for each query q (M, key_dim). exclude (M,): a document id per query whose
        chunks must not be returned (the document being read, so the model cannot look up the text it is about to
        predict). The rule is applied inside the search, so every query gets k results from other documents.
        (An earlier version fetched 4k candidates and dropped excluded ones afterwards. The nearest chunks are usually
        from the same document, so reads often came back empty in training but never in evaluation, and the model
        broke when evaluated with the datastore: experiments/rag_text.py.) Returns indices (M, k') and a validity mask
        (M, k'), k' = min(k, N)."""
        if self.keys is None:
            raise RuntimeError("Datastore.refresh() must run before search()")
        q = q.float()
        kk = min(k, len(self))
        best_s = best_i = None
        for b0 in range(0, len(self), block):                          # blockwise top-k keeps memory bounded
            s = q @ self.keys[b0:b0 + block].T
            if exclude is not None:
                s = s.masked_fill(self.doc_ids[b0:b0 + block].view(1, -1) == exclude.view(-1, 1), float("-inf"))
            s, i = s.topk(min(kk, s.shape[1]), dim=1)
            i = i + b0
            if best_s is not None:
                s, j = torch.cat([best_s, s], 1).topk(kk, dim=1)
                i = torch.cat([best_i, i], 1).gather(1, j)
            best_s, best_i = s, i
        return best_i, torch.isfinite(best_s)
