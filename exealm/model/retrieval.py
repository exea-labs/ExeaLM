"""RG: the retrieval circuit. The model runs its own RAG engine from scratch.

Nothing here is borrowed from an outside retriever. The model learns all four parts of RAG:

  1. its own embedding model: ChunkEncoder turns a chunk of text into per-token states and one key vector;
  2. what to search for: an RG slot's stream A, read at the end of each chunk, is its query;
  3. which results to trust: a softmax over the top-k results' scores, learned end to end;
  4. what to take from them: each position reads the retrieved tokens with attention (query from stream B), gated by
     stream C.

Retrieval happens once per chunk of `cfg.chunk` tokens. The query for chunk c is made at the last position of chunk
c - 1, so a position never retrieves with information from its own future. The datastore never returns chunks from the
document being read (exealm/memory/datastore.py), so the model cannot look up the answer it is being trained to
predict.

Every layer starts with one RG slot and the search can never remove the last one (exealm/model/search.py). Without a
datastore (for example in unit tests or a plain language-model baseline) RG slots output zeros.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP, RMSNorm


class ChunkEncoder(nn.Module):
    """The model's own retrieval embedding model, shared by every RG slot in every layer, so there is one datastore
    index. It reads the model's token embeddings (passed in, not owned, so they stay tied), lets the tokens of a chunk
    see each other (one attention layer: a stored chunk is entirely in the past, so no causal mask), then an MLP.
    Mixing inside the chunk matters: without it a token's state knows only itself, so a retrieved value cannot carry
    the key it belongs to, and the read cannot match the question to the right value (experiments/rag_lookup.py)."""

    def __init__(self, cfg):
        super().__init__()
        e = cfg.enc_dim
        self.proj = nn.Linear(cfg.d, e, bias=False)
        self.pos = nn.Parameter(torch.randn(cfg.rag_chunk_len, e) * 0.02)
        self.mix_norm = RMSNorm(e)
        self.heads = max(1, e // 32)
        self.qkv = nn.Linear(e, 3 * e, bias=False)
        self.mix_out = nn.Linear(e, e, bias=False)
        self.mlp = MLP(e, 2 * e)
        self.norm = RMSNorm(e)
        self.key = nn.Linear(e, cfg.rag_key_dim or cfg.h, bias=False)
        self.lexical = cfg.rag_lexical
        if self.lexical:                                     # fixed random code per token, weighted by rarity (idf)
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("lex", torch.randn(cfg.vocab_size, cfg.rag_lexical, generator=g))
            self.register_buffer("idf", torch.ones(cfg.vocab_size))

    def forward(self, tokens, emb):
        """tokens (N, Lc) -> per-token states (N, Lc, enc_dim) and unit-length keys (N, h)."""
        N, Lc = tokens.shape
        z = self.proj(emb(tokens)) + self.pos[:Lc]
        q, k, v = self.qkv(self.mix_norm(z)).view(N, Lc, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        z = z + self.mix_out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, Lc, -1))
        z = self.norm(self.mlp(z))
        key = F.normalize(self.key(z.mean(1)), dim=-1)
        if self.lexical:                                     # a bag of rare-word codes: same words, similar key
            lex = F.normalize((self.lex[tokens] * self.idf[tokens].unsqueeze(-1)).sum(1), dim=-1)
            key = torch.cat([key, lex.to(key.dtype)], -1) / math.sqrt(2)
        return z, key

    @torch.no_grad()
    def set_idf(self, tokens):
        """Word rarity from a datastore's chunks: idf = log(N / (1 + number of chunks containing the token))."""
        if not self.lexical:
            return
        N, V = tokens.shape[0], self.idf.shape[0]
        df = torch.zeros(V, dtype=torch.long)
        for i in range(0, N, 65536):
            t = tokens[i:i + 65536].cpu().long()
            rows = torch.arange(t.shape[0]).view(-1, 1).expand_as(t)
            pairs = torch.unique(rows * V + t)                       # each token counted once per chunk
            df += torch.bincount(pairs % V, minlength=V)
        self.idf.copy_(torch.log(N / (1 + df.float())).clamp(min=0).to(self.idf.device))


class RetrievalCircuit(nn.Module):
    """Runs every RG slot of one layer.

    Search. The query for chunk c is the model's own encoder applied to chunk c-1 (the text just read), plus the
    slot's own stream A from chunk c-1 scaled by a learned weight that starts at 0. The stored keys come from the same
    encoder, so from the first step "find stored text like what I just read" works, and training sharpens it.

    Read. Retrieved tokens are put in the layer's own stream space, exactly as if they appeared in the context: the
    token embedding goes through the layer's norm and this slot's own input rows. Keys use the same rows (stream B) as
    the reading positions' queries but look one token back, so "find the key, read what follows it" is one attention
    step and a retrieved token matches the same token in context from the start. An earlier design read the encoder's
    states through a separate projection, and in experiments/rag_lookup.py it never learned to retrieve.
    """

    def __init__(self, cfg):
        super().__init__()
        self.chunk, self.topk, self.h, self.query = cfg.chunk, cfg.rag_topk, cfg.h, cfg.rag_query
        self.alpha = nn.Parameter(torch.zeros(()))                         # weight of the slot's stream in the query
        kd = (cfg.rag_key_dim or cfg.h) + cfg.rag_lexical
        self.own = nn.Linear(cfg.h, kd, bias=False) if kd != cfg.h else None   # the slot's stream into key space
        self.log_tau = nn.Parameter(torch.zeros(()))                       # sharpness of "which result to trust" (starts at 1)
        self.record = False         # if True, keep what the last forward pass retrieved in self.last (for inspection)
        self.last = None

    @staticmethod
    def context_keys(mem, C, n):
        """The encoder's key for every chunk of the text being read: (B, n, h). Computed once per forward pass."""
        if "ctx_keys" not in mem.cache:
            tok = mem.tokens
            B, T = tok.shape
            tok = F.pad(tok, (0, n * C - T))
            mem.cache["ctx_keys"] = mem.encoder(tok.reshape(B * n, C), mem.emb)[1].view(B, n, -1)
        return mem.cache["ctx_keys"]

    def forward(self, S, mem, layer, slots):
        A, Q, _ = S                                                        # each (B, T, g, h)
        B, T, g, h = A.shape
        C = self.chunk
        n = -(-T // C)
        ds = None if mem is None else mem.datastore
        self.last = None
        if ds is None or ds.keys is None or mem.encoder is None or mem.tokens is None or n < 2:
            return A * 0.0                                                 # zeros, still attached to the graph
        ends = torch.arange(1, n, device=A.device) * C - 1                 # last position of chunks 0 .. n-2
        if self.query == "mean":
            own = F.pad(A, (0, 0, 0, 0, 0, n * C - T)).view(B, n, C, g, h)[:, :n - 1].float().mean(2)
        else:
            own = A[:, ends].float()                                       # (B, n-1, g, h)
        base = self.context_keys(mem, C, n)[:, :n - 1].unsqueeze(2).float()        # (B, n-1, 1, h)
        if self.own is not None:
            own = self.own(own)
        q = F.normalize(base + self.alpha * F.normalize(own, dim=-1), dim=-1)
        M = B * (n - 1) * g
        kd = q.shape[-1]
        qf = q.reshape(M, kd)
        docs = None
        if mem.doc_ids is not None:
            docs = mem.doc_ids[:, ends].unsqueeze(-1).expand(B, n - 1, g).reshape(M)
        idx, valid = ds.search(qf.detach(), self.topk, exclude=docs)       # candidates from the stored index
        k = idx.shape[1]
        tok = ds.tokens[idx].reshape(M * k, -1)
        keys = mem.encoder(tok, mem.emb)[1]                                # re-encoded live: gradients flow
        Lc = tok.shape[1]
        score = self.log_tau.exp() * (qf.unsqueeze(1) * keys.view(M, k, kd).float()).sum(-1)
        logp = torch.log_softmax(score.masked_fill(~valid, -1e4), dim=1)   # (M, k)
        if self.record:                                                    # see exealm/eval/show_retrieval.py
            self.last = dict(idx=idx.view(B, n - 1, g, k).cpu(), weight=logp.detach().exp().view(B, n - 1, g, k).cpu(),
                             valid=valid.view(B, n - 1, g, k).cpu())
        # Retrieved tokens in this layer's own stream space, through this slot's own input rows. Plain token
        # embeddings: an earlier version also added the encoder's in-chunk states through a zero-initialised matrix,
        # and within ~100 steps that term swamped the token identity the match relies on (retrieval accuracy 1.5% with
        # it, 97% without it, experiments/rag_lookup.py). Context for retrieved tokens is an open design question.
        r = layer.norm(mem.emb(tok)).view(B * (n - 1), g, k, Lc, -1)
        sl = torch.as_tensor(slots, device=A.device)
        W = layer.inp.weight.view(layer.K, 3, h, -1).index_select(0, sl)   # (g, 3, h, d)
        b = layer.inp.bias.view(layer.K, 3, h).index_select(0, sl)         # (g, 3, h)
        # keys look one token back (like the lag of TONN's attention circuits): matching a retrieved token finds what
        # FOLLOWS it, so "find the key, read what comes after it" (an induction step) is a single attention lookup
        r_prev = F.pad(r, (0, 0, 1, 0))[:, :, :, :Lc]
        kr = (torch.einsum("pgkld,ghd->pgklh", r_prev, W[:, 1]) + b[:, 1].view(1, g, 1, 1, h)).reshape(M, k, Lc, h).float()
        vr = (torch.einsum("pgkld,ghd->pgklh", r, W[:, 0]) + b[:, 0].view(1, g, 1, 1, h)).reshape(M, k, Lc, h).float()
        Qc = F.pad(Q, (0, 0, 0, 0, 0, n * C - T))[:, C:]                   # read queries of chunks 1 .. n-1
        Qc = Qc.reshape(B, n - 1, C, g, h).permute(0, 1, 3, 2, 4).reshape(M, C, h).float()
        att = torch.einsum("mch,mkrh->mckr", Qc, kr) / math.sqrt(h) + logp.view(M, 1, k, 1)
        att = att.masked_fill(~valid.view(M, 1, k, 1), -1e4)
        att = att.reshape(M, C, k * Lc).softmax(-1).view(M, C, k, Lc)
        y = torch.einsum("mckr,mkrh->mch", att, vr) * valid.any(1).view(M, 1, 1)
        y = y.view(B, n - 1, g, C, h).permute(0, 1, 3, 2, 4).reshape(B, (n - 1) * C, g, h)
        y = F.pad(y, (0, 0, 0, 0, C, 0))[:, :T]                            # chunk 0 retrieves nothing
        # No output gate: an earlier version multiplied the read by silu(stream C) of the reading position. That gate
        # changes with the reading token, and it scrambled the retrieved value (a linear probe recovered it 68% of
        # the time before the gate and 2.5% after it), so retrieval never learned. Stream C is unused for now.
        return y.to(A.dtype)


def neighbour_loss(model, tokens, chunk, tau=10.0):
    """Self-supervised training for the model's own retrieval encoder (the inverse-cloze idea REALM used to start its
    retriever): each chunk of a training window should find the NEXT chunk of the same window among every next-chunk in
    the batch. Without it, the encoder learns only through the language-model loss, which on real text was too weak for
    the search to find related text (results/README.md, section 6). tokens (B, T); returns a scalar loss."""
    B, T = tokens.shape
    n = T // chunk
    if model.encoder is None or n < 2:
        return tokens.new_zeros((), dtype=torch.float32)
    keys = model.encoder(tokens[:, :n * chunk].reshape(B * n, chunk), model.emb)[1].view(B, n, -1)
    q = keys[:, :-1].reshape(-1, keys.shape[-1])                    # chunk i
    k = keys[:, 1:].reshape(-1, keys.shape[-1])                     # chunk i + 1 (its positive)
    logits = tau * q @ k.T
    return F.cross_entropy(logits.float(), torch.arange(len(q), device=tokens.device))
