"""AG: the agent-board circuit. Agents (copies of the model) talk to each other through a shared read-write board.

Once per chunk of `cfg.board_chunk` tokens (default: `cfg.chunk`), every AG slot POSTS one message to its channel of the board:
    value     = the chunk's average of stream A          (what the message says)
    key       = the chunk's average of stream B, unit length   (what it is about)
    strength  = sigmoid(w . average of stream C + b)      (how much the agent wants it read; the model can stay quiet)
and at every position it READS the channel with attention: query = stream B at that position, over the messages the
other agents of its swarm (and, if cfg.board_read_self, itself) posted in EARLIER chunks. A "nothing" entry is always available, so an
agent can choose to read nothing. The result is gated by silu(stream C).

Training: the rows of a batch are grouped into swarms of cfg.swarm_size consecutive rows. Each swarm has its own in-batch
board, so posting and reading both get gradients, and swarms never hear each other (docs/swarms.md). Inference and multi-agent post-training: a persistent AgentBoard
(exealm/memory/board.py) adds messages from agents that ran earlier or elsewhere, and new posts are written to it.

Every layer starts with one AG slot and the search can never remove the last one (exealm/model/search.py).

Measuring board use: set `record = True` on a BoardCircuit and each forward pass leaves a summary in `last` (see
usage_stats); pretrain.py does this on its validation batches and logs the result with every eval.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoardCircuit(nn.Module):
    """Runs every AG slot of one layer. Each AG slot is its own channel of the board."""

    def __init__(self, cfg):
        super().__init__()
        self.chunk, self.read_self = cfg.board_chunk or cfg.chunk, cfg.board_read_self
        self.post_w = nn.Parameter(torch.zeros(cfg.h))
        self.post_b = nn.Parameter(torch.zeros(()))
        self.log_tau = nn.Parameter(torch.tensor(math.log(10.0)))
        self.pool_w = nn.Parameter(torch.zeros(cfg.h)) if cfg.chunk_pool == "attn" else None   # which positions to post
        self.record = False                                               # if True, keep usage_stats(...) in self.last
        self.last = None

    def forward(self, S, mem, layer, slots):
        A, K, G = S                                                       # each (B, T, g, h)
        B, T, g, h = A.shape
        C = self.chunk
        n = -(-T // C)
        sw = mem.swarm_size if mem is not None else 1                 # agents per swarm
        if sw < 1 or B % sw:
            sw = 1
        ns = B // sw                                                    # swarms in this batch
        chunks = lambda u: F.pad(u, (0, 0, 0, 0, 0, n * C - T)).view(B, n, C, g, h).float()
        if self.pool_w is None:
            val = chunks(A).mean(2)                                       # (B, n, g, h) one post per chunk
            key = F.normalize(chunks(K).mean(2), dim=-1)
        else:                                                             # learned weights over the chunk, from stream C
            sc = chunks(G) @ self.pool_w.float()                          # (B, n, C, g)
            pad = (torch.arange(n * C, device=A.device) >= T).view(1, n, C, 1)
            a = sc.masked_fill(pad, float("-inf")).softmax(2).unsqueeze(-1)
            val = (a * chunks(A)).sum(2)
            key = F.normalize((a * chunks(K)).sum(2), dim=-1)
        log_w = F.logsigmoid(chunks(G).mean(2) @ self.post_w.float() + self.post_b.float())   # (B, n, g)
        q = F.normalize(K.float(), dim=-1)
        tau = self.log_tau.exp()

        # in-batch board: every swarm member's posts, visible from the next chunk on
        E = sw * n
        keyT, valT = key.view(ns, E, g, h), val.view(ns, E, g, h)
        lin = tau * torch.einsum("atsgh,aegh->atsge", q.view(ns, sw, T, g, h), keyT) \
            + log_w.view(ns, 1, 1, E, g).permute(0, 1, 2, 4, 3)
        e_chunk = torch.arange(n, device=A.device).repeat(sw)
        e_author = torch.arange(sw, device=A.device).repeat_interleave(n)
        r_chunk = torch.arange(T, device=A.device) // C
        vis = e_chunk.view(1, 1, 1, 1, E) < r_chunk.view(1, 1, T, 1, 1)
        if not self.read_self:
            vis = vis & (e_author.view(1, 1, 1, 1, E) != torch.arange(sw, device=A.device).view(1, sw, 1, 1, 1))
        lin = lin.masked_fill(~vis, float("-inf")).reshape(B, T, g, E)
        parts = [lin]

        # persistent board: messages from other agents that ran before (inference / multi-agent post-training)
        ext = None
        if mem is not None and mem.board is not None:
            ext = mem.board.read(layer, slots, A.device)
        if ext is not None:
            eK, eV, eW, eA, ok = ext                                      # (g, E2, h), (g, E2, h), (g, E2) x3
            lex = tau * torch.einsum("btgh,geh->btge", q, eK) + eW.view(1, 1, g, -1)
            seen = ok.view(1, g, -1).expand(B, g, -1)
            if mem.agent_ids is not None:                                 # an agent's own old posts come from context
                seen = seen & (eA.view(1, g, -1) != mem.agent_ids.view(B, 1, 1))
            parts.append(lex.masked_fill(~seen.view(B, 1, g, -1), float("-inf")))
        parts.append(lin.new_zeros(B, T, g, 1))                          # "read nothing"
        att = torch.cat(parts, -1).softmax(-1)
        if self.record:
            self.last = usage_stats(att, log_w, E, sw, n, C)
        y = torch.einsum("atsge,aegh->atsgh", att[..., :E].reshape(ns, sw, T, g, E), valT).reshape(B, T, g, h)
        if ext is not None:
            y = y + torch.einsum("btge,geh->btgh", att[..., E:-1], eV)

        if mem is not None and mem.board is not None and mem.write_board:
            authors = mem.agent_ids if mem.agent_ids is not None else torch.arange(B)
            mem.board.post(layer, slots, key, val, log_w, authors, n_complete=T // C)
        return y.to(A.dtype) * F.silu(G)


@torch.no_grad()
def usage_stats(att, log_w, E, sw, n, C):
    """How a layer's AG slots used the board in one forward pass. att (B, T, g, E + E2 + 1): attention over the E
    in-batch posts, E2 persistent-board posts and "nothing"; log_w (B, n, g) log post strengths. Reading is averaged
    over positions from the second chunk on (before that no post is visible yet), so the shares add up to 1:
        read_nothing   attention on the "read nothing" entry
        read_others    attention on posts by the other agents of the swarm
        read_self      attention on the agent's own earlier posts (0 unless cfg.board_read_self)
        read_board     attention on the persistent board (0 in pretraining)
    and for posting:
        post_strength  mean sigmoid write strength of the posts
        quiet_posts    share of posts with strength below 0.1 (the agent chose to stay quiet)
    Returns None when the text is too short for anyone to read a post."""
    B, T, g, _ = att.shape
    if T <= C:
        return None
    a = att[:, C:].float()                                            # (B, T - C, g, ...)
    inb = a[..., :E].reshape(B // sw, sw, T - C, g, E)
    own = (torch.arange(sw, device=att.device).repeat_interleave(n).view(1, E)
           == torch.arange(sw, device=att.device).view(sw, 1))       # (sw, E): which posts are the row's own
    self_mass = (inb * own.view(1, sw, 1, 1, E)).sum(-1)
    w = log_w.float().exp()
    return dict(read_nothing=float(a[..., -1].mean()), read_others=float((inb.sum(-1) - self_mass).mean()),
                read_self=float(self_mass.mean()), read_board=float(a[..., E:-1].sum(-1).mean()),
                post_strength=float(w.mean()), quiet_posts=float((w < 0.1).float().mean()))
