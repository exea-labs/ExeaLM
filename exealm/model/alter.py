"""WA: the weight-alter circuit. A slot that changes a weight matrix of the model while it reads, for itself or for
the other agents of its swarm.

Once per chunk of `cfg.board_chunk` tokens (default: `cfg.chunk`), every WA slot emits one rank-1 change to the TARGET
weight matrix, the output matrix W_o of a gated MLP (hidden width f -> stream width d):

    delta W  =  strength * u v^T
    u (d)      = U . chunk summary of stream A          what to write (the output side)
    v (f)      = unit( V . chunk summary of stream B )  which inputs it applies to (the input side)
    (each summary is a learned weighting of the chunk's positions that starts as the plain average)
    strength   = gain * sigmoid(w . chunk average of stream C + b)      the slot can stay quiet

A slot's writes are corrected so they do not disturb each other: after any number of writes, the changed matrix
returns exactly u at key v for every write so far (exact_writes below).

The change takes effect from the next chunk on (never inside the chunk that produced it, so nothing sees the future)
and lasts for the rest of the pass: at a later position the target computes (W_o + sum of the changes so far) . hidden.
`cfg.wa_scope` says whose copy of the weights is changed: "self" (the agent's own), "others" (the other agents of its
swarm) or "all". Each agent of a swarm therefore runs with its own, different weights in that one matrix.
`cfg.wa_target` picks the layer whose MLP is changed: -1 (default) is the MLP right after the emitting TONN layer, or
a layer number at or after it (a change to an earlier layer could not take effect in the same pass).

Everything is differentiable, so what to write is learned from the ordinary loss: an agent learns to edit a
teammate's weights from the teammate's errors. The slot adds nothing to the residual stream itself; its whole effect
is the weight change. Switching the slot off (the search's worth measurement) removes its changes.

What this is, plainly: the changes live for one pass, like the in-batch board. They are fast weights (Schmidhuber
1992; Ba et al. 2016) written by a learned rule, a small hypernetwork (Ha et al. 2016) inside the layer. Permanent
edits to the stored parameters, across training steps, are not implemented.

WA is not in the default menu: add it with `menu` or `pinned` in the config (old checkpoints load unchanged).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightAlter(nn.Module):
    """Runs every WA slot of one TONN layer. Each slot emits its own changes."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.wa_scope not in ("self", "others", "all"):
            raise ValueError("wa_scope must be 'self', 'others' or 'all'")
        if not cfg.ffn_width:
            raise ValueError("the weight-alter circuit edits an MLP: the model needs one (ffn != 0)")
        self.chunk, self.scope, self.target = cfg.board_chunk or cfg.chunk, cfg.wa_scope, cfg.wa_target
        self.persist = getattr(cfg, "wa_persist", 0.0)
        if self.persist and cfg.swarm_size != 1:
            raise ValueError("wa_persist carries one row's weight change to its next window: it needs swarm_size 1")
        if cfg.wa_matrix not in ("out", "in"):
            raise ValueError("wa_matrix must be 'out' or 'in'")
        self.matrix = cfg.wa_matrix
        if cfg.wa_key not in ("learned", "hidden"):
            raise ValueError("wa_key must be 'learned' or 'hidden'")
        # "learned": the key v is a learned linear map of the slot's stream. "hidden": the key is what the target
        # matrix itself receives at the picked positions of the writer's chunk (the MLP's hidden activation for
        # "out", its normed input for "in"), so a reader that meets the same token produces the same key without
        # having to learn the match (the key used by ROME, Meng et al. 2022)
        self.key = cfg.wa_key
        # "out": the MLP's output matrix W_o (hidden f -> stream d). "in": its input matrix W_i (stream d -> 2f), so
        # the change passes through the MLP's gate: a key that matches twice as well opens it more than twice as far
        rows, cols = (cfg.d, cfg.ffn_width) if self.matrix == "out" else (2 * cfg.ffn_width, cfg.d)
        self.to_u = nn.Linear(cfg.h, rows, bias=False)
        self.to_v = nn.Linear(cfg.h, cols, bias=False)
        self.gate_w = nn.Parameter(torch.zeros(cfg.h))
        self.gate_b = nn.Parameter(torch.zeros(()))
        self.log_gain = nn.Parameter(torch.zeros(()))
        # which positions of the chunk the value (u) and the key (v) are read from: a learned softmax over the chunk,
        # scored from stream C. Zero start = the plain average.
        self.pick_u = nn.Parameter(torch.zeros(cfg.h))
        self.pick_v = nn.Parameter(torch.zeros(cfg.h))
        nn.init.normal_(self.to_u.weight, std=0.02)                       # start close to "change nothing"

    def forward(self, S, mem, layer, slots):
        A, K, G = S                                                       # each (B, T, g, h)
        B, T, g, h = A.shape
        C = self.chunk
        n = -(-T // C)
        sw = mem.swarm_size if mem is not None else 1
        if sw < 1 or B % sw:
            sw = 1
        chunks = lambda x: F.pad(x, (0, 0, 0, 0, 0, n * C - T)).view(B, n, C, g, h).float()
        pool = lambda x: chunks(x).mean(2)                                                        # (B, n, g, h)
        pad = (torch.arange(n * C, device=A.device) >= T).view(1, n, C, 1)

        def weights(w):                                    # a learned weighting of the chunk's positions (B, n, C, g)
            return (chunks(G) @ w.float()).masked_fill(pad, float("-inf")).softmax(2)

        def picked(x, w):
            return (weights(w).unsqueeze(-1) * chunks(x)).sum(2)
        on = layer.mask[torch.as_tensor(slots, device=A.device)].float()                         # 0: switched off
        strength = self.log_gain.exp() * torch.sigmoid(pool(G) @ self.gate_w.float() + self.gate_b.float()) * on
        u = self.to_u(picked(A, self.pick_u)) * strength.unsqueeze(-1)    # (B, n, g, d)
        v = F.normalize(self.to_v(picked(K, self.pick_v)), dim=-1)        # (B, n, g, f)
        raw = (u, v)
        here = mem.cache["layer"]
        target = here if self.target < 0 else self.target
        if target < here:
            raise ValueError(f"wa_target {target} is before the emitting layer {here}")
        stream = getattr(mem, "stream", None)
        prior = (stream.get(("wa", target), {}).get(self.matrix)
                 if (self.persist and stream is not None) else None)     # the change carried from earlier windows
        extra = dict(prior=prior)
        if self.key == "hidden":                          # the key is filled in by the target MLP (apply_edits)
            extra.update(key="hidden", pick=weights(self.pick_v))
        else:
            u, v = exact_writes(correct_for_carried(u, v, prior), v)
        mem.cache.setdefault("wa", {}).setdefault(target, []).append(
            dict(u=u, v=v, raw_u=raw[0], raw_v=raw[1], matrix=self.matrix, chunk=C, swarm=sw, scope=self.scope, complete=T // C, **extra))
        return A.new_zeros(B, T, g, h)


def correct_for_carried(u, v, prior):
    """The delta rule across windows: a write of value u at key v is corrected for what the carried change already
    returns at v, so writing a key again replaces what it held instead of adding to it. prior (B, rows, cols) or None."""
    if prior is None:
        return u
    return u - torch.einsum("brc,bngc->bngr", prior.float(), v.float()).to(u.dtype)


def fold_edits(carried, edits, keep):
    """The weight change to carry into the next window: the one carried into this window plus every change written in
    it, times keep. carried: {matrix: (B, rows, cols)} or None; returns the same for "in" and "out"."""
    out = {}
    for matrix in ("in", "out"):
        total = None if carried is None else carried.get(matrix)
        for e in edits:
            if e["matrix"] == matrix:
                d = torch.einsum("bngr,bngc->brc", e["u"].float(), e["v"].float())
                total = d if total is None else total + d
        if total is not None:
            out[matrix] = keep * total
    return out


def exact_writes(u, v, floor=0.1):
    """Turn each slot's sequence of writes (value u_e at key v_e) into rank-1 changes that do not disturb each other.

    Adding u_e v_e^T as it is would also change what every earlier, similar key returns (crosstalk). Instead each
    key is first made orthogonal to the slot's earlier keys, and the value is corrected for what the matrix already
    returns at that key. After any number of writes, the changed matrix returns exactly u_e at key v_e for every
    write so far (recursive least squares; the exact form of the delta rule). A key that repeats an earlier one
    (less than `floor` of it is new) is only partly rewritten, to keep the change bounded.
    u (B, n, g, d), v (B, n, g, f) unit length. Returns the corrected (u, v), same shapes."""
    us, vs = [], []
    for e in range(u.shape[1]):
        ue, ve = u[:, e], v[:, e]
        if vs:
            Vp, Up = torch.stack(vs, 2), torch.stack(us, 2)               # (B, g, e, f), (B, g, e, d)
            c = torch.einsum("bgef,bgf->bge", Vp, ve)                     # overlap with each earlier key
            ve = ve - torch.einsum("bge,bgef->bgf", c, Vp)
            ue = ue - torch.einsum("bge,bged->bgd", c, Up)                # what the matrix returns there already
        r = ve.norm(dim=-1, keepdim=True)
        vs.append(ve / r.clamp_min(1e-6))
        us.append(ue / r.clamp_min(floor))
    return torch.stack(us, 1), torch.stack(vs, 1)


def visible(edit, T, device):
    """(sw, T, E) bool: may the reader (agent, position) use change e? E = sw * n * g, ordered author, chunk, slot."""
    B, n, g, _ = edit["u"].shape
    sw, C = edit["swarm"], edit["chunk"]
    author = torch.arange(sw, device=device).repeat_interleave(n * g)
    chunk = torch.arange(n, device=device).repeat_interleave(g).repeat(sw)
    vis = chunk.view(1, 1, -1) < (torch.arange(T, device=device) // C).view(1, T, 1)             # earlier chunks only
    same = author.view(1, 1, -1) == torch.arange(sw, device=device).view(sw, 1, 1)
    if edit["scope"] == "self":
        return vis & same
    if edit["scope"] == "others":
        return vis & ~same
    return vis.expand(sw, T, -1)


def apply_edits(hidden, edits):
    """What the changed weights add to the target's output: sum over visible changes of u (v . hidden).
    hidden (B, T, f). Returns (B, T, d)."""
    B, T, f = hidden.shape
    out = 0.0
    for e in edits:
        sw = e["swarm"]
        ns = B // sw
        if e.get("key") == "hidden" and "done" not in e:   # keys from what this matrix receives in the writer's chunk
            a = e["pick"]                                                  # (B, n, C, g)
            n, C = a.shape[1], a.shape[2]
            hc = F.pad(hidden.float(), (0, 0, 0, n * C - T)).view(B, n, C, f)
            v = F.normalize(torch.einsum("bncg,bncf->bngf", a, hc), dim=-1)
            e["raw_v"] = v
            e["u"], e["v"] = exact_writes(correct_for_carried(e["raw_u"], v, e.get("prior")), v)
            e["done"] = True
        u = e["u"].reshape(ns, -1, e["u"].shape[-1])                      # (ns, E, d)
        v = e["v"].reshape(ns, -1, f)                                     # (ns, E, f)
        coef = torch.einsum("astf,aef->aste", hidden.float().view(ns, sw, T, f), v)
        coef = coef * visible(e, T, hidden.device).unsqueeze(0)
        out = out + torch.einsum("aste,aed->astd", coef, u).reshape(B, T, -1)
    return out.to(hidden.dtype)


def weight_delta(edit, row, position):
    """The literal change (d, f) to the target matrix that agent `row` runs with at `position`: for inspection and
    tests. Equals what apply_edits uses."""
    B, n, g, d = edit["u"].shape
    sw = edit["swarm"]
    s, r = divmod(row, sw)
    u = edit["u"].reshape(B // sw, -1, d)[s]
    v = edit["v"].reshape(B // sw, -1, edit["v"].shape[-1])[s]
    vis = visible(edit, position + 1, u.device)[r, position].to(u.dtype)
    return torch.einsum("e,ed,ef->df", vis, u, v)
