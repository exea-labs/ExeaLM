"""ExeaLM: a TONN language model with retrieval (RG) and agent-board (AG) circuits built into every layer.

    tokens -> embedding -> [TONN layer -> gated MLP] x n_layers -> norm -> next-token logits

A TONN layer reads the residual stream into `slots` small circuits through ONE input projection (3 streams of h
channels per slot), runs each circuit, scales every slot's output to unit size, and writes back through ONE output
projection. Which circuit sits in which slot is not fixed by hand: the search (search.py) measures what each kind of
circuit is worth and replaces the useless ones while the model trains.
"""
import math
import random
from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..memory import MemoryContext
from . import circuits as Cc
from .baseline import Attention, first_layer_supplies_values, mix, mixer, mlp_options, name_for_streaming  # noqa: F401
from .alter import WeightAlter, fold_edits
from .board import BoardCircuit
from .common import MLP, RMSNorm
from .retrieval import ChunkEncoder, RetrievalCircuit


# --------------------------------------------------------------------------------------------- choosing circuits
def rand_spec(rng, cfg, allowed=None, weights=None):
    """A random circuit: type (uniform, or in proportion to `weights`), lag and starting decay."""
    allowed = list(allowed or cfg.menu)
    if weights:
        w = [max(0.0, weights.get(t, 0.0)) for t in allowed]
        t = rng.choices(allowed, w)[0] if sum(w) > 0 else rng.choice(allowed)
    else:
        t = rng.choice(allowed)
    return dict(tmpl=t, lag=rng.choice(cfg.lags), lam0=rng.choice([0.2, 0.5, 0.9, -0.5]))


def initial_specs(cfg, seed=None):
    """Every layer starts with its pinned circuits (one RG and one AG by default); the other slots are random draws
    from the rest of the menu. The search may later add more RG or AG slots, never fewer than the pinned number."""
    rng = random.Random(cfg.seed if seed is None else seed)
    fill = [t for t in cfg.menu if t not in cfg.pinned]
    out = []
    for _ in range(cfg.n_layers - len(cfg.attn_layers)):
        row = [dict(tmpl=t, lag=0, lam0=0.5) for t, c in cfg.pinned.items() for _ in range(c)]
        if len(row) > cfg.slots:
            raise ValueError("more pinned circuits than slots")
        w = {t: cfg.init_weights.get(t, 1.0) for t in fill} if cfg.init_weights else None   # unlisted types: 1
        row += [rand_spec(rng, cfg, fill, w) for _ in range(cfg.slots - len(row))]
        out.append(row)
    return out


def slot_cost(spec, cfg, T):
    """Rough multiply-adds per token of one slot (its share of the projections plus its circuit). The search compares
    worth per cost, so a cheap circuit that helps a little can beat an expensive one that helps a little more."""
    t, d, h = spec["tmpl"], cfg.d, cfg.h
    proj = 3 * d * h + h * d
    if t == "RP":
        return proj + 2 * h * T
    if t == "GD":
        # Per token: chunk dot products and triangular solves, plus three products with the matrix state.
        return proj + 4 * min(cfg.chunk, T) * h + 3 * h * h + 6 * h
    if t == "RG":
        e, k, Lc = cfg.enc_dim, cfg.rag_topk, cfg.rag_chunk_len
        return proj + k * Lc * (d * e + 4 * e * e + 2 * e * h) / cfg.chunk + 2 * k * Lc * h
    if t == "AG":
        return proj + 2 * h * cfg.swarm_size * T / (cfg.board_chunk or cfg.chunk)
    if t == "WA":                                    # its share of applying the changes in the target MLP
        return proj + (cfg.ffn_width + d) * cfg.swarm_size * T / (cfg.board_chunk or cfg.chunk)
    return proj + 6 * h


# --------------------------------------------------------------------------------------------- the TONN layer
class TONNLayer(nn.Module):
    def __init__(self, cfg, specs, index):
        super().__init__()
        d, K, h = cfg.d, len(specs), cfg.h
        self.cfg, self.index, self.K, self.h = cfg, index, K, h
        self.norm = RMSNorm(d)
        self.inp = nn.Linear(d, K * 3 * h)
        self.out = nn.Linear(K * h, d, bias=False)
        self.onorm = RMSNorm(d) if getattr(cfg, "peri_ln", False) else None   # Peri-LN, as in the attention layers
        self.lam = nn.Parameter(torch.zeros(K, h))        # decays or GD gate direction, through tanh(3 lam)
        self.beta = nn.Parameter(torch.ones(K))           # attention sharpness or GD write bias
        self.register_buffer("mask", torch.ones(K))       # 0 switches a slot off (worth measurement)
        self.register_buffer("scale", torch.zeros(K))     # running output size per slot (0: not measured yet)
        self.rg = RetrievalCircuit(cfg)
        self.ag = BoardCircuit(cfg)
        # the weight-alter circuit exists only when the config can use it, so older checkpoints load unchanged
        self.wa = WeightAlter(cfg) if ("WA" in cfg.menu or "WA" in cfg.pinned) else None
        nn.init.normal_(self.out.weight, std=0.5 / math.sqrt(K * h))
        self.specs = [None] * K
        for k, s in enumerate(specs):
            self.set_spec(k, s, reinit=False)

    def set_spec(self, k, spec, reinit=True, out_scale=0.1):
        """Put a circuit in slot k. reinit: fresh input rows and small output columns for this slot only."""
        self.specs[k] = dict(spec)
        with torch.no_grad():
            self.scale[k] = 0.0
            self.lam[k].fill_(math.atanh(max(-0.97, min(0.97, spec["lam0"]))) / 3)
            self.beta[k].fill_(1.0)
            if reinit:
                r = slice(k * 3 * self.h, (k + 1) * 3 * self.h)
                nn.init.kaiming_uniform_(self.inp.weight[r], a=math.sqrt(5))
                self.inp.bias[r].zero_()
                c = slice(k * self.h, (k + 1) * self.h)
                nn.init.normal_(self.out.weight[:, c], std=out_scale / math.sqrt(self.K * self.h))

    def plan(self, device):
        """Slots sorted by circuit type: the order, each type's range in it, and the lags in that order. Cached per
        architecture (the key changes whenever a slot gets a new circuit)."""
        key = (tuple((sp["tmpl"], sp["lag"]) for sp in self.specs), str(device))
        if getattr(self, "_plan_key", None) != key:
            order = sorted(range(self.K), key=lambda k: self.specs[k]["tmpl"])
            ranges, start = {}, 0
            for t in sorted({sp["tmpl"] for sp in self.specs}):
                n = sum(1 for sp in self.specs if sp["tmpl"] == t)
                ranges[t] = (start, start + n)
                start += n
            running = {t: bounds for t, bounds in ranges.items() if t in Cc.RUNNING_SUM_TYPES}
            self._plan = dict(order=torch.tensor(order, device=device), slots=order, ranges=ranges, running=running,
                              lags=torch.tensor([self.specs[k]["lag"] for k in order], device=device))
            self._plan_key = key
        return self._plan

    def forward(self, x, mem=None):
        """Slots are computed grouped by type. Instead of gathering each group's inputs and putting the outputs back in
        slot order (two copies of large activations every step), the small weight matrices are permuted into type
        order once per step."""
        B, T, _ = x.shape
        K, h = self.K, self.h
        pl = self.plan(x.device)
        o = pl["order"]
        W = self.inp.weight.view(K, 3 * h, -1).index_select(0, o).reshape(K * 3 * h, -1)
        b = self.inp.bias.view(K, 3 * h).index_select(0, o).reshape(-1)
        u = F.linear(self.norm(x), W, b).view(B, T, K, 3, h)                 # slots in type order
        lam = torch.tanh(self.lam * 3).index_select(0, o)
        beta = self.beta.index_select(0, o)
        stream = getattr(mem, "stream", None) if mem is not None else None
        P, prev, new = 0, None, {}
        if stream is not None:                    # one window of a longer text: lags also read the last P positions
            P = 2 * max(self.cfg.lags)            # of the window before, and recurrences start from where it ended
            prev = stream.get(("tonn", self.index))
            tail = prev["tail"].to(u.dtype) if prev is not None else u.new_zeros(B, P, K, 3, h)
            u = torch.cat([tail, u], 1)
        running, carry = {}, None
        if pl["running"] and stream is None:
            running = Cc.run_running_sums(u, pl["lags"], lam, pl["running"])
        elif pl["running"]:
            running, carry = Cc.run_running_sums(u, pl["lags"], lam, pl["running"], tail=P,
                                                 s0=None if prev is None else prev["run"], return_state=True)
        outs = []
        for t, (a, c) in pl["ranges"].items():
            if t in running:
                outs.append(running[t])
                continue
            S = [u[:, :, a:c, j] for j in range(3)]
            slots = pl["slots"][a:c]                                          # original slot numbers
            if t in ("RG", "AG", "WA"):
                S = [z[:, P:] for z in S]
            if t == "RG":
                y = self.rg(S, mem, self, slots)
            elif t == "AG":
                y = self.ag(S, mem, self.index, slots)
            elif t == "WA":
                y = self.wa(S, mem, self, slots)
            elif stream is not None:
                y, new[t] = Cc.run_stream(t, S, pl["lags"][a:c], lam[a:c], beta[a:c], chunk=self.cfg.chunk, tail=P,
                                          state=None if prev is None else prev.get(t))
            else:
                y = Cc.run(t, S, pl["lags"][a:c], lam[a:c], beta[a:c], chunk=self.cfg.chunk)
            outs.append(y)
        if stream is not None:
            stream[("tonn", self.index)] = dict(tail=u[:, -P:], run=carry, **new)
        y = torch.cat(outs, 2) if len(outs) > 1 else outs[0]
        y = self._unit_scale(y, o) * self.mask.index_select(0, o).view(1, 1, -1, 1)
        Wo = self.out.weight.view(-1, K, h).index_select(1, o).reshape(-1, K * h)
        out = F.linear(y.reshape(B, T, K * h), Wo)
        return x + (out if self.onorm is None else self.onorm(out))

    def _unit_scale(self, y, o):
        """Divide each slot by the size its output had on PAST batches (a running average), so a prediction never
        depends on other rows of the same batch and no slot drowns out the others. y is in type order o."""
        with torch.no_grad():
            rms = y.detach().float().pow(2).mean(dim=(0, 1, 3)).sqrt()
            sc = self.scale.index_select(0, o)
            # A new/replaced slot has no past measurement. Using this batch's
            # RMS would leak future tokens and unrelated rows into its outputs.
            used = torch.where(sc > 0, sc, torch.ones_like(sc))
            if self.training:
                self.scale.index_copy_(0, o, torch.where(sc > 0, 0.99 * sc + 0.01 * rms, rms))
        return y / (used.view(1, 1, -1, 1).to(y.dtype) + 1e-3)


# --------------------------------------------------------------------------------------------- the model
class ExeaLM(nn.Module):
    def __init__(self, cfg: ModelConfig, specs=None):
        super().__init__()
        self.cfg = cfg
        specs = specs or initial_specs(cfg)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList([TONNLayer(cfg, specs[i], i) for i in range(cfg.n_layers - len(cfg.attn_layers))])
        self.attn = nn.ModuleList([mixer(cfg, i) for i in cfg.attn_layers])          # attention or Gated DeltaNet
        first_layer_supplies_values(self.attn)
        name_for_streaming(self.attn, cfg.attn_layers)
        f = cfg.ffn_width
        self.mlps = nn.ModuleList([MLP(cfg.d, f, **mlp_options(cfg)) for _ in range(cfg.n_layers)]) if f else None
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight
        uses_rg = "RG" in cfg.menu or any(s["tmpl"] == "RG" for row in specs for s in row)
        self.encoder = ChunkEncoder(cfg) if uses_rg else None

    @property
    def specs(self):
        return [[dict(s) for s in lay.specs] for lay in self.layers]

    def forward(self, idx, mem=None, targets=None):
        """idx (B, T) token ids. Returns (logits (B, T, V), loss or None)."""
        if mem is None:
            mem = MemoryContext()
        mem = replace(mem, encoder=self.encoder, emb=self.emb, tokens=idx, cache={})
        x = self.emb(idx)
        ti = ai = 0
        ctx = {"stream": mem.stream}                          # shared by the attention layers
        for i in range(self.cfg.n_layers):
            if i in self.cfg.attn_layers:                     # a hybrid model's full attention block
                x = mix(self.attn[ai], x, ctx)
                ai += 1
            else:
                mem.cache["layer"] = i
                x = self.layers[ti](x, mem)
                ti += 1
            if self.mlps is not None:
                edits = mem.cache.get("wa", {}).get(i)
                persist = getattr(self.cfg, "wa_persist", 0.0) and mem.stream is not None
                carried = mem.stream.get(("wa", i)) if persist else None
                x = self.mlps[i](x, edits, carried)
                if persist and (edits or carried):            # this window's changes join the carried one
                    mem.stream[("wa", i)] = fold_edits(carried, edits or [], self.cfg.wa_persist)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def architecture(self):
        """How many slots of each circuit type each layer holds."""
        rows = []
        for lay in self.layers:
            c = {}
            for s in lay.specs:
                c[s["tmpl"]] = c.get(s["tmpl"], 0) + 1
            rows.append(c)
        return rows

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, mem=None):
        """Plain sampling. It re-reads the whole context for every new token, which is slow.
        TODO(speed): most TONN circuits are recurrences, so generation can carry a small state per slot and cost the
        same for every new token (docs/ideas.md)."""
        for _ in range(max_new_tokens):
            logits = self(idx, mem)[0][:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = logits.topk(top_k)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], 1)
            if mem is not None and mem.doc_ids is not None:
                mem = replace(mem, doc_ids=torch.cat([mem.doc_ids, mem.doc_ids[:, -1:]], 1))
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())
