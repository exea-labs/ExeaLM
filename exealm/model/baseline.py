"""The baseline every architecture change must beat: a plain Llama-style transformer (RoPE attention, SwiGLU MLP,
RMSNorm, tied embeddings) with the same width, depth and MLP as the TONN model, so the two have about the same number
of parameters. Same interface as ExeaLM: forward(idx, mem=None, targets=None) -> (logits, loss)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP, Canon, RMSNorm, rotate
from .delta import GatedDeltaNet


class Attention(nn.Module):
    """Causal multi-head attention with RoPE. Two options from Qwen: qk_norm (RMSNorm on each head's queries and keys,
    Qwen3) and gate (each output channel multiplied by a sigmoid of the input, "gated attention", Qiu et al. 2025,
    used in Qwen3-Next and Qwen3.5). Four more, each off unless asked for:
        value_residual  every attention layer after the first mixes in the first one's values, v = a v_first + b v,
                        a and b learned from 0.5 (ResFormer, arXiv 2410.17897)
        xsa             each head's output loses its component along the token's own value (exclusive self-attention,
                        arXiv 2603.09078)
        peri_ln         an RMSNorm on the output before it joins the residual stream (Peri-LN, arXiv 2502.02732)
        canon           a Canon layer on the normalized input (arXiv 2512.17351)"""

    def __init__(self, d, heads, gate=False, qk_norm=False, value_residual=False, xsa=False, peri_ln=False,
                 canon=False):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=False) if gate else None
        self.qn, self.kn = (RMSNorm(d // heads), RMSNorm(d // heads)) if qk_norm else (None, None)
        self.value_residual = value_residual
        self.vr = nn.Parameter(torch.full((2,), 0.5)) if value_residual else None   # dropped on the first layer
        self.xsa = xsa
        self.onorm = RMSNorm(d) if peri_ln else None
        self.canon = Canon(d) if canon else None

    def forward(self, x, ctx=None):
        """ctx: a dict shared by the model's attention layers in one forward pass (holds the first layer's values)."""
        B, T, d = x.shape
        h = self.norm(x)
        if self.canon is not None:
            h = self.canon(h)
        q, k, v = self.qkv(h).view(B, T, 3, self.heads, d // self.heads).unbind(2)
        if self.value_residual and ctx is not None:
            if "v_first" not in ctx:
                ctx["v_first"] = v
            elif self.vr is not None:
                v = self.vr[0] * ctx["v_first"] + self.vr[1] * v
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        stream = ctx.get("stream") if ctx is not None else None
        key = getattr(self, "stream_key", None)
        if stream is not None and key is not None:                      # reading a long text window by window: the
            prev = stream.get(key)                                      # keys and values of the last window too
            Tc = 0 if prev is None else prev[0].shape[1]
            ka, va = (k, v) if prev is None else (torch.cat([prev[0], k], 1), torch.cat([prev[1], v], 1))
            stream[key] = (k, v)
            seen = torch.arange(Tc + T, device=x.device).view(1, -1) <= torch.arange(T, device=x.device).view(-1, 1) + Tc
            o = F.scaled_dot_product_attention(rotate(q, offset=Tc).transpose(1, 2), rotate(ka).transpose(1, 2),
                                               va.transpose(1, 2), attn_mask=seen).transpose(1, 2)
        else:
            q, k = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True).transpose(1, 2)   # (B, T, H, hd)
        if self.xsa:
            vn = F.normalize(v, dim=-1)
            o = o - (o * vn).sum(-1, keepdim=True) * vn
        y = o.reshape(B, T, d)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(h))
        out = self.out(y)
        if self.onorm is not None:
            out = self.onorm(out)
        return x + out


def mixer(cfg, i, head_dim=64):
    """The sequence mixer of non-TONN layer i: Gated DeltaNet if i is in cfg.delta_layers, else attention."""
    if i in getattr(cfg, "delta_layers", ()):
        return GatedDeltaNet(cfg.d, head_dim)
    return Attention(cfg.d, max(1, cfg.d // head_dim), getattr(cfg, "attn_gate", False), getattr(cfg, "qk_norm", False),
                     getattr(cfg, "value_residual", False), getattr(cfg, "xsa", False), getattr(cfg, "peri_ln", False),
                     getattr(cfg, "canon", False))


def name_for_streaming(mixers, layers):
    """Give each attention layer the key it keeps its keys and values under when a text is read window by window."""
    for m, i in zip(mixers, layers):
        m.stream_key = ("attn", i)


def first_layer_supplies_values(mixers):
    """With value_residual the first attention layer only supplies the values the others mix in: drop its weights."""
    first = next((m for m in mixers if isinstance(m, Attention)), None)
    if first is not None and first.vr is not None:
        del first.vr
        first.vr = None


def mix(layer, x, ctx):
    """Run one sequence mixer (attention or Gated DeltaNet) with the context its layers share."""
    return layer(x, ctx)


def mlp_options(cfg):
    return dict(canon=getattr(cfg, "canon", False), peri_ln=getattr(cfg, "peri_ln", False))


class Llama(nn.Module):
    def __init__(self, cfg, head_dim=64):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        f = cfg.ffn_width or int(8 * cfg.d / 3) // 8 * 8
        self.attn = nn.ModuleList([mixer(cfg, i, head_dim) for i in range(cfg.n_layers)])
        first_layer_supplies_values(self.attn)
        name_for_streaming(self.attn, range(cfg.n_layers))
        self.mlps = nn.ModuleList([MLP(cfg.d, f, **mlp_options(cfg)) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight

    def forward(self, idx, mem=None, targets=None):
        x = self.emb(idx)
        ctx = {"stream": getattr(mem, "stream", None)}
        for a, m in zip(self.attn, self.mlps):
            x = m(mix(a, x, ctx))
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss
