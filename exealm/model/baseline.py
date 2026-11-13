"""The baseline every architecture change must beat: a plain Llama-style transformer (RoPE attention, SwiGLU MLP,
RMSNorm, tied embeddings) with the same width, depth and MLP as the TONN model, so the two have about the same number
of parameters. Same interface as ExeaLM: forward(idx, mem=None, targets=None) -> (logits, loss)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP, RMSNorm, rotate
from .delta import GatedDeltaNet


class Attention(nn.Module):
    """Causal multi-head attention with RoPE. Two options from Qwen: qk_norm (RMSNorm on each head's queries and keys,
    Qwen3) and gate (each output channel multiplied by a sigmoid of the input, "gated attention", Qiu et al. 2025,
    used in Qwen3-Next and Qwen3.5)."""

    def __init__(self, d, heads, gate=False, qk_norm=False):
        super().__init__()
        self.heads = heads
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=False) if gate else None
        self.qn, self.kn = (RMSNorm(d // heads), RMSNorm(d // heads)) if qk_norm else (None, None)

    def forward(self, x):
        B, T, d = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).view(B, T, 3, self.heads, d // self.heads).unbind(2)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        q, k = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T, d)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(h))
        return x + self.out(y)


def mixer(cfg, i, head_dim=64):
    """The sequence mixer of non-TONN layer i: Gated DeltaNet if i is in cfg.delta_layers, else attention."""
    if i in getattr(cfg, "delta_layers", ()):
        return GatedDeltaNet(cfg.d, head_dim)
    return Attention(cfg.d, max(1, cfg.d // head_dim), getattr(cfg, "attn_gate", False), getattr(cfg, "qk_norm", False))


class Llama(nn.Module):
    def __init__(self, cfg, head_dim=64):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d)
        nn.init.normal_(self.emb.weight, std=0.02)
        f = cfg.ffn_width or int(8 * cfg.d / 3) // 8 * 8
        self.attn = nn.ModuleList([mixer(cfg, i, head_dim) for i in range(cfg.n_layers)])
        self.mlps = nn.ModuleList([MLP(cfg.d, f) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.weight

    def forward(self, idx, mem=None, targets=None):
        x = self.emb(idx)
        for a, m in zip(self.attn, self.mlps):
            x = m(a(x))
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss
