"""Building blocks shared by TONN and the baseline transformer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Canon(nn.Module):
    """Canon layer (Allen-Zhu, Physics of Language Models part 4.1, arXiv 2512.17351): h plus a causal depthwise
    convolution of h over the last k positions. The convolution starts at zero, so a model starts as it would without."""

    def __init__(self, d, k=4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, h):
        T = h.shape[1]
        return h + self.conv(h.transpose(1, 2))[..., :T].transpose(1, 2)


class MLP(nn.Module):
    """Gated MLP (SwiGLU) with its own norm and residual: x + W_o(silu(a) * b), [a, b] = W_i norm(x).

    In our quick tests TONN layers alone tied a same-size transformer on text; with this MLP after every TONN layer
    TONN won, and the MLP was the part whose removal cost the most.
    canon: a Canon layer on the normalized input. peri_ln: an RMSNorm on the output before it joins the residual
    stream (Peri-LN, arXiv 2502.02732)."""

    def __init__(self, d, hidden, canon=False, peri_ln=False):
        super().__init__()
        self.norm = RMSNorm(d)
        self.wi = nn.Linear(d, 2 * hidden, bias=False)
        self.wo = nn.Linear(hidden, d, bias=False)
        self.canon = Canon(d) if canon else None
        self.onorm = RMSNorm(d) if peri_ln else None

    def forward(self, x, edits=None, carried=None):
        """edits: weight changes to W_o from weight-alter circuits (exealm/model/alter.py), or None. carried: the
        change carried from earlier windows of the same text, {"in": (B, 2f, d), "out": (B, d, f)}, used at every
        position (docs/brain.md)."""
        xn = self.norm(x)
        if self.canon is not None:
            xn = self.canon(xn)
        pre = self.wi(xn)
        if carried and "in" in carried:
            pre = pre + torch.einsum("btd,bkd->btk", xn.float(), carried["in"]).to(pre.dtype)
        if edits:
            from .alter import apply_edits
            to_in = [e for e in edits if e["matrix"] == "in"]
            if to_in:
                pre = pre + apply_edits(xn, to_in)
        a, b = pre.chunk(2, -1)
        hidden = F.silu(a) * b
        out = self.wo(hidden)
        if edits:
            to_out = [e for e in edits if e["matrix"] == "out"]
            if to_out:
                out = out + apply_edits(hidden, to_out)
        if carried and "out" in carried:
            out = out + torch.einsum("btf,bdf->btd", hidden.float(), carried["out"]).to(out.dtype)
        if self.onorm is not None:
            out = self.onorm(out)
        return x + out


def rotate(x, base=10000.0, offset=0):
    """Rotary position embedding (RoPE) on the last dim of x (B, T, g, h): channel i is paired with channel i + h/2 and
    each pair is rotated by an angle proportional to its position, so a query-key dot product depends only on how far
    apart the two positions are. (Pairing halves rather than neighbours avoids a stack and reshape: same function
    class, faster.) offset: the position of the first row, when x continues an earlier window."""
    B, T, g, h = x.shape
    half = h // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32).view(T, 1) * freq.view(1, half)
    cos, sin = ang.cos().view(1, T, 1, half).to(x.dtype), ang.sin().view(1, T, 1, half).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
