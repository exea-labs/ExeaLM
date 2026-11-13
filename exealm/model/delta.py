"""Gated DeltaNet, the linear-attention layer of Qwen3.5 (and Qwen3-Next), in plain PyTorch.

Each head keeps a matrix memory S (head_dim x head_dim). At every token it forgets a little (gate alpha_t), erases what
it stored under the current key, and writes the new value there (strength beta_t):

    S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T          o_t = S_t q_t

With alpha = 1 this is the delta rule (Schlag et al. 2021; Yang et al. 2024, "Parallelizing Linear Transformers with
the Delta Rule over Sequence Length"); the gate is from Yang, Kautz and Hatamizadeh, "Gated Delta Networks" (2025).
The fast kernels (flash-linear-attention) are Triton and need an NVIDIA GPU; `chunk_gated_delta` below computes the
same thing chunk by chunk with batched matrix products, which runs anywhere, and `gated_delta_reference` is the
token-by-token recurrence it is tested against (tests/test_delta.py).

The layer around it follows Qwen3-Next: q, k, v from one projection, a short causal depthwise convolution and silu on
each, L2-normalised q and k, per-head gates alpha (a learned decay rate times softplus) and beta (sigmoid), and an
output that is RMS-normalised per head and gated by silu of another projection of the input.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RMSNorm


def gated_delta_reference(q, k, v, g, beta):
    """The recurrence, one token at a time. q, k (B, H, T, Dk); v (B, H, T, Dv); g (B, H, T) log alpha <= 0;
    beta (B, H, T). Returns o (B, H, T, Dv). The state is kept as P = S^T (Dk x Dv), so o_t = q_t^T P_t."""
    B, H, T, Dk = q.shape
    P = q.new_zeros(B, H, Dk, v.shape[-1])
    out = []
    for t in range(T):
        a, b = g[:, :, t].exp()[..., None, None], beta[:, :, t][..., None, None]
        kt = k[:, :, t].unsqueeze(-1)                                       # (B, H, Dk, 1)
        P = a * (P - b * kt @ (kt.transpose(-1, -2) @ P)) + b * kt @ v[:, :, t].unsqueeze(-2)
        out.append((q[:, :, t].unsqueeze(-2) @ P).squeeze(-2))
    return torch.stack(out, 2)


def _unit_lower_solve(L, R):
    """Solve L X = R for unit lower-triangular L (..., C, C)."""
    try:
        return torch.linalg.solve_triangular(L, R, upper=False, unitriangular=True)
    except (NotImplementedError, RuntimeError):                           # a backend without it: forward substitution
        X = torch.zeros_like(R)
        for t in range(L.shape[-1]):
            X[..., t, :] = R[..., t, :] - (L[..., t, :t].unsqueeze(-2) @ X[..., :t, :]).squeeze(-2)
        return X


def chunk_gated_delta(q, k, v, g, beta, chunk=64):
    """The same recurrence, chunk by chunk. Inside a chunk, with gamma_t the product of the gates so far, the update
    values u_t solve a unit lower-triangular system, so a chunk costs a few (chunk x chunk) matrix products and one
    triangular solve; only the chunks run in sequence. Shapes as gated_delta_reference."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    dtype = v.dtype
    pad = (-T) % chunk
    if pad:                                                             # padding: k = 0, beta = 0 leaves S unchanged
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    N, C = (T + pad) // chunk, chunk
    q, k, v = (t.float().reshape(B, H, N, C, -1) for t in (q, k, v))
    g, beta = g.float().reshape(B, H, N, C), beta.float().reshape(B, H, N, C)
    lg = g.cumsum(-1)                                                   # log gamma_t inside each chunk
    diff = lg.unsqueeze(-1) - lg.unsqueeze(-2)                          # log(gamma_t / gamma_j), row t, column j
    ones = torch.ones(C, C, dtype=torch.bool, device=q.device)
    strict, incl = torch.tril(ones, -1), torch.tril(ones)
    neg = torch.tensor(float("-inf"), device=q.device)
    A = beta.unsqueeze(-1) * torch.where(strict, diff, neg).exp() * (k @ k.transpose(-1, -2))
    L = A + torch.eye(C, device=q.device)
    U0 = _unit_lower_solve(L, beta.unsqueeze(-1) * v)                   # (I + A)^-1 diag(beta) V
    W = _unit_lower_solve(L, (beta * lg.exp()).unsqueeze(-1) * k)       # (I + A)^-1 diag(beta gamma) K
    QK = (q @ k.transpose(-1, -2)) * torch.where(incl, diff, neg).exp()
    P = q.new_zeros(B, H, Dk, Dv)
    outs = []
    for n in range(N):
        U = U0[:, :, n] - W[:, :, n] @ P                                # the update values u_t of this chunk
        outs.append((lg[:, :, n].exp().unsqueeze(-1) * q[:, :, n]) @ P + QK[:, :, n] @ U)
        tail = (lg[:, :, n, -1:] - lg[:, :, n]).exp().unsqueeze(-1)     # gamma_C / gamma_j
        P = lg[:, :, n, -1].exp()[..., None, None] * P + k[:, :, n].transpose(-1, -2) @ (tail * U)
    return torch.stack(outs, 2).reshape(B, H, N * C, Dv)[:, :, :T].to(dtype)


class GatedDeltaNet(nn.Module):
    """A Qwen3.5-style linear-attention block with its own norm and residual, the same interface as Attention:
    x (B, T, d) -> x + out(...)."""

    def __init__(self, d, head_dim=64, conv=4, chunk=64):
        super().__init__()
        self.H, self.D, self.chunk = max(1, d // head_dim), min(head_dim, d), chunk
        self.norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(3 * d, 3 * d, conv, groups=3 * d, padding=conv - 1, bias=False)
        self.ab = nn.Linear(d, 2 * self.H, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        dt = torch.exp(torch.empty(self.H).uniform_(math.log(1e-3), math.log(0.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))      # softplus(dt_bias) = dt at the start
        self.gate = nn.Linear(d, d, bias=False)
        self.onorm = RMSNorm(self.D)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        Bn, T, d = x.shape
        h = self.norm(x)
        qkv = F.silu(self.conv(self.qkv(h).transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = (t.reshape(Bn, T, self.H, self.D).transpose(1, 2) for t in qkv.split(d, -1))
        q, k = F.normalize(q.float(), dim=-1) / math.sqrt(self.D), F.normalize(k.float(), dim=-1)
        a, b = self.ab(h).float().split(self.H, -1)                       # (B, T, H) each
        g = (-self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())).transpose(1, 2)
        beta = torch.sigmoid(b).transpose(1, 2)
        o = chunk_gated_delta(q, k, v.float(), g, beta, self.chunk).transpose(1, 2)   # (B, T, H, D)
        o = self.onorm(o.to(x.dtype)) * F.silu(self.gate(h)).view(Bn, T, self.H, self.D)
        return x + self.out(o.reshape(Bn, T, d))
