"""The two running-sum recurrences most TONN circuits are made of, in plain PyTorch.

    decay_scan:  s_t = lam * s_{t-1} + u_t     (one fixed decay per channel, may be negative)
    gated_scan:  s_t = g_t * s_{t-1} + u_t     (a per-token decay; also supports signed fixed decays broadcast over time)

Both are exact and causal, in plain PyTorch: an FFT convolution for the first, a parallel prefix scan for the second.
Both take an optional starting state s0 (B, C), the state the previous window of the same text ended with, so a long
text can be read window by window and give the same result as one long pass (streaming, docs/brain.md).
A fused GPU kernel (Triton or CUDA) for these two is still one of the most useful contributions anyone can make
(docs/ideas.md).
"""
import torch
import torch.nn.functional as F


def decay_scan(u, lam, s0=None):
    """u (B, T, C), lam (C,) with |lam| < 1. One causal FFT convolution with the kernel lam^n, in float32. With s0, the
    state before the first position: lam^(t+1) s0 is added at position t."""
    B, T, C = u.shape
    dtype = u.dtype
    u, lam = u.float(), lam.float()
    n = torch.arange(T, device=u.device, dtype=torch.float32).view(T, 1)
    kern = lam.sign().view(1, C) ** n * lam.abs().clamp_min(1e-6).view(1, C) ** n       # (T, C): lam^n, sign kept
    nfft = 1 << (2 * T - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(u, n=nfft, dim=1) * torch.fft.rfft(kern, n=nfft, dim=0).unsqueeze(0),
                        n=nfft, dim=1)[:, :T]
    if s0 is not None:
        n1 = n + 1
        y = y + (lam.sign().view(1, C) ** n1 * lam.abs().clamp_min(1e-6).view(1, C) ** n1).unsqueeze(0) * s0.float().unsqueeze(1)
    return y.to(dtype)


def gated_scan(u, g, chunk=None, s0=None):
    """u, g (B, T, C), |g| < 1. Exact, as a parallel prefix scan (Hillis-Steele): log2(T) rounds of elementwise
    work, each combining every position with the one 2^r steps back:
        S_t <- S_t + G_t * S_{t-2^r},   G_t <- G_t * G_{t-2^r}
    The decays only ever multiply, so it stays stable. It replaced a chunked version that built a chunk x chunk matrix
    per channel and made GI and SM circuits 6-7x slower than an attention block (experiments/profile_circuits.py).
    `chunk` is accepted for compatibility and ignored. With s0, the state before the first position: after the rounds G_t
    is the product of all decays up to t, so G_t s0 is added at position t."""
    T = u.shape[1]
    dtype = u.dtype
    s, a = u.float(), g.float()
    k = 1
    while k < T:
        s = s + a * F.pad(s, (0, 0, k, 0))[:, :T]
        a = a * F.pad(a, (0, 0, k, 0), value=1.0)[:, :T]
        k *= 2
    if s0 is not None:
        s = s + a * s0.float().unsqueeze(1)
    return s.to(dtype)
