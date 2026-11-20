"""TONN's circuit types: the small computations a slot can hold.

Every slot gets three input streams A, B, C (each h channels, read from the residual stream by the layer) and returns
h channels. A circuit may also use a time lag (how far back it looks) and a decay per channel (how fast it forgets).
Slots of the same type in a layer run together as one batched operation, on tensors of shape (batch, time, g, h) for
g slots.

RG (retrieval) and AG (agent board) need outside memory, so they live in their own files: retrieval.py and board.py.
"""
import math

import torch
import torch.nn.functional as F

from .common import rotate
from .delta import chunk_gated_delta
from .scan import decay_scan, gated_scan

# type: (streams used, uses a lag, uses decays, what it computes)
CIRCUITS = {
    "D": (1, True, False, "delay: A[t - lag]"),
    "I": (1, False, True, "leaky running sum: s_t = lam s_{t-1} + A_t"),
    "GI": (2, False, False, "gated running sum: s_t = sigmoid(B_t) s_{t-1} + A_t"),
    "M": (2, True, False, "product: A[t - lag] * B_t"),
    "MI": (2, True, True, "running sum of a product: I(A[t - lag] * B_t)"),
    "MIM": (3, True, True, "gated running sum of a product: C_t * I(A[t - lag] * B_t), an induction-like memory"),
    "EP": (2, False, True, "softmax pooling over the past: I(exp(A) * B) / I(exp(A))"),
    "G": (2, False, False, "gated unit: silu(A_t) * B_t"),
    "C": (3, True, False, "short convolution: silu(A_t + B[t - a] + C[t - 2a])"),
    "RP": (3, True, False, "rotary attention: softmax_j(rot(A_t) . rot(B[j - lag])) C_j over j <= t"),
    "SM": (3, False, True, "selective state space (Mamba-like): s_t = exp(-dt_t a) s_{t-1} + dt_t A_t, dt = softplus(B)"),
    "OS": (2, False, True, "oscillating memory (S4D-like): complex z_t = r e^{iw} z_{t-1} + A_t, times silu(B_t)"),
    "GD": (3, False, True, "gated delta-rule memory: query and write to a matrix state (delta.py)"),
    "RG": (3, False, False, "retrieval from the read-only datastore, with the model's own embeddings (retrieval.py)"),
    "AG": (3, False, False, "agent board: post messages, read other agents' messages (board.py)"),
    "WA": (3, False, False, "weight alter: change a weight matrix of this agent or of its swarm (alter.py); "
                            "not in the default menu"),
}
MEMORY_CIRCUITS = ("RG", "AG", "WA")


def shift(U, lags):
    """U (B, T, g, h), lags (g,) integer: U[:, t - lag] per slot, zero before the start."""
    B, T, g, h = U.shape
    if g == 0:
        return U
    idx = torch.arange(T, device=U.device).view(1, T, 1, 1) - lags.view(1, 1, g, 1)
    return U.gather(1, idx.clamp(min=0).expand(B, T, g, h)) * (idx >= 0)


def oscillating_memory(u, lam):
    """z_t = r e^{iw} z_{t-1} + u_t on channel pairs (u_2i + i u_2i+1), as one FFT convolution. The slot's decays give
    the radius r = |lam_2i| and the frequency w = pi lam_2i+1 of each pair."""
    B, T, g, h = u.shape
    dtype = u.dtype
    u, lam = u.float(), lam.float()
    r = lam[:, 0::2].abs().clamp(1e-6, 0.999)
    w = math.pi * lam[:, 1::2]
    n = torch.arange(T, device=u.device, dtype=torch.float32).view(T, 1, 1)
    kern = torch.exp(n * torch.log(r)) * torch.exp(1j * (n * w))           # (T, g, h/2) complex
    z_in = torch.complex(u[..., 0::2], u[..., 1::2])
    nfft = 1 << (2 * T - 1).bit_length()
    z = torch.fft.ifft(torch.fft.fft(z_in, n=nfft, dim=1) * torch.fft.fft(kern, n=nfft, dim=0).unsqueeze(0),
                       n=nfft, dim=1)[:, :T]
    return torch.stack([z.real, z.imag], -1).reshape(B, T, g, h).to(dtype)


def gated_delta_slot(key, value, query, direction, write_bias, chunk):
    """Run one matrix state per slot using a shared learned direction to read the two gates."""
    h = key.shape[-1]
    dtype = value.dtype
    with torch.autocast(device_type=key.device.type, enabled=False):
        k, v, q = (u.transpose(1, 2).float() for u in (key, value, query))  # (B, g, T, h)
        direction = direction.float().unsqueeze(1) / math.sqrt(h)
        log_decay = -F.softplus((k * direction).sum(-1) - 2)
        write = torch.sigmoid((v * direction).sum(-1) + write_bias.float().unsqueeze(-1))
        q, k = F.normalize(q, dim=-1) / math.sqrt(h), F.normalize(k, dim=-1)
        out = chunk_gated_delta(q, k, v, log_decay, write, chunk=min(chunk, key.shape[1]))
    return out.transpose(1, 2).to(dtype)


def run(t, S, lags, lam, beta, *, chunk=64):
    """Run g slots of type t. S: [A, B, C] each (batch, T, g, h); lags (g,); lam (g, h) decays in (-1, 1); beta (g,)
    attention sharpness. GD uses lam as a gate direction and beta as a write bias. Returns (batch, T, g, h)."""
    A, Bs, Cs = S
    Bn, T, g, h = A.shape
    flat = lambda u: u.reshape(Bn, T, g * h)
    back = lambda u: u.view(Bn, T, g, h)
    if t == "D":
        return shift(A, lags)
    if t == "I":
        return back(decay_scan(flat(A), lam.reshape(-1)))
    if t == "GI":
        return back(gated_scan(flat(A), torch.sigmoid(flat(Bs) + 2)))
    if t == "M":
        return shift(A, lags) * Bs
    if t == "MI":
        return back(decay_scan(flat(shift(A, lags) * Bs), lam.reshape(-1)))
    if t == "MIM":
        return Cs * back(decay_scan(flat(shift(A, lags) * Bs), lam.reshape(-1)))
    if t == "EP":                                     # positive decay keeps the denominator positive
        e = torch.exp(A.clamp(-8, 8))
        lp = lam.abs().reshape(-1)
        both = decay_scan(torch.cat([flat(e * Bs), flat(e)], -1), torch.cat([lp, lp]))   # one FFT for both sums
        num, den = both.split(g * h, -1)
        return back(num) / (back(den) + 1e-3)
    if t == "G":
        return F.silu(A) * Bs
    if t == "C":
        a = lags.clamp(min=1)
        return F.silu(A + shift(Bs, a) + shift(Cs, 2 * a))
    if t == "RP":
        q = rotate(A * beta.view(1, 1, g, 1)).transpose(1, 2)
        k = rotate(shift(Bs, lags)).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, Cs.transpose(1, 2), is_causal=True).transpose(1, 2)
    if t == "SM":
        dt = F.softplus(Bs - 2)
        decay = torch.exp(-dt * torch.exp(2 * lam).view(1, 1, g, h))
        return back(gated_scan(flat(dt * A), flat(decay))) * F.silu(Cs)
    if t == "OS":
        return oscillating_memory(A, lam) * F.silu(Bs)
    if t == "GD":
        return gated_delta_slot(A, Bs, Cs, lam, beta, chunk)
    raise ValueError(f"unknown circuit type {t}")
