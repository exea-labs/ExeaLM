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
    "RG": (3, False, False, "retrieval from the read-only datastore, with the model's own embeddings (retrieval.py)"),
    "AG": (3, False, False, "agent board: post messages, read other agents' messages (board.py)"),
    "WA": (3, False, False, "weight alter: change a weight matrix of this agent or of its swarm (alter.py); "
                            "not in the default menu"),
}
MEMORY_CIRCUITS = ("RG", "AG", "WA")
RUNNING_SUM_TYPES = frozenset(("I", "GI", "MI", "MIM", "EP", "SM"))


def shift(U, lags):
    """U (B, T, g, h), lags (g,) integer: U[:, t - lag] per slot, zero before the start."""
    B, T, g, h = U.shape
    if g == 0:
        return U
    idx = torch.arange(T, device=U.device).view(1, T, 1, 1) - lags.view(1, 1, g, 1)
    return U.gather(1, idx.clamp(min=0).expand(B, T, g, h)) * (idx >= 0)


def _running_sum_inputs(t, A, B, lags, lam):
    """Scan inputs (batch, T, channels) and either fixed (channels,) or per-token decays."""
    if t == "GI":
        return A.flatten(2), torch.sigmoid(B.flatten(2) + 2)
    if t == "SM":
        dt = F.softplus(B - 2)
        decay = torch.exp(-dt * torch.exp(2 * lam))
        return (dt * A).flatten(2), decay.flatten(2)
    if t == "EP":
        weights = torch.exp(A.clamp(-8, 8))
        decay = lam.abs().flatten()
        return torch.cat(((weights * B).flatten(2), weights.flatten(2)), -1), decay.repeat(2)
    values = A if t == "I" else shift(A, lags) * B
    return values.flatten(2), lam.flatten()


def _running_sum_output(t, summed, C):
    if t == "EP":
        num, den = summed.chunk(2, -1)
        return (num / (den + 1e-3)).reshape_as(C)
    summed = summed.reshape_as(C)
    if t == "MIM":
        return C * summed
    if t == "SM":
        return summed * F.silu(C)
    return summed


def run_running_sums(u, lags, lam, ranges):
    """Run one scan across the layer's real running-sum slots, then restore each group's shape and dtype."""
    inputs = []
    for t, (a, b) in ranges.items():
        streams = u[:, :, a:b].unbind(3)
        values, decay = _running_sum_inputs(t, *streams[:2], lags[a:b], lam[a:b])
        inputs.append((t, values, decay, streams[2]))
    if not inputs:
        return {}
    if len(inputs) == 1:
        t, values, decay, C = inputs[0]
        summed = decay_scan(values, decay) if decay.ndim == 1 else gated_scan(values, decay)
        return {t: _running_sum_output(t, summed, C)}

    values = torch.cat([v for _, v, _, _ in inputs], -1)
    if all(decay.ndim == 1 for _, _, decay, _ in inputs):
        summed = decay_scan(values, torch.cat([decay for _, _, decay, _ in inputs]))
    else:
        decays = []
        for _, v, decay, _ in inputs:
            decay = decay.float()
            if decay.ndim == 1:
                # Match decay_scan's signed magnitude floor before expanding over tokens.
                decay = (decay.sign() * decay.abs().clamp_min(1e-6)).expand_as(v)
            decays.append(decay)
        summed = gated_scan(values, torch.cat(decays, -1))
    parts = summed.split([v.size(-1) for _, v, _, _ in inputs], -1)
    return {t: _running_sum_output(t, part.to(v.dtype), C) for (t, v, _, C), part in zip(inputs, parts)}


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


def run(t, S, lags, lam, beta):
    """Run g slots of type t. S: [A, B, C] each (batch, T, g, h); lags (g,); lam (g, h) decays in (-1, 1); beta (g,)
    attention sharpness. Returns (batch, T, g, h)."""
    A, Bs, Cs = S
    if t in RUNNING_SUM_TYPES:
        values, decay = _running_sum_inputs(t, A, Bs, lags, lam)
        summed = decay_scan(values, decay) if decay.ndim == 1 else gated_scan(values, decay)
        return _running_sum_output(t, summed, Cs)
    if t == "D":
        return shift(A, lags)
    if t == "M":
        return shift(A, lags) * Bs
    if t == "G":
        return F.silu(A) * Bs
    if t == "C":
        a = lags.clamp(min=1)
        return F.silu(A + shift(Bs, a) + shift(Cs, 2 * a))
    if t == "RP":
        g = A.size(2)
        q = rotate(A * beta.view(1, 1, g, 1)).transpose(1, 2)
        k = rotate(shift(Bs, lags)).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, Cs.transpose(1, 2), is_causal=True).transpose(1, 2)
    if t == "OS":
        return oscillating_memory(A, lam) * F.silu(Bs)
    raise ValueError(f"unknown circuit type {t}")
