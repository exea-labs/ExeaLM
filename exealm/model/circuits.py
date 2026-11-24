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


def _scan(values, decay, s0):
    if decay.ndim == 1:
        return decay_scan(values, decay) if s0 is None else decay_scan(values, decay, s0)
    return gated_scan(values, decay) if s0 is None else gated_scan(values, decay, None, s0)


def run_running_sums(u, lags, lam, ranges, tail=0, s0=None, return_state=False):
    """Run one scan across the layer's real running-sum slots, then restore each group's shape and dtype.
    Streaming (docs/brain.md): u starts with `tail` positions carried from the previous window (only lags read them),
    s0 is the scan state that window ended with, and return_state also returns the state this window ends with."""
    inputs = []
    for t, (a, b) in ranges.items():
        streams = u[:, :, a:b].unbind(3)
        values, decay = _running_sum_inputs(t, *streams[:2], lags[a:b], lam[a:b])
        if tail:
            values, decay = values[:, tail:], (decay[:, tail:] if decay.ndim == 3 else decay)
        inputs.append((t, values, decay, streams[2][:, tail:]))
    if not inputs:
        return ({}, None) if return_state else {}
    if len(inputs) == 1:
        t, values, decay, C = inputs[0]
        summed = _scan(values, decay, s0)
        out = {t: _running_sum_output(t, summed, C)}
        return (out, summed[:, -1]) if return_state else out

    values = torch.cat([v for _, v, _, _ in inputs], -1)
    if all(decay.ndim == 1 for _, _, decay, _ in inputs):
        summed = _scan(values, torch.cat([decay for _, _, decay, _ in inputs]), s0)
    else:
        decays = []
        for _, v, decay, _ in inputs:
            decay = decay.float()
            if decay.ndim == 1:
                # Match decay_scan's signed magnitude floor before expanding over tokens.
                decay = (decay.sign() * decay.abs().clamp_min(1e-6)).expand_as(v)
            decays.append(decay)
        summed = _scan(values, torch.cat(decays, -1), s0)
    parts = summed.split([v.size(-1) for _, v, _, _ in inputs], -1)
    out = {t: _running_sum_output(t, part.to(v.dtype), C) for (t, v, _, C), part in zip(inputs, parts)}
    return (out, summed[:, -1]) if return_state else out


def oscillating_memory(u, lam, z0=None, return_state=False):
    """z_t = r e^{iw} z_{t-1} + u_t on channel pairs (u_2i + i u_2i+1), as one FFT convolution. The slot's decays give
    the radius r = |lam_2i| and the frequency w = pi lam_2i+1 of each pair. z0 (B, g, h/2) complex: the state the
    previous window ended with; return_state also returns the state this window ends with."""
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
    if z0 is not None:
        n1 = n + 1
        z = z + (torch.exp(n1 * torch.log(r)) * torch.exp(1j * (n1 * w))).unsqueeze(0) * z0.unsqueeze(1)
    y = torch.stack([z.real, z.imag], -1).reshape(B, T, g, h).to(dtype)
    return (y, z[:, -1]) if return_state else y


def gated_delta_slot(key, value, query, direction, write_bias, chunk, state=None, return_state=False):
    """Run one matrix state per slot using a shared learned direction to read the two gates. state (B, g, h, h): the
    matrices the previous window ended with; return_state also returns the ones this window ends with."""
    h = key.shape[-1]
    dtype = value.dtype
    with torch.autocast(device_type=key.device.type, enabled=False):
        k, v, q = (u.transpose(1, 2).float() for u in (key, value, query))  # (B, g, T, h)
        direction = direction.float().unsqueeze(1) / math.sqrt(h)
        log_decay = -F.softplus((k * direction).sum(-1) - 2)
        write = torch.sigmoid((v * direction).sum(-1) + write_bias.float().unsqueeze(-1))
        q, k = F.normalize(q, dim=-1) / math.sqrt(h), F.normalize(k, dim=-1)
        out, P = chunk_gated_delta(q, k, v, log_decay, write, chunk=min(chunk, key.shape[1]), state=state,
                                   return_state=True)
    out = out.transpose(1, 2).to(dtype)
    return (out, P) if return_state else out


def run(t, S, lags, lam, beta, *, chunk=64):
    """Run g slots of type t. S: [A, B, C] each (batch, T, g, h); lags (g,); lam (g, h) decays in (-1, 1); beta (g,)
    attention sharpness. GD uses lam as a gate direction and beta as a write bias. Returns (batch, T, g, h)."""
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
    if t == "GD":
        return gated_delta_slot(A, Bs, Cs, lam, beta, chunk)
    raise ValueError(f"unknown circuit type {t}")


def run_stream(t, S, lags, lam, beta, *, chunk=64, tail=0, state=None):
    """run() for one window of a longer text (docs/brain.md). S: [A, B, C], each starting with `tail` positions carried
    from the previous window, which only the lags read; state: what this type carried (None at the start). Returns the
    output for the window's own positions and the state to carry on (None for types that keep none). Running-sum types
    go through run_running_sums."""
    A, Bs, Cs = S
    own = lambda z: z[:, tail:]                                                         # noqa: E731
    if t == "D":
        return own(shift(A, lags)), None
    if t == "M":
        return own(shift(A, lags)) * own(Bs), None
    if t == "G":
        return F.silu(own(A)) * own(Bs), None
    if t == "C":
        a = lags.clamp(min=1)
        return F.silu(own(A) + own(shift(Bs, a)) + own(shift(Cs, 2 * a))), None
    if t == "OS":
        y, z = oscillating_memory(own(A), lam, state, return_state=True)
        return y * F.silu(own(Bs)), z
    if t == "GD":
        return gated_delta_slot(own(A), own(Bs), own(Cs), lam, beta, chunk, state, return_state=True)
    if t == "RP":
        raise ValueError("RP circuits cannot read a text window by window yet: they need a key cache")
    raise ValueError(f"run_stream does not take {t}")
