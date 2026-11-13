"""Gated DeltaNet (exealm/model/delta.py): the chunked computation equals the token-by-token recurrence, and the layer
is causal."""
import pytest
import torch
import torch.nn.functional as F

from exealm.model.delta import GatedDeltaNet, chunk_gated_delta, gated_delta_reference


@pytest.mark.parametrize("T,chunk", [(37, 8), (64, 16), (50, 64)])
def test_chunked_equals_the_recurrence(T, chunk):
    torch.manual_seed(0)
    B, H, D = 2, 3, 8
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    g = -F.softplus(torch.randn(B, H, T, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    ref = gated_delta_reference(q, k, v, g, beta)
    out = chunk_gated_delta(q, k, v, g, beta, chunk)
    assert torch.allclose(out.double(), ref, atol=1e-4), (out.double() - ref).abs().max()


def test_layer_is_causal_and_trains():
    torch.manual_seed(0)
    m = GatedDeltaNet(32, head_dim=16, chunk=8)
    x = torch.randn(2, 30, 32, requires_grad=True)
    y = m(x)
    x2 = x.detach().clone()
    x2[:, 17] += 1.0
    y2 = m(x2)
    assert torch.allclose(y[:, :17], y2[:, :17], atol=1e-5)
    assert not torch.allclose(y[:, 17:], y2[:, 17:])
    y.sum().backward()
    assert m.qkv.weight.grad is not None and m.A_log.grad is not None
