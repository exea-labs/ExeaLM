"""Batched layer scans agree with the original per-circuit scans and their gradients."""
from copy import deepcopy

import pytest
import torch

from exealm import get_config
from exealm.model import circuits
from exealm.model.tonn import TONNLayer


def unbatched(u, lags, lam, ranges):
    return {t: circuits.run(t, u[:, :, a:b].unbind(3), lags[a:b], lam[a:b], None)
            for t, (a, b) in ranges.items()}


@pytest.mark.parametrize("types,kernel", [
    (("I", "MI", "MIM", "EP"), "decay_scan"),
    (("GI", "SM"), "gated_scan"),
    (("EP", "GI", "I", "MI", "MIM", "SM"), "gated_scan"),
    (("EP",), "decay_scan"),
])
@pytest.mark.parametrize("length", [1, 37])
def test_packed_scan_matches_separate_scans_and_dispatches_once(monkeypatch, types, kernel, length):
    torch.manual_seed(0)
    # Two slots of each type, with lags extending past the window and positive, negative, and near-zero decays.
    K, h = len(types) * 2, 4
    streams = torch.randn(2, length, K, 3, h, requires_grad=True)
    lags = torch.tensor([2, 40] * len(types))
    lam = torch.tensor([[-0.95, -1e-8, 0.0, 0.95], [0.4, 1e-8, -0.3, 0.8]] * len(types), requires_grad=True)
    ranges = {t: (2 * i, 2 * i + 2) for i, t in enumerate(types)}
    expected = unbatched(streams, lags, lam, ranges)

    calls = []
    decay_scan, gated_scan = circuits.decay_scan, circuits.gated_scan

    def traced_decay(*args):
        calls.append("decay_scan")
        return decay_scan(*args)

    def traced_gated(*args):
        calls.append("gated_scan")
        return gated_scan(*args)

    monkeypatch.setattr(circuits, "decay_scan", traced_decay)
    monkeypatch.setattr(circuits, "gated_scan", traced_gated)
    actual = circuits.run_running_sums(streams, lags, lam, ranges)
    assert calls == [kernel]

    weights = [torch.randn_like(output) for output in expected.values()]
    for t in types:
        torch.testing.assert_close(actual[t], expected[t], rtol=3e-5, atol=4e-6)

    def grads(outputs):
        loss = sum((out * weight).sum() for out, weight in zip(outputs.values(), weights))
        return torch.autograd.grad(loss, (streams, lam), allow_unused=True)

    for a, b in zip(grads(actual), grads(expected)):
        # Length-one gated scans can omit their unused decay inputs entirely.
        if a is None or b is None:
            grad = b if a is None else a
            if grad is not None:
                assert grad.count_nonzero() == 0
        else:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-5)


def test_layer_outputs_and_parameter_gradients_match_unbatched(monkeypatch):
    torch.manual_seed(1)
    cfg = get_config("tiny")
    types = ("MIM", "GI", "I", "G", "EP", "MI", "I", "SM")
    specs = [dict(tmpl=t, lag=i % 3, lam0=(-0.6 if i % 2 else 0.8)) for i, t in enumerate(types)]
    layer = TONNLayer(cfg, specs, index=0)
    layer.mask[4] = 0
    separate = deepcopy(layer)
    x = torch.randn(2, 33, cfg.d, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    packed_out = layer(x)
    monkeypatch.setattr(circuits, "run_running_sums", unbatched)
    separate_out = separate(y)
    torch.testing.assert_close(packed_out, separate_out, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(layer.scale, separate.scale, rtol=1e-5, atol=2e-6)

    weights = torch.randn_like(packed_out)
    (packed_out * weights).sum().backward()
    (separate_out * weights).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=8e-6)
    for p, q in zip(layer.parameters(), separate.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-5)
