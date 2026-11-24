"""Gated DeltaNet (exealm/model/delta.py): the chunked computation equals the token-by-token recurrence, and the layer
is causal."""
import math
import random

import pytest
import torch
import torch.nn.functional as F

from exealm import ExeaLM, count_params, get_config
from exealm.model.circuits import run
from exealm.model.delta import GatedDeltaNet, _unit_lower_solve, chunk_gated_delta, gated_delta_reference
from exealm.model.search import denoise_step, group_worth
from exealm.model.tonn import slot_cost


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


@pytest.mark.parametrize("T", [5, 19])
def test_delta_slots_match_reference_and_are_causal(T):
    torch.manual_seed(0)
    B, slots, h = 2, 3, 8
    streams = [torch.randn(B, T, slots, h) for _ in range(3)]
    lags = torch.zeros(slots, dtype=torch.long)
    lam = (torch.rand(slots, h) * 2 - 1).requires_grad_()
    beta = torch.randn(slots, requires_grad=True)
    out = run("GD", streams, lags, lam, beta, chunk=8)

    key, value, query = [u.transpose(1, 2) for u in streams]
    direction = lam[:, None, :] / math.sqrt(h)
    decay = -F.softplus((key * direction).sum(-1) - 2)
    write = torch.sigmoid((value * direction).sum(-1) + beta[:, None])
    ref = gated_delta_reference(F.normalize(query, dim=-1) / math.sqrt(h), F.normalize(key, dim=-1), value,
                                decay, write)
    assert out.shape == (B, T, slots, h) and out.dtype == value.dtype
    torch.testing.assert_close(out, ref.transpose(1, 2))

    low_precision = [u.bfloat16() for u in streams]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed = run("GD", low_precision, lags, lam, beta, chunk=8)
    assert mixed.dtype == torch.bfloat16
    torch.testing.assert_close(mixed, run("GD", low_precision, lags, lam, beta, chunk=8))

    split = T // 2
    changed = [u.clone() for u in streams]
    for u in changed:
        u[0, split:, 0] += torch.randn_like(u[0, split:, 0])
    other = run("GD", changed, lags, lam, beta, chunk=8)
    torch.testing.assert_close(out[:, :split], other[:, :split])
    torch.testing.assert_close(out[1], other[1])
    torch.testing.assert_close(out[..., 1:, :], other[..., 1:, :])
    assert not torch.allclose(out[0, split:, 0], other[0, split:, 0])
    out.square().mean().backward()
    for p in (lam, beta):
        assert torch.isfinite(p.grad).all() and p.grad.count_nonzero() == p.numel()


def test_triangular_fallback_has_native_gradients(monkeypatch):
    torch.manual_seed(0)
    left = (torch.randn(2, 4, 4).tril(-1) + torch.eye(4)).requires_grad_()
    right = torch.randn(2, 4, 3, requires_grad=True)
    expected = _unit_lower_solve(left, right)
    expected_grads = torch.autograd.grad(expected.square().sum(), (left, right))

    def unsupported(*args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(torch.linalg, "solve_triangular", unsupported)
    actual = _unit_lower_solve(left, right)
    torch.testing.assert_close(actual, expected)
    for grad, expected_grad in zip(torch.autograd.grad(actual.square().sum(), (left, right)), expected_grads):
        torch.testing.assert_close(grad, expected_grad)


def test_tiny_search_can_add_ablate_and_replace_a_delta_slot():
    torch.manual_seed(0)
    default = get_config("tiny")
    cfg = get_config("tiny", menu=default.menu + ("GD",), init_weights={"GD": 0})
    model = ExeaLM(cfg)
    assert all("GD" not in layer for layer in model.architecture())
    before = count_params(model)
    assert before == count_params(ExeaLM(default))
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randint(0, cfg.vocab_size, (2, 21))
    y = torch.roll(x, -1, dims=1)
    lay = model.layers[0]
    slot = next(k for k, sp in enumerate(lay.specs) if sp["tmpl"] not in cfg.pinned)
    W = torch.ones(cfg.n_layers, cfg.slots)
    W[0, slot] = -1

    def replace_with(t):
        changed = denoise_step(model, opt, W, sigma=1.0 / W.numel(), rng=random.Random(0), T=x.size(1),
                               weights={t: 1.0}, p_mutate=0)
        assert [(i, k, sp["tmpl"]) for i, k, sp in changed] == [(0, slot, t)]

    replace_with("GD")
    assert slot_cost(lay.specs[slot], cfg, x.size(1)) > slot_cost(dict(tmpl="G"), cfg, x.size(1))
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 21, cfg.vocab_size) and torch.isfinite(loss)
    loss.backward()
    opt.step()
    for p in (lay.lam, lay.beta):
        assert opt.state[p]["exp_avg"][slot].count_nonzero()

    worth, per_type, base = group_worth(model, [(x, y, None)])
    assert model.training and lay.mask[slot] == 1 and "GD" in per_type
    model.eval()
    clone = ExeaLM(cfg, model.specs).eval()
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(clone(x)[0], model(x)[0])
        lay.mask[slot] = 0
        cut = float(model(x, targets=y)[1]) / math.log(2)
        lay.mask[slot] = 1
    assert worth[0, slot].item() == pytest.approx(cut - base, abs=1e-6)

    replace_with("G")
    assert count_params(model) == before and lay.scale[slot] == 0
    assert all(opt.state[p]["exp_avg"][slot].count_nonzero() == 0 for p in (lay.lam, lay.beta))
    assert "GD" not in lay.plan(x.device)["ranges"]
