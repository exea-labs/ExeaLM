import copy

import pytest
import torch

from exealm import ExeaLM, get_config


def model():
    torch.manual_seed(0)
    return ExeaLM(get_config('tiny', menu=('I',), pinned={}, n_layers=1, ffn=0, lags=(0,)))


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('change', ['future', 'other_row'])
def test_unobserved_scales_cannot_leak_future_or_other_rows(training, change):
    a = model().train(training)
    b = copy.deepcopy(a)
    x = torch.arange(48).reshape(2, 24) % 256
    changed = x.clone()
    if change == 'future':
        changed[0, -1] = (changed[0, -1] + 17) % 256
    else:
        changed[1] = (changed[1] + 39) % 256
    with torch.no_grad():
        original, perturbed = a(x)[0], b(changed)[0]
    assert torch.allclose(original[0, :12], perturbed[0, :12], atol=1e-6, rtol=1e-5)
    if training:
        assert (a.layers[0].scale > 0).all()  # still learns measurements for the next batch
    else:
        assert (a.layers[0].scale == 0).all()  # evaluation never mutates training statistics


def test_replaced_slot_uses_a_causal_fallback_until_it_is_observed():
    a = model()
    x = torch.arange(48).reshape(2, 24) % 256
    with torch.no_grad():
        a(x)
    assert (a.layers[0].scale > 0).all()
    a.layers[0].set_spec(0, dict(a.layers[0].specs[0]))
    assert a.layers[0].scale[0] == 0
    a.eval()
    b = copy.deepcopy(a)
    changed = x.clone()
    changed[0, -1] += 17
    with torch.no_grad():
        first, second = a(x)[0], b(changed)[0]
    assert torch.allclose(first[:, :12], second[:, :12], atol=1e-6, rtol=1e-5)


def test_measured_slot_outputs_and_statistics_match_existing_rule():
    layer = model().layers[0].train()
    torch.manual_seed(3)
    layer.scale.copy_(torch.linspace(.1, .8, layer.K))
    before = layer.scale.clone()
    order = torch.arange(layer.K).flip(0)
    y = torch.randn(2, 13, layer.K, layer.h)
    previous = before.index_select(0, order)
    expected = y / (previous.view(1, 1, -1, 1) + 1e-3)
    actual = layer._unit_scale(y, order)
    assert torch.equal(actual, expected)
    expected_scale = before.clone()
    rms = y.float().square().mean((0, 1, 3)).sqrt()
    expected_scale.index_copy_(0, order, .99 * previous + .01 * rms)
    assert torch.equal(layer.scale, expected_scale)
