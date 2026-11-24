"""The four changes from recent papers (value_residual, xsa, peri_ln, canon): each trains, keeps the model causal, and
does what it says."""
import pytest
import torch

from exealm import ExeaLM, Llama, get_config
from exealm.model.baseline import Attention

SWITCHES = ("value_residual", "xsa", "peri_ln", "canon")
CORE = dict(pinned={}, menu=("D", "I", "GI", "M", "MI", "MIM", "EP", "G", "C", "RP", "SM", "OS"), swarm_size=1)


def models(**on):
    torch.manual_seed(0)
    cfg = get_config("tiny", n_layers=3, attn_layers=(1, 2), attn_gate=True, qk_norm=True, **CORE, **on)
    return {"hybrid": ExeaLM(cfg), "llama": Llama(cfg)}


@pytest.mark.parametrize("on", [dict(value_residual=True), dict(xsa=True), dict(peri_ln=True), dict(canon=True),
                                {s: True for s in SWITCHES}])
def test_every_switch_trains_and_stays_causal(on):
    x = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(1))
    for name, model in models(**on).items():
        logits, loss = model(x, targets=x)
        loss.backward()
        assert torch.isfinite(loss), name
        new = [(n, p.grad is not None) for n, p in model.named_parameters() if any(k in n for k in ("vr", "onorm", "canon"))]
        assert all(g for _, g in new), (name, new)                      # every new parameter is trained
        assert bool(new) == any(on.get(k) for k in ("value_residual", "peri_ln", "canon")), name
        model.eval()
        with torch.no_grad():
            y = x.clone()
            y[:, 12:] = (y[:, 12:] + 1) % 256                           # change only the second half
            a, b = model(x)[0], model(y)[0]
        torch.testing.assert_close(a[:, :12], b[:, :12])                # the first half cannot see it, name


def test_the_switches_add_what_they_should():
    base = {k: sum(p.numel() for p in m.parameters()) for k, m in models().items()}
    for switch in ("value_residual", "peri_ln", "canon"):                # xsa adds no parameters
        more = {k: sum(p.numel() for p in m.parameters()) for k, m in models(**{switch: True}).items()}
        assert all(more[k] > base[k] for k in base), switch
    assert sum(p.numel() for p in models(value_residual=True)["llama"].parameters()) - base["llama"] == 2 * 2   # not layer 0


def test_exclusive_self_attention_removes_what_a_token_reads_from_itself():
    torch.manual_seed(0)
    x = torch.randn(1, 5, 64)
    plain, xsa = Attention(64, 4), Attention(64, 4, xsa=True)
    xsa.load_state_dict(plain.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(xsa(x)[:, 0], x[:, 0])               # position 0 attends only to its own value
        assert not torch.allclose(plain(x)[:, 0], x[:, 0])
        assert not torch.allclose(xsa(x)[:, 1:], x[:, 1:])


def test_value_residual_mixes_in_the_first_attention_layers_values():
    torch.manual_seed(0)
    first, second = Attention(64, 4, value_residual=True), Attention(64, 4, value_residual=True)
    x = torch.randn(1, 6, 64)
    ctx = {}
    first(x, ctx)
    assert "v_first" in ctx
    with torch.no_grad():
        second.vr.copy_(torch.tensor([0.0, 1.0]))                      # all its own values: the plain layer
        own = second(x, dict(ctx))
        second.vr.copy_(torch.tensor([1.0, 0.0]))                      # all the first layer's values
        borrowed = second(x, dict(ctx))
    assert not torch.allclose(own, borrowed)
