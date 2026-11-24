"""Weight-alter changes that persist across the windows of one text (wa_persist, docs/brain.md): each window's changes
join a weight change carried into the next, and a key written again is replaced, not added to (the delta rule)."""
import pytest
import torch
import torch.nn.functional as F

from exealm import ExeaLM, MemoryContext, get_config
from exealm.model.alter import correct_for_carried, fold_edits
from exealm.model.common import MLP


@pytest.mark.parametrize("keep", [1.0, 0.9])
def test_writing_a_key_again_replaces_its_value(keep):
    torch.manual_seed(0)
    v = F.normalize(torch.randn(1, 1, 1, 6), dim=-1)
    u1, u2 = torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4)
    first = fold_edits(None, [dict(matrix="out", u=correct_for_carried(u1, v, None), v=v)], keep)["out"]
    torch.testing.assert_close(first @ v[0, 0, 0], keep * u1[0, 0])        # the carried change returns u1 at v
    second = fold_edits({"out": first}, [dict(matrix="out", u=correct_for_carried(u2, v, first), v=v)], keep)["out"]
    torch.testing.assert_close(second @ v[0, 0, 0], keep * u2[0, 0], atol=1e-5, rtol=1e-4)   # u2, not u1 + u2


def test_the_mlp_uses_the_carried_change_at_every_position():
    torch.manual_seed(0)
    mlp = MLP(8, 16)
    x = torch.randn(2, 5, 8)
    a, b = mlp.wi(mlp.norm(x)).chunk(2, -1)
    hidden = F.silu(a) * b
    out_change, in_change = torch.randn(2, 8, 16), torch.randn(2, 32, 8)
    torch.testing.assert_close(mlp(x, None, {"out": out_change}), mlp(x) + torch.einsum("btf,bdf->btd", hidden, out_change))
    assert not torch.allclose(mlp(x, None, {"in": in_change}), mlp(x))


def brain(persist, key="hidden"):
    torch.manual_seed(0)
    cfg = get_config("tiny", n_layers=2, attn_layers=(), pinned={"WA": 1}, menu=("I", "SM", "C", "WA"), swarm_size=1,
                     wa_key=key, wa_persist=persist)
    return ExeaLM(cfg)


@pytest.mark.parametrize("key", ["hidden", "learned"])
def test_a_text_read_in_windows_carries_its_weight_change(key):
    model = brain(0.9, key)
    x = torch.randint(0, 256, (2, 48), generator=torch.Generator().manual_seed(1))
    state = {}
    for i in range(3):
        logits, loss = model(x[:, 16 * i:16 * (i + 1)], MemoryContext(stream=state), x[:, 16 * i:16 * (i + 1)])
    carried = [v for k, v in state.items() if k[0] == "wa"]
    assert carried and all(c["out"].shape == (2, 64, model.cfg.ffn_width) and c["out"].abs().sum() > 0 for c in carried)
    loss.backward()                                                         # the third window's loss
    wa = [p for n, p in model.named_parameters() if ".wa." in n]
    assert wa and any(p.grad is not None and p.grad.abs().sum() > 0 for p in wa)


def test_without_persist_nothing_is_carried_and_swarms_are_refused():
    model = brain(0.0)
    state = {}
    model(torch.randint(0, 256, (2, 16)), MemoryContext(stream=state))
    assert not [k for k in state if k[0] == "wa"]
    with pytest.raises(ValueError, match="swarm_size 1"):
        ExeaLM(get_config("tiny", pinned={"WA": 1}, menu=("I", "WA"), swarm_size=2, wa_persist=0.9))
