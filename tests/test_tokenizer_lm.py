"""experiments/tokenizer_lm.py: a batch run as micro-batches (--accum) gives the same loss and gradient as the whole batch,
and the search's selection batches cut the same way give the same worth."""
import argparse
import copy
import importlib.util
import pathlib

import pytest
import torch

from exealm import ExeaLM, Llama, MemoryContext, get_config
from exealm.model.search import group_worth
from exealm.train.pretrain import make_optimizer

path = pathlib.Path(__file__).parents[1] / "experiments" / "tokenizer_lm.py"   # CI runs bare pytest: load by path
spec = importlib.util.spec_from_file_location("tokenizer_lm", path)
tokenizer_lm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_lm)


@pytest.mark.parametrize("accum", [1, 2, 4])
@pytest.mark.parametrize("arch", ["llama", "tonn"])
def test_train_step_takes_the_whole_batch_loss_and_gradient(arch, accum, monkeypatch):
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: None)   # clipping would hide a scale error
    torch.manual_seed(0)
    model = Llama(get_config("tiny")) if arch == "llama" else ExeaLM(get_config("tiny"))
    if arch == "tonn":
        model.eval()          # in training mode the running slot scale updates once per micro-batch (see train_step)
    x, y = torch.randint(0, 256, (4, 16)), torch.randint(0, 256, (4, 16))
    ref = copy.deepcopy(model)
    want_loss = ref(x, MemoryContext(), y)[1]
    want_loss.backward()
    want = dict(ref.named_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in want.values())
    for p in model.parameters():
        p.grad = torch.ones_like(p)                     # stale gradients that the step must clear first
    loss = tokenizer_lm.train_step(model, make_optimizer(model, 1e-2, 0.1), x, y, torch.device("cpu"), accum)
    assert loss == pytest.approx(float(want_loss.detach()), abs=1e-6)
    for name, p in model.named_parameters():
        if want[name].grad is None:
            assert p.grad is None, name
        else:
            assert p.grad is not None and torch.allclose(p.grad, want[name].grad, rtol=1e-4, atol=1e-7), name


def test_train_step_rejects_a_batch_that_does_not_split_evenly():
    model = Llama(get_config("tiny"))
    x = torch.randint(0, 256, (6, 16))
    with pytest.raises(ValueError):
        tokenizer_lm.train_step(model, make_optimizer(model, 1e-2, 0.1), x, x, torch.device("cpu"), 4)


class _Shards:
    def batch(self, B, T, rng):
        t = torch.from_numpy(rng.integers(0, 256, (B, T + 1)))
        return t[:, :-1], t[:, 1:], None


def test_search_selection_batches_cut_by_accum_keep_the_tokens_and_the_worth():
    a = argparse.Namespace(batch=4, seq=16, accum=1)
    whole = tokenizer_lm.selection_batches(_Shards(), a, torch.device("cpu"), 0)
    a.accum = 2
    cut = tokenizer_lm.selection_batches(_Shards(), a, torch.device("cpu"), 0)
    assert len(whole) == 4 and len(cut) == 8 and all(x.shape == (2, 16) for x, _, _ in cut)
    for i, (x, y, _) in enumerate(whole):
        assert torch.equal(x, torch.cat([cut[2 * i][0], cut[2 * i + 1][0]]))
        assert torch.equal(y, torch.cat([cut[2 * i][1], cut[2 * i + 1][1]]))
    torch.manual_seed(0)
    model = ExeaLM(get_config("tiny"))
    W, per_type, base = group_worth(model, whole)
    W_cut, per_type_cut, base_cut = group_worth(model, cut)
    assert W.abs().max() > 1e-4                         # the slots matter, so equal worth is not just zeros
    assert base_cut == pytest.approx(base, abs=1e-5)
    assert torch.allclose(W_cut, W, atol=1e-5)
    assert per_type_cut.keys() == per_type.keys()
    assert model.training
