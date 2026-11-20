import json

import pytest
import torch

from exealm import ExeaLM, Llama, MemoryContext, get_config
from exealm.model.search import reset_adam_rows
from exealm.train.muon import Muon, orthogonalize
from exealm.train.pretrain import load_model, main, make_optimizer


@pytest.mark.parametrize("shape", [(768, 256), (256, 680), (32, 8), (256, 256)])
def test_orthogonalize_is_close_to_the_nearest_semi_orthogonal_matrix(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape)
    O = orthogonalize(G)
    assert O.shape == G.shape and O.dtype == G.dtype
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    target = U @ Vh
    assert float((O * target).sum() / (O.norm() * target.norm())) > 0.97
    if shape[0] != shape[1]:                         # a square Gaussian matrix has a few near-zero singular values
        s = torch.linalg.svdvals(O)
        assert 0.6 < float(s.min()) and float(s.max()) < 1.25


def test_muon_step_has_about_the_rms_of_an_adamw_update():
    torch.manual_seed(0)
    for shape in [(256, 256), (768, 256), (256, 680)]:
        p = torch.nn.Parameter(torch.zeros(*shape))
        p.grad = torch.randn(*shape)
        Muon([dict(params=[p], use_muon=True)], lr=1.0).step()
        assert 0.15 < float(p.detach().pow(2).mean().sqrt()) < 0.22


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs a recent PyTorch")
@pytest.mark.parametrize("shape", [(768, 256), (256, 680), (64, 64)])
def test_muon_groups_take_the_same_steps_as_torch_muon(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape) * 0.02
    a, b = torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.clone())
    ours = Muon([dict(params=[a], use_muon=True, weight_decay=0.1)], lr=2e-3)
    ref = torch.optim.Muon([b], lr=2e-3, weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
    for _ in range(5):
        a.grad = torch.randn(*shape)
        b.grad = a.grad.clone()
        ours.step()
        ref.step()
    assert (a - w).abs().max() > 1e-3                                         # the weights really moved
    assert torch.allclose(a, b, atol=1e-7, rtol=0)


def test_groups_without_muon_take_the_same_steps_as_torch_adamw():
    torch.manual_seed(0)
    init = [torch.randn(5, 3), torch.randn(7)]
    a = [torch.nn.Parameter(t.clone()) for t in init]
    b = [torch.nn.Parameter(t.clone()) for t in init]
    groups = lambda ps: [dict(params=[ps[0]], weight_decay=0.1), dict(params=[ps[1]], weight_decay=0.0)]
    ours = Muon(groups(a), lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
    ref = torch.optim.AdamW(groups(b), lr=3e-3, betas=(0.9, 0.95), eps=1e-8)
    for _ in range(6):
        for x, y in zip(a, b):
            x.grad = torch.randn_like(x)
            y.grad = x.grad.clone()
        ours.step()
        ref.step()
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)


def test_muon_rejects_parameters_that_are_not_matrices():
    with pytest.raises(ValueError):
        Muon([dict(params=[torch.nn.Parameter(torch.zeros(4))], use_muon=True)])


def split(model, opt):
    names = {id(p): n for n, p in model.named_parameters()}
    out = {"muon": set(), "decay": set(), "no_decay": set()}
    for g in opt.param_groups:
        key = "muon" if g.get("use_muon") else "decay" if g["weight_decay"] > 0 else "no_decay"
        out[key] |= {names[id(p)] for p in g["params"]}
    return out


def test_make_optimizer_puts_only_linear_weights_on_muon():
    model = ExeaLM(get_config("small", vocab_size=512))
    s = split(model, make_optimizer(model, 2e-3, 0.1, "muon"))
    every = [n for n, _ in model.named_parameters()]
    assert sorted(s["muon"] | s["decay"] | s["no_decay"]) == sorted(every)
    assert len(s["muon"]) + len(s["decay"]) + len(s["no_decay"]) == len(every)
    assert {"layers.0.inp.weight", "layers.0.out.weight", "attn.0.qkv.weight", "mlps.0.wi.weight"} <= s["muon"]
    assert "emb.weight" in s["no_decay"]                                     # tied to the output head
    assert {"layers.0.lam", "encoder.pos"} <= s["decay"]                    # 2-D, but not weight matrices
    assert {"norm.w", "layers.0.beta", "layers.0.inp.bias"} <= s["no_decay"]
    adamw = split(model, make_optimizer(model, 2e-3, 0.1))                   # the AdamW default is unchanged
    assert adamw["muon"] == set() and adamw["decay"] == s["muon"] | s["decay"]

    llama = Llama(get_config("small", vocab_size=512, tie_embeddings=False))
    s = split(llama, make_optimizer(llama, 2e-3, 0.1, "muon"))
    assert "head.weight" in s["decay"] and "emb.weight" in s["no_decay"]
    with pytest.raises(ValueError):
        make_optimizer(llama, 2e-3, 0.1, "sgd")


def test_search_slot_reset_clears_muon_momentum():
    torch.manual_seed(0)
    model = ExeaLM(get_config("tiny"))
    opt = make_optimizer(model, 1e-2, 0.1, "muon")
    x = torch.randint(0, 256, (2, 16))
    model(x, MemoryContext(), x)[1].backward()
    opt.step()
    lay, k = model.layers[0], 1
    reset_adam_rows(opt, lay, k)
    rows, cols = slice(k * 3 * lay.h, (k + 1) * 3 * lay.h), slice(k * lay.h, (k + 1) * lay.h)
    inp, out = opt.state[lay.inp.weight]["momentum_buffer"], opt.state[lay.out.weight]["momentum_buffer"]
    assert inp[rows].abs().sum() == 0 and inp.abs().sum() > 0
    assert out[:, cols].abs().sum() == 0 and out.abs().sum() > 0
    assert opt.state[lay.lam]["exp_avg"][k].abs().sum() == 0                  # AdamW rows inside the same optimizer


def test_pretraining_with_muon_learns_and_resumes_exactly(tmp_path, monkeypatch):
    from exealm.model import search

    real_worth = search.group_worth

    def replacement_worth(model, batches):                                    # force slot replacements after resume
        worth, per_type, base = real_worth(model, batches)
        worth.fill_(-1.0)
        return worth, {kind: [-1.0] for kind in per_type}, base

    monkeypatch.setattr(search, "group_worth", replacement_worth)
    common = ["--preset", "tiny", "--data", "synthetic", "--steps", "40", "--eval_every", "20", "--search_every", "4",
              "--search_until", "0.9", "--datastore_chunks", "300", "--refresh_every", "15", "--device", "cpu",
              "--optimizer", "muon"]
    a, b = tmp_path / "a", tmp_path / "b"
    main(common + ["--out", str(a)])
    main(common + ["--out", str(b), "--stop_at", "6"])
    main(common + ["--out", str(b), "--resume", "auto"])
    evals = [json.loads(l) for l in (a / "log.jsonl").read_text().splitlines()]
    bits = [e["val_bits_per_token"] for e in evals if e["event"] == "eval"]
    assert bits[-1] < bits[0]
    full, _ = load_model(str(a / "model.pt"))
    resumed, ck = load_model(str(b / "model.pt"))
    assert ck["step"] == 40 and isinstance(ck["opt"]["param_groups"][0]["use_muon"], bool)
    assert full.specs == resumed.specs
    for (k, v), w in zip(full.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(v, w), k
