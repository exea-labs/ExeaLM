import json
import random
from dataclasses import replace

import pytest
import torch

from exealm import ExeaLM, Llama, MemoryContext, ModelConfig, count_params, get_config, llama_baseline
from exealm.config import MENU
from exealm.model.circuits import CIRCUITS
from exealm.model.scan import decay_scan, gated_scan
from exealm.model.search import denoise_step
from exealm.model.tonn import initial_specs


def tiny(**kw):
    return get_config("tiny", **kw)


def every_type_specs(cfg):
    """Specs that put every circuit type in the menu into each layer at least once."""
    types = list(cfg.menu)
    return [[dict(tmpl=types[k % len(types)], lag=[0, 1, 3][k % 3], lam0=0.5) for k in range(cfg.slots)]
            for _ in range(cfg.n_layers)]


def test_scans_match_a_plain_loop():
    torch.manual_seed(0)
    u, lam, g = torch.randn(2, 37, 5), torch.rand(5) * 1.8 - 0.9, torch.rand(2, 37, 5)
    s1, s2, r1, r2 = torch.zeros(2, 5), torch.zeros(2, 5), [], []
    for t in range(37):
        s1 = lam * s1 + u[:, t]
        s2 = g[:, t] * s2 + u[:, t]
        r1.append(s1)
        r2.append(s2)
    assert torch.allclose(decay_scan(u, lam), torch.stack(r1, 1), atol=1e-4)
    assert torch.allclose(gated_scan(u, g, chunk=8), torch.stack(r2, 1), atol=1e-4)


def test_every_circuit_type_runs_and_trains():
    cfg = tiny(slots=len(CIRCUITS))
    m = ExeaLM(cfg, every_type_specs(cfg))
    x = torch.randint(0, cfg.vocab_size, (4, 40))
    logits, loss = m(x, targets=x)
    assert logits.shape == (4, 40, cfg.vocab_size) and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_every_layer_starts_with_one_retrieval_and_one_board_circuit():
    cfg = get_config("100m")
    for row in initial_specs(cfg):
        kinds = [s["tmpl"] for s in row]
        assert kinds.count("RG") == 1 and kinds.count("AG") == 1 and len(kinds) == cfg.slots


def test_search_never_removes_the_last_pinned_circuit():
    cfg = tiny()
    m = ExeaLM(cfg)
    W = -torch.ones(cfg.n_layers, cfg.slots)                 # pretend every slot is useless
    denoise_step(m, None, W, sigma=1.0, rng=random.Random(0), T=64)
    for lay in m.layers:
        kinds = [s["tmpl"] for s in lay.specs]
        assert kinds.count("RG") >= 1 and kinds.count("AG") >= 1


def test_100m_preset_is_about_a_tenth_of_a_billion_parameters():
    cfg = get_config("100m")
    with torch.device("meta"):
        tonn, base = ExeaLM(cfg), Llama(llama_baseline(cfg))
    assert 90e6 < count_params(tonn) < 115e6
    assert abs(count_params(tonn) - count_params(base)) / count_params(base) < 0.05   # a fair baseline


def test_hybrid_presets_use_gated_attention_and_qk_norm_and_the_baseline_stays_plain():
    for name in ("100m", "small"):
        cfg = get_config(name)
        assert cfg.attn_gate and cfg.qk_norm
        assert not llama_baseline(cfg).attn_gate and not llama_baseline(cfg).qk_norm
        with torch.device("meta"):
            att = [m for m in ExeaLM(cfg).modules() if type(m).__name__ == "Attention"]
            plain = [m for m in Llama(llama_baseline(cfg)).modules() if type(m).__name__ == "Attention"]
        assert att and all(m.gate is not None and m.qn is not None for m in att)
        assert plain and all(m.gate is None and m.qn is None for m in plain)


def test_configs_saved_before_the_gate_existed_load_as_trained():
    d = json.loads(get_config("small").to_json())
    for k in ("attn_gate", "qk_norm"):
        d.pop(k)
    old = ModelConfig.from_dict(d)
    assert not old.attn_gate and not old.qk_norm
    gated, ungated = ExeaLM(replace(old, attn_gate=True, qk_norm=True)), ExeaLM(old)
    with pytest.raises(RuntimeError):                          # an ungated checkpoint never loads into a gated model
        gated.load_state_dict(ungated.state_dict())


def test_baseline_runs():
    cfg = tiny()
    m = Llama(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    assert torch.isfinite(m(x, targets=x)[1])


def test_generate_appends_tokens():
    cfg = tiny()
    m = ExeaLM(cfg).eval()
    out = m.generate(torch.zeros(1, 5, dtype=torch.long), 3, mem=MemoryContext())
    assert out.shape == (1, 8)


def test_hybrid_attention_layers_and_starting_mix():
    cfg = tiny(n_layers=3, attn_layers=(1,), init_weights={t: (1.0 if t == "RP" else 0.0) for t in MENU})
    m = ExeaLM(cfg)
    assert len(m.layers) == 2 and len(m.attn) == 1
    for lay in m.layers:                                   # the starting mix: only RP besides the pinned circuits
        assert {s["tmpl"] for s in lay.specs} <= {"RP", "RG", "AG"}
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    logits, loss = m(x, targets=x)
    loss.backward()
    assert logits.shape == (2, 24, cfg.vocab_size) and torch.isfinite(loss)
