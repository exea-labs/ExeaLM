"""Reading a long text window by window (MemoryContext(stream={}), docs/brain.md) gives the same predictions as one
pass over the whole text: exactly for TONN layers, which carry their recurrent state and the inputs their lags need,
and for attention as long as the text is two windows (each keeps the keys and values of the window before)."""
import pytest
import torch

from exealm import ExeaLM, Llama, MemoryContext, get_config
from exealm.config import LAGS

STREAMABLE = ("D", "I", "GI", "M", "MI", "MIM", "EP", "G", "C", "SM", "OS", "GD")
CORE = dict(pinned={}, menu=STREAMABLE, swarm_size=1)


def every_type(cfg):
    """Specs that put every streamable circuit type in the model, with lags from 0 up to the longest."""
    n = cfg.n_layers - len(cfg.attn_layers)
    return [[dict(tmpl=STREAMABLE[(l * cfg.slots + k) % len(STREAMABLE)], lag=LAGS[(l * cfg.slots + 3 * k) % len(LAGS)],
                  lam0=0.5 + 0.04 * k) for k in range(cfg.slots)] for l in range(n)]


def streamed(model, x, window):
    state = {}
    with torch.no_grad():
        return torch.cat([model(x[:, i:i + window], MemoryContext(stream=state))[0]
                          for i in range(0, x.shape[1], window)], 1)


def whole(model, x):
    with torch.no_grad():
        return model(x)[0]


def tokens(T, seed=1):
    return torch.randint(0, 256, (2, T), generator=torch.Generator().manual_seed(seed))


def test_a_tonn_model_reads_a_text_in_windows_exactly_as_in_one_pass():
    torch.manual_seed(0)
    cfg = get_config("tiny", n_layers=2, attn_layers=(), **CORE)
    model = ExeaLM(cfg, specs=every_type(cfg)).eval()
    assert {s["tmpl"] for row in model.specs for s in row} == set(STREAMABLE)
    x = tokens(64)
    torch.testing.assert_close(streamed(model, x, 16), whole(model, x), atol=2e-4, rtol=1e-3)   # 4 windows
    torch.testing.assert_close(streamed(model, x, 8), whole(model, x), atol=2e-4, rtol=1e-3)    # 8 windows, lag > window


@pytest.mark.parametrize("kind", ["hybrid", "llama"])
def test_attention_keeps_the_window_before(kind):
    torch.manual_seed(0)
    cfg = get_config("tiny", n_layers=3, attn_layers=(1,), attn_gate=True, qk_norm=True, value_residual=True, xsa=True,
                     **CORE)
    model = (ExeaLM(cfg, specs=every_type(cfg)) if kind == "hybrid" else Llama(cfg)).eval()
    x = tokens(32)
    torch.testing.assert_close(streamed(model, x, 16), whole(model, x), atol=2e-4, rtol=1e-3)   # two windows: exact
    three = tokens(48)
    a, b = streamed(model, three, 16), whole(model, three)
    torch.testing.assert_close(a[:, :32], b[:, :32], atol=2e-4, rtol=1e-3)
    assert not torch.allclose(a[:, 32:], b[:, 32:], atol=1e-3)             # the third window no longer sees the first


def test_the_carried_state_keeps_the_gradient():
    torch.manual_seed(0)
    cfg = get_config("tiny", n_layers=2, attn_layers=(), **CORE)
    model = ExeaLM(cfg, specs=every_type(cfg))
    x = tokens(32)
    state = {}
    logits = None
    for chunk in (x[:, :16], x[:, 16:]):
        logits = model(chunk, MemoryContext(stream=state))[0]
    logits.sum().backward()                                                  # the second window's output only
    assert state and all(v is None or not isinstance(v, dict) or v["tail"].requires_grad for v in state.values())
    rows = model.emb.weight.grad[x[:, :16].unique()].abs().sum(-1)
    assert (rows > 0).all()                                                  # reaches the first window's tokens


def test_rotary_attention_circuits_refuse_to_stream():
    cfg = get_config("tiny", n_layers=1, attn_layers=(), pinned={}, menu=("RP",), swarm_size=1)
    model = ExeaLM(cfg).eval()
    with pytest.raises(ValueError, match="RP circuits"):
        streamed(model, tokens(16), 8)


def test_the_qwen_style_baseline_streams_its_delta_layers_too():
    torch.manual_seed(0)
    cfg = get_config("tiny", n_layers=4, attn_gate=True, qk_norm=True, delta_layers=(0, 1, 2), **CORE)
    model = Llama(cfg).eval()
    x = tokens(32)
    torch.testing.assert_close(streamed(model, x, 16), whole(model, x), atol=2e-4, rtol=1e-3)
    gdn_only = Llama(get_config("tiny", n_layers=2, delta_layers=(0, 1), **CORE)).eval()
    x = tokens(64)
    torch.testing.assert_close(streamed(gdn_only, x, 16), whole(gdn_only, x), atol=2e-4, rtol=1e-3)   # any length



def test_streaming_can_start_late_with_the_same_tokens_per_step(tmp_path):
    import json
    from exealm.train import pretrain
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(CORE))
    pretrain.main(["--preset", "tiny", "--data", "synthetic", "--seq", "16", "--batch", "2", "--stream", "4",
                   "--stream_from", "0.5", "--steps", "4", "--eval_every", "4", "--search", "0", "--device", "cpu",
                   "--config", str(cfg), "--out", str(tmp_path / "late")])
    log = [json.loads(l) for l in open(tmp_path / "late" / "log.jsonl")]
    assert log[0]["tokens_per_step"] == 2 * 16 * 4 and any(r.get("event") == "eval" for r in log)


def test_stream_mix_reads_half_the_micro_batches_as_streams(tmp_path, monkeypatch):
    import json
    from exealm.train import pretrain
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(CORE))
    from exealm.data.loader import SyntheticShards
    seen, orig = [], SyntheticShards.batch                      # (rows, length) of every batch the run draws

    def spy(self, rows, length, *args, **kw):
        seen.append((rows, length))
        return orig(self, rows, length, *args, **kw)
    monkeypatch.setattr(SyntheticShards, "batch", spy)
    pretrain.main(["--preset", "tiny", "--data", "synthetic", "--seq", "16", "--batch", "2", "--stream", "4",
                   "--stream_from", "0.5", "--stream_mix", "0.5", "--accum", "2", "--steps", "4", "--eval_every", "4",
                   "--search", "0", "--device", "cpu", "--config", str(cfg), "--out", str(tmp_path / "mix")])
    train = seen[-8:]                                           # the validation draws come first
    assert train[:4] == [(8, 16)] * 4                            # steps 0-1: windows alone
    assert train[4:8] == [(2, 64), (8, 16), (2, 64), (8, 16)]   # steps 2-3: one stream and one alone each step
    import pytest
    with pytest.raises(SystemExit):
        pretrain.main(["--preset", "tiny", "--data", "synthetic", "--stream", "4", "--stream_mix", "0.4",
                       "--steps", "1", "--device", "cpu", "--out", str(tmp_path / "bad")])
