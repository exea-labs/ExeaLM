"""The options for long runs on shared GPUs, in one process: gradient accumulation, time limits and stop signals.
tests/test_ddp.py checks the same with two processes."""
import json
import signal

import pytest
import torch

from exealm.train import pretrain
from exealm.train.pretrain import load_model, main

TINY = ["--preset", "tiny", "--data", "synthetic", "--device", "cpu"]


def events(folder):
    return [json.loads(line) for line in (folder / "log.jsonl").read_text().splitlines()]


def same_weights(a, b):
    for (k, v), w in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.equal(v, w), k


def test_accumulated_steps_match_one_larger_batch(tmp_path):
    """A Llama has no swarms and no running statistics, so 2 passes of 4 rows must give the step of 1 pass of 8."""
    common = TINY + ["--model", "llama", "--steps", "5", "--eval_every", "5"]
    main(common + ["--batch", "8", "--out", str(tmp_path / "one")])
    main(common + ["--batch", "4", "--accum", "2", "--out", str(tmp_path / "two")])
    assert events(tmp_path / "two")[0]["tokens_per_step"] == events(tmp_path / "one")[0]["tokens_per_step"] == 8 * 128
    one, two = load_model(str(tmp_path / "one" / "model.pt"))[0], load_model(str(tmp_path / "two" / "model.pt"))[0]
    for (k, v), w in zip(one.state_dict().items(), two.state_dict().values()):
        assert torch.allclose(v, w, atol=1e-4), k


def test_a_time_limit_stops_cleanly_and_the_next_session_finishes_the_same_run(tmp_path):
    common = TINY + ["--steps", "12", "--eval_every", "6", "--search_every", "4", "--datastore_chunks", "200"]
    a, b = tmp_path / "a", tmp_path / "b"
    main(common + ["--out", str(a)])
    main(common + ["--out", str(b), "--max_minutes", "1e-9"])             # the limit is over after the first step
    stop = [e for e in events(b) if e["event"] == "stop"]
    assert [(e["step"], e["why"]) for e in stop] == [(1, "max_minutes")] and (b / "ckpt.pt").exists()
    main(common + ["--out", str(b), "--resume", "auto"])
    same_weights(load_model(str(a / "model.pt"))[0], load_model(str(b / "model.pt"))[0])


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="no SIGUSR1 on Windows")
def test_a_stop_signal_saves_and_stops_within_a_step(tmp_path, monkeypatch):
    """The signal arrives late in step 3, after that step's decision to go on, so the run stops after step 4."""
    real, calls = torch.nn.utils.clip_grad_norm_, []

    def clip_and_signal(*args, **kwargs):                                 # called once per step, inside the loop
        calls.append(1)
        if len(calls) == 3:
            signal.raise_signal(signal.SIGUSR1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pretrain.torch.nn.utils, "clip_grad_norm_", clip_and_signal)
    before = signal.getsignal(signal.SIGUSR1)
    main(TINY + ["--steps", "20", "--eval_every", "10", "--out", str(tmp_path)])
    assert signal.getsignal(signal.SIGUSR1) == before                     # the run put the old handler back
    stop = [e for e in events(tmp_path) if e["event"] == "stop"]
    assert [(e["step"], e["why"]) for e in stop] == [(4, "signal")]
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["step"] == 4


def test_save_minutes_writes_checkpoints_on_the_clock(tmp_path):
    main(TINY + ["--steps", "4", "--eval_every", "4", "--save_minutes", "1e-9", "--out", str(tmp_path)])
    assert torch.load(tmp_path / "ckpt.pt", weights_only=False)["step"] == 3       # the last step writes model.pt
