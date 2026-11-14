import json

import torch

from exealm.train.dpo import dpo_loss
from exealm.train.grpo import group_advantages, grpo_loss, reward_exact_answer
from exealm.train.pretrain import load_model, main


def test_pretraining_smoke_run_lowers_the_loss_and_saves(tmp_path):
    main(["--preset", "tiny", "--data", "synthetic", "--steps", "120", "--eval_every", "60", "--search_every", "30",
          "--datastore_chunks", "500", "--refresh_every", "50", "--device", "cpu", "--out", str(tmp_path)])
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    evals = [json.loads(l) for l in lines]
    bits = [e["val_bits_per_token"] for e in evals if e.get("event") == "eval"]
    assert bits[-1] < 8.0 and bits[-1] < bits[0] + 1e-6
    assert any(e.get("event") == "search" for e in evals)
    model, ck = load_model(str(tmp_path / "model.pt"))
    assert ck["step"] == 120


def test_grpo_advantages_and_loss():
    adv = group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]), group_size=4)
    assert torch.allclose(adv[4:], torch.zeros(4), atol=1e-4)            # a group that all agree learns nothing
    assert adv[0] > 0 > adv[1]
    logp = torch.zeros(8, 5, requires_grad=True)
    loss = grpo_loss(logp, logp.detach(), logp.detach(), adv, torch.ones(8, 5))
    loss.backward()
    assert logp.grad[0].sum() < 0 < logp.grad[1].sum()                   # raise good answers, lower bad ones
    assert reward_exact_answer("so the answer is 42", 42) == 1.0


def test_dpo_prefers_the_chosen_answer():
    lc = torch.zeros(3, requires_grad=True)
    dpo_loss(lc, torch.zeros(3), torch.zeros(3), torch.zeros(3)).backward()
    assert (lc.grad < 0).all()


def test_resumed_run_matches_an_uninterrupted_one(tmp_path, monkeypatch):
    from exealm.model import search

    real_worth = search.group_worth

    def replacement_worth(model, batches):
        # Exercise search RNG, grace periods and optimizer row resets after resume
        # regardless of whether this platform/data sample happens to learn useful slots.
        worth, per_type, base = real_worth(model, batches)
        worth.fill_(-1.0)
        return worth, {kind: [-1.0] for kind in per_type}, base

    monkeypatch.setattr(search, "group_worth", replacement_worth)
    common = ["--preset", "tiny", "--data", "synthetic", "--steps", "40", "--eval_every", "20", "--search_every", "4",
              "--search_until", "0.9", "--datastore_chunks", "300", "--refresh_every", "15", "--device", "cpu"]
    a, b = tmp_path / "a", tmp_path / "b"
    main(common + ["--out", str(a)])                                          # straight through
    main(common + ["--out", str(b), "--stop_at", "6"])                       # between two rounds that replace slots
    main(common + ["--out", str(b), "--resume", "auto"])
    full, _ = load_model(str(a / "model.pt"))
    resumed, ck = load_model(str(b / "model.pt"))
    assert ck["step"] == 40
    events = [json.loads(l) for l in (b / "log.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events if e["event"] == "resume"] == [6]
    after = [e for e in events if e["event"] == "search" and e["step"] > 6]
    assert any(e["replaced"] > 0 for e in after)                             # so the search's own state mattered
    assert full.specs == resumed.specs                                       # the search made the same choices
    for (k, v), w in zip(full.state_dict().items(), resumed.state_dict().values()):
        assert torch.equal(v, w), k


def test_resume_auto_starts_fresh_without_a_checkpoint(tmp_path):
    main(["--preset", "tiny", "--data", "synthetic", "--steps", "5", "--eval_every", "5", "--device", "cpu",
          "--out", str(tmp_path), "--resume", "auto"])
    events = [json.loads(l)["event"] for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert events[0] == "start" and "resume" not in events
