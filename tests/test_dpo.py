"""DPO: the preference loader, answer log-probabilities, and the training loop."""
import gzip
import json
import math

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")
from exealm import ExeaLM, get_config
from exealm.train import dpo, sft
from exealm.train.pretrain import make_optimizer, save

WORDS = ["red", "blue", "green", "black", "white", "pink"]
GOOD = dict(zip(WORDS, ["apple", "sky", "leaf", "night", "snow", "rose"]))
BAD = dict(zip(WORDS, ["sky", "leaf", "night", "snow", "rose", "apple"]))


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from exealm.data.tokenizer import train
    d = tmp_path_factory.mktemp("tok")
    with gzip.open(d / "a.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps(dict(text=" ".join(WORDS + list(GOOD.values())) + f" What is {i}?")) + "\n")
    return train([str(d / "a.jsonl.gz")], 330, str(d / "tokenizer")), str(d / "tokenizer")


def test_loss_is_log_2_at_the_reference_and_falls_when_the_chosen_answer_gains():
    z = torch.zeros(3)
    assert float(dpo.dpo_loss(z - 5, z - 7, z - 5, z - 7)) == pytest.approx(math.log(2))
    assert float(dpo.dpo_loss(z - 4, z - 8, z - 5, z - 7, beta=1.0)) < float(dpo.dpo_loss(z - 5, z - 7, z - 5, z - 7))


def test_loader_reads_both_forms_and_skips_broken_lines(tmp_path):
    lines = [dict(prompt="q", chosen="a", rejected="b"),
             dict(messages=[dict(role="system", content="s"), dict(role="user", content="q")], chosen="a", rejected="b"),
             dict(prompt="q", chosen="a", rejected="a"), dict(prompt="q", chosen="a"),
             dict(messages=[dict(role="robot", content="q")], chosen="a", rejected="b")]
    path = tmp_path / "p.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    prefs, skipped = dpo.load_preferences(str(path))
    assert skipped == 4 and len(prefs) == 2 and [m["role"] for m in prefs[1][0]] == ["system", "user"]


def test_encode_masks_only_the_answer_and_cuts_the_prompt_from_the_left(tok):
    t, _ = tok
    (ci, cm), (ri, rm) = dpo.encode([([dict(role="user", content="red")], "apple", "sky")], t, 64)[0]
    answer = sft.format_chat([dict(role="assistant", content="apple")], t)
    assert ci[-len(answer[0]):] == answer[0] and sum(cm) == sum(answer[1]) and cm[:len(ci) - len(answer[0])].count(1) == 0
    assert ci[:len(ci) - len(answer[0])] == ri[:len(ri) - len(sft.format_chat([dict(role="assistant", content="sky")], t)[0])]
    long = dpo.encode([([dict(role="user", content="red " * 40)], "apple", "sky")], t, 16)[0]
    assert len(long[0][0]) == 16 and long[0][0][-len(answer[0]):] == answer[0]          # the answer is kept whole
    assert dpo.encode([([dict(role="user", content="red")], "apple " * 40, "sky")], t, 16) == []


def test_answer_logp_matches_one_row_at_a_time(tok):
    t, _ = tok
    torch.manual_seed(0)
    model = ExeaLM(get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1))
    model.train()
    model(torch.randint(0, t.get_vocab_size(), (4, 32)))
    model.eval()
    ex = dpo.encode([([dict(role="user", content=w)], GOOD[w], BAD[w] + " " + BAD[w]) for w in WORDS[:3]], t, 64)
    pad = t.token_to_id("<|pad|>")
    with torch.no_grad():
        pc, pr = dpo.batch_logps(model, ex, pad, "cpu")
        for k, (c, r) in enumerate(ex):
            for got, (ids, mask) in ((pc[k], c), (pr[k], r)):
                logits = model(torch.tensor([ids[:-1]]))[0][0]
                lp = torch.log_softmax(logits, -1)[torch.arange(len(ids) - 1), torch.tensor(ids[1:])]
                assert float(got) == pytest.approx(float((lp * torch.tensor(mask[1:])).sum()), abs=1e-4)


def test_dpo_loop_makes_a_tiny_model_prefer_the_chosen_answers(tmp_path, tok):
    t, tok_dir = tok
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1)
    base = ExeaLM(cfg)
    save(str(tmp_path / "base.pt"), base, make_optimizer(base, 1e-3, 0.1), 0, cfg, "tonn")
    rng = np.random.default_rng(0)
    with open(tmp_path / "prefs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for _ in range(240):
            w = WORDS[int(rng.integers(len(WORDS)))]
            f.write(json.dumps(dict(prompt=w, chosen=GOOD[w], rejected=BAD[w])) + "\n")
    out = tmp_path / "run"
    before = {k: v.clone() for k, v in base.state_dict().items()}
    dpo.main(["--ckpt", str(tmp_path / "base.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "prefs.jsonl"),
              "--steps", "150", "--batch", "16", "--lr", "3e-3", "--beta", "1.0", "--eval_every", "50",
              "--device", "cpu", "--out", str(out)])
    log = [json.loads(line) for line in open(out / "log.jsonl", encoding="utf-8")]
    assert log[0]["event"] == "start" and log[0]["preferences"] == 228 and log[0]["val_preferences"] == 12
    assert log[0]["val_loss"] == pytest.approx(math.log(2), abs=1e-3)          # the policy starts as the reference
    assert log[-1]["step"] == 150 and log[-1]["val_loss"] < 0.2 * log[0]["val_loss"]
    assert log[-1]["val_prefers_chosen"] == 1.0
    ck = torch.load(out / "model.pt", weights_only=False)
    assert any(not torch.equal(ck["model"][k], before[k]) for k in before)
