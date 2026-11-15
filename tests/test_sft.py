"""SFT: the conversation loader, batches that keep conversations apart, and the training loop."""
import gzip
import json

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")
from exealm import ExeaLM, get_config
from exealm.chat import answer
from exealm.hub import from_pretrained
from exealm.train import sft
from exealm.train.pretrain import make_optimizer, save

WORDS = ["red", "blue", "green", "black", "white", "pink"]
REPLY = dict(zip(WORDS, ["apple", "sky", "leaf", "night", "snow", "rose"]))


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from exealm.data.tokenizer import train
    d = tmp_path_factory.mktemp("tok")
    with gzip.open(d / "a.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps(dict(text=" ".join(WORDS + list(REPLY.values())) + f" What is {i}? こんにちは")) + "\n")
    t = train([str(d / "a.jsonl.gz")], 330, str(d / "tokenizer"))
    return t, str(d / "tokenizer")


def test_loader_reads_both_forms_and_skips_broken_lines(tmp_path):
    lines = [dict(messages=[dict(role="user", content="hi"), dict(role="assistant", content="hello")]),
             dict(conversations=[{"from": "system", "value": "s"}, {"from": "human", "value": "q"},
                                 {"from": "gpt", "value": "a"}]),
             dict(messages=[dict(role="user", content="no answer")]),
             dict(messages=[dict(role="robot", content="x"), dict(role="assistant", content="y")]),
             dict(text="not a conversation")]
    path = tmp_path / "c.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\n\nnot json\n")
    convs, skipped = sft.load_conversations(str(path))
    assert skipped == 4 and len(convs) == 2
    assert [m["role"] for m in convs[1]] == ["system", "user", "assistant"]


def test_encode_cuts_long_conversations_and_drops_ones_with_no_answer_left(tok):
    t, _ = tok
    short = [dict(role="user", content="red"), dict(role="assistant", content="apple")]
    late = [dict(role="user", content="red " * 30), dict(role="assistant", content="apple")]
    out = sft.encode([short, late], t, max_len=12)
    assert len(out) == 1 and len(out[0][0]) <= 12 and sum(out[0][1]) > 0


def test_batches_group_similar_lengths_and_cover_every_conversation():
    examples = [([k] * n, [0] + [1] * (n - 1)) for k, n in enumerate([3, 9, 4, 10, 3, 9, 4, 10])]
    stream = sft.batches(examples, 2, np.random.default_rng(0), pad=99, pool=4)
    seen = []
    for _ in range(4):
        ids, mask = next(stream)
        lengths = (ids != 99).sum(1).tolist()
        assert abs(lengths[0] - lengths[1]) <= 1                       # neighbours in length share a batch
        assert (mask[ids == 99] == 0).all()
        seen += ids[:, 0].tolist()
    assert sorted(seen) == list(range(8))                               # one pass covers everything once


def test_a_padded_row_scores_the_same_alone_as_in_a_batch(tok):
    """Conversations cannot see each other: right padding and the other rows change nothing."""
    t, _ = tok
    torch.manual_seed(0)
    model = ExeaLM(get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1))
    model.train()
    model(torch.randint(0, t.get_vocab_size(), (4, 32)))               # measure the circuit scales once
    model.eval()
    convs = [[dict(role="user", content=w), dict(role="assistant", content=REPLY[w])] for w in WORDS[:3]]
    convs[1][0]["content"] = "red blue green black white pink red blue"   # a longer row, so the others are padded
    ex = sft.encode(convs, t, 64)
    pad = t.token_to_id("<|pad|>")
    ids, mask = sft.pad_batch(ex, pad)
    assert ids.shape[1] == len(ex[1][0]) > len(ex[0][0])
    with torch.no_grad():
        together = model(ids)[0]
        for r, (i, _) in enumerate(ex):
            alone = model(torch.tensor([i]))[0][0]
            assert torch.allclose(together[r, :len(i)], alone, atol=1e-5)
        one_by_one = sum(float(sft.sft_loss(model, *sft.pad_batch([e], pad))) * sum(e[1][1:]) for e in ex)
    assert sft.evaluate(model, ex, 3, pad, "cpu") == pytest.approx(one_by_one / sum(sum(e[1][1:]) for e in ex), abs=1e-5)
    assert not model.training


def test_sft_loop_teaches_a_tiny_model_to_answer(tmp_path, tok):
    t, tok_dir = tok
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1)
    base = ExeaLM(cfg)
    save(str(tmp_path / "base.pt"), base, make_optimizer(base, 1e-3, 0.1), 0, cfg, "tonn")
    rng = np.random.default_rng(0)
    with open(tmp_path / "sft.jsonl", "w") as f:
        for _ in range(240):
            w = WORDS[int(rng.integers(len(WORDS)))]
            f.write(json.dumps(dict(messages=[dict(role="user", content=w),
                                              dict(role="assistant", content=REPLY[w])])) + "\n")
    out = tmp_path / "run"
    sft.main(["--ckpt", str(tmp_path / "base.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "sft.jsonl"),
              "--steps", "150", "--batch", "16", "--lr", "3e-3", "--eval_every", "50", "--device", "cpu",
              "--out", str(out)])
    log = [json.loads(line) for line in open(out / "log.jsonl")]
    assert log[0]["event"] == "start" and log[0]["conversations"] == 228 and log[0]["val_conversations"] == 12
    assert log[-1]["step"] == 150 and log[-1]["val_loss"] < 0.2 * log[0]["val_loss"]
    model, _ = from_pretrained(str(out / "model.pt"), tokenizer=False)
    said = answer(model, t, [[dict(role="user", content=w)] for w in WORDS], max_new_tokens=6, temperature=0)
    assert [s.strip() for s in said] == [REPLY[w] for w in WORDS]


def test_sft_refuses_a_tokenizer_that_does_not_fit(tmp_path, tok):
    _, tok_dir = tok
    cfg = get_config("tiny")
    model = ExeaLM(cfg)
    save(str(tmp_path / "base.pt"), model, make_optimizer(model, 1e-3, 0.1), 0, cfg, "tonn")
    (tmp_path / "d.jsonl").write_text("")
    with pytest.raises(ValueError, match="tokenizer has"):
        sft.main(["--ckpt", str(tmp_path / "base.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "d.jsonl")])
