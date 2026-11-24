"""SFT: conversation loading, solo and swarm batches, and the training loop."""
import gzip
import json

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")
from exealm import ExeaLM, MemoryContext, get_config
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


def test_swarm_loader_and_encoder_keep_whole_tasks(tmp_path, tok):
    t, _ = tok
    helper = dict(conversations=[dict(**{"from": "human", "value": "red"}),
                                 dict(**{"from": "gpt", "value": "apple"})])
    answer = dict(messages=[dict(role="user", content="blue"), dict(role="assistant", content="sky")])
    late = dict(messages=[dict(role="user", content="blue " * 30), *answer["messages"][1:]])
    no_answer = dict(messages=[dict(role="user", content="blue")])
    records = [[helper, answer], [helper], [helper, late], [helper, no_answer], [None, answer]]
    path = tmp_path / "tasks.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(dict(agents=agents)) + "\n" for agents in records)

    tasks, skipped = sft.load_conversations(str(path), swarm=2, answer_agent=1)
    assert len(tasks) == 2 and skipped == 3
    encoded = sft.encode_swarms(tasks, t, max_len=16, answer_agent=1)
    assert len(encoded) == 1
    writer, reader = encoded[0]
    writer_ids, writer_mask = sft.format_chat(tasks[0][0], t)
    assert writer[0] == writer_ids and any(writer_mask) and not any(writer[1])
    assert reader == sft.format_chat(answer["messages"], t)


def test_answer_loss_trains_other_agents_ag_posts():
    torch.manual_seed(0)
    cfg = get_config("tiny", board_read_self=False)
    model = ExeaLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (3, 2 * cfg.chunk))
    mask = torch.zeros_like(ids)
    mask[1, cfg.chunk + 1] = 1
    board = model.layers[0].ag
    inputs, outputs = [], []
    capture_ag = board.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    capture_head = model.head.register_forward_hook(lambda _, args, output: outputs.append(output))
    try:
        for swarm in (3, 1):
            loss = sft.sft_loss(model, ids, mask, MemoryContext(swarm_size=swarm))
            logits = outputs.pop()
            expected = torch.nn.functional.cross_entropy(logits[1, cfg.chunk].unsqueeze(0), ids[1, cfg.chunk + 1:cfg.chunk + 2])
            torch.testing.assert_close(loss, expected)
            values, keys, _ = inputs.pop()
            dvalues, dkeys, dpost_w, dpost_b, dlogits = torch.autograd.grad(
                loss, (values, keys, board.post_w, board.post_b, logits))
            writers = [0, 2]
            assert dlogits[writers].count_nonzero() == 0
            # At the first read position only the helpers' first-chunk posts can supply these gradients.
            assert dvalues[1].count_nonzero() == 0
            assert dvalues[writers, cfg.chunk:].count_nonzero() == 0
            grads = [*dvalues[writers, :cfg.chunk], *dkeys[writers, :cfg.chunk], dpost_w, dpost_b]
            assert all(torch.isfinite(grad).all() for grad in grads)
            assert all(bool(grad.count_nonzero()) == (swarm == 3) for grad in grads)
    finally:
        capture_ag.remove()
        capture_head.remove()


def test_swarm_sft_tiny_learns_split_information(tmp_path, tok):
    t, tok_dir = tok
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1, board_read_self=False)
    base = ExeaLM(cfg)
    ckpt = tmp_path / "base.pt"
    save(str(ckpt), base, make_optimizer(base, 1e-3, 0.1), 0, cfg, "tonn")
    question = "What is the word the other agent has?"
    tasks = [dict(agents=[dict(messages=[dict(role="user", content=w)]),
                          dict(messages=[dict(role="user", content=question),
                                         dict(role="assistant", content=REPLY[w])])]) for w in WORDS[:2]]
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text("\n".join(map(json.dumps, tasks * 5)))
    val.write_text("\n".join(map(json.dumps, tasks)))
    out = tmp_path / "run"
    sft.main(["--ckpt", str(ckpt), "--tokenizer", tok_dir, "--data", str(train), "--val", str(val),
              "--swarm", "2", "--answer_agent", "1", "--steps", "60", "--batch", "3", "--lr", "3e-3",
              "--device", "cpu", "--out", str(out)])

    start, end = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert (start["swarms"], start["val_swarms"], start["swarm_size"], start["answer_agent"]) == (10, 2, 2, 1)
    assert end["step"] == 60 and np.isfinite(end["train_loss"])
    trained, _ = from_pretrained(str(out / "model.pt"), tokenizer=False)
    val_tasks, _ = sft.load_conversations(str(val), swarm=2, answer_agent=1)
    encoded = sft.encode_swarms(val_tasks, t, max_len=1024, answer_agent=1)
    cut = sft.evaluate(trained, [rows[1] for rows in encoded], 3, t.token_to_id("<|pad|>"), "cpu")
    assert end["val_loss"] < 0.2 * start["val_loss"]
    assert end["val_loss"] < 0.5 * cut
