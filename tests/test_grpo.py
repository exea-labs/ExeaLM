"""GRPO: rewards, the task loader, sampling a group, and the loop."""
import gzip
import json

import numpy as np
import pytest
import torch

pytest.importorskip("tokenizers")
from exealm import ExeaLM, MemoryContext, get_config
from exealm.chat import chat_prompt
from exealm.train import grpo, sft
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


def test_rewards():
    assert grpo.reward_math(r"first \boxed{\frac{2}{3}}, then \boxed{2x-8}.", "2x-8") == 1.0
    assert grpo.reward_math(r"\boxed{7}", "8") == 0.0
    assert grpo.reward_math("so the total is 1,024.0 apples", "1024") == 1.0
    assert grpo.reward_math("no number here", "3") == 0.0
    assert grpo.reward_exact(" sky \n", "sky") == 1.0 and grpo.reward_exact("the sky", "sky") == 0.0
    assert grpo.reward_code("```python\ndef add(a, b):\n    return a + b\n```\nDone.", "assert add(2, 3) == 5") == 1.0
    assert grpo.reward_code("def add(a, b):\n    return a - b", "assert add(2, 3) == 5") == 0.0
    assert grpo.reward_code("while True:\n    pass", "pass", timeout=1.0) == 0.0


def test_loader_reads_both_forms_and_skips_broken_lines(tmp_path):
    lines = [dict(prompt="2+2?", answer="4"), dict(messages=[dict(role="user", content="q")], answer="a", reward="exact"),
             dict(prompt="q", answer="a", reward="unknown"), dict(prompt="q"), dict(prompt="q", answer="")]
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json\n")
    tasks, skipped = grpo.load_tasks(str(tmp_path / "t.jsonl"))
    assert skipped == 4 and [t[2] for t in tasks] == ["math", "exact"]


def test_a_sampled_group_scores_the_answer_and_its_ending(tok):
    t, _ = tok
    torch.manual_seed(0)
    model = ExeaLM(get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1))
    task = ([dict(role="user", content="red")], "apple", "exact")
    rows = grpo.sample_group(model, t, task, 5, max_new_tokens=6, temperature=1.0)
    end = t.token_to_id("<|im_end|>")
    assert len(rows) == 5 and model.training
    for ids, mask, text in rows:
        n = sum(mask)
        assert len(ids) == len(mask) and mask[:len(mask) - n].count(1) == 0 and 1 <= n <= 6
        assert (ids[-1] == end) == (n < 6 or ids[-1] == end and len(t.encode(text).ids) < 6)
        assert end not in ids[len(ids) - n:-1]                              # the ending appears only as the last token


def test_swarm_task_loader_validates_all_agent_prompts(tmp_path):
    reader = dict(messages=[dict(role="user", content="What is the color?")])
    helper = dict(prompt="The color is blue")
    good = dict(agents=[helper, reader], answer="blue", reward="exact")
    wrong_size = dict(good, agents=[reader])
    wrong_role = dict(good, agents=[helper, dict(messages=[dict(role="unknown", content="q")])])
    path = tmp_path / "swarm.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in [good, wrong_size, wrong_role])
    tasks, skipped = grpo.load_tasks(str(path), swarm=2)
    assert skipped == 2
    assert tasks == [([[dict(role="user", content=helper["prompt"])], reader["messages"]], "blue", "exact")]


def test_swarm_advantages_are_normalized_per_task_then_shared_by_agents():
    rewards = torch.tensor([0.0, 1.0, 0.5, 5.0, 5.0, 5.0])
    adv = grpo.group_advantages(rewards, 3, swarm=2)
    expected = torch.tensor([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0] + [0.0] * 6)
    torch.testing.assert_close(adv, expected, atol=3e-6, rtol=0)
    logp = torch.zeros(12, 2, requires_grad=True)
    mask = torch.tensor([[True, False]] * 12)
    loss = grpo.grpo_loss(logp, logp.detach(), logp.detach(), adv, mask)
    loss.backward()
    torch.testing.assert_close(logp.grad[:, 0], -adv / 12)
    assert logp.grad[:, 1].count_nonzero() == 0


def test_swarm_rollout_replay_preserves_boards_and_stopped_agents(tok):
    t, _ = tok
    torch.manual_seed(0)
    model = ExeaLM(get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1, board_read_self=False)).eval()
    messages = [[dict(role="user", content="red")],
                [dict(role="user", content="What is the word the other agent has?")]]
    short, long = [chat_prompt(agent, t) for agent in messages]
    end = t.token_to_id("<|im_end|>")
    red, blue = [t.encode(w).ids[0] for w in ("red", "blue")]
    tokens = [[end, red, blue, red], [end, red, end, blue], [end, end, end, red]]
    rollout_logits = []
    swarm_sizes = []

    def force_tokens(_, args, logits):
        chosen = tokens[len(rollout_logits)]
        rollout_logits.append(logits[:, -1].detach())
        return torch.full_like(logits, -1000).scatter_(-1, torch.tensor(chosen)[:, None, None].expand(
            -1, logits.size(1), 1), 0)

    head = model.head.register_forward_hook(force_tokens)
    board = model.layers[0].ag.register_forward_pre_hook(lambda _, args: swarm_sizes.append(args[1].swarm_size))
    temperature = 0.7
    try:
        attempts = grpo.sample_swarm_group(model, t, (messages, "red", "exact"), 2, 3, temperature)
    finally:
        head.remove()
        board.remove()
    assert swarm_sizes == [2, 2, 2]
    assert [[sum(mask) for _, mask, _ in attempt] for attempt in attempts] == [[1, 3], [2, 3]]
    assert attempts[0][0][0] == [end] * (len(long) - len(short)) + short + [end]
    rows = [(ids, mask) for attempt in attempts for ids, mask, _ in attempt]
    ids, mask = sft.pad_batch(rows, end)
    assert ids[0, -3:].tolist() == [end, end, end]
    replay = grpo.token_logprobs(model, ids, MemoryContext(swarm_size=2), temperature)
    for step, logits in enumerate(rollout_logits):
        pos = len(long) + step
        active = mask[:, pos].bool()
        sampled = (logits / temperature).log_softmax(-1).gather(1, ids[:, pos:pos + 1]).squeeze(1)
        torch.testing.assert_close(replay[active, pos - 1], sampled[active])

    isolated = grpo.token_logprobs(model, ids[2:], MemoryContext(swarm_size=2), temperature)
    torch.testing.assert_close(replay[2:], isolated)
    cut = grpo.token_logprobs(model, ids, MemoryContext(swarm_size=1), temperature)
    assert not torch.allclose(replay[mask[:, 1:].bool()], cut[mask[:, 1:].bool()])


def test_swarm_grpo_tiny_updates_shared_board(tmp_path, tok):
    t, tok_dir = tok
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1, board_read_self=False)
    base = ExeaLM(cfg).eval()
    ckpt = tmp_path / "base.pt"
    save(str(ckpt), base, make_optimizer(base, 1e-3, 0.0), 0, cfg, "tonn")

    messages = [[dict(role="user", content="The other agent needs to know that the answer is red.")],
                [dict(role="user", content="What is the word the other agent has?")]]
    group, temperature = 16, 0.8
    attempts = grpo.sample_swarm_group(base, t, (messages, "", "exact"), group, 1, temperature,
                                       torch.Generator().manual_seed(0))
    helper_texts = {attempt[0][2].strip() for attempt in attempts}
    answers = [attempt[1][2].strip() for attempt in attempts]
    answer = next(text for text in answers if text and text not in helper_texts and answers.count(text) < group)
    task = dict(agents=[dict(messages=agent) for agent in messages], answer=answer, reward="exact")
    path = tmp_path / "tasks.jsonl"
    path.write_text((json.dumps(task) + "\n") * 2)
    out = tmp_path / "run"
    trained = grpo.main(["--ckpt", str(ckpt), "--tokenizer", tok_dir, "--data", str(path), "--swarm", "2",
                         "--answer_agent", "1", "--prompts", "1", "--group", str(group), "--max_new_tokens", "1",
                         "--steps", "1", "--epochs", "2", "--lr", "1e-3", "--temperature", str(temperature),
                         "--device", "cpu", "--out", str(out)]).eval()
    start, end = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert (start["swarm_size"], start["answer_agent"], end["step"]) == (2, 1, 1)
    assert end["sampled_reward"] == pytest.approx(answers.count(answer) / group, abs=1e-4)
    assert not torch.equal(trained.layers[0].ag.post_w, base.layers[0].ag.post_w)
    rows = [(ids, mask) for attempt in attempts for ids, mask, _ in attempt]
    ids, mask = sft.pad_batch(rows, t.token_to_id("<|im_end|>"))
    mem = MemoryContext(swarm_size=2)
    winners = torch.tensor([text == answer for text in answers]).repeat_interleave(2)
    chosen = winners[:, None] & mask[:, 1:].bool()
    with torch.no_grad():
        before = grpo.token_logprobs(base, ids, mem, temperature)
        after = grpo.token_logprobs(trained, ids, mem, temperature)
    assert torch.isfinite(after).all() and after[chosen].mean() > before[chosen].mean()
    assert (out / "model.pt").is_file()


@pytest.mark.parametrize("held_out", [False, True])
def test_code_rewards_are_refused_without_the_flag(tmp_path, tok, held_out):
    _, tok_dir = tok
    cfg = get_config("tiny", vocab_size=330, swarm_size=1)
    model = ExeaLM(cfg)
    save(str(tmp_path / "base.pt"), model, make_optimizer(model, 1e-3, 0.1), 0, cfg, "tonn")
    code = tmp_path / "code.jsonl"
    code.write_text(json.dumps(dict(prompt="write add", answer="assert add(1, 1) == 2", reward="code")) + "\n")
    if held_out:
        train = tmp_path / "train.jsonl"
        train.write_text(json.dumps(dict(prompt="1+1", answer="2", reward="math")) + "\n")
        args = ["--data", str(train), "--val", str(code)]
    else:
        args = ["--data", str(code)]
    with pytest.raises(ValueError, match="allow_code_exec"):
        grpo.main(["--ckpt", str(tmp_path / "base.pt"), "--tokenizer", tok_dir, *args])


def test_grpo_fixes_a_model_whose_most_likely_answer_is_wrong(tmp_path, tok):
    """SFT on data that gives the wrong reply two times in three, then GRPO with a reward for the right one."""
    t, tok_dir = tok
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=t.get_vocab_size(), swarm_size=1)
    base = ExeaLM(cfg)
    save(str(tmp_path / "base.pt"), base, make_optimizer(base, 1e-3, 0.1), 0, cfg, "tonn")
    rng = np.random.default_rng(0)
    with open(tmp_path / "sft.jsonl", "w") as f:
        for k in range(360):
            w = WORDS[k % len(WORDS)]
            reply = GOOD[w] if rng.random() < 0.35 else BAD[w]
            f.write(json.dumps(dict(messages=[dict(role="user", content=w), dict(role="assistant", content=reply)])) + "\n")
    sft.main(["--ckpt", str(tmp_path / "base.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "sft.jsonl"),
              "--steps", "200", "--batch", "16", "--lr", "3e-3", "--eval_every", "200", "--device", "cpu",
              "--out", str(tmp_path / "sft")])
    with open(tmp_path / "tasks.jsonl", "w") as f:
        for k in range(60):
            w = WORDS[k % len(WORDS)]
            f.write(json.dumps(dict(prompt=w, answer=GOOD[w], reward="exact")) + "\n")
    out = tmp_path / "run"
    grpo.main(["--ckpt", str(tmp_path / "sft" / "model.pt"), "--tokenizer", tok_dir, "--data", str(tmp_path / "tasks.jsonl"),
               "--val_frac", "0.1", "--steps", "60", "--prompts", "6", "--group", "16", "--epochs", "2",
               "--max_new_tokens", "6", "--lr", "1e-3", "--beta", "0.01", "--eval_every", "20", "--device", "cpu",
               "--out", str(out)])
    log = [json.loads(line) for line in open(out / "log.jsonl")]
    assert log[0]["event"] == "start" and log[0]["tasks"] == 54 and log[0]["val_tasks"] == 6
    assert log[0]["val_reward"] <= 0.2                                      # the wrong reply is the likely one
    assert log[-1]["step"] == 60 and log[-1]["val_reward"] >= 0.8
    assert log[-1]["sampled_reward"] > log[1]["sampled_reward"] > 0.2
