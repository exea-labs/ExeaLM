"""Measuring how agents use the board: what they read (nothing, each other, themselves) and how loudly they post."""
import json

import torch

from exealm import AgentBoard, ExeaLM, MemoryContext, get_config
from exealm.train.pretrain import main


def run_recorded(mem, B=4, T=40, **cfg_kw):
    torch.manual_seed(0)
    cfg = get_config("tiny", **cfg_kw)
    m = ExeaLM(cfg).eval()
    for lay in m.layers:
        lay.ag.record = True
    with torch.no_grad():
        m(torch.randint(0, cfg.vocab_size, (B, T)), mem)
    return [lay.ag.last for lay in m.layers]


def test_reading_shares_add_up_and_posts_are_measured():
    for st in run_recorded(MemoryContext(swarm_size=2)):
        assert st is not None
        read = st["read_nothing"] + st["read_others"] + st["read_self"] + st["read_board"]
        assert abs(read - 1) < 1e-4
        assert st["read_others"] > 0 and st["read_board"] == 0
        assert 0 < st["post_strength"] < 1 and 0 <= st["quiet_posts"] <= 1


def test_a_lone_agent_can_only_read_itself_or_nothing():
    for st in run_recorded(MemoryContext(swarm_size=1)):
        assert st["read_others"] == 0 and st["read_self"] > 0
    for st in run_recorded(MemoryContext(swarm_size=1), board_read_self=False):
        assert st["read_self"] == 0 and abs(st["read_nothing"] - 1) < 1e-5


def test_reading_the_persistent_board_is_counted():
    torch.manual_seed(0)
    cfg = get_config("tiny")
    m = ExeaLM(cfg).eval()
    board = AgentBoard()
    x = torch.randint(0, cfg.vocab_size, (2, 40))
    with torch.no_grad():
        m(x[:1], MemoryContext(board=board, agent_ids=torch.tensor([0]), write_board=True))
        for lay in m.layers:
            lay.ag.record = True
        m(x[1:], MemoryContext(board=board, agent_ids=torch.tensor([1])))
    assert all(lay.ag.last["read_board"] > 0 for lay in m.layers)


def test_too_short_to_read_gives_no_stats():
    assert all(st is None for st in run_recorded(MemoryContext(swarm_size=2), T=8))


def test_pretraining_logs_board_use_and_board_worth(tmp_path):
    main(["--preset", "tiny", "--data", "synthetic", "--steps", "40", "--eval_every", "20", "--search_every", "10",
          "--device", "cpu", "--out", str(tmp_path)])
    recs = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    evals = [r for r in recs if r["event"] == "eval"]
    searches = [r for r in recs if r["event"] == "search"]
    cfg = get_config("tiny")
    assert evals and all(len(r["board"]["read_nothing_per_layer"]) == cfg.n_layers for r in evals)
    assert all(0 <= r["board"]["read_nothing"] <= 1 for r in evals)
    assert searches and all(len(r["board_worth"]) == cfg.n_layers for r in searches)
    assert all(w is not None for w in searches[0]["board_worth"])          # every layer keeps its pinned AG slot


def test_baseline_runs_have_no_board_stats(tmp_path):
    main(["--preset", "tiny", "--data", "synthetic", "--steps", "10", "--eval_every", "10", "--model", "llama",
          "--device", "cpu", "--out", str(tmp_path)])
    evals = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert all("board" not in r for r in evals if r["event"] == "eval")
