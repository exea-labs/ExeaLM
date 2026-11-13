"""The search logs the architecture it leaves after every round, and experiments/plot_search.py reads it back."""
import importlib.util
import json
import pathlib

import pytest

from exealm import get_config
from exealm.config import MENU
from exealm.train.pretrain import main

path = pathlib.Path(__file__).parents[1] / "experiments" / "plot_search.py"   # CI runs bare pytest: load by path
spec = importlib.util.spec_from_file_location("plot_search", path)
plot_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_search)


def write_log(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(path)


def arch(*rows):
    return [dict(r) for r in rows]


CFG = dict(n_layers=2, slots=4, attn_layers=[])
A0 = arch({"RG": 1, "AG": 1, "C": 2}, {"RG": 1, "AG": 1, "EP": 2})
A1 = arch({"RG": 1, "AG": 1, "C": 1, "EP": 1}, {"RG": 1, "AG": 1, "EP": 2})
A2 = arch({"RG": 1, "AG": 2, "C": 1}, {"RG": 1, "AG": 1, "EP": 2})


def test_search_records_carry_the_architecture_and_a_resumed_log_reads_the_same(tmp_path):
    common = ["--preset", "tiny", "--data", "synthetic", "--steps", "40", "--eval_every", "20", "--search_every", "4",
              "--search_until", "0.9", "--device", "cpu"]
    a, b = tmp_path / "a", tmp_path / "b"
    main(common + ["--out", str(a)])
    main(common + ["--out", str(b), "--stop_at", "6"])
    main(common + ["--out", str(b), "--resume", "auto"])
    cfg = get_config("tiny")
    recs = [json.loads(l) for l in (a / "log.jsonl").read_text().splitlines()]
    searches = [r for r in recs if r["event"] == "search"]
    assert searches and any(r["replaced"] > 0 for r in searches)
    for r in searches:
        assert len(r["architecture"]) == cfg.n_layers
        assert all(sum(row.values()) == cfg.slots and row["RG"] >= 1 and row["AG"] >= 1 for row in r["architecture"])
    assert searches[-1]["architecture"] == recs[-1]["architecture"]           # frozen after the last round
    straight, resumed = plot_search.read_log(str(a / "log.jsonl")), plot_search.read_log(str(b / "log.jsonl"))
    assert straight["rounds"] == [r["step"] for r in searches]
    assert straight["points"][0] == (0, recs[0]["architecture"]) and straight["end"] == 40
    assert resumed["points"] == straight["points"] and resumed["replaced"] == straight["replaced"]


def test_rounds_run_again_after_a_resume_replace_the_old_ones(tmp_path):
    log = write_log(tmp_path / "log.jsonl", [
        dict(event="start", model="tonn", cfg=CFG, architecture=A0),
        dict(event="search", step=10, replaced=1, architecture=A1),
        dict(event="eval", step=20),
        dict(event="search", step=20, replaced=1, architecture=A0),      # lost: the run crashed after step 20
        dict(event="eval", step=30),
        dict(event="resume", step=20, total_steps=40),                     # back from the step-20 checkpoint
        dict(event="search", step=20, replaced=1, architecture=A2),
        dict(event="eval", step=40),
        dict(event="end", architecture=A2),
    ])
    run = plot_search.read_log(log)
    assert run["rounds"] == [10, 20] and run["replaced"] == [1, 1]
    assert run["points"] == [(0, A0), (10, A1), (20, A2), (40, A2)]
    assert run["layers"] == [0, 1] and run["slots"] == 4


def test_the_last_run_in_a_reused_folder_is_read_and_hybrid_layers_keep_their_numbers(tmp_path):
    hybrid = dict(CFG, n_layers=6, attn_layers=[1, 2, 4, 5])
    log = write_log(tmp_path / "log.jsonl", [
        dict(event="start", model="tonn", cfg=CFG, architecture=A1),
        dict(event="eval", step=10),
        dict(event="start", model="tonn", cfg=hybrid, architecture=A0),
        dict(event="search", step=5, replaced=0, architecture=A0),
        dict(event="eval", step=10),
        dict(event="end", architecture=A0),
    ])
    run = plot_search.read_log(log)
    assert run["layers"] == [0, 3]                                           # model layer numbers, not TONN indices
    assert run["points"] == [(0, A0), (5, A0), (10, A0)]


def test_logs_without_an_architecture_are_refused_clearly(tmp_path):
    old = write_log(tmp_path / "old.jsonl", [dict(event="start", model="tonn", cfg=CFG, architecture=A0),
                                             dict(event="search", step=10, replaced=0)])
    with pytest.raises(ValueError, match="architecture"):
        plot_search.read_log(old)
    llama = write_log(tmp_path / "llama.jsonl", [dict(event="start", model="llama", cfg=CFG, architecture=None)])
    with pytest.raises(ValueError, match="not a TONN run"):
        plot_search.read_log(llama)


def test_every_circuit_type_has_one_family_and_one_colour():
    assert sorted(plot_search.ORDER) == sorted(MENU) and len(plot_search.ORDER) == len(set(plot_search.ORDER))
    assert len(plot_search.COLORS) == len(plot_search.FAMILIES)


def test_table_lists_the_start_the_rounds_that_replaced_something_and_the_end(tmp_path):
    log = write_log(tmp_path / "log.jsonl", [
        dict(event="start", model="tonn", cfg=CFG, architecture=A0),
        dict(event="search", step=10, replaced=1, architecture=A1),
        dict(event="search", step=20, replaced=0, architecture=A1),
        dict(event="eval", step=40),
        dict(event="end", architecture=A1),
    ])
    rows = plot_search.table(plot_search.read_log(log)).splitlines()[2:]
    assert [r.split("|")[1].strip() for r in rows] == ["start", "10", "end"]
    assert rows[1] == "| 10 | 1 | RG, AG, C, EP | RG, AG, EP ×2 |"


def test_plot_writes_an_image(tmp_path):
    pytest.importorskip("matplotlib")
    log = write_log(tmp_path / "log.jsonl", [
        dict(event="start", model="tonn", cfg=CFG, architecture=A0),
        dict(event="search", step=10, replaced=1, architecture=A2),
        dict(event="eval", step=40),
        dict(event="end", architecture=A2),
    ])
    out = tmp_path / "fig" / "search.png"
    plot_search.main([log, log, "--labels", "a", "b", "--out", str(out)])
    assert out.stat().st_size > 1000
