"""experiments/muon_table.py picks each optimizer's learning rate on seed 0 and pairs the two on every shared seed."""
import importlib.util
import json
import pathlib

path = pathlib.Path(__file__).parents[1] / "experiments" / "muon_table.py"   # CI runs bare pytest: load by path
spec = importlib.util.spec_from_file_location("muon_table", path)
muon_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muon_table)


def write(folder, opt, lr, seed, bpb):
    row = dict(model="tonn_core", tag=f"m13_{opt}_lr{lr:g}_s{seed}", lr=lr, optimizer=opt, seed=seed, steps=3000,
               secs=100.0, bits_per_byte=dict(en=bpb, ja_wiki=bpb, ja_web=bpb, code=bpb))
    if opt == "adamw" and lr == 1e-3:
        del row["optimizer"]                                   # rows from before --optimizer existed are AdamW
    (folder / f"lm_compare_{row['tag']}.json").write_text(json.dumps(dict(rows=[row])))


def test_best_rates_come_from_seed_0_and_the_difference_is_paired(tmp_path, monkeypatch, capsys):
    for opt, lr, seed, bpb in [("adamw", 1e-3, 0, 1.52), ("adamw", 2e-3, 0, 1.50), ("adamw", 4e-3, 0, 1.51),
                               ("adamw", 4e-3, 1, 1.40),       # better on seed 1, but seed 0 picks the rate
                               ("adamw", 2e-3, 1, 1.51), ("muon", 2e-3, 0, 1.49), ("muon", 4e-3, 0, 1.495),
                               ("muon", 2e-3, 1, 1.50)]:
        write(tmp_path, opt, lr, seed, bpb)
    monkeypatch.setattr("sys.argv", ["muon_table.py", str(tmp_path)])
    muon_table.main()
    out = json.loads((tmp_path / "summary.json").read_text())
    assert out["best_lr"] == {"adamw": 2e-3, "muon": 2e-3}
    assert out["seeds"] == [0, 1]
    assert [round(d, 4) for d in out["diffs"]] == [0.01, 0.01]    # AdamW minus Muon: positive means Muon is better
    assert out["runs"]["adamw_lr0.001"] == {"0": 1.52}
    text = capsys.readouterr().out
    assert "better on 2 of 2 seeds" in text and "| muon | 0.002 | 1.4900 | 1.5000 | 1.4950 | 100 |" in text
