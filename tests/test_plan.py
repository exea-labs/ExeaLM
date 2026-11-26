"""The compute plan (experiments/expanse/plan.py): its sizes are the code's, its hours are the document's, and every
run it prints is one the training script can start."""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pytest
import torch

from exealm import ExeaLM, Llama, count_params, get_config
from exealm.train import pretrain

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "experiments", "expanse"), os.path.join(ROOT, "experiments")]
import eval_text  # noqa: E402
import plan  # noqa: E402
import throughput  # noqa: E402


def printed(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        plan.main(argv)
    return out.getvalue()


def overrides(name):
    return json.load(open(os.path.join(ROOT, "experiments", "expanse", name)))


def test_the_plans_parameter_counts_are_the_codes():
    core = overrides("exealm_core.json")
    for preset, s in plan.SIZES.items():
        with torch.device("meta"):                                        # the plain Llama: what pretrain builds for
            counts = dict(exealm=count_params(ExeaLM(get_config(preset, vocab_size=32768))),      # --model llama
                          core=count_params(ExeaLM(get_config(preset, vocab_size=32768, **core))),
                          llama=count_params(Llama(get_config(preset, vocab_size=32768, attn_gate=False,
                                                              qk_norm=False))))
        if "qwen" in s:
            with torch.device("meta"):
                counts["qwen"] = count_params(Llama(get_config(preset, vocab_size=32768,
                                                               **overrides(f"qwen_{preset}.json"))))
        for model, n in counts.items():
            assert round(n / 1e6, 2) == s[model], (preset, model, n)
    cfg = get_config("500m", **core)
    assert "RG" not in cfg.menu and "AG" not in cfg.menu and cfg.swarm_size == 1 and not cfg.pinned
    assert cfg.value_residual and cfg.xsa and cfg.tonn_every == 4                   # ExeaLM is the whole recipe
    for preset in plan.SIZES:                                             # the base Qwen3.5-style model: no ExeaLM parts
        q = overrides(f"qwen_{preset}.json")
        assert q["attn_gate"] and q["qk_norm"] and not q.get("value_residual") and not q.get("xsa")
    with torch.device("meta"):
        nine = count_params(ExeaLM(get_config("9b", vocab_size=32768)))
    assert 8.5e9 < nine < 9.5e9                                           # the run the study is evidence for


def test_the_table_fits_the_allocation_and_is_the_one_in_the_budget_document():
    text = printed([])
    rows = [l for l in text.splitlines() if "| `" in l]
    hours = {l.split("`")[1]: int(l.split("|")[-2].replace(",", "")) for l in rows}
    assert hours["exealm_3b_4.1x_s0"] == 733 and hours["qwen_3b_4.1x_s0"] == 736    # about 12B tokens at 81 TFLOP/s
    assert len(hours) == 28 and all(f"qwen_{n[7:]}" in hours for n in hours if n.startswith("exealm_"))
    assert all(f"llama_{n[7:]}" in hours for n in hours if n.startswith("exealm_") and "_4.1x_" in n and "_3b_" not in n)
    assert {"exealm_100m_111x_s0", "qwen_100m_111x_s0"} <= set(hours)              # the 9B run's tokens per parameter
    assert "llama_3b_4.1x_s0" not in hours and not [n for n in hours if n.startswith("gated_")]
    assert {n.split("_")[1] for n in hours} == {"100m", "250m", "500m", "3b"}       # four sizes on one curve
    assert {l.split("|")[5].strip() for l in rows if "exealm" in l and "_4.1x_" in l} == {"4.0"}  # one curve (core's steps)
    assert "ExeaLM at 9B on 1T tokens" in text and "about 190,8" in text           # the price of the 9B run
    reserve = int(text.split("**Reserve of the 2,000** | | | | | | **")[1].split("**")[0].replace(",", ""))
    assert 50 < reserve < 400
    budget = open(os.path.join(ROOT, "docs", "budget.md")).read()
    for line in text.splitlines():
        if line.startswith("|"):
            assert line in budget, f"docs/budget.md is missing this line of plan.py's table:\n{line}"


def test_measured_rates_and_the_four_gpu_speed_up_rescale_the_plan(tmp_path):
    rows = [dict(size="0.5B", model="hybrid", precision="bf16", effective_TFLOPs=100.0, compile=0, tf32=0),
            dict(size="0.5B", model="exealm", precision="bf16", effective_TFLOPs=162.0, compile=0, tf32=0),
            dict(size="0.5B", model="exealm", precision="bf16", effective_TFLOPs=500.0, compile=1, tf32=0),
            dict(size="0.5B", model="llama", precision="fp32", effective_TFLOPs=999.0, compile=0, tf32=0),
            dict(size="0.5B", model="gated", precision="bf16", effective_TFLOPs=777.0, compile=0, tf32=0),
            dict(size="1B", model="exealm", precision="bf16", effective_TFLOPs=0.0, compile=0, tf32=0),
            dict(size="3B", model="exealm", precision="bf16", error="out of memory at batch 1")]
    path = tmp_path / "throughput_h100.json"
    path.write_text(json.dumps(rows))
    rates, source = plan.measured(rows)
    assert rates["500m"] == dict(exealm=100.0, llama=124.8, qwen=100.0)       # the memory-free hybrid's row, as trained;
    assert source["500m"] == dict(exealm="hybrid", llama="MI300X", qwen="MI300X")  # a gated Llama is not the baseline
    assert rates["1b"] == dict(plan.MI300X["1b"], qwen=81.0)             # nothing measured: the MI300X figure stays
    text = printed(["--rates", str(path), "--speedup", "3.2"])
    row = next(l for l in text.splitlines() if "| `exealm_500m_4.1x_s0`" in l)
    full = plan.gpu_hours(plan.SIZES["500m"]["mid"], plan.steps_of("500m", plan.TPP) * plan.STEP_TOKENS, 81.0)
    assert int(row.split("|")[-2]) == round(full * 81 / 100 * 4 / 3.2)    # the measured rate, 4 / 3.2 the charge
    one = next(l for l in text.splitlines() if "| `exealm_250m_4.1x_s0`" in l)
    assert int(one.split("|")[-2]) == round(plan.gpu_hours(plan.SIZES["250m"]["mid"], plan.steps_of("250m", plan.TPP) * plan.STEP_TOKENS, 56.2))


def test_every_run_is_a_whole_number_of_swarms_and_passes_per_step():
    for name, part, preset, model, seed, steps in plan.runs("mid"):
        text = printed(["--run", name, "--lr", "1e-3"])
        args = text.split('ARGS="')[1].split('"')[0].split()
        get = lambda k: args[args.index(k) + 1]                           # noqa: E731
        gpus = 1 if "nairr-gpu-shared" in text else 4
        assert gpus == plan.SIZES[preset]["gpus"]
        rows, stream = int(get("--batch")), int(get("--stream")) if "--stream" in args else 1
        assert stream == (plan.STREAM if model == "exealm" else 1)             # streaming is part of ExeaLM only
        assert rows * int(get("--accum")) * int(get("--seq")) * stream == plan.STEP_TOKENS
        assert rows * stream == plan.SIZES[preset]["batch"] * gpus                # windows per GPU x GPUs
        assert rows % gpus == 0 and rows * stream * int(get("--eval_batches")) == 512   # the same 512 windows
        assert int(get("--steps")) == steps and name.startswith(f"{model}_{preset}_")
        assert get("--seed") == str(seed) and get("--preset") == preset
        assert get("--compile") == "1" if model in ("llama", "qwen") and gpus == 1 else "--compile" not in args
        assert (get("--zero") == "1") if preset == "3b" else "--zero" not in args
        if model == "llama":                                              # the plain Llama: no --config
            assert get("--model") == "llama" and "--config" not in args
        elif model == "qwen":
            assert get("--model") == "llama" and os.path.exists(os.path.join(ROOT, get("--config")))
        else:
            assert get("--search") == "0" and get("--config") == f"experiments/expanse/exealm_mid_{preset}.json"
        assert ("--ntasks-per-node=1" in text) == (gpus == 1)
        assert f"SBATCH_JOB_NAME=exealm-{name} " in text                  # the job is named after its run
        hh = int(text.split("SBATCH_TIMELIMIT=")[1].split(":")[0])        # a limit sized to the run, not always 48 h
        assert 1 <= hh <= 48 and (hh == 48) == (preset == "3b")
        save = int(text.split("SAVE_MIN=")[1].split()[0])                # checkpoints on a timer only where they pay
        assert save == (120 if preset == "3b" else 60 if "_111x_" in name else 0), (name, save)
    memory = printed(["--run", "exealm_3b_4.1x_s0", "--lr", "1e-3", "--variant", "memory"])
    assert "--config" not in memory.split('ARGS="')[1].split('"')[0]
    with pytest.raises(SystemExit, match="needs --lr"):
        plan.main(["--run", "exealm_3b_4.1x_s0"])
    with pytest.raises(SystemExit, match="does not divide"):
        plan.main(["--run", "exealm_3b_4.1x_s0", "--lr", "1e-3", "--batch", "6"])
    check = printed(["--lr_check", "500m", "--lrs", "3e-4,6e-4"])
    assert check.count("sbatch ") == 6 and "--steps 1000 --warmup 100" in check and "qwen_500m.json" in check
    assert check.count("SAVE_MIN=0") == 6                                  # a learning-rate check keeps only its log
    assert printed(["--lr_check", "3b", "--lrs", "3e-4,6e-4"]).count("sbatch ") == 4    # no plain Llama at 3B


def test_a_run_too_long_for_its_data_is_flagged(tmp_path):
    pack = tmp_path / "en"
    pack.mkdir()
    (pack / "meta.json").write_text(json.dumps(dict(n_tokens=400_000_000)))
    mix = tmp_path / "mix.json"
    mix.write_text(json.dumps({"sources": {"en": {"paths": str(pack), "weight": 1.0}}}))
    assert plan.pack_tokens(str(mix)) == 400_000_000
    assert "NOT READY" in printed(["--run", "exealm_3b_4.1x_s0", "--lr", "1e-3", "--data", str(mix)])
    (pack / "meta.json").write_text(json.dumps(dict(n_tokens=3_000_000_000)))
    assert "NOT READY" not in printed(["--run", "exealm_3b_4.1x_s0", "--lr", "1e-3", "--data", str(mix)])


def test_throughput_times_the_plans_two_models():
    dev = torch.device("cpu")
    built = {k: throughput.build("tiny", k, 256, dev) for k in throughput.KINDS}
    n = {k: count_params(m) for k, (m, _) in built.items()}
    assert n["gated"] > n["llama"]                                        # the gate and QK-norm are parameters
    assert built["exealm"][1].swarm_size == get_config("tiny").swarm_size and built["hybrid"][1].swarm_size == 1
    kinds = lambda m: {s["tmpl"] for lay in m.specs for s in lay}          # noqa: E731
    assert {"RG", "AG"} <= kinds(built["exealm"][0]) and not {"RG", "AG"} & kinds(built["hybrid"][0])
    assert built["qwen"][0].cfg.delta_layers and n["qwen"] != n["gated"]   # Gated DeltaNet in 3 of every 4 layers
    with pytest.raises(SystemExit):
        throughput.build("tiny", "mamba", 256, dev)


def pack(path, n_tokens=6000, vocab=256, seed=0):
    path.mkdir(parents=True)
    np.random.default_rng(seed).integers(0, vocab, n_tokens).astype(np.uint16).tofile(path / "tokens.bin")
    offsets = np.arange(0, n_tokens, 100, dtype=np.int64)
    np.save(path / "offsets.npy", offsets)
    (path / "meta.json").write_text(json.dumps(dict(dtype="uint16", vocab_size=vocab, n_tokens=n_tokens,
                                                    n_docs=len(offsets), bytes_per_token=2.0)))
    return str(path)


def test_eval_text_scores_swarm_trained_and_plain_models_alike_and_groups_seeds(tmp_path, capsys):
    train = pack(tmp_path / "data" / "train")
    for i, name in enumerate(("val_a", "val_b")):
        pack(tmp_path / "data" / name, seed=i + 1)
    common = ["--preset", "tiny", "--data", train, "--seq", "16", "--batch", "8", "--steps", "4", "--eval_every", "4",
              "--search", "0", "--device", "cpu"]
    runs = []
    for name, extra in (("llama_s0", ["--model", "llama"]), ("exealm_s0", []), ("exealm_s1", ["--seed", "1"])):
        runs.append(str(tmp_path / name))
        pretrain.main(common + extra + ["--out", runs[-1]])
    capsys.readouterr()
    out = tmp_path / "text.json"
    rows, means = eval_text.main(runs + ["--root", str(tmp_path / "data"), "--sets", "val_a,val_b", "--batch", "4",
                                         "--seq", "16", "--eval_batches", "2", "--device", "cpu", "--out", str(out)])
    assert [r["model"] for r in rows] == ["llama", "tonn", "tonn"] and all(r["step"] == 4 for r in rows)
    assert set(rows[0]["bits_per_byte"]) == {"a", "b"}
    assert all(3.5 < r["mean"] < 4.5 for r in rows)                       # random tokens: 8 bits each, 2 bytes each
    assert list(means) == ["llama", "exealm"]
    assert means["exealm"] == pytest.approx((rows[1]["mean"] + rows[2]["mean"]) / 2, abs=1e-4)
    assert json.load(open(out))["means"] == means
    again, _ = eval_text.main(runs[:1] + ["--root", str(tmp_path / "data"), "--sets", "val_a,val_b", "--batch", "4",
                                          "--seq", "16", "--eval_batches", "2", "--device", "cpu"])
    assert again[0]["bits_per_byte"] == rows[0]["bits_per_byte"]          # the same windows every time


def test_the_runner_picks_learning_rates_and_submits_every_phase(tmp_path):
    import subprocess
    import pick_lr
    keep = tmp_path / "runs"
    for name, scores, done in (("lr_exealm_100m_0.0003", [2.1, 1.9], True), ("lr_exealm_100m_0.001", [2.0, 1.7], True),
                               ("lr_qwen_100m_0.001", [2.2, float("nan")], True), ("lr_qwen_100m_0.0003", [2.3, 1.95], True),
                               ("lr_exealm_250m_0.001", [2.0], False)):
        (keep / name).mkdir(parents=True)
        lines = [json.dumps(dict(event="eval", step=i, val_bits_per_byte=v)) for i, v in enumerate(scores)]
        lines += [json.dumps(dict(event="end"))] if done else []
        (keep / name / "log.jsonl").write_text("\n".join(lines) + "\n")
    out = tmp_path / "lr_choice.json"
    chosen = pick_lr.main(["--keep", str(keep), "--out", str(out)])
    assert chosen == {"exealm_100m": 0.001, "qwen_100m": 0.0003}          # nan loses; 250m is not finished
    text = printed(["--run", "qwen_100m_4.1x_s1", "--lr", "auto", "--lr_file", str(out)])
    assert "--lr 0.0003" in text and "--anneal_data data/mix_pretrain_anneal.json --anneal_from 0.9" in text
    with pytest.raises(SystemExit, match="no learning rate"):
        plan.main(["--run", "exealm_3b_4.1x_s0", "--lr", "auto", "--lr_file", str(out)])
    assert printed(["--list", "3B"]).split() == ["exealm_3b_4.1x_s0", "qwen_3b_4.1x_s0"]
    assert len(printed(["--list", "curve"]).split()) == 24
    go = os.path.join(ROOT, "experiments", "expanse", "go.sh")
    env = dict(os.environ, DRY="1")
    lr = subprocess.run(["bash", go, "lr"], capture_output=True, text=True, env=env, cwd=ROOT).stdout
    assert lr.count("sbatch ") == 3 * 3 * 3 + 2 * 2                         # 3 rates and three models below 3B;
                                                                            # 2 rates and two models at 3B
    first = subprocess.run(["bash", go, "first"], capture_output=True, text=True, env=env, cwd=ROOT).stdout
    assert "first_contact.sb" in first
    nine = subprocess.run(["bash", go, "nine_b"], capture_output=True, text=True, env=env, cwd=ROOT).stdout
    assert nine.count("nine_b.sb") == 3 and nine.count("afterok") == 2          # the larger jobs wait for one node
    assert os.path.isdir(os.path.join(ROOT, "logs", "expanse"))                  # Slurm writes job output there


def test_the_anneal_mix_takes_over_for_the_last_steps(tmp_path, monkeypatch):
    from exealm.data.loader import SyntheticShards
    from exealm.train import pretrain as pt
    calls = []
    real = SyntheticShards.batch
    monkeypatch.setattr(SyntheticShards, "batch", lambda self, *a, **k: (calls.append(id(self)), real(self, *a, **k))[1])
    import exealm.train.pretrain as mod
    made = []
    orig = mod.open_data
    monkeypatch.setattr(mod, "open_data", lambda spec: (made.append(spec), SyntheticShards() if spec == "synthetic" else orig(spec))[1])
    pt.main(["--preset", "tiny", "--data", "synthetic", "--anneal_data", "synthetic", "--anneal_from", "0.5", "--seq", "16",
             "--batch", "2", "--steps", "4", "--eval_every", "4", "--search", "0", "--device", "cpu", "--out", str(tmp_path / "a")])
    assert made.count("synthetic") >= 2                                     # the anneal mix was opened
    assert len(set(calls)) >= 2                                             # and batches came from both


def test_the_report_finds_the_gaps_the_multiplier_and_the_price(tmp_path):
    import report
    rows, A, k = [], 40.0, 0.05                              # the Qwen3.5-style curve: bits per byte = A x C^-k
    for name, part, preset, model, seed, steps in plan.runs():
        params = plan.SIZES[preset]["core" if model == "exealm" else model]
        c = 6 * params * 1e6 * steps * plan.STEP_TOKENS
        if model == "exealm":
            c *= 2                                           # ExeaLM scores what the baseline needs twice the compute for
        score = A * c ** -k + (0.05 if model == "llama" else 0) + 0.001 * seed
        rows.append(dict(run=f"/keep/{name}", params_M=params, mean=score, step=steps,
                         bits_per_byte={"code": score + 0.1, "en": score - 0.1}))
    rows.append(dict(run="/keep/lr_exealm_100m_0.001", params_M=1, mean=9.0, step=1, bits_per_byte={"en": 9.0}))
    (tmp_path / "text.json").write_text(json.dumps(dict(runs=rows, means={})))
    out = report.main(["--results", str(tmp_path)])
    assert [r["size"] for r in out["curve"]] == ["100m", "250m", "500m", "3b"]
    assert all(r["gap_qwen"] > 0 and r["ahead_qwen"] in ("3 of 3", "2 of 2", "1 of 1") for r in out["curve"])
    assert out["curve"][-1]["llama"] is None                 # no plain Llama at 3B
    qwen = [r for r in out["multiplier"] if r["baseline"] == "qwen"]
    assert len(qwen) == 4 and all(abs(r["flops_multiplier"] - 2.0) < 0.05 for r in qwen)
    assert all(r["gpu_hour_multiplier"] == r["flops_multiplier"] for r in qwen)    # the same assumed speed
    assert out["long"][0]["tpp"] == 111 and out["long"][0]["gap_qwen"] > 0
    assert out["nine_b"]["gpu_hours"] == round(plan.gpu_hours(plan.NINE_B_PARAMS / 1e6, 1e12, 81.0))
    assert out["nine_b"]["rate_from"] == "ExeaLM at 3B"
    text = (tmp_path / "report.md").read_text()
    assert "Compute multiplier" in text and "| 3b | ExeaLM |" in text
    (tmp_path / "nine_b.json").write_text(json.dumps([dict(nodes=1, effective_TFLOPs_per_gpu=50.0),
                                                     dict(nodes=4, effective_TFLOPs_per_gpu=60.0),
                                                     dict(nodes=4, error="out of memory at batch 4")]))
    assert report.main(["--results", str(tmp_path)])["nine_b"]["rate_TFLOPs"] == 60.0     # the most nodes measured
