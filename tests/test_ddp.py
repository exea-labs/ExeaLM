"""Pretraining in several processes (DistributedDataParallel, gloo on the CPU): the processes stay one model through
search rounds that replace circuits, train like one process on the same total batch (also with --accum), resume
exactly, notice if they ever drift apart, and stop together at a time limit."""
import json
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from exealm.train.pretrain import main

pytestmark = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs gloo")

# Search rounds every 4 steps that replace circuits (as in test_training.py's resume test), and retrieval on.
ARGS = ["--preset", "tiny", "--data", "synthetic", "--steps", "40", "--eval_every", "20", "--search_every", "4",
        "--search_until", "0.9", "--datastore_chunks", "300", "--refresh_every", "15", "--device", "cpu"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank, world, port, argv, out, drift=False):
    """One process of a torchrun-style run: the same environment torchrun sets, then pretrain's main(). drift: process
    1 nudges one weight before every optimizer step, which DDP alone would never let happen."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank),
                      LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    if drift and rank == 1:
        real = torch.nn.utils.clip_grad_norm_

        def clip_and_nudge(params, *args, **kwargs):                  # called once per step, just before the update
            params = list(params)
            out = real(params, *args, **kwargs)
            with torch.no_grad():
                params[0].add_(1e-3)
            return out

        torch.nn.utils.clip_grad_norm_ = clip_and_nudge
    model = main(argv)
    torch.save({n: p.detach().clone() for n, p in model.named_parameters()}, os.path.join(out, f"rank{rank}.pt"))


def run(argv, out, world=2, drift=False):
    os.makedirs(out, exist_ok=True)
    mp.spawn(worker, args=(world, free_port(), argv, str(out), drift), nprocs=world, join=True)
    return [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(world)]


def records(folder, event):
    return [r for r in map(json.loads, (folder / "log.jsonl").read_text().splitlines()) if r["event"] == event]


def test_two_processes_stay_one_model_and_train_like_one(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    main(ARGS + ["--out", str(one)])
    ranks = run(ARGS + ["--out", str(two)], two)
    for name, p in ranks[0].items():                                  # through every replacement and every update
        assert torch.equal(p, ranks[1][name]), name
    s1, s2 = records(one, "search"), records(two, "search")
    assert sum(r["replaced"] for r in s2) > 0                         # rounds really replaced circuits
    assert [r["architecture"] for r in s1] == [r["architecture"] for r in s2]   # the same choices as one process
    v1, v2 = records(one, "eval"), records(two, "eval")
    assert len(v2) == len(v1) and all(abs(a["val_bits_per_token"] - b["val_bits_per_token"]) < 0.01
                                      for a, b in zip(v1, v2))
    starts = records(two, "start")
    assert len(starts) == 1 and starts[0]["processes"] == 2          # process 0 logs for both
    assert starts[0]["tokens_per_step"] == records(one, "start")[0]["tokens_per_step"] == 8 * 128
    assert all(e["processes_in_sync"] for e in v2) and v2[-1]["tokens"] == 40 * 8 * 128


def test_a_stopped_two_process_run_resumes_exactly(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run(ARGS + ["--out", str(a)], tmp_path / "wa")
    run(ARGS + ["--out", str(b), "--stop_at", "6"], tmp_path / "wb1")
    run(ARGS + ["--out", str(b), "--resume", "auto"], tmp_path / "wb2")
    assert [(r["step"], r["saved_processes"]) for r in records(b, "resume")] == [(6, 2)]
    assert any(r["replaced"] for r in records(b, "search") if r["step"] > 6)
    full, resumed = torch.load(a / "model.pt", weights_only=False), torch.load(b / "model.pt", weights_only=False)
    assert full["specs"] == resumed["specs"]
    for k, v in full["model"].items():
        assert torch.equal(v, resumed["model"][k]), k


def test_the_total_batch_must_split_into_whole_swarms(tmp_path):
    with pytest.raises(Exception, match="whole swarms"):             # 6 rows: 3 per process, but swarms of 2
        run(ARGS + ["--batch", "6", "--out", str(tmp_path / "x")], tmp_path / "x")


def test_accumulated_steps_on_two_processes_match_one_process(tmp_path):
    """A Llama has no swarms and no running statistics: 2 passes of 8 rows split over 2 processes, with the gradients
    averaged on the second pass only, must give the step of 1 process."""
    common = ["--preset", "tiny", "--data", "synthetic", "--device", "cpu", "--model", "llama", "--steps", "5",
              "--eval_every", "5", "--batch", "8", "--accum", "2"]
    main(common + ["--out", str(tmp_path / "one")])
    ranks = run(common + ["--out", str(tmp_path / "two")], tmp_path / "two")
    assert records(tmp_path / "two", "start")[0]["tokens_per_step"] == 2 * 8 * 128
    one = torch.load(tmp_path / "one" / "model.pt", weights_only=False)["model"]
    for name, p in ranks[0].items():
        assert torch.equal(p, ranks[1][name]), name
        assert torch.allclose(p, one[name], atol=1e-4), name


def test_processes_that_drift_apart_are_noticed_and_put_back(tmp_path):
    args = ARGS + ["--steps", "10", "--eval_every", "5", "--search", "0", "--out", str(tmp_path / "run")]
    ranks = run(args, tmp_path / "run", drift=True)
    assert [e["processes_in_sync"] for e in records(tmp_path / "run", "eval")] == [False, False]
    for name, p in ranks[0].items():                                  # process 0's model, copied to process 1
        assert torch.equal(p, ranks[1][name]), name


def test_a_llama_that_drifts_apart_is_put_back_too(tmp_path):
    # Expanse jobs 56074847-51: the put-back read model.specs, which only ExeaLM has, and crashed every baseline run.
    args = ["--preset", "tiny", "--model", "llama", "--data", "synthetic", "--steps", "10", "--eval_every", "5",
            "--search", "0", "--device", "cpu", "--out", str(tmp_path / "run")]
    ranks = run(args, tmp_path / "run", drift=True)
    assert [e["processes_in_sync"] for e in records(tmp_path / "run", "eval")] == [False, False]
    for name, p in ranks[0].items():
        assert torch.equal(p, ranks[1][name]), name


ZERO = ["--preset", "tiny", "--data", "synthetic", "--device", "cpu", "--steps", "8", "--eval_every", "4",
        "--batch", "8", "--search", "0"]


@pytest.mark.parametrize("model", ["tonn", "llama"])
def test_zero_splits_adam_state_and_trains_exactly_like_adamw(tmp_path, model):
    # --zero 1 (ZeRO stage 1) is how a 3B model fits four 80 GB GPUs: each process keeps a part of AdamW's state.
    common = ZERO + ["--model", model]
    plain = run(common + ["--out", str(tmp_path / "plain")], tmp_path / "plain")
    zero = run(common + ["--zero", "1", "--out", str(tmp_path / "zero")], tmp_path / "zero")
    for name, p in plain[0].items():
        assert torch.equal(zero[0][name], zero[1][name]), name           # the processes stay one model
        assert torch.allclose(p, zero[0][name], atol=1e-6), name         # and it is AdamW's model
    assert all(e["processes_in_sync"] for e in records(tmp_path / "zero", "eval"))
    ck = torch.load(tmp_path / "zero" / "model.pt", weights_only=False)  # process 0 saved every part of the state
    assert len(ck["opt"]["state"]) == len(torch.load(tmp_path / "plain" / "model.pt", weights_only=False)["opt"]["state"])


def test_a_zero_run_stopped_by_its_time_limit_resumes_exactly(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    common = ZERO + ["--model", "llama", "--zero", "1"]
    run(common + ["--out", str(a)], tmp_path / "wa")
    run(common + ["--out", str(b), "--stop_at", "3"], tmp_path / "wb1")
    run(common + ["--out", str(b), "--resume", "auto"], tmp_path / "wb2")
    full, resumed = torch.load(a / "model.pt", weights_only=False), torch.load(b / "model.pt", weights_only=False)
    for k, v in full["model"].items():
        assert torch.equal(v, resumed["model"][k]), k


def test_a_time_limit_stops_both_processes_at_the_same_step(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    common = ARGS + ["--steps", "12", "--eval_every", "6"]
    run(common + ["--out", str(a)], tmp_path / "wa")
    run(common + ["--out", str(b), "--max_minutes", "1e-9"], tmp_path / "wb1")     # over after the first step
    assert [(r["step"], r["why"]) for r in records(b, "stop")] == [(1, "max_minutes")]
    run(common + ["--out", str(b), "--resume", "auto"], tmp_path / "wb2")
    full, resumed = torch.load(a / "model.pt", weights_only=False), torch.load(b / "model.pt", weights_only=False)
    for k, v in full["model"].items():
        assert torch.equal(v, resumed["model"][k]), k


def test_a_zero_checkpoint_is_saved_in_parts_and_resumes_exactly(tmp_path):
    # The timed ckpt.pt under --zero: every process writes its own part of AdamW's state (ckpt.pt.zero<rank>), not
    # gathered on process 0, and a resume reads each part back on its process.
    a, b = tmp_path / "a", tmp_path / "b"
    common = ZERO + ["--model", "tonn", "--zero", "1"]
    run(common + ["--out", str(a)], tmp_path / "wa")
    run(common + ["--out", str(b), "--stop_at", "3"], tmp_path / "wb1")
    ck = torch.load(b / "ckpt.pt", weights_only=False)
    assert ck["opt"] is None and ck["state"]["zero_shards"] == 2
    parts = [torch.load(b / f"ckpt.pt.zero{r}", weights_only=False) for r in (0, 1)]
    assert [p["step"] for p in parts] == [3, 3] and all(p["opt"]["state"] for p in parts)
    run(common + ["--out", str(b), "--resume", "auto"], tmp_path / "wb2")
    full, resumed = torch.load(a / "model.pt", weights_only=False), torch.load(b / "model.pt", weights_only=False)
    for k, v in full["model"].items():
        assert torch.equal(v, resumed["model"][k]), k
    assert resumed["opt"] is not None                                   # model.pt stays complete in one file


@pytest.mark.parametrize("model,zero", [("tonn", "0"), ("llama", "0"), ("llama", "1")])
def test_accumulation_averages_gradients_over_processes(tmp_path, model, zero):
    # --accum > 1 on several processes: every optimizer step must average the processes' gradients (DDP syncs on the
    # last pass), so two processes train exactly the model one process trains on the same batches. A shadowed loop
    # variable once skipped the averaging on every pass: the processes drifted apart (or, with --zero, each updated
    # its part of the weights from its own rows alone), the 0.5B and 3B runs of 25-26 November.
    common = ["--preset", "tiny", "--model", model, "--data", "synthetic", "--device", "cpu", "--steps", "8",
              "--eval_every", "4", "--batch", "8", "--search", "0", "--accum", "3", "--zero", zero]
    one, two = tmp_path / "one", tmp_path / "two"
    single = main(common + ["--out", str(one)])
    ranks = run(common + ["--out", str(two)], two)
    assert all(e["processes_in_sync"] for e in records(two, "eval"))         # the processes never drift apart
    for name, p in single.named_parameters():
        assert torch.equal(ranks[0][name], ranks[1][name]), name
    # and they take one process's updates: AdamW turns the different order of float sums into small differences, but
    # each process alone (half the rows, no averaging) would move every weight by a whole step's worth
    # (ExeaLM's TONN layer also measures its running output size on each process's own rows, so its weights may differ
    # from one process's by more; its held-out score below may not)
    far = max(float((p.detach() - ranks[0][n]).abs().max()) for n, p in single.named_parameters())
    assert far < (2e-3 if model == "llama" else 5e-2), far
    v1, v2 = records(one, "eval"), records(two, "eval")
    assert all(abs(a["val_bits_per_token"] - b["val_bits_per_token"]) < 0.01 for a, b in zip(v1, v2))
