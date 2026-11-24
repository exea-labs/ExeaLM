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
