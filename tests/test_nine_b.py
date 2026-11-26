"""The 9B measurement (experiments/expanse/nine_b.py) on the CPU: its model is the study's ExeaLM, what FSDP needs from
the model changes nothing it computes (with activation checkpointing too), a run writes the row the report reads, and
running out of memory leaves an error row. With two processes (gloo): AdamW's fused step on sharded tensors, and, where
FSDP2 can train on the CPU, a sharded run against one process. In torch 2.6 it cannot (FSDP2 needs CUDA-style streams
the CPU lacks), so that part is skipped there and only the cluster runs it."""
import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from exealm import ExeaLM, count_params, get_config

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "experiments", "expanse")]
import nine_b  # noqa: E402

SB = os.path.join(ROOT, "experiments", "expanse", "nine_b.sb")
TINY = ["--preset", "tiny", "--vocab", "256", "--device", "cpu", "--batch", "2", "--seq", "32", "--warmup", "1",
        "--steps", "3"]


def tiny(**extra):
    """The tiny preset as the study trains ExeaLM, with 8 layers: TONN at 0 and 4, six attention layers for value
    residual to connect."""
    torch.manual_seed(0)
    return nine_b.build("tiny", 256, n_layers=8, **extra)


def test_the_model_is_the_studys_exealm_at_9b():
    with torch.device("meta"):
        model = nine_b.build()
        mid = json.load(open(os.path.join(os.path.dirname(nine_b.CORE), "exealm_mid_9b.json")))
        want = count_params(ExeaLM(get_config("9b", vocab_size=32768, **mid)))
    cfg = model.cfg
    assert (cfg.d, cfg.n_layers, cfg.vocab_size) == (4096, 42, 32768)
    assert cfg.value_residual and cfg.xsa and cfg.swarm_size == 1 and not cfg.pinned
    assert cfg.attn_layers == tuple(i for i in range(42) if i != 21)                # one TONN layer, at 21
    assert not {s["tmpl"] for row in model.specs for s in row} & set(nine_b.MEMORY)
    assert count_params(model) == want and 9.2e9 < want < 9.3e9


def grads(model):
    return {n.replace("unit.inner.", ""): p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("ckpt", [0, 1])
def test_what_fsdp_needs_changes_nothing_the_model_computes(ckpt):
    """prepare() hands each layer its arguments beside the call, drops the unused memory circuits and, with
    checkpointing, recomputes each layer from the TONN sizes its forward pass saw: the same losses, gradients and
    running sizes as the plain model, over two passes (the first measures the sizes, the second uses them)."""
    plain = tiny()
    ready = copy.deepcopy(plain)
    layers = nine_b.prepare(ready, ckpt)
    assert len(layers) == 8 and all(m.inner is not None for pair in layers for m in pair)
    assert not any(n for n, _ in ready.named_parameters() if ".rg." in n or ".ag." in n)
    x = torch.randint(0, 256, (2, 33), generator=torch.Generator().manual_seed(1))
    for _ in range(2):
        a, b = (m(x[:, :-1], None, x[:, 1:])[1] for m in (plain, ready))
        for m, loss in ((plain, a), (ready, b)):
            m.zero_grad(set_to_none=True)
            loss.backward()
        assert torch.allclose(a, b, atol=1e-6)
        ga, gb = grads(plain), grads(ready)
        assert set(gb) <= set(ga) and set(ga) - set(gb) <= {n for n in ga if ".rg." in n or ".ag." in n}
        for n, g in gb.items():
            assert torch.allclose(ga[n], g, atol=1e-5, rtol=1e-4), n
        vr = [g for n, g in gb.items() if n.endswith(".vr")]
        assert len(vr) == len(plain.cfg.attn_layers) - 1 and all(g.abs().sum() > 0 for g in vr)   # value residual is alive
        for p, r in zip(plain.layers, ready.layers):
            assert torch.equal(p.scale, r.unit.inner.scale) and (p.scale > 0).any()   # updated once per pass


def test_prepare_refuses_a_model_with_memory_slots():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="memory-free"):
        nine_b.prepare(ExeaLM(get_config("tiny")), 1)                             # RG and AG pinned in every layer


def test_the_rows_arithmetic():
    r = nine_b.rates(9.1e9, 4, 512, 16, 2.0)
    assert r["tokens_per_s"] == 16384 and r["tokens_per_s_per_gpu"] == 1024 and r["s_per_step"] == 2.0
    assert r["effective_TFLOPs_per_gpu"] == round(6 * 9.1e9 * 1024 / 1e12, 1) == 55.9
    assert r["params_M"] == 9100.0


def test_one_process_writes_the_row_the_report_reads(tmp_path, capsys):
    out = tmp_path / "results" / "nine_b.json"
    first = nine_b.main(TINY + ["--checkpoint", "0", "--out", str(out)])
    second = nine_b.main(TINY + ["--out", str(out)])
    rows = json.loads(out.read_text())
    assert rows == [first, second]                                              # appended, the first one kept
    assert [r["checkpointing"] for r in rows] == [False, True]
    for r in rows:
        assert (r["nodes"], r["gpus"], r["sharding"], r["batch_per_gpu"], r["seq"]) == (1, 1, "none", 2, 32)
        assert r["tokens_per_s"] == r["tokens_per_s_per_gpu"]
        assert abs(r["tokens_per_s"] * r["s_per_step"] / (2 * 32) - 1) < 0.02  # s_per_step is rounded
        assert r["peak_mem_gb"] is None and "error" not in r                    # no GPU memory to measure on the CPU
        assert r["loss_end"] < r["loss_start"]                                  # the same batch: the steps train
    assert '"effective_TFLOPs_per_gpu"' in capsys.readouterr().out
    import report                                                               # what prices the 9B run
    rate, nodes = report.nine_b_rate(str(out.parent))
    assert nodes == 1 and rate == max(r["effective_TFLOPs_per_gpu"] for r in rows)


def test_running_out_of_memory_leaves_an_error_row(tmp_path, monkeypatch, capsys):
    def full(*args, where=None, **kwargs):
        where["stage"] = "step 1"
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(nine_b, "measure", full)
    out = tmp_path / "nine_b.json"
    with pytest.raises(SystemExit) as e:
        nine_b.main(TINY + ["--out", str(out)])
    assert e.value.code == 1
    [row] = json.loads(out.read_text())
    assert row["error"] == "out of memory on process 0 at step 1" and row["batch_per_gpu"] == 2
    assert "tokens_per_s" not in row
    assert json.loads(capsys.readouterr().out.splitlines()[0]) == row


def test_the_slurm_script():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    subprocess.run([bash, "-n", SB], check=True)
    text = open(SB).read()
    assert "#SBATCH --partition=nairr-gpu" in text and "#SBATCH --gpus-per-node=h100:4" in text
    assert "#SBATCH --time=03:00:00" in text and "OUT=results/expanse/nine_b.json" in text
    assert "--checkpoint 1" in text and "experiments/expanse/nine_b.py" in text
    assert 'scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1' in text and "--nproc_per_node 4" in text


# ------------------------------------------------------------------------------------------------- two processes
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def environ(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank),
                      LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE=str(world))
    torch.set_num_threads(1)


def spawn(fn, *args, world=2):
    mp.spawn(fn, args=(world, free_port()) + args, nprocs=world, join=True)


def fused_worker(rank, world, port, out):
    environ(rank, world, port)
    dist.init_process_group("gloo")
    ok = nine_b.fused_adamw_works("cpu", world)
    open(os.path.join(out, f"fused{rank}"), "w").write(str(ok))
    dist.destroy_process_group()


def probe_worker(rank, world, port, out):
    """A plain two-layer model through one FSDP2 training step on the CPU."""
    environ(rank, world, port)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    dist.init_process_group("gloo")
    try:
        mesh = init_device_mesh("cpu", (world,))
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        for m in model:
            fully_shard(m, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model(torch.randn(2, 8)).sum().backward()
        msg = ""
    except Exception as e:                                                      # what keeps FSDP2 off the CPU
        msg = f"{type(e).__name__}: {e}".splitlines()[0][:200]
    open(os.path.join(out, f"probe{rank}"), "w").write(msg)
    dist.destroy_process_group()


def needs_fsdp2_on_cpu(out):
    """Skips the test, saying why, if FSDP2 cannot train on the CPU in this torch."""
    spawn(probe_worker, str(out))
    why = open(os.path.join(out, "probe0")).read()
    if why:
        pytest.skip(f"FSDP2 cannot train on the CPU with gloo in torch {torch.__version__} ({why}); the cluster job "
                    "is the first sharded run")


gloo = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs gloo")


@gloo
def test_fused_adamw_steps_on_sharded_parameters(tmp_path):
    spawn(fused_worker, str(tmp_path))
    assert [open(tmp_path / f"fused{r}").read() for r in range(2)] == ["True", "True"]


def main_worker(rank, world, port, argv):
    environ(rank, world, port)
    nine_b.main(argv)


def step_worker(rank, world, port, out):
    """One step of the sharded tiny model in 32-bit, each process on its half of a batch of 4 rows."""
    environ(rank, world, port)
    try:
        _, world, dev = nine_b.dist_setup(torch.device("cpu"))
        model = tiny()
        nine_b.shard(model, nine_b.prepare(model, 1), nine_b.make_mesh(dev, world, world, False), param_dtype=None)
        opt = nine_b.make_optimizer(model, 1e-3, 0.1)
        x = torch.randint(0, 256, (4, 33), generator=torch.Generator().manual_seed(1))[2 * rank:2 * rank + 2]
        model(x[:, :-1], None, x[:, 1:])[1].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        full = {n.replace("unit.inner.", ""): p.full_tensor() for n, p in model.named_parameters()}
        if rank == 0:
            torch.save(full, os.path.join(out, "sharded.pt"))
    except Exception:
        traceback.print_exc()
        raise
    dist.barrier()
    dist.destroy_process_group()


@gloo
def test_a_sharded_run_writes_one_row_and_steps_like_one_process(tmp_path):
    needs_fsdp2_on_cpu(tmp_path)
    out = tmp_path / "nine_b.json"
    spawn(main_worker, TINY + ["--out", str(out)])
    [row] = json.loads(out.read_text())                                         # process 0 writes, once
    assert (row["gpus"], row["nodes"], row["sharding"]) == (2, 1, "fsdp") and row["loss_end"] < row["loss_start"]

    spawn(step_worker, str(tmp_path))
    sharded = torch.load(tmp_path / "sharded.pt")
    model = tiny()
    nine_b.prepare(model, 0)
    opt = nine_b.make_optimizer(model, 1e-3, 0.1)
    x = torch.randint(0, 256, (4, 33), generator=torch.Generator().manual_seed(1))
    model(x[:, :-1], None, x[:, 1:])[1].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    for n, p in model.named_parameters():
        assert torch.allclose(p.detach(), sharded[n.replace("unit.inner.", "")], atol=1e-5), n
