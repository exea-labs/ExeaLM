"""Training speed of the real model: tokens per second for one training step (forward + backward + optimizer) of the
`100m` preset against the same-size Llama baseline, with and without the memory circuits doing work (a datastore for
RG, swarms of 4 for AG). This is the number DATA.md's compute plan needs.

python experiments/speed.py --device mps --batch 8 --seq 2048 --out results/speed
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exealm import Datastore, ExeaLM, Llama, MemoryContext, count_params, get_config, llama_baseline  # noqa: E402
from exealm.train.pretrain import make_optimizer, pick_device  # noqa: E402


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench(name, model, mem_fn, a, dev):
    opt = make_optimizer(model, 1e-4, 0.1)
    x = torch.randint(0, model.cfg.vocab_size, (a.batch, a.seq), device=dev)
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else None
    times = []
    for i in range(a.warmup + a.reps):
        sync(dev)
        t0 = time.time()
        with torch.autocast(**amp) if amp else torch.enable_grad():
            loss = model(x, mem_fn(), x)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(dev)
        if i >= a.warmup:
            times.append(time.time() - t0)
    s = sorted(times)[len(times) // 2]
    rec = dict(name=name, params_M=round(count_params(model) / 1e6, 1), batch=a.batch, seq=a.seq,
               secs_per_step=round(s, 3), tokens_per_sec=round(a.batch * a.seq / s))
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="100m")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--datastore_chunks", type=int, default=100_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/speed")
    a = ap.parse_args()
    dev = pick_device(a.device)
    cfg = get_config(a.preset)
    rows = []
    torch.manual_seed(0)
    rows.append(bench("llama", Llama(llama_baseline(cfg)).to(dev), lambda: None, a, dev))
    model = ExeaLM(cfg).to(dev)
    rows.append(bench("tonn, memory idle", model, lambda: MemoryContext(), a, dev))
    ds = Datastore(torch.randint(0, cfg.vocab_size, (a.datastore_chunks, cfg.rag_chunk_len)),
                   torch.arange(a.datastore_chunks)).to(dev)
    ds.refresh(model.encoder, model.emb)
    docs = torch.full((a.batch, a.seq), -1, device=dev)
    rows.append(bench("tonn, retrieval on", model, lambda: MemoryContext(datastore=ds, doc_ids=docs), a, dev))
    rows.append(bench("tonn, retrieval on + swarms of 4", model,
                      lambda: MemoryContext(datastore=ds, doc_ids=docs, swarm_size=4), a, dev))
    t0 = time.time()
    ds.refresh(model.encoder, model.emb)
    sync(dev)
    refresh = round(time.time() - t0, 2)
    os.makedirs(a.out, exist_ok=True)
    out = dict(args=vars(a), device=str(dev), rows=rows, datastore_refresh_secs=refresh)
    json.dump(out, open(os.path.join(a.out, f"speed_{dev.type}.json"), "w"), indent=1)
    print(json.dumps(dict(datastore_refresh_secs=refresh, chunks=a.datastore_chunks)))


if __name__ == "__main__":
    main()
