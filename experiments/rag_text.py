"""Experiment 4: built-in RAG on real text. Does a datastore of training text make the model predict held-out text
better?

Data: any packed text (bytes or tokens). The datastore is chunks of the training documents, or of a separate library
(--store, e.g. Wikipedia); it never returns chunks of the document being read, and validation documents are not in it.
Arms (same model, seed and steps):
  rag    trained with the datastore attached (re-encoded with the model's own encoder every --refresh_every steps)
  none   trained without a datastore (RG slots output zeros): the control
Every model is scored on held-out text with the datastore attached and removed, in bits per byte.

python experiments/rag_text.py --data data/bytes/train --val data/bytes/val --chunks 200000 --steps 4000 --out results/rag_text
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exealm import Datastore, ExeaLM, MemoryContext, count_params, get_config  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.config import HYBRID  # noqa: E402
from exealm.train.pretrain import lr_at, make_optimizer, pick_device  # noqa: E402


@torch.no_grad()
def score(model, batches, ds, dev):
    model.eval()
    nll = 0.0
    for x, y, d in batches:
        mem = MemoryContext(datastore=ds, doc_ids=d.to(dev) + 10 ** 9) if ds is not None else MemoryContext()
        nll += float(model(x.to(dev), mem, y.to(dev))[1])
    model.train()
    return round(nll / len(batches) / math.log(2), 4)


def HYBRID_KW(layers):
    return dict(attn_layers=tuple(i for i in range(layers) if i % 3), **HYBRID)   # every third layer TONN (default)


def run(arm, a, dev, train, evals, ds):
    torch.manual_seed(a.seed)
    cfg = get_config("small-tonn", vocab_size=int(train.meta["vocab_size"]), d=a.d, n_layers=a.layers, slots=a.slots,
                     h=a.h, chunk=a.chunk, rag_chunk_len=a.chunk, rag_topk=a.topk, enc_dim=a.enc_dim, seed=a.seed,
                     **(HYBRID_KW(a.layers) if a.hybrid else {}))
    model = ExeaLM(cfg).to(dev)
    opt = make_optimizer(model, a.lr, 0.1)
    rng = np.random.default_rng(a.seed)
    use = ds if arm == "rag" else None
    if use is not None:
        use.refresh(model.encoder, model.emb, step=0)
    curve, t0 = [], time.time()
    for step in range(a.steps):
        if use is not None and step > 0 and step % a.refresh_every == 0:
            use.refresh(model.encoder, model.emb, step=step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, 200, 0.2)
        x, y, d = train.batch(a.batch, a.seq, rng)
        mem = MemoryContext(datastore=use, doc_ids=d.to(dev)) if use is not None else MemoryContext()
        loss = model(x.to(dev), mem, y.to(dev))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ds.refresh(model.encoder, model.emb, step=step)        # keys from this model, for scoring
            rec = dict(arm=arm, step=step + 1, secs=round(time.time() - t0, 1), train_loss=round(float(loss.detach()), 4),
                       attached=score(model, evals, ds, dev), removed=score(model, evals, None, dev))
            rec["retrieval_gain"] = round(rec["removed"] - rec["attached"], 4)
            curve.append(rec)
            print(json.dumps(rec), flush=True)
    return dict(arm=arm, params=count_params(model), final=curve[-1], curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--arms", default="rag,none")
    ap.add_argument("--chunks", type=int, default=200_000, help="datastore size (chunks of --chunk tokens)")
    ap.add_argument("--store", default=None, help="packed library for the datastore (default: the training data)")
    ap.add_argument("--refresh_every", type=int, default=500)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--enc_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hybrid", type=int, default=0, help="1: the default ExeaLM hybrid (attention on odd layers)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/rag_text")
    a = ap.parse_args()
    dev = pick_device(a.device)
    train, val = TokenShards(a.data), TokenShards(a.val)
    store = TokenShards(a.store) if a.store else train              # a separate library, or the training text
    ds = Datastore.from_shards(store, a.chunks, a.chunk, seed=a.seed + 1)
    if a.store:
        ds.doc_ids += 2 * 10 ** 9                                     # library documents are never the one being read
    ds = ds.to(dev)
    erng = np.random.default_rng(10_000)
    evals = [val.batch(a.batch, a.seq, erng) for _ in range(a.eval_batches)]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"rag_text_seed{a.seed}.json")
    runs = []
    for arm in a.arms.split(","):
        runs.append(run(arm, a, dev, train, evals, ds))
        json.dump(dict(args=vars(a), device=str(dev), runs=runs), open(path, "w", encoding="utf-8", newline="\n"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
