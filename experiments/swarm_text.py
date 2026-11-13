"""Experiment 2: swarms on real text. Does training agents together, talking through the board, help them model text?

Data: FineWeb-Edu as raw bytes, packed with document boundaries (exealm.data.pack --tokenizer bytes).
Arms (same model, same seed, same architecture, same steps and bytes per agent; the architecture search is off so the
only difference is the swarm):
  solo    every agent alone (swarm_size 1), random windows: ordinary language-model training
  split   swarms of 4 reading 4 consecutive windows of one stretch of text, sharing a board
  mixed   swarms of 4; each swarm is split or random with even odds (the training default)
  split_solo  the same batches as split (4 consecutive windows per stretch) but every agent alone: the control for
          batch diversity, since a split batch holds 4x fewer independent stretches of text
Every model is scored on the same held-out split swarms, with the board shared and with it cut (swarm_size 1), in bits
per byte, overall and by the agent's position in the stretch (agent 0 reads the first window, agent 3 the last).
The loader hides each split swarm's seam targets (agent i's last target is agent i + 1's first byte, which agent i can
hear through the board; exealm/data/loader.py), in training and in scoring, for every arm. Results in
results/swarm_text*/ from before that change scored those targets.
The gain from the board = bits with the board cut - bits with it shared.

python experiments/swarm_text.py --data data/bytes/train --val data/bytes/val --steps 3000 --out results/swarm_text
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
from exealm import ExeaLM, MemoryContext, count_params, get_config  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.config import HYBRID  # noqa: E402
from exealm.train.pretrain import lr_at, make_optimizer, pick_device  # noqa: E402

# arm: (agents sharing a board while training, rows grouped per data draw, data policy)
ARMS = {"solo": (1, 1, "random"), "split": (4, 4, "split"), "mixed": (4, 4, "mixed"), "split_solo": (1, 4, "split")}


@torch.no_grad()
def score(model, batches, swarm_size, n_agents, dev):
    """Bits per byte, overall and per agent position, with the given swarm size at evaluation."""
    model.eval()
    tot = torch.zeros(n_agents)
    for x, y in batches:
        logits = model(x.to(dev), MemoryContext(swarm_size=swarm_size))[0].float()
        y = y.to(dev)
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, reduction="none", ignore_index=-100)
        nll = (nll.sum(1) / (y != -100).sum(1)).cpu()                  # mean over scored targets (seams are hidden)
        tot += nll.view(-1, n_agents).mean(0)
    model.train()
    per = (tot / len(batches) / math.log(2)).tolist()
    return dict(bpb=round(sum(per) / n_agents, 4), per_agent=[round(v, 4) for v in per])


def HYBRID_KW(layers):
    return dict(attn_layers=tuple(i for i in range(layers) if i % 3), **HYBRID)   # every third layer TONN (default)


def run(arm, a, dev, train, evals):
    train_swarm, group, policy = ARMS[arm]
    group = a.swarm if group > 1 else 1
    torch.manual_seed(a.seed)
    cfg = get_config("small-tonn", vocab_size=int(train.meta["vocab_size"]), d=a.d, n_layers=a.layers, slots=a.slots,
                     h=a.h, chunk=a.chunk, swarm_size=a.swarm, seed=a.seed,
                     **(HYBRID_KW(a.layers) if a.hybrid else {}))
    model = ExeaLM(cfg).to(dev)
    opt = make_optimizer(model, a.lr, 0.1)
    rng = np.random.default_rng(a.seed)
    curve, t0 = [], time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, 200, 0.2)
        x, y, _ = train.batch(a.batch, a.seq, rng, group, policy, mask_seams=bool(a.mask_seams))
        loss = model(x.to(dev), MemoryContext(swarm_size=train_swarm), y.to(dev))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            sh, cut = score(model, evals, a.swarm, a.swarm, dev), score(model, evals, 1, a.swarm, dev)
            rec = dict(arm=arm, step=step + 1, secs=round(time.time() - t0, 1), train_loss=round(float(loss.detach()), 4),
                       shared=sh, cut=cut, board_gain=round(cut["bpb"] - sh["bpb"], 4))
            curve.append(rec)
            print(json.dumps(rec), flush=True)
    return dict(arm=arm, params=count_params(model), final=curve[-1], curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--arms", default="solo,split,mixed")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32, help="rows per batch (agents); a multiple of --swarm")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--swarm", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hybrid", type=int, default=0, help="1: the default ExeaLM hybrid (every third layer TONN)")
    ap.add_argument("--mask_seams", type=int, default=1, help="0: score and train on the seam targets too (as before the fix)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/swarm_text")
    a = ap.parse_args()
    dev = pick_device(a.device)
    train, val = TokenShards(a.data), TokenShards(a.val)
    erng = np.random.default_rng(10_000)
    evals = [val.batch(a.batch, a.seq, erng, a.swarm, "split", mask_seams=bool(a.mask_seams))[:2] for _ in range(a.eval_batches)]
    os.makedirs(a.out, exist_ok=True)
    runs = []
    path = os.path.join(a.out, f"text_seed{a.seed}.json")
    for arm in a.arms.split(","):
        runs.append(run(arm, a, dev, train, evals))
        json.dump(dict(args=vars(a), device=str(dev), runs=runs), open(path, "w"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
