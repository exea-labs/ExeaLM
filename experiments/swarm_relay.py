"""Experiment 1, proof of mechanism: can agents learn to pass information to each other through the board, during
training, from nothing but one agent's loss?

The task splits the information between the agents of a swarm (--agents N, default 2):
  agents 0 .. N-2 read FACTS:  one fact per chunk: [FACT, key, value, filler ...], then filler
  agent N-1 reads QUESTIONS:   filler, then one question per chunk: [ASK, key, value ...] about every fact held by any
                               other agent, in a new random order
Only the last agent is scored, and only on predicting each question's value from its key. Values are random for every
example, so it can do better than chance (1 in 64) only if the others pass their facts over the board. The fact
holders get no loss of their own: whatever they learn to post, they learn from the asker's errors flowing back through
the board. Questions come after all facts have been posted (a post is readable from the next chunk on).
With N = 4 the asker has to gather facts from three agents at once.

Arms (same model, same data, same steps):
  swarm   agents 0 and 1 share a board while training
  solo    every agent has a private board (swarm_size 1): the control, which must stay at chance
Every trained model is scored with the board shared and with it cut (swarm_size 1 at evaluation).

python experiments/swarm_relay.py --steps 3000 --out results/swarm_relay
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
from exealm.train.pretrain import pick_device  # noqa: E402

N_FILL, KEY0, VAL0, N_KEYS = 64, 64, 128, 64
FACT, ASK = 192, 193


def make_batch(n_swarms, n_facts, C, rng, fill_kinds=N_FILL, agents=2):
    """Rows s*agents .. s*agents+agents-1 are swarm s: fact holders, then the asker. n_facts per holder.
    Returns x (B, T), y (B, T)."""
    holders = agents - 1
    total = holders * n_facts
    T = (n_facts + total) * C
    B = agents * n_swarms
    x = rng.integers(0, fill_kinds, size=(B, T))
    y = np.full((B, T), -100)
    for s in range(n_swarms):
        r0 = s * agents
        keys = rng.choice(N_KEYS, total, replace=False)
        vals = rng.integers(0, N_KEYS, total)
        for f in range(total):
            holder, i = divmod(f, n_facts)
            p = i * C
            x[r0 + holder, p:p + 3] = (FACT, KEY0 + keys[f], VAL0 + vals[f])
        asker = r0 + holders
        for j, f in enumerate(rng.permutation(total)):
            p = (n_facts + j) * C
            x[asker, p:p + 3] = (ASK, KEY0 + keys[f], VAL0 + vals[f])
            y[asker, p + 1] = VAL0 + vals[f]                       # at the key, predict its value
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def score(model, batches, swarm_size, dev):
    model.eval()
    hit = n = 0
    bits = 0.0
    for x, y in batches:
        x, y = x.to(dev), y.to(dev)
        logits = model(x, MemoryContext(swarm_size=swarm_size))[0].float()
        m = y >= 0
        lp = torch.log_softmax(logits[m], -1)
        bits += float(-lp.gather(1, y[m].view(-1, 1)).sum()) / math.log(2)
        hit += int((lp.argmax(-1) == y[m]).sum())
        n += int(m.sum())
    model.train()
    return dict(acc=round(hit / n, 4), bits=round(bits / n, 3))


def run(arm, a, dev):
    torch.manual_seed(a.seed)
    cfg = get_config("tiny", vocab_size=256, chunk_pool=a.pool, board_chunk=a.board_chunk, d=a.d, n_layers=a.layers, slots=a.slots, h=a.h, chunk=a.chunk,
                     swarm_size=a.agents, seed=a.seed)
    model = ExeaLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    erng = np.random.default_rng(10_000)
    evals = [make_batch(a.swarms, a.facts, a.chunk, erng, a.fill, a.agents) for _ in range(a.eval_batches)]
    train_swarm = a.agents if arm == "swarm" else 1
    curve, t0 = [], time.time()
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / 200)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = make_batch(a.swarms, a.facts, a.chunk, rng, a.fill, a.agents)
        loss = model(x.to(dev), MemoryContext(swarm_size=train_swarm), y.to(dev))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            rec = dict(arm=arm, step=step + 1, secs=round(time.time() - t0, 1), shared=score(model, evals, a.agents, dev),
                       cut=score(model, evals, 1, dev))
            curve.append(rec)
            print(json.dumps(rec), flush=True)
    return dict(arm=arm, params=count_params(model), final=curve[-1], curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="swarm,solo")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--agents", type=int, default=2, help="agents per swarm: agents-1 fact holders and one asker")
    ap.add_argument("--swarms", type=int, default=32, help="swarms per batch")
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--fill", type=int, default=N_FILL, help="how many different filler tokens")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--board_chunk", type=int, default=0, help="AG posts every this many tokens (0: every chunk)")
    ap.add_argument("--pool", default="mean", choices=["mean", "attn"], help="how a chunk becomes a post or a key")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/swarm_relay")
    a = ap.parse_args()
    dev = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    out = dict(args=vars(a), device=str(dev), chance=dict(acc=round(1 / N_KEYS, 4), bits=round(math.log2(N_KEYS), 3)),
               runs=[run(arm, a, dev) for arm in a.arms.split(",")])
    tag = "" if (a.agents, a.facts) == (2, 4) else f"_a{a.agents}_f{a.facts}"
    path = os.path.join(a.out, f"relay{tag}_seed{a.seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
