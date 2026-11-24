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
Every trained model is scored with communication enabled and with it cut (swarm_size 1 at evaluation).

python experiments/swarm_relay.py --steps 3000 --out results/swarm_relay
python experiments/swarm_relay.py --sweep --seeds 0,1 --device cpu
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exealm import ExeaLM, MemoryContext, count_params, get_config  # noqa: E402
from exealm.config import MENU  # noqa: E402
from exealm.train.pretrain import pick_device  # noqa: E402

N_FILL, KEY0, VAL0, N_KEYS = 64, 64, 128, 64
FACT, ASK = 192, 193
EVAL_SEED = 10_000


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
    pos = None
    for x, y in batches:
        x, y = x.to(dev), y.to(dev)
        logits = model(x, MemoryContext(swarm_size=swarm_size))[0].float()
        m = y >= 0
        lp = torch.log_softmax(logits[m], -1)
        bits += float(-lp.gather(1, y[m].view(-1, 1)).sum()) / math.log(2)
        ok = (lp.argmax(-1) == y[m]).float()
        hit += int(ok.sum())
        per = ok.view(int(m.any(1).sum()), -1).sum(0)      # by question number: 1st, 2nd, ... question of each asker
        pos = per if pos is None else pos + per
        n += int(m.sum())
    model.train()
    rows = n / len(pos)
    return dict(acc=round(hit / n, 4), bits=round(bits / n, 3), by_question=[round(float(p) / rows, 3) for p in pos])


def relay_config(a, matched=False):
    extra = {}
    base_menu = tuple(t for t in MENU if t not in ("RG", "AG"))
    if matched:
        # With one pinned slot in either arm, the same seed draws the same other circuits.
        extra = dict(pinned={a.memory: 1}, menu=base_menu + (a.memory,), board_read_self=False)
    elif a.memory == "WA":
        extra = dict(pinned={"WA": a.wa_slots}, menu=base_menu + ("WA",))
    elif a.memory == "none":
        extra = dict(pinned={}, menu=base_menu)
    if a.memory == "WA":
        extra.update(wa_scope=a.wa_scope, wa_matrix=a.wa_matrix, wa_key=a.wa_key)
    return get_config("tiny", vocab_size=256, chunk_pool=a.pool, board_chunk=a.board_chunk, d=a.d,
                      n_layers=a.layers, slots=a.slots, h=a.h, chunk=a.chunk, swarm_size=a.agents,
                      seed=a.seed, **extra)


def run(arm, a, dev, matched=False):
    torch.manual_seed(a.seed)
    cfg = relay_config(a, matched)
    model = ExeaLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    erng = np.random.default_rng(EVAL_SEED)
    evals = [make_batch(a.swarms, a.facts, a.chunk, erng, a.fill, a.agents) for _ in range(a.eval_batches)]
    train_swarm = a.agents if arm == "swarm" else 1
    curve, t0 = [], time.perf_counter()
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
            rec = dict(arm=arm, step=step + 1)
            if matched:
                rec["train_loss"] = round(loss.detach().item(), 4)
                if dev.type == "mps":
                    torch.mps.synchronize()
                elif dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            eval_start = time.perf_counter()
            rec.update(secs=round(eval_start - t0, 1), shared=score(model, evals, a.agents, dev),
                       cut=score(model, evals, 1, dev))
            if matched:
                rec["eval_secs"] = round(time.perf_counter() - eval_start, 1)
            curve.append(rec)
            print(json.dumps(rec), flush=True)
    result = dict(arm=arm, params=count_params(model), final=curve[-1], curve=curve)
    if matched:
        result.update(config=asdict(cfg), specs=model.specs)
    return result


def run_case(a, dev, matched=False):
    print(f"RELAY memory={a.memory} facts={a.facts} seed={a.seed} device={dev}", flush=True)
    out = dict(args=vars(a), device=str(dev), chance=dict(acc=round(1 / N_KEYS, 4), bits=round(math.log2(N_KEYS), 3)),
               runs=[run(arm, a, dev, matched) for arm in a.arms.split(",")])
    if matched:
        out.update(eval_seed=EVAL_SEED, seq_len=a.agents * a.facts * a.chunk, torch_threads=torch.get_num_threads())
    tag = "" if (a.agents, a.facts) == (2, 4) else f"_a{a.agents}_f{a.facts}"
    if a.memory != "AG":
        tag += f"_{a.memory.lower()}" + (f"{a.wa_slots}" if a.memory == "WA" and a.wa_slots != 1 else "")
        tag += "_in" if a.memory == "WA" and a.wa_matrix == "in" else ""
        tag += "_hkey" if a.memory == "WA" and a.wa_key == "hidden" else ""
    path = Path(a.out) / f"relay{tag}_seed{a.seed}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("WROTE", path, flush=True)
    return path, out


def run_sweep(a, dev):
    lines = [
        "# Relay capacity: board vs hidden-key weight-alter",
        "",
        f"**Setup.** {a.agents} agents, {a.swarms} swarms per batch, {a.chunk} tokens per chunk, "
        f"{a.steps:,} steps on `{dev}`, held-out batch seed {EVAL_SEED} ({a.eval_batches} batches).",
        "Chance accuracy is 1.56%. Loss is the last training batch's cross-entropy in nats. "
        "Held-out cross-entropy is in bits per answer token.",
        "",
        "| Facts per holder | Seed | Memory | Tokens per agent | Parameters | Step | Train loss | Shared accuracy | Cut accuracy | Shared bits | Cut bits | JSON |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for facts in a.fact_counts:
        for seed in a.seeds:
            for memory in ("AG", "WA"):
                case = argparse.Namespace(**vars(a))
                case.facts, case.seed, case.memory = facts, seed, memory
                case.wa_key = "hidden" if memory == "WA" else "learned"
                path, out = run_case(case, dev, matched=True)
                run_result = out["runs"][0]
                final = run_result["final"]
                shared, cut = final["shared"], final["cut"]
                lines.append(
                    f"| {facts} | {seed} | {memory} | {out['seq_len']:,} | {run_result['params']:,} | "
                    f"{final['step']:,} | {final['train_loss']:.4f} | {shared['acc']:.2%} | {cut['acc']:.2%} | "
                    f"{shared['bits']:.3f} | {cut['bits']:.3f} | [JSON]({path.name}) |"
                )
    (Path(a.out) / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def int_list(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as err:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from err
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("values must be distinct")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", help="default: swarm,solo (swarm only in sweep mode)")
    ap.add_argument("--sweep", action="store_true",
                    help="compare board and hidden-key weight edits with one cross-agent slot per layer")
    ap.add_argument("--fact_counts", type=int_list, help="sweep facts per holder (default: 4,8,16,32,64)")
    ap.add_argument("--seeds", type=int_list, help="sweep training seeds (default: 0,1)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--agents", type=int, default=2, help="agents per swarm: agents-1 fact holders and one asker")
    ap.add_argument("--swarms", type=int, help="swarms per batch (default: 32, or 8 in sweep mode)")
    ap.add_argument("--facts", type=int, default=4, help="facts per holder for a single run")
    ap.add_argument("--fill", type=int, default=N_FILL, help="how many different filler tokens")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0, help="training seed for a single run")
    ap.add_argument("--board_chunk", type=int, default=0, help="AG/WA writes every this many tokens (0: every chunk)")
    ap.add_argument("--pool", default="mean", choices=["mean", "attn"], help="how a chunk becomes a post or a key")
    ap.add_argument("--memory", default="AG", choices=["AG", "WA", "none"],
                    help="single-run path: the agent board (AG), weight changes (WA), or neither")
    ap.add_argument("--wa_slots", type=int, default=1, help="weight-alter slots per layer (with --memory WA)")
    ap.add_argument("--wa_scope", default="others", choices=["self", "others", "all"])
    ap.add_argument("--wa_matrix", default="out", choices=["out", "in"])
    ap.add_argument("--wa_key", default="learned", choices=["learned", "hidden"],
                    help="key source for a single WA run (sweep always uses hidden)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", help="results directory (default: results/swarm_relay or results/swarm_relay_capacity)")
    a = ap.parse_args()
    if a.swarms is None:
        a.swarms = 8 if a.sweep else 32
    if a.agents < 2 or a.chunk < 3 or not 1 <= a.fill <= N_FILL:
        ap.error(f"the relay needs at least two agents, three tokens per chunk, and 1 to {N_FILL} filler tokens")
    if min(a.swarms, a.steps, a.eval_every, a.eval_batches) < 1:
        ap.error("swarms, steps, eval_every and eval_batches must be positive")
    if a.sweep:
        a.arms = a.arms or "swarm"
        a.fact_counts = a.fact_counts or [4, 8, 16, 32, 64]
        a.seeds = a.seeds or [0, 1]
        a.out = a.out or "results/swarm_relay_capacity"
        if len(a.seeds) < 2 or min(a.seeds) < 0 or EVAL_SEED in a.seeds:
            ap.error(f"the sweep requires at least two nonnegative training seeds distinct from {EVAL_SEED}")
        if a.arms != "swarm" or a.wa_slots != 1 or a.wa_scope != "others":
            ap.error("the sweep trains the swarm arm with one cross-agent slot per layer")
        if a.board_chunk not in (0, a.chunk):
            ap.error("the sweep posts once per task chunk (--board_chunk 0 or equal to --chunk)")
    else:
        if a.fact_counts is not None or a.seeds is not None:
            ap.error("--fact_counts and --seeds require --sweep")
        a.arms = a.arms or "swarm,solo"
        a.out = a.out or "results/swarm_relay"
    for facts in a.fact_counts if a.sweep else [a.facts]:
        if facts < 1 or (a.agents - 1) * facts > N_KEYS:
            ap.error(f"facts must be positive and the swarm can hold at most {N_KEYS} distinct facts")
    dev = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    if a.sweep:
        run_sweep(a, dev)
    else:
        run_case(a, dev)


if __name__ == "__main__":
    main()
