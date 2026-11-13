"""Record what a trained full model does inside its memory circuits on one relay example and one lookup example:
which board posts agent B reads at each question, which tables the search pulls from the library, and which retrieved
token the read lands on. Writes JSON for docs/figures/memory_gif.py (the animation).

python experiments/trace_memory.py --ckpt results/full_small/exealm_skillsA_seed0_ckpt.pt --out docs/figures/memory_trace.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from exealm import Datastore, MemoryContext  # noqa: E402
from exealm.train.pretrain import pick_device  # noqa: E402
from rag_lookup import ANS as L_ANS, ASK as L_ASK, FACT as L_FACT, KEY0 as LKEY0, VAL0 as LVAL0, make_kb_dense, make_questions_dense  # noqa: E402
from show_skills import load  # noqa: E402
from swarm_relay import ASK as R_ASK, FACT as R_FACT, KEY0 as RKEY0, VAL0 as RVAL0, make_batch as relay_batch  # noqa: E402


def label(t, kind):
    """A token as people read it: K07, V39, FACT, ASK, ANS, or '.' for filler."""
    if kind == "relay":
        if t == R_FACT:
            return "FACT"
        if t == R_ASK:
            return "ASK"
        if RKEY0 <= t < RKEY0 + 64:
            return f"K{t - RKEY0:02d}"
        if RVAL0 <= t < RVAL0 + 64:
            return f"V{t - RVAL0:02d}"
        return "."
    if t == L_FACT:
        return "|"
    if t == L_ASK:
        return "ASK"
    if t == L_ANS:
        return "ANS"
    if LKEY0 <= t < LKEY0 + 64:
        return f"K{t - LKEY0:02d}"
    if LVAL0 <= t < LVAL0 + 64:
        return f"V{t - LVAL0:02d}"
    return "."


def top3(logits, base, kind):
    p = logits.float().softmax(-1)
    v, i = p.topk(3)
    return [[label(int(t) - base, kind), round(float(q), 3)] for q, t in zip(v, i)]


@torch.no_grad()
def relay(model, cfg, dev, rng):
    base, C, nf = cfg.vocab_size - 256, 8, 4
    x, y = relay_batch(1, nf, C, rng, agents=2)
    x = x + base
    T = x.shape[1]
    out = dict(chunk=C, A=[label(int(t) - base, "relay") for t in x[0]], B=[label(int(t) - base, "relay") for t in x[1]])
    for lay in model.layers:
        lay.ag.record = True
    runs = {}
    for name, sw in (("shared", 2), ("cut", 1)):
        logits = model(x.to(dev), MemoryContext(swarm_size=sw))[0][1]
        runs[name] = (logits, [lay.ag.last_att for lay in model.layers])
    n = T // C
    qs = []
    for j in range(nf):
        p = (nf + j) * C + 1                                             # B's key position: it predicts the value
        q = dict(pos=p, key=label(int(x[1, p]) - base, "relay"), answer=label(int(y[1, p]), "relay"),
                 shared=top3(runs["shared"][0][p], base, "relay"), cut=top3(runs["cut"][0][p], base, "relay"), layers=[])
        for li, att in enumerate(runs["shared"][1]):
            a = att[1, p, 0]                                              # B's row, its AG slot: E = 2 * n posts + nothing
            q["layers"].append(dict(A=[round(float(v), 4) for v in a[:n]], B=[round(float(v), 4) for v in a[n:2 * n]],
                                    nothing=round(float(a[-1]), 4)))
        qs.append(q)
    out["questions"] = qs
    for lay in model.layers:
        lay.ag.record = False
    return out


@torch.no_grad()
def lookup(model, cfg, dev, rng, n_facts=1024):
    base, C = cfg.vocab_size - 512, cfg.chunk
    kb = make_kb_dense(n_facts // (C // 4), C, rng)
    x, y = make_questions_dense(1, 1, C, kb, rng)
    x, y = x + base, torch.where(y >= 0, y + base, y)
    ds = Datastore(torch.from_numpy(kb[0] + base), torch.arange(len(kb[0]))).to(dev)
    ds.refresh(model.encoder, model.emb)
    for lay in model.layers:
        lay.rg.record = True
    logits = model(x.to(dev), MemoryContext(datastore=ds, doc_ids=torch.full(x.shape, -1, device=dev)))[0][0]
    recs = [dict(lay.rg.last) for lay in model.layers]                  # before the next pass resets them
    removed = model(x.to(dev), MemoryContext())[0][0]
    tables = kb[0]
    # which table holds the facts this question chunk asks about
    k1, k2 = int(x[0, 1]) - base, int(x[0, 2]) - base
    right = next(i for i in range(len(tables)) if any(tables[i][4 * f] == k1 and tables[i][4 * f + 1] == k2 for f in range(C // 4)))
    out = dict(chunk=C, library_tables=len(tables), right_table=int(right),
               question=[label(int(t) - base, "lookup") for t in x[0, :C]], answer_chunk=[label(int(t) - base, "lookup") for t in x[0, C:]],
               layers=[])
    for rec in recs:
        idx, w = rec["idx"][0, 0, 0], rec["weight"][0, 0, 0]               # chunk 1's search: (k,), (k,)
        out["layers"].append(dict(tables=[int(i) for i in idx], weights=[round(float(v), 4) for v in w],
                                  table_tokens=[[label(int(t), "lookup") for t in tables[int(i)]] for i in idx]))
    qs = []
    for p in (y[0] >= 0).nonzero().flatten().tolist():
        c = p - C                                                        # position inside the answer chunk
        q = dict(pos=p, keys=[label(int(x[0, p - 1]) - base, "lookup"), label(int(x[0, p]) - base, "lookup")],
                 answer=label(int(y[0, p]) - base, "lookup"), attached=top3(logits[p], base, "lookup"),
                 removed=top3(removed[p], base, "lookup"), read=[])
        for rec in recs:
            r = rec["read"][0, 0, 0, c]                          # (k, Lc) attention over the retrieved tokens
            flat = r.flatten()
            top = flat.topk(3)
            q["read"].append([[int(i) // r.shape[1], int(i) % r.shape[1], round(float(v), 4)] for v, i in zip(top.values, top.indices)])
        qs.append(q)
    out["questions"] = qs
    for lay in model.layers:
        lay.rg.record = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/full_small/exealm_skillsA_seed0_ckpt.pt")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="docs/figures/memory_trace.json")
    a = ap.parse_args()
    dev = pick_device(a.device)
    model, cfg = load(a.ckpt, dev)
    rng = np.random.default_rng(a.seed)
    out = dict(ckpt=a.ckpt, relay=relay(model, cfg, dev, rng), lookup=lookup(model, cfg, dev, rng))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
