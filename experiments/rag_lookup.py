"""Experiment 3, proof of mechanism: can the model build its own RAG engine from scratch? Its own embeddings for the
datastore, its own search queries, its own reading of the results, all learned from the next-token loss.

The task: a datastore of facts, one per chunk, [FACT, k1, k2, value, filler ...], with two-token keys (4,096 possible)
and values drawn fresh for every training step. The facts change every step, so they cannot be memorized in the
weights. The model reads questions, one per chunk, [ASK, k1, k2, filler ...], and in the next chunk must produce the
answer: [ANS, value ...]. It is scored only on the value. The value appears nowhere in the model's own text, so it
can beat chance (1 in 64) only by retrieving the right fact. The datastore is re-encoded with the model's own
encoder every step (cheap at this size); the chosen facts are re-encoded live, so gradients reach the encoder.

Cold start: with a random encoder the right fact is rarely in the top results, so there is little to learn from.
--grow starts with a small datastore and doubles it on a schedule, up to --facts.

Arms: the same model trained with retrieval (datastore attached) and without it (the control, which must stay at
chance). Every trained model is scored with the datastore attached and removed.

python experiments/rag_lookup.py --facts 1024 --steps 4000 --out results/rag_lookup
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
from exealm.train.pretrain import pick_device  # noqa: E402

N_FILL, KEY0, N_KEY, VAL0, N_VAL = 64, 64, 64, 128, 64
FACT, ASK, ANS = 192, 193, 194


def make_kb(n, C, rng):
    """n facts with distinct two-token keys and random values, one per datastore chunk of C tokens."""
    flat = rng.choice(N_KEY * N_KEY, n, replace=False)
    k1, k2 = flat // N_KEY, flat % N_KEY
    vals = rng.integers(0, N_VAL, n)
    toks = rng.integers(0, N_FILL, size=(n, C))
    toks[:, 0], toks[:, 1], toks[:, 2], toks[:, 3] = FACT, KEY0 + k1, KEY0 + k2, VAL0 + vals
    return toks, k1, k2, vals


def make_questions(B, n_q, C, kb, rng, restate=False):
    """Rows of n_q (question chunk, answer chunk) pairs about random facts of the knowledge base. restate: the answer
    chunk repeats the key before the value, [ANS, k1, k2, value], and the value is predicted at k2 (the reading position
    then holds the key itself); otherwise the answer chunk is [ANS, value] and the value is predicted at ANS, so the
    model must also carry the key over from the question chunk."""
    _, k1, k2, vals = kb
    T = 2 * n_q * C
    x = rng.integers(0, N_FILL, size=(B, T))
    y = np.full((B, T), -100)
    for b in range(B):
        for i, f in enumerate(rng.integers(0, len(vals), n_q)):
            p = 2 * i * C
            x[b, p:p + 3] = (ASK, KEY0 + k1[f], KEY0 + k2[f])
            if restate:
                x[b, p + C:p + C + 4] = (ANS, KEY0 + k1[f], KEY0 + k2[f], VAL0 + vals[f])
                y[b, p + C + 2] = VAL0 + vals[f]               # at k2, predict the value
            else:
                x[b, p + C:p + C + 2] = (ANS, VAL0 + vals[f])
                y[b, p + C] = VAL0 + vals[f]                   # at ANS, predict the value
    return torch.from_numpy(x), torch.from_numpy(y)


def make_kb_dense(n_chunks, C, rng):
    """Tables: every datastore chunk holds C // 4 facts back to back, [k1, k2, value, FACT] each, the way a chunk of
    real text holds many facts. Returns the chunks (n_chunks, C) and per chunk its facts' keys and values."""
    F = C // 4
    flat = rng.choice(N_KEY * N_KEY, n_chunks * F, replace=False).reshape(n_chunks, F)
    k1, k2 = flat // N_KEY, flat % N_KEY
    vals = rng.integers(0, N_VAL, (n_chunks, F))
    toks = np.stack([KEY0 + k1, KEY0 + k2, VAL0 + vals, np.full_like(k1, FACT)], -1).reshape(n_chunks, C)
    return toks, k1, k2, vals


def make_questions_dense(B, n_q, C, kb, rng):
    """Rows of n_q (question chunk, answer chunk) pairs, each pair about every fact of one random table. The question
    chunk lists the keys, [ASK, k1, k2, filler] per fact; the answer chunk restates them with the values,
    [ANS, k1, k2, value], and the value is predicted at k2. The search for the answer chunk is made from the
    question chunk, so it has to find the table; the read then finds the key in it and reads the value after it."""
    _, k1, k2, vals = kb
    F = C // 4
    T = 2 * n_q * C
    x = rng.integers(0, N_FILL, size=(B, T))
    y = np.full((B, T), -100)
    for b in range(B):
        for i, t in enumerate(rng.integers(0, len(vals), n_q)):
            p = 2 * i * C
            for j, f in enumerate(rng.permutation(F)):
                q, a_ = p + 4 * j, p + C + 4 * j
                x[b, q:q + 3] = (ASK, KEY0 + k1[t, f], KEY0 + k2[t, f])
                x[b, a_:a_ + 4] = (ANS, KEY0 + k1[t, f], KEY0 + k2[t, f], VAL0 + vals[t, f])
                y[b, a_ + 2] = VAL0 + vals[t, f]
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def score(model, kbs, batches, attach, dev):
    model.eval()
    hit = n = 0
    bits = 0.0
    for kb, (x, y) in zip(kbs, batches):
        mem = MemoryContext()
        if attach:
            ds = Datastore(torch.from_numpy(kb[0]), torch.arange(len(kb[0]))).to(dev)
            ds.refresh(model.encoder, model.emb)
            mem = MemoryContext(datastore=ds, doc_ids=torch.full(x.shape, -1, device=dev))
        x, y = x.to(dev), y.to(dev)
        logits = model(x, mem)[0].float()
        m = y >= 0
        lp = torch.log_softmax(logits[m], -1)
        bits += float(-lp.gather(1, y[m].view(-1, 1)).sum()) / math.log(2)
        hit += int((lp.argmax(-1) == y[m]).sum())
        n += int(m.sum())
    model.train()
    return dict(acc=round(hit / n, 4), bits=round(bits / n, 3))


def kb_size(step, a):
    if not a.grow:
        return a.facts
    return min(a.facts, a.grow * 2 ** (step // a.grow_every))


def new_kb(n_facts, a, rng):
    return make_kb_dense(max(1, n_facts // (a.chunk // 4)), a.chunk, rng) if a.dense else make_kb(n_facts, a.chunk, rng)


def new_questions(kb, a, rng):
    if a.dense:
        return make_questions_dense(a.batch, a.questions, a.chunk, kb, rng)
    return make_questions(a.batch, a.questions, a.chunk, kb, rng, a.restate)


def run(arm, a, dev):
    torch.manual_seed(a.seed)
    cfg = get_config("tiny", vocab_size=256, chunk_pool=a.pool, d=a.d, n_layers=a.layers, slots=a.slots, h=a.h, chunk=a.chunk,
                     rag_chunk_len=a.chunk, rag_topk=a.topk, enc_dim=a.enc_dim, rag_query=a.query, seed=a.seed)
    model = ExeaLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    erng = np.random.default_rng(10_000)
    ekbs = [new_kb(a.facts, a, erng) for _ in range(a.eval_batches)]
    evals = [new_questions(kb, a, erng) for kb in ekbs]
    curve, t0 = [], time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 200)
        kb = new_kb(kb_size(step, a), a, rng)
        x, y = new_questions(kb, a, rng)
        mem = MemoryContext()
        if arm == "rag":
            ds = Datastore(torch.from_numpy(kb[0]), torch.arange(len(kb[0]))).to(dev)
            ds.refresh(model.encoder, model.emb)
            mem = MemoryContext(datastore=ds, doc_ids=torch.full(x.shape, -1, device=dev))
        loss = model(x.to(dev), mem, y.to(dev))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            rec = dict(arm=arm, step=step + 1, kb=kb_size(step, a), secs=round(time.time() - t0, 1),
                       attached=score(model, ekbs, evals, True, dev), removed=score(model, ekbs, evals, False, dev))
            curve.append(rec)
            print(json.dumps(rec), flush=True)
    return dict(arm=arm, params=count_params(model), final=curve[-1], curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="rag,none")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--facts", type=int, default=1024, help="facts in the datastore (at evaluation, and the cap)")
    ap.add_argument("--grow", type=int, default=16, help="start the datastore this small and double it (0: off)")
    ap.add_argument("--grow_every", type=int, default=400)
    ap.add_argument("--questions", type=int, default=4)
    ap.add_argument("--restate", type=int, default=0, help="1: the answer chunk repeats the key before the value")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--query", default="last", choices=["last", "mean"], help="RG query: chunk's last position or mean")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--enc_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense", type=int, default=0, help="1: datastore chunks are tables of chunk/4 facts (make_kb_dense)")
    ap.add_argument("--pool", default="mean", choices=["mean", "attn"], help="how a chunk becomes a post or a key")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/rag_lookup")
    a = ap.parse_args()
    dev = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    out = dict(args=vars(a), device=str(dev), chance=dict(acc=round(1 / N_VAL, 4), bits=round(math.log2(N_VAL), 3)),
               runs=[run(arm, a, dev) for arm in a.arms.split(",")])
    path = os.path.join(a.out, f"lookup_f{a.facts}_q{a.query}_r{a.restate}_seed{a.seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
