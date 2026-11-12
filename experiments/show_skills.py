"""Show what a trained full model (experiments/full_small.py) can do with its board and its library, in a form a
person can read. Two demos, each run twice, with the circuit on and off:

  relay    two agents. Agent A reads facts "key -> value"; agent B is asked for the values and never sees the facts.
           Only the board connects them. Board shared vs cut.
  lookup   one agent and a library of tables it has never seen (1,024 facts). It reads a list of keys and must
           give each value. Library attached vs removed.

Keys and values are synthetic tokens, printed as K00..K63 and V00..V63.

python experiments/show_skills.py --ckpt results/full_small/exealm_skills_ckpt.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from exealm import Datastore, ExeaLM, MemoryContext  # noqa: E402
from exealm.config import ModelConfig  # noqa: E402
from exealm.train.pretrain import pick_device  # noqa: E402
from rag_lookup import KEY0 as LKEY0, VAL0 as LVAL0, make_kb_dense, make_questions_dense  # noqa: E402
from swarm_relay import KEY0 as RKEY0, VAL0 as RVAL0, make_batch as relay_batch  # noqa: E402


def fmt(g, v):
    """A predicted value, marked right or wrong; anything that isn't a value token at all is shown as (other)."""
    return f"V{g:02d} {'ok' if g == v else '--'}" if 0 <= g < 64 else "(other)"


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    model = ExeaLM(cfg, ck.get("specs"))
    model.load_state_dict(ck["model"])
    model.encoder.idf_ready = True                     # keep the word rarity learned from the training library
    return model.to(dev).eval(), cfg


@torch.no_grad()
def relay_demo(model, cfg, dev, rng, chunk, n_facts):
    base = cfg.vocab_size - 256
    x, y = relay_batch(1, n_facts, chunk, rng, agents=2)              # row 0 holds the facts, row 1 asks
    x = x + base
    preds = {}
    for name, sw in (("board shared", 2), ("board cut", 1)):
        logits = model(x.to(dev), MemoryContext(swarm_size=sw))[0]
        preds[name] = logits[1].argmax(-1).cpu() - base
    print("RELAY: agent A reads the facts, agent B is asked for them. Only the board connects them.")
    facts = [(int(x[0, i * chunk + 1]) - base - RKEY0, int(x[0, i * chunk + 2]) - base - RVAL0) for i in range(n_facts)]
    print("  agent A reads:  " + "   ".join(f"K{k:02d} -> V{v:02d}" for k, v in facts))
    right = {n: 0 for n in preds}
    print("  agent B is asked     answer   board shared   board cut")
    for j in range(n_facts):
        p = (n_facts + j) * chunk + 1                                  # the key position; it predicts the value
        k, v = int(x[1, p]) - base - RKEY0, int(y[1, p]) - RVAL0
        got = {n: int(preds[n][p]) - RVAL0 for n in preds}
        for n in preds:
            right[n] += got[n] == v
        print(f"  K{k:02d} ?                V{v:02d}      {fmt(got['board shared'], v):14s} {fmt(got['board cut'], v)}")
    return right, n_facts


@torch.no_grad()
def lookup_demo(model, cfg, dev, rng, n_facts, show=8):
    base, C = cfg.vocab_size - 512, cfg.chunk
    kb = make_kb_dense(n_facts // (C // 4), C, rng)
    x, y = make_questions_dense(1, 1, C, kb, rng)                      # one question chunk, one answer chunk
    x, y = x + base, torch.where(y >= 0, y + base, y)
    ds = Datastore(torch.from_numpy(kb[0] + base), torch.arange(len(kb[0]))).to(dev)
    ds.refresh(model.encoder, model.emb)
    m = (y[0] >= 0).nonzero().flatten()
    out = {}
    for name, mem in (("library attached", MemoryContext(datastore=ds, doc_ids=torch.full(x.shape, -1, device=dev))),
                      ("library removed", MemoryContext())):
        out[name] = model(x.to(dev), mem)[0][0].argmax(-1).cpu()
    print(f"LOOKUP: a library of {len(kb[0])} tables ({n_facts} facts) the model has never seen. It reads a list of keys")
    print("  and must give each value. The values are in the library and nowhere else.")
    print("  key        answer   library attached   library removed")
    right = {n: 0 for n in out}
    for i, p in enumerate(m.tolist()):
        k1, k2 = int(x[0, p - 1]) - base - LKEY0, int(x[0, p]) - base - LKEY0
        v = int(y[0, p]) - base - LVAL0
        got = {n: int(out[n][p]) - base - LVAL0 for n in out}
        for n in out:
            right[n] += got[n] == v
        if i < show:
            print(f"  K{k1:02d} K{k2:02d}    V{v:02d}      {fmt(got['library attached'], v):18s} {fmt(got['library removed'], v)}")
    print(f"  ... {len(m) - show} more keys in this list")
    return right, len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/full_small/exealm_skills_ckpt.pt")
    ap.add_argument("--relay_chunk", type=int, default=8)
    ap.add_argument("--relay_facts", type=int, default=4)
    ap.add_argument("--facts", type=int, default=1024)
    ap.add_argument("--trials", type=int, default=50, help="examples behind the totals printed at the end")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = pick_device(a.device)
    model, cfg = load(a.ckpt, dev)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, {a.ckpt}\n")
    rng = np.random.default_rng(a.seed)
    relay_demo(model, cfg, dev, rng, a.relay_chunk, a.relay_facts)
    print()
    lookup_demo(model, cfg, dev, rng, a.facts)
    tot_r, tot_l, nr, nl = {}, {}, 0, 0
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(a.trials):
            r, n = relay_demo(model, cfg, dev, rng, a.relay_chunk, a.relay_facts)
            nr += n
            tot_r = {k: tot_r.get(k, 0) + v for k, v in r.items()}
            r, n = lookup_demo(model, cfg, dev, rng, a.facts)
            nl += n
            tot_l = {k: tot_l.get(k, 0) + v for k, v in r.items()}
    if not a.trials:
        return
    print(f"\nover {a.trials} more examples of each (chance is 1 in 64):")
    print("  relay:  " + ", ".join(f"{k} {100 * v / nr:.1f}%" for k, v in tot_r.items()))
    print("  lookup: " + ", ".join(f"{k} {100 * v / nl:.1f}%" for k, v in tot_l.items()))


if __name__ == "__main__":
    main()
