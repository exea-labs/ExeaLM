"""Do the memory circuits help on real text? A knowledge test on held-out Wikipedia, for a trained full model
(experiments/full_small.py checkpoints).

Each held-out article is split in half. The model reads the start of the SECOND half (512 tokens). What it needs to
know is in the FIRST half, which it never saw in training and which is not in its context. Three ways to hand it the
first half, each a different path through the model:

  library     RG: the first halves of every held-out article form a library the model never trained on; the model's
              own search must find the right one (and "library, perfect search": OracleDatastore, the right article)
  board       AG, live: a second agent reads the first half at the same time, in a swarm of 2, and the reader hears
              its posts through the in-batch board (a post is visible from the next 8-token stretch on)
  notes       AG, from a knowledge base: a reader agent reads the first half once, ahead of time, and its posts are
              stored on a persistent board (exealm/memory/board.py); the model reads those notes from its first token.
              "notes, own article": only that article's notes. "notes, all articles": every held-out article's notes
              at once, so the model must pick the right ones by their keys

Scored in bits per token on three sets of target tokens in the second half:
  all         every token
  re-mention  a rare word (outside the 1,000 most common tokens) that appears in the first half and not yet in the
              text being read: the model can only get it from memory or by guessing
  novel       a rare word that appears in neither: no memory can help (a control; any "gain" here is noise or harm)
and top-1 accuracy on re-mentions.

python experiments/irl_eval.py --ckpt results/full_small/exealm_skillsA_seed0_ckpt.pt --articles 256
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from exealm import Datastore, ExeaLM, MemoryContext  # noqa: E402
from exealm.config import ModelConfig  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.memory.board import AgentBoard  # noqa: E402
from exealm.train.pretrain import pick_device  # noqa: E402
from rag_wiki_split import TEXT_DOC_OFFSET, OracleDatastore, split_articles  # noqa: E402


class FrozenBoard(AgentBoard):
    """A persistent board whose messages are read many times: read() is cached on the device."""

    def read(self, layer, slots, device):
        key = (layer, tuple(int(k) for k in slots), str(device))
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key not in self._cache:
            self._cache[key] = super().read(layer, slots, device)
        return self._cache[key]


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["cfg"])
    model = ExeaLM(cfg, ck.get("specs"))
    model.load_state_dict(ck["model"])
    model.encoder.idf_ready = True                     # keep the word rarity learned from the training library
    return model.to(dev).eval(), cfg


def pick(arts, n, T, rng):
    """Articles whose first half has at least one chunk and whose second half holds T + 1 tokens."""
    ok = [a for a in arts if len(a[2]) >= T + 1]
    idx = rng.choice(len(ok), size=min(n, len(ok)), replace=False)
    return [ok[i] for i in sorted(idx)]


def targets(arts, T, common):
    """x, y, doc ids (second-half windows) and the masks of re-mention and novel targets."""
    xs, ys, ds, rem, nov = [], [], [], [], []
    for d, first, second in arts:
        w = second[:T + 1].astype(np.int64)
        x, y = w[:-1], w[1:]
        known = set(first.reshape(-1).tolist())
        seen = set()
        r, v = np.zeros(T, bool), np.zeros(T, bool)
        for t in range(T):
            seen.add(int(x[t]))
            tok = int(y[t])
            if tok in common or tok in seen:
                continue
            r[t], v[t] = tok in known, tok not in known
        xs.append(x); ys.append(y); ds.append(np.full(T, d + TEXT_DOC_OFFSET)); rem.append(r); nov.append(v)
    t = lambda a, dt: torch.from_numpy(np.stack(a)).to(dt)
    return t(xs, torch.long), t(ys, torch.long), t(ds, torch.long), t(rem, torch.bool), t(nov, torch.bool)


def first_windows(arts, T):
    """Each article's first half as reader windows of T tokens (the last one may be shorter)."""
    out = []
    for d, first, _ in arts:
        toks = first.reshape(-1).astype(np.int64)
        out.append([torch.from_numpy(toks[s:s + T]) for s in range(0, len(toks), T)])
    return out


@torch.no_grad()
def write_notes(model, windows, dev, capacity):
    """A board holding the posts of a reader agent that read each window once. One author per window."""
    board = FrozenBoard(capacity=capacity)
    author = 0
    for wins in windows:
        for w in wins:
            if len(w) < (model.cfg.board_chunk or model.cfg.chunk):
                continue
            tmp = AgentBoard(capacity=capacity)                        # the reader hears no one else's notes
            model(w.view(1, -1).to(dev), MemoryContext(board=tmp, agent_ids=torch.tensor([author], device=dev), write_board=True))
            for ch, msgs in tmp.channels.items():
                board.channels.setdefault(ch, {}).update(msgs)
            author += 1
    return board, author


def scores(nll, ok, y, rem, nov):
    """Bits per token on all, re-mention and novel targets, and top-1 accuracy on re-mentions."""
    out = {}
    for name, m in (("all", torch.ones_like(rem)), ("re-mention", rem), ("novel", nov)):
        out[name] = round(float(nll[m].mean()) / math.log(2), 4) if m.any() else None
    out["re-mention top-1"] = round(float(ok[rem].float().mean()), 4) if rem.any() else None
    return out


@torch.no_grad()
def run(model, x, y, mem_fn, dev, batch, rows=None):
    """Per-token nll and top-1 hits of the reading rows. mem_fn(i0, i1) -> (inputs, MemoryContext, row index)."""
    nlls, oks = [], []
    for i0 in range(0, len(x), batch):
        i1 = min(len(x), i0 + batch)
        inp, mem, keep = mem_fn(i0, i1)
        logits = model(inp.to(dev), mem)[0].float()[keep]
        yy = y[i0:i1].to(dev)
        nlls.append(F.cross_entropy(logits.transpose(1, 2), yy, reduction="none").cpu())
        oks.append((logits.argmax(-1) == yy).cpu())
    return torch.cat(nlls), torch.cat(oks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wiki", default="data/tokpack/bpe32k/store_wiki")
    ap.add_argument("--articles", type=int, default=256)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--min_tokens", type=int, default=1024)
    ap.add_argument("--common", type=int, default=1000, help="the most common tokens never count as re-mentions")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = pick_device(a.device)
    model, cfg = load(a.ckpt, dev)
    T, C = a.seq, cfg.chunk
    wiki = TokenShards(a.wiki)
    train_arts, eval_arts = split_articles(wiki, C, a.min_tokens, 0, 0.1)     # the split full_small.py trained on
    counts = np.bincount(np.concatenate([f.reshape(-1) for _, f, _ in train_arts]), minlength=cfg.vocab_size)
    common = set(np.argsort(-counts)[:a.common].tolist())
    rng = np.random.default_rng(a.seed)
    arts = pick(eval_arts, a.articles, T, rng)
    x, y, d, rem, nov = targets(arts, T, common)
    print(f"{len(arts)} held-out articles, {int(rem.sum())} re-mention and {int(nov.sum())} novel targets "
          f"in {x.numel()} tokens", flush=True)

    lib_tok = np.concatenate([f for _, f, _ in eval_arts]).astype(np.int64)
    lib_doc = np.concatenate([np.full(len(f), dd) for dd, f, _ in eval_arts])
    lib = Datastore(torch.from_numpy(lib_tok), torch.from_numpy(lib_doc)).to(dev)
    lib.refresh(model.encoder, model.emb)
    oracle = OracleDatastore(torch.from_numpy(lib_tok), torch.from_numpy(lib_doc)).to(dev)
    oracle.refresh(model.encoder, model.emb)
    res = {}

    def plain(mem):
        return lambda i0, i1: (x[i0:i1], mem(i0, i1), slice(None))

    res["none"] = scores(*run(model, x, y, plain(lambda i0, i1: MemoryContext()), dev, a.batch), y, rem, nov)
    res["library"] = scores(*run(model, x, y, plain(lambda i0, i1: MemoryContext(datastore=lib, doc_ids=d[i0:i1].to(dev))), dev, a.batch), y, rem, nov)
    res["library, perfect search"] = scores(*run(model, x, y, plain(lambda i0, i1: MemoryContext(datastore=oracle, doc_ids=d[i0:i1].to(dev))), dev, a.batch), y, rem, nov)

    # board, live: rows [first-half window, second-half window] as swarms of 2; score the second
    firsts = first_windows(arts, T)
    fw = torch.stack([F.pad(w[0], (0, T - len(w[0])), value=0) if len(w[0]) < T else w[0] for w in firsts])

    def live(i0, i1):
        inp = torch.stack([fw[i0:i1], x[i0:i1]], 1).reshape(-1, T)
        return inp, MemoryContext(swarm_size=2), slice(1, None, 2)

    res["board, live (swarm of 2)"] = scores(*run(model, x, y, live, dev, a.batch), y, rem, nov)
    res["board, live, cut"] = scores(*run(model, x, y, lambda i0, i1: (torch.stack([fw[i0:i1], x[i0:i1]], 1).reshape(-1, T), MemoryContext(swarm_size=1), slice(1, None, 2)), dev, a.batch), y, rem, nov)

    # notes: posts written ahead of time by a reader agent
    reader = 10 ** 6                                                   # the reading rows' own agent id
    nll_o, ok_o = [], []
    for i, wins in enumerate(firsts):
        board, _ = write_notes(model, [wins], dev, capacity=4096)
        mem = MemoryContext(board=board, agent_ids=torch.tensor([reader], device=dev))
        n_, o_ = run(model, x[i:i + 1], y[i:i + 1], lambda i0, i1: (x[i:i + 1], mem, slice(None)), dev, 1)
        nll_o.append(n_); ok_o.append(o_)
    res["notes, own article"] = scores(torch.cat(nll_o), torch.cat(ok_o), y, rem, nov)
    board, n_auth = write_notes(model, firsts, dev, capacity=10 ** 7)
    res["notes, all articles"] = scores(*run(model, x, y, plain(lambda i0, i1: MemoryContext(board=board, agent_ids=torch.full((i1 - i0,), reader, device=dev))), dev, a.batch), y, rem, nov)

    base = res["none"]
    print(f"\n{'':28s} {'all':>8s} {'re-mention':>11s} {'novel':>8s} {'re-mention top-1':>17s}")
    for k, v in res.items():
        print(f"{k:28s} {v['all']:8.4f} {v['re-mention']:11.4f} {v['novel']:8.4f} {100 * v['re-mention top-1']:16.1f}%"
              + ("" if k == "none" else f"   re-mention gain {base['re-mention'] - v['re-mention']:+.3f}"))
    out = dict(ckpt=a.ckpt, articles=len(arts), re_mentions=int(rem.sum()), novel=int(nov.sum()),
               reader_windows_in_notes=n_auth, args=vars(a), results=res)
    path = a.out or os.path.join(os.path.dirname(a.ckpt), "irl_" + os.path.basename(a.ckpt).replace("_ckpt.pt", ".json"))
    json.dump(out, open(path, "w"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
