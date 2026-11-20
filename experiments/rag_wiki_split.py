"""Experiment 5: built-in RAG on a knowledge task. Can the model use a library that holds what it needs?

Each Wikipedia article is split in half. The FIRST half goes into the library (the datastore), the SECOND half is the
text the model reads and predicts. The second half keeps mentioning names, dates and facts from the first, so a model
that looks up the start of the article should predict it better. The library never contains the text being
predicted: the two halves count as different documents, and the first half is not in the model's context.

Train on 90% of the articles (their first halves are the training library); evaluate on the held-out 10% with a fresh
library made of THEIR first halves, so nothing can be memorized: the model has to retrieve.
Arms: rag (trained with the library) and none (trained without). Both are scored on held-out articles with the
library attached and removed, in bits per token.

python experiments/rag_wiki_split.py --wiki data/tokpack/bpe32k/store_wiki --steps 3000 --out results/rag_wiki_split
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
from exealm.config import HYBRID  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.model.retrieval import neighbour_loss  # noqa: E402
from exealm.train.pretrain import lr_at, make_optimizer, pick_device  # noqa: E402

TEXT_DOC_OFFSET = 10 ** 9                # second halves get different document ids from their first halves


def split_articles(shards, chunk, min_tokens, seed, eval_frac):
    """Per article (long enough): first-half chunks for the library and the second half as text."""
    toks = np.asarray(shards.tokens)
    starts = list(shards.offsets) + [len(toks)]
    arts = []
    for d in range(len(shards.offsets)):
        a, b = int(starts[d]), int(starts[d + 1])
        if b - a < min_tokens:
            continue
        mid = a + (b - a) // 2
        n = (mid - a) // chunk
        if n < 1:
            continue
        arts.append((d, toks[a:a + n * chunk].reshape(n, chunk), toks[mid:b]))
    rng = np.random.default_rng(seed)
    rng.shuffle(arts)
    k = max(1, int(len(arts) * eval_frac))
    return arts[k:], arts[:k]


class OracleDatastore(Datastore):
    """A diagnostic: the search always looks inside the right article (the one whose second half is being read) and
    returns its k chunks closest to the query. If retrieval helps with this perfect article-level search but not with
    the learned one, the search is the bottleneck; if it doesn't help even here, the model isn't using what it reads."""

    def search(self, q, k, exclude=None, block=262144):
        if exclude is None:
            return super().search(q, k)
        art = exclude - TEXT_DOC_OFFSET
        if not hasattr(self, "_rows"):
            ids = self.doc_ids.cpu().numpy()
            order = np.argsort(ids, kind="stable")
            uniq, start = np.unique(ids[order], return_index=True)
            ends = list(start[1:]) + [len(ids)]
            self._rows = {int(u): torch.as_tensor(order[a:b], device=self.keys.device) for u, a, b in zip(uniq, start, ends)}
        idx = torch.zeros(len(q), k, dtype=torch.long, device=q.device)
        ok = torch.zeros(len(q), k, dtype=torch.bool, device=q.device)
        for i, a in enumerate(art.tolist()):
            rows = self._rows.get(int(a))
            if rows is None:
                continue
            sc = self.keys[rows] @ q[i].float()
            top = sc.topk(min(k, len(rows))).indices
            idx[i, :len(top)], ok[i, :len(top)] = rows[top], True
        return idx, ok


def library(arts, oracle=False):
    tokens = np.concatenate([lib for _, lib, _ in arts]).astype(np.int64)
    docs = np.concatenate([np.full(len(lib), d) for d, lib, _ in arts])
    return (OracleDatastore if oracle else Datastore)(tokens, docs)


def text_batch(arts, B, T, rng):
    """B windows of second-half text (shorter halves are padded by sampling another article)."""
    xs, ys, ds = [], [], []
    while len(xs) < B:
        d, _, txt = arts[int(rng.integers(len(arts)))]
        if len(txt) < T + 1:
            continue
        s = int(rng.integers(0, len(txt) - T))
        w = torch.from_numpy(txt[s:s + T + 1].astype(np.int64))
        xs.append(w[:-1]); ys.append(w[1:]); ds.append(torch.full((T,), d + TEXT_DOC_OFFSET))
    return torch.stack(xs), torch.stack(ys), torch.stack(ds)


@torch.no_grad()
def hit_rate(model, batches, ds, chunk, dev, k=4):
    """How often the search, with the encoder's own key for the chunk just read, returns a chunk of the same
    article's first half among its top k (what retrieval needs to find to help)."""
    hits = n = 0
    for x, _, d in batches:
        B, T = x.shape
        m = T // chunk
        keys = model.encoder(x[:, :m * chunk].reshape(B * m, chunk).to(dev), model.emb)[1]
        idx, _ = ds.search(keys, k)
        art = (d[:, 0] - TEXT_DOC_OFFSET).to(dev).repeat_interleave(m)
        hits += int((ds.doc_ids[idx] == art.view(-1, 1)).any(1).sum())
        n += len(art)
    return round(hits / max(1, n), 4)


@torch.no_grad()
def score(model, batches, ds, dev):
    model.eval()
    nll = 0.0
    for x, y, d in batches:
        mem = MemoryContext(datastore=ds, doc_ids=d.to(dev)) if ds is not None else MemoryContext()
        nll += float(model(x.to(dev), mem, y.to(dev))[1])
    model.train()
    return round(nll / len(batches) / math.log(2), 4)


def run(arm, a, dev, train_arts, evals, lib_train, lib_eval, V):
    torch.manual_seed(a.seed)
    cfg = get_config("small-tonn", vocab_size=V, chunk=a.chunk, rag_chunk_len=a.chunk, seed=a.seed, rag_key_dim=a.key_dim, rag_lexical=a.lexical,
                     attn_layers=tuple(i for i in range(6) if i % 3), **HYBRID)
    model = ExeaLM(cfg).to(dev)
    opt = make_optimizer(model, a.lr, 0.1)
    rng = np.random.default_rng(a.seed)
    use = lib_train if arm == "rag" else None
    if use is not None:
        use.refresh(model.encoder, model.emb, step=0)
    curve, t0 = [], time.time()
    for step in range(a.steps):
        if use is not None and step > 0 and step % a.refresh_every == 0:
            use.refresh(model.encoder, model.emb, step=step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, 200, 0.2)
        x, y, d = text_batch(train_arts, a.batch, a.seq, rng)
        mem = MemoryContext(datastore=use, doc_ids=d.to(dev)) if use is not None else MemoryContext()
        loss = model(x.to(dev), mem, y.to(dev))[1]
        if a.neighbour and arm == "rag":                # train the search: a chunk should find its neighbour
            loss = loss + a.neighbour * neighbour_loss(model, x.to(dev), a.chunk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            lib_eval.refresh(model.encoder, model.emb, step=step)
            rec = dict(arm=arm, step=step + 1, secs=round(time.time() - t0, 1),
                       attached=score(model, evals, lib_eval, dev), removed=score(model, evals, None, dev),
                       same_article_hits=hit_rate(model, evals, lib_eval, a.chunk, dev))
            rec["retrieval_gain"] = round(rec["removed"] - rec["attached"], 4)
            curve.append(rec)
            print(json.dumps(rec), flush=True)
    return dict(arm=arm, params=count_params(model), final=curve[-1], curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True, help="packed Wikipedia articles (exealm.data.pack)")
    ap.add_argument("--arms", default="rag,none")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--min_tokens", type=int, default=1024, help="articles shorter than this are skipped")
    ap.add_argument("--eval_frac", type=float, default=0.1)
    ap.add_argument("--refresh_every", type=int, default=500)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neighbour", type=float, default=0.0, help="weight of the encoder's neighbour-finding loss")
    ap.add_argument("--tag", default="")
    ap.add_argument("--oracle", type=int, default=0, help="1: the search always looks inside the right article")
    ap.add_argument("--key_dim", type=int, default=0, help="size of retrieval keys (0: one circuit's width)")
    ap.add_argument("--lexical", type=int, default=0, help="size of the lexical (rare-word) part of retrieval keys")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/rag_wiki_split")
    a = ap.parse_args()
    dev = pick_device(a.device)
    shards = TokenShards(a.wiki)
    train_arts, eval_arts = split_articles(shards, a.chunk, a.min_tokens, a.seed, a.eval_frac)
    lib_train, lib_eval = library(train_arts, a.oracle).to(dev), library(eval_arts, a.oracle).to(dev)
    erng = np.random.default_rng(10_000)
    evals = [text_batch(eval_arts, a.batch, a.seq, erng) for _ in range(a.eval_batches)]
    print(json.dumps(dict(train_articles=len(train_arts), eval_articles=len(eval_arts), library_train=len(lib_train),
                          library_eval=len(lib_eval))), flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"rag_wiki_split{('_' + a.tag) if a.tag else ''}_seed{a.seed}.json")
    runs = []
    for arm in a.arms.split(","):
        runs.append(run(arm, a, dev, train_arts, evals, lib_train, lib_eval, int(shards.meta["vocab_size"])))
        json.dump(dict(args=vars(a), device=str(dev), runs=runs), open(path, "w", encoding="utf-8", newline="\n"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
