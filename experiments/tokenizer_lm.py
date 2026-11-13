"""Tokenizer size (and architecture), decided by a small model's loss. For each candidate tokenizer, train the same small model for the
same number of TOKENS (our plan fixes the token count, so a bigger vocabulary also means more text seen) on English,
Japanese and code, and score bits per BYTE on held-out text of each kind. Bits per byte compare across tokenizers;
bits per token do not.

Expects data packed per tokenizer: <root>/<tok>/train_{en,ja,code} and <root>/<tok>/val_{en,ja_wiki,ja_web,code}.

python experiments/tokenizer_lm.py --root data/tokpack --toks bpe32k,bpe48k,bpe64k --steps 3000 --out results/tokenizer
python experiments/tokenizer_lm.py --toks bpe32k --models tonn,llama --seeds 0,1 --out results/arch   # TONN vs Llama
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
from exealm import ExeaLM, Llama, MemoryContext, count_params, get_config  # noqa: E402
from exealm.config import MENU  # noqa: E402
from exealm.model.search import Searcher  # noqa: E402
from exealm.data.loader import MultiShards, TokenShards  # noqa: E402
from exealm.train.pretrain import lr_at, make_optimizer, pick_device  # noqa: E402

TRAIN = ("train_en", "train_ja", "train_code")
VAL = ("val_en", "val_ja_wiki", "val_ja_web", "val_code")


@torch.no_grad()
def bits_per_byte(model, val, a, dev):
    model.eval()
    rng = np.random.default_rng(10_000)
    nll = 0.0
    for _ in range(a.eval_batches):
        x, y, _ = val.batch(a.batch, a.seq, rng)
        nll += float(model(x.to(dev), MemoryContext(), y.to(dev))[1])
    model.train()
    return round(nll / a.eval_batches / math.log(2) / val.meta["bytes_per_token"], 4)


def run(tok, a, dev, kind="tonn", seed=0):
    root = os.path.join(a.root, tok)
    train = MultiShards([os.path.join(root, t) for t in TRAIN], weights=[float(w) for w in a.mix.split(",")])
    vals = {v: TokenShards(os.path.join(root, v)) for v in VAL}
    torch.manual_seed(seed)
    # kinds: tonn (default: one RG and one AG circuit pinned per layer; idle here: no datastore, no swarm),
    # tonn_core (no memory circuits: every slot models language), tonn_core_search (the same with TONN's search), llama
    over = {}
    if kind.startswith("tonn_core"):
        over = dict(pinned={}, menu=tuple(t for t in MENU if t not in ("RG", "AG")))
    if kind != "llama":
        over.update(json.loads(a.cfg))                 # e.g. {"init_weights": {"RP": 4}, "attn_layers": [1, 3, 5]}
        for k in ("attn_layers", "menu", "lags"):
            if k in over:
                over[k] = tuple(over[k])
    if kind == "llama":
        over.update(json.loads(a.llama_cfg))           # e.g. Qwen3.5-style: {"delta_layers": [0, 1, 2, 4], "attn_gate": true}
        if "delta_layers" in over:
            over["delta_layers"] = tuple(over["delta_layers"])
    elif "delta_layers" in over:
        over["delta_layers"] = tuple(over["delta_layers"])
    over = {**json.loads(a.size), **over}                  # model size for every model, Llama included
    cfg = get_config("small-tonn", vocab_size=int(train.meta["vocab_size"]), seed=seed, **over)
    model = (Llama(cfg) if kind == "llama" else ExeaLM(cfg)).to(dev)
    searcher, sel = None, None
    if kind.endswith("_search"):
        searcher = Searcher(every=250, until=0.5, seed=seed)
        srng = np.random.default_rng(20_000 + seed)
        sel = [(x.to(dev), y.to(dev), MemoryContext()) for x, y, _ in (train.batch(a.batch, a.seq, srng) for _ in range(4))]
    opt = make_optimizer(model, a.lr, 0.1)
    rng = np.random.default_rng(seed)
    t0, bytes_seen, curve, recent = time.time(), 0.0, [], []
    total = a.steps
    step = 0
    while step < total:
        if a.seconds and step == 50:                      # equal compute: fit the run to a time budget
            if dev.type == "mps":
                torch.mps.synchronize()
            per = (time.time() - t_after_10) / 40
            total = max(100, int(a.seconds / per))
        if step == 10:
            if dev.type == "mps":
                torch.mps.synchronize()
            t_after_10 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, a.lr, 200, 0.2)
        if searcher is not None:
            searcher.maybe_round(model, opt, step, total, sel, a.seq)
        x, y, _ = train.batch(a.batch, a.seq, rng)
        loss = model(x.to(dev), MemoryContext(), y.to(dev))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bytes_seen += a.batch * a.seq * train.meta["bytes_per_token"]
        recent.append(float(loss.detach()))
        if (step + 1) % 250 == 0:                          # training-loss curve (spikes show instability)
            curve.append(round(sum(recent) / len(recent), 4))
            recent = []
        if dev.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()
        step += 1
    rec = dict(model=kind, tag=a.tag, cfg=a.cfg if kind != "llama" else "{}", lr=a.lr, seed=seed, tokenizer=tok,
               vocab=int(train.meta["vocab_size"]), params_M=round(count_params(model) / 1e6, 2),
               steps=total, seconds_budget=a.seconds, tokens_seen=total * a.batch * a.seq, bytes_seen=int(bytes_seen),
               secs=round(time.time() - t0, 1),
               bits_per_byte={v[4:]: bits_per_byte(model, s, a, dev) for v, s in vals.items()},
               architecture=model.architecture() if kind != "llama" else None, train_loss_every_250=curve)
    print(json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tokpack")
    ap.add_argument("--toks", default="bpe32k,bpe48k,bpe64k")
    ap.add_argument("--mix", default="0.5,0.3,0.2", help="weights of English, Japanese and code in training")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=0, help="equal compute: set the step count to fill this many seconds")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--models", default="tonn", help="any of tonn, tonn_core, tonn_core_search, llama")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cfg", default="{}", help="JSON ModelConfig overrides for the TONN models")
    ap.add_argument("--llama_cfg", default="{}", help="JSON overrides for the Llama (delta_layers, attn_gate, qk_norm)")
    ap.add_argument("--size", default="{}", help='JSON size overrides for every model, e.g. {"d": 384, "n_layers": 8}')
    ap.add_argument("--tag", default="", help="a name for this run in the output rows")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/tokenizer")
    a = ap.parse_args()
    dev = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    name = "lm_seed0.json" if (a.models, a.seeds, a.tag) == ("tonn", "0", "") else f"lm_compare{('_' + a.tag) if a.tag else ''}.json"
    for tok in a.toks.split(","):
        for kind in a.models.split(","):
            for seed in [int(x) for x in a.seeds.split(",")]:
                rows.append(run(tok, a, dev, kind, seed))
                json.dump(dict(args=vars(a), device=str(dev), rows=rows), open(os.path.join(a.out, name), "w"), indent=1)


if __name__ == "__main__":
    main()
