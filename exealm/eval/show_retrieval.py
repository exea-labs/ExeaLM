"""Print what the RG circuits retrieved for a prompt: for every chunk of the prompt, every layer and RG slot, the
datastore chunks the model pulled in and how much weight it gave each.

RG is the model's own RAG engine (exealm/model/retrieval.py). Looking at what it actually fetches is the quickest way
to tell whether retrieval learned anything: early in training the picks look random; a model that uses its datastore
should pull in chunks on the same topic as the text it is reading.

    # a trained checkpoint, the datastore rebuilt the way pretrain.py builds it, a prompt as text
    python -m exealm.eval.show_retrieval --ckpt runs/small/model.pt --data data/packed/train --datastore_chunks 20000 \\
        --tokenizer tokenizer --prompt "The capital of Japan is Tokyo. It became the capital in 1868, when"

    # no checkpoint or tokenizer needed: an untrained tiny model on synthetic data, the prompt a window of the data
    python -m exealm.eval.show_retrieval --preset tiny --data synthetic --datastore_chunks 2000

The datastore is rebuilt with Datastore.from_shards and the same seed pretrain.py uses (--seed + 1), then re-encoded
with the checkpoint's own encoder. Retrieval happens once per chunk of cfg.chunk tokens, from the second chunk on, so
the prompt must be longer than one chunk (64 tokens for the 100m and small presets, 8 for tiny).
"""
import argparse

import numpy as np
import torch

from ..config import get_config
from ..data.loader import SyntheticShards, TokenShards
from ..memory import Datastore, MemoryContext
from ..model import ExeaLM


@torch.no_grad()
def retrievals(model, ids, mem, top=None):
    """Run the model on ids (1, T) and return what every RG slot retrieved, as a list of dicts:
    chunk (the chunk of the prompt being read), layer (the model's layer number; in a hybrid only the TONN layers have
    RG slots), slot, and hits: [(datastore index, weight)] sorted by weight, at most `top` per slot."""
    was = model.training
    model.eval()
    rgs = [lay.rg for lay in model.layers]
    for rg in rgs:
        rg.record = True
    try:
        model(ids, mem)
    finally:
        for rg in rgs:
            rg.record = False
        model.train(was)
    out = []
    layer_no = [i for i in range(model.cfg.n_layers) if i not in model.cfg.attn_layers]   # TONN layer -> model layer
    for i, lay in enumerate(model.layers):
        rec = lay.rg.last
        if rec is None:
            continue
        slots = [k for k, s in enumerate(lay.specs) if s["tmpl"] == "RG"]
        n = rec["idx"].shape[1]
        for c in range(n):
            for j, k in enumerate(slots):
                idx, w, ok = rec["idx"][0, c, j], rec["weight"][0, c, j], rec["valid"][0, c, j]
                hits = sorted(((int(a), float(b)) for a, b, v in zip(idx, w, ok) if v), key=lambda h: -h[1])
                out.append(dict(chunk=c + 1, layer=layer_no[i], slot=k, hits=hits[:top] if top else hits))
    return out


def show(text_of, ids, recs, ds, chunk):
    """Print the prompt chunk by chunk, each followed by what was retrieved while reading it."""
    by_chunk = {}
    for r in recs:
        by_chunk.setdefault(r["chunk"], []).append(r)
    T = ids.shape[1]
    for c in range(-(-T // chunk)):
        span = ids[0, c * chunk:(c + 1) * chunk]
        print(f"=== chunk {c} (tokens {c * chunk}-{c * chunk + len(span) - 1}): {text_of(span)!r}")
        if c not in by_chunk:
            print("    (the first chunk retrieves nothing: its query is only ready at the chunk's end)")
            continue
        for r in by_chunk[c]:
            print(f"  layer {r['layer']} slot {r['slot']}:")
            for i, w in r["hits"]:
                print(f"    {w:6.3f}  #{i} (doc {int(ds.doc_ids[i])})  {text_of(ds.tokens[i])!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None, help="model.pt from exealm.train.pretrain (default: an untrained model)")
    ap.add_argument("--preset", default="tiny", help="used only without --ckpt")
    ap.add_argument("--data", default="synthetic",
                    help="packed directory the datastore is sampled from, or 'synthetic'")
    ap.add_argument("--datastore_chunks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="the --seed of the training run (the datastore uses seed + 1)")
    ap.add_argument("--tokenizer", default=None, help="to encode --prompt and print text instead of token ids")
    ap.add_argument("--prompt", default=None, help="text (needs --tokenizer); default: a window of --data")
    ap.add_argument("--prompt_len", type=int, default=None, help="length of the window from --data (default: 4 chunks)")
    ap.add_argument("--top", type=int, default=None, help="print at most this many hits per slot")
    a = ap.parse_args(argv)

    data = SyntheticShards() if a.data == "synthetic" else TokenShards(a.data)
    if a.ckpt:
        from ..train.pretrain import load_model
        model, ck = load_model(a.ckpt)
        if ck["kind"] != "tonn":
            raise SystemExit("this checkpoint is the Llama baseline, which has no retrieval")
    else:
        model = ExeaLM(get_config(a.preset, vocab_size=int(data.meta["vocab_size"])))
    cfg = model.cfg
    if a.ckpt and model.encoder is not None:
        model.encoder.idf_ready = True                     # keep the checkpoint's own word rarities (rag_lexical)
    ds = Datastore.from_shards(data, a.datastore_chunks, cfg.rag_chunk_len, seed=a.seed + 1)
    ds.refresh(model.encoder, model.emb)

    tok = None
    if a.tokenizer:
        from ..data.tokenizer import load
        tok = load(a.tokenizer)
    text_of = (lambda t: tok.decode([int(x) for x in t])) if tok else (lambda t: " ".join(str(int(x)) for x in t))

    if a.prompt is not None:
        if tok is None:
            raise SystemExit("--prompt needs --tokenizer")
        ids = torch.tensor([tok.encode(a.prompt).ids])
        docs = torch.full_like(ids, -1)                                  # a new text: no datastore chunk is excluded
    else:
        L = a.prompt_len or 4 * cfg.chunk
        start = int(np.random.default_rng(a.seed + 2).integers(0, len(data.tokens) - L))
        ids = torch.from_numpy(np.asarray(data.tokens[start:start + L]).astype(np.int64)).view(1, L)
        docs = torch.from_numpy(data.doc_of(np.arange(start, start + L)).astype(np.int64)).view(1, L)
    if ids.shape[1] <= cfg.chunk:
        raise SystemExit(f"the prompt is {ids.shape[1]} tokens; retrieval needs more than one chunk ({cfg.chunk})")
    recs = retrievals(model, ids, MemoryContext(datastore=ds, doc_ids=docs), a.top)
    show(text_of, ids, recs, ds, cfg.chunk)
    return recs


if __name__ == "__main__":
    main()
