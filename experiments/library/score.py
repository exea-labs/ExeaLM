"""Score checkpoints on a distilled library's question/answer pairs and morph tasks: bits per byte on the target
tokens only, with the library attached or removed, or with the right entry pasted before the task.

The question is whether a model relies on its library and adapts what it finds. Every item of the packs that
experiments/library/build.py pack writes (qa/<split>, tasks/<split>) is a prefix (a question, or an instruction and
the user's context) and a target. Conditions:

    library   RG searches the library (models with RG slots only)
    none      no library and nothing pasted: the model alone
    pasted    the item's source entry pasted before the prefix: ordinary RAG with the right document, an upper bound
              on what a perfect search could give

For ExeaLM, "library" and "none" are the attached and removed comparison; for a Llama baseline, "pasted" and "none" are
RAG with the right document and the model with nothing. With the library, it also reports how often the model's own
search returns a chunk of the item's source entry among its top-k: "search_first" at the target's first chunk, whose
search reads the end of the prefix, and "search_any" at any chunk of the target. Both count the items whose entry has a
chunk in the datastore ("searchable"), over every RG slot of every layer.

Layout. RG searches once per chunk of cfg.chunk tokens, and the query for chunk c is chunk c - 1. So "library" and
"none" rows start with as many <|endoftext|> tokens as it takes (at least one) for the prefix to end on a chunk
boundary: the target's first search reads the prefix. "pasted" rows are <|endoftext|>, the entry, a blank line, the
prefix and the target. Every scored row has its own document number, and the library's are shifted past them, so a
scored row never hides a library entry from the search (RG never returns the document being read).

The datastore holds every chunk of every entry (cut every cfg.rag_chunk_len tokens, the last chunk ending at the
entry's end), or --datastore_chunks chunks sampled as pretraining samples them.

    python experiments/library/score.py --library data/library/v1 runs/exealm/model.pt runs/llama/model.pt \\
        --out results/library/score.jsonl
"""
import argparse
import contextlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.hub import build_model  # noqa: E402
from exealm.memory import Datastore, MemoryContext  # noqa: E402

CONDITIONS = ("library", "none", "pasted")


def checkpoint(path):
    """A run folder stands for its model.pt, or its ckpt.pt while the run is still training."""
    if os.path.isdir(path):
        for name in ("model.pt", "ckpt.pt"):
            if os.path.exists(os.path.join(path, name)):
                return os.path.join(path, name)
        raise SystemExit(f"{path} holds neither model.pt nor ckpt.pt")
    return path


def label(path):
    """A short name for a checkpoint: its run folder for model.pt and ckpt.pt, else its file name."""
    base = os.path.basename(path)
    return os.path.basename(os.path.dirname(path)) if base in ("model.pt", "ckpt.pt") else base


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_items(lib_dir, sets, split, limit=0):
    """The items of the chosen sets, each a dict with set, kind, type, source_doc, prefix and target token ids and
    target_bytes. Also returns the ids of <|endoftext|> and of a blank line, from the packs' meta.json."""
    items, eot, sep = [], None, None
    for s in sets:
        d = os.path.join(lib_dir, s, split)
        if not os.path.exists(os.path.join(d, "meta.json")):
            print(f"{d}: no such pack, skipped")
            continue
        shards = TokenShards(d)
        eot, sep = shards.meta["eot_id"], shards.meta["sep_ids"]
        with open(os.path.join(d, "items.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for r in rows[:limit or None]:
            a = int(shards.offsets[r["doc"]])
            ids = [int(t) for t in shards.tokens[a:a + r["target_end"]]]
            items.append(dict(set=s, kind=r["kind"], type=r.get("type"), source_doc=r["source_doc"],
                              prefix=ids[:r["target_start"]], target=ids[r["target_start"]:],
                              target_bytes=r["target_bytes"]))
    if not items:
        raise SystemExit(f"nothing to score in {lib_dir} ({', '.join(sets)}, split {split})")
    return items, eot, sep


def entry_ids(lib, d):
    """Library document d's tokens without its <|endoftext|>."""
    end = int(lib.offsets[d + 1]) if d + 1 < len(lib.offsets) else len(lib.tokens)
    return [int(t) for t in lib.tokens[int(lib.offsets[d]):end - 1]]


def library_store(lib, chunk_len, n_chunks=0, seed=1):
    """Every chunk of every entry: cut every chunk_len tokens, the last chunk ending at the entry's end so the end is
    in too. Entries shorter than chunk_len have none. n_chunks > 0: that many sampled as pretraining samples them."""
    if n_chunks:
        return Datastore.from_shards(lib, n_chunks, chunk_len, seed=seed)
    toks = np.asarray(lib.tokens)
    ends = np.append(lib.offsets[1:], len(toks))
    starts, docs = [], []
    for d, (a, b) in enumerate(zip(lib.offsets.tolist(), ends.tolist())):
        if b - a < chunk_len:
            continue
        s = list(range(a, b - chunk_len + 1, chunk_len))
        if s[-1] != b - chunk_len:
            s.append(b - chunk_len)
        starts += s
        docs += [d] * len(s)
    if not starts:
        raise SystemExit(f"no library entry is as long as one retrieval chunk ({chunk_len} tokens)")
    idx = np.array(starts)[:, None] + np.arange(chunk_len)[None]
    return Datastore(toks[idx].astype(np.int64), np.array(docs))


def layout(item, condition, chunk, eot, sep, entry=None):
    """(token ids, position where the target starts) of one item in one condition (see the module docstring)."""
    pre, tgt = item["prefix"], item["target"]
    head = [eot] + entry + sep if condition == "pasted" else [eot] * (1 + (-(1 + len(pre))) % chunk)
    return head + pre + tgt, len(head) + len(pre)


def rg_circuits(model):
    """The RG circuits of the layers that have RG slots (none for a Llama, or an ExeaLM without RG)."""
    if getattr(model, "encoder", None) is None:
        return []
    return [lay.rg for lay in model.layers if any(s["tmpl"] == "RG" for s in lay.specs)]


@torch.no_grad()
def score(model, items, condition, eot, sep, lib, ds=None, offset=0, batch=8, dev="cpu", amp=None):
    """Bits on each item's target, and whether the search returned a chunk of the item's source entry (at the
    target's first chunk, and at any of its chunks; only with a datastore). Library document d is document
    d + offset in ds; scored row i is document i."""
    C = model.cfg.chunk
    rows = [layout(it, condition, C, eot, sep, entry_ids(lib, it["source_doc"]) if condition == "pasted" else None)
            for it in items]
    bits = np.zeros(len(items))
    first, anyhit = np.zeros(len(items), bool), np.zeros(len(items), bool)
    rgs = rg_circuits(model) if ds is not None else []
    lib_docs = ds.doc_ids.cpu() if ds is not None else None
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    for b0 in range(0, len(order), batch):
        part = order[b0:b0 + batch]
        T = max(len(rows[i][0]) for i in part)
        x = torch.full((len(part), T), eot, dtype=torch.long)
        y = torch.full((len(part), T), -100, dtype=torch.long)
        for r, i in enumerate(part):
            ids, ts = rows[i]
            x[r, :len(ids)] = torch.tensor(ids)
            y[r, ts - 1:len(ids) - 1] = torch.tensor(ids[ts:])        # position p predicts token p + 1
        docs = torch.tensor(part, dtype=torch.long).view(-1, 1).expand(-1, T).contiguous()
        mem = MemoryContext(datastore=ds, doc_ids=docs.to(dev), swarm_size=1)
        for rg in rgs:
            rg.record = True
        try:
            with torch.autocast(**amp) if amp else contextlib.nullcontext():
                logits = model(x.to(dev), mem)[0]
        finally:
            for rg in rgs:
                rg.record = False
        sel = y != -100
        nll = F.cross_entropy(logits[sel.to(dev)].float(), y[sel].to(dev), reduction="none").cpu()
        bits[part] = torch.zeros(len(part)).index_add_(0, sel.nonzero()[:, 0], nll).numpy() / math.log(2)
        want = torch.tensor([items[i]["source_doc"] + offset for i in part]).view(-1, 1, 1, 1)
        for rg in rgs:
            if rg.last is None:
                continue
            match = (lib_docs[rg.last["idx"]] == want) & rg.last["valid"]       # (rows, chunks - 1, slots, k)
            for r, i in enumerate(part):
                ids, ts = rows[i]
                c0, c1 = ts // C, (len(ids) - 1) // C                    # the target's first and last chunk
                first[i] |= bool(match[r, c0 - 1].any())                 # chunk c's search is entry c - 1
                anyhit[i] |= bool(match[r, c0 - 1:c1].any())
            rg.last = None
    return bits, first, anyhit


def summarize(items, bits, idx):
    """Bits per byte and per token over the items idx, pooled (total bits over total target bytes)."""
    n_bytes = sum(items[i]["target_bytes"] for i in idx)
    n_tok = sum(len(items[i]["target"]) for i in idx)
    total = float(sum(bits[i] for i in idx))
    return dict(items=len(idx), target_tokens=n_tok, target_bytes=n_bytes,
                bits_per_byte=round(total / max(1, n_bytes), 4), bits_per_token=round(total / max(1, n_tok), 4))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpts", nargs="+", help="checkpoints or run folders: ExeaLM and Llama baselines alike")
    ap.add_argument("--library", required=True, help="the packed library from experiments/library/build.py pack")
    ap.add_argument("--sets", default="qa,tasks")
    ap.add_argument("--split", default="heldout", choices=["heldout", "train"])
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--datastore_chunks", type=int, default=0,
                    help="0: every chunk of every entry. N: N chunks sampled as pretraining samples them")
    ap.add_argument("--seed", type=int, default=0, help="the run's --seed: a sampled datastore uses seed + 1, as in "
                                                        "pretraining")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score at most this many items of each set (0: all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "none"], help="auto: bfloat16 on CUDA and ROCm")
    ap.add_argument("--out", default=None, help="a .jsonl file for the rows")
    a = ap.parse_args(argv)

    conditions = [c for c in a.conditions.split(",") if c]
    if set(conditions) - set(CONDITIONS):
        raise SystemExit(f"--conditions takes {', '.join(CONDITIONS)}")
    dev = pick_device(a.device)
    use_amp = a.amp == "bf16" or (a.amp == "auto" and dev.type == "cuda")
    amp = dict(device_type=dev.type, dtype=torch.bfloat16) if use_amp else None
    lib = TokenShards(a.library)
    items, eot, sep = load_items(a.library, [s for s in a.sets.split(",") if s], a.split, a.limit)
    top = max(max(max(it["prefix"] + it["target"]) for it in items), int(np.max(lib.tokens)))
    out = []
    for path in a.ckpts:
        path = checkpoint(path)
        model = build_model(torch.load(path, map_location="cpu", weights_only=False)).to(dev).eval()
        if top >= model.cfg.vocab_size:
            raise SystemExit(f"{path} has {model.cfg.vocab_size} tokens but the library holds token {top}: pack the "
                             "library with the tokenizer the model was trained with")
        name = type(model).__name__
        for cond in conditions:
            ds = None
            if cond == "library":
                if not rg_circuits(model):
                    print(f"{path}: {name} has no RG slots, so no library condition")
                    continue
                ds = library_store(lib, model.cfg.rag_chunk_len, a.datastore_chunks, a.seed + 1)
                ds.doc_ids += len(items)                                 # scored rows are documents 0 .. n - 1
                ds = ds.to(dev)
                ds.refresh(model.encoder, model.emb)
            bits, first, anyhit = score(model, items, cond, eot, sep, lib, ds, len(items), a.batch, dev, amp)
            in_store = set((ds.doc_ids.cpu() - len(items)).tolist()) if ds is not None else set()
            for s in dict.fromkeys(it["set"] for it in items):
                idx = [i for i, it in enumerate(items) if it["set"] == s]
                row = dict(ckpt=path, model=name, set=s, condition=cond, **summarize(items, bits, idx))
                row["by_kind"] = {k: summarize(items, bits, [i for i in idx if items[i]["kind"] == k])["bits_per_byte"]
                                  for k in dict.fromkeys(items[i]["kind"] for i in idx)}
                if ds is not None:
                    ok = [i for i in idx if items[i]["source_doc"] in in_store]
                    row.update(searchable=len(ok), search_first=round(float(first[ok].mean()), 4) if ok else None,
                               search_any=round(float(anyhit[ok].mean()), 4) if ok else None)
                out.append(row)
    print(f"{'checkpoint':32s} {'model':7s} {'set':6s} {'condition':9s} {'items':>6s} {'bits/byte':>9s} "
          f"{'search@1st':>10s} {'search@any':>10s}")
    for r in out:
        hit = ("" if r.get("search_first") is None else f"{r['search_first']:.3f}",
               "" if r.get("search_any") is None else f"{r['search_any']:.3f}")
        print(f"{label(r['ckpt']):32.32s} {r['model']:7s} {r['set']:6s} {r['condition']:9s} {r['items']:6d} "
              f"{r['bits_per_byte']:9.4f} {hit[0]:>10s} {hit[1]:>10s}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r) + "\n" for r in out)
    return out


if __name__ == "__main__":
    main()
