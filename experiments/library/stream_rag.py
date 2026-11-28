"""The library through the stream: how much of a retrieved entry's help survives when the model read it one or two
windows before the question, carried in its state (MemoryContext(stream=...)), rather than in the question's window.

For each held-out library item, the entry BM25 ranks first for its prefix (score.bm25_pick), truncated to --entry
tokens, and these readings, bits per byte on the target:

    none    the question alone
    same    the entry, a blank line, the question: one window (ordinary RAG, as score.py's bm25)
    back1   window 1 holds the entry (padded on the left to a full window); window 2 holds the question
    back2   as back1 with a window of unrelated held-out text between them

Attention reaches one window back through the carried keys and values; two windows back only a recurrent state
(ExeaLM's TONN layer, the Qwen3.5-style model's Gated DeltaNet) can carry anything. The share of the entry's help
kept at back2 measures that state as memory for retrieved knowledge.

    python experiments/library/stream_rag.py runs/exealm_250m_4.1x_s0 runs/strongllama_250m_4.1x_s0 \\
        --library data/library/v1 --filler data/exealm-pretrain/slice-30b-r2/packed/wikipedia_en/val \\
        --out results/expanse/stream_rag_250m.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm import MemoryContext  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.train.pretrain import load_model  # noqa: E402
from score import bm25_pick, checkpoint, entry_ids, load_items, pick_device, summarize  # noqa: E402

CONDITIONS = ("none", "same", "back1", "back2")


def windows_for(item, cond, entry, filler, seq, eot, sep):
    """The windows to read in order, each a list of token ids, and where the target starts in the last one."""
    q = list(sep) + item["prefix"] + item["target"]
    if cond == "none":
        last = [eot] + item["prefix"] + item["target"]
        return [last], 1 + len(item["prefix"])
    if cond == "same":
        last = [eot] + entry + q
        return [last], 1 + len(entry) + len(sep) + len(item["prefix"])
    first = [eot] * (seq - len(entry)) + entry                    # the entry ends exactly at the window's end
    ws = [first] + ([filler] if cond == "back2" else []) + [q]
    return ws, len(sep) + len(item["prefix"])


@torch.no_grad()
def score(model, items, cond, picks, lib, filler_rows, a, eot, sep, dev):
    bits = np.zeros(len(items))
    for b0 in range(0, len(items), a.batch):
        idx = list(range(b0, min(len(items), b0 + a.batch)))
        rows = []
        for i in idx:
            ent = entry_ids(lib, int(picks[i]))[:a.entry]
            fill = [int(t) for t in filler_rows[i % len(filler_rows)]]
            rows.append(windows_for(items[i], cond, ent, fill, a.seq, eot, sep))
        n_win = len(rows[0][0])
        state = {}
        for w in range(n_win):
            T = max(len(r[0][w]) for r in rows)
            x = torch.full((len(rows), T), eot, dtype=torch.long)
            for r, (ws, _) in enumerate(rows):
                x[r, :len(ws[w])] = torch.tensor(ws[w])
            mem = MemoryContext(stream=state) if n_win > 1 else MemoryContext()
            logits = model(x.to(dev), mem)[0]
        y = torch.full((len(rows), T), -100, dtype=torch.long)
        for r, (ws, ts) in enumerate(rows):
            ids = ws[-1]
            y[r, ts - 1:len(ids) - 1] = torch.tensor(ids[ts:])     # position p predicts token p + 1
        sel = (y != -100).to(dev)
        nll = F.cross_entropy(logits[sel].float(), y.to(dev)[sel], reduction="none")
        rr = sel.nonzero()[:, 0]
        bits[idx] = (torch.zeros(len(rows), device=dev).index_add_(0, rr, nll) / math.log(2)).cpu().numpy()
    return bits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--library", required=True)
    ap.add_argument("--filler", required=True, help="a packed held-out set: windows of unrelated text for back2")
    ap.add_argument("--sets", default="qa,tasks")
    ap.add_argument("--limit", type=int, default=2000, help="items of each set")
    ap.add_argument("--entry", type=int, default=448, help="entry tokens kept (its start)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--max_item", type=int, default=448, help="items with a longer question and target are left out")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = pick_device(a.device)
    lib = TokenShards(a.library)
    items, eot, sep = [], None, None
    for s in a.sets.split(","):
        it, eot, sep = load_items(a.library, [s], "heldout", a.limit)
        items += [x for x in it if len(x["prefix"]) + len(x["target"]) + len(sep) <= a.max_item]
    picks, top1, _ = bm25_pick(lib, items, eot, sep, dev=dev)
    fl = TokenShards(a.filler)
    ends = np.append(fl.offsets[1:], len(fl.tokens))
    starts = [int(s) for s, e in zip(fl.offsets, ends) if e - s >= a.seq][:4096]
    filler_rows = [np.asarray(fl.tokens[s:s + a.seq]) for s in starts]
    print(f"{len(items)} items, BM25 source first for {top1.mean():.3f}; {len(filler_rows)} filler windows", flush=True)
    res = dict(args=vars(a), items=len(items), bm25_top1=round(float(top1.mean()), 4), runs=[])
    sets = list(dict.fromkeys(x["set"] for x in items))
    for run in a.runs:
        model, ck = load_model(checkpoint(run), dev)
        model.eval()
        rec = dict(run=os.path.basename(run.rstrip("/")), model=ck["kind"], rows={})
        for cond in CONDITIONS:
            bits = score(model, items, cond, picks, lib, filler_rows, a, eot, sep, dev)
            for s in sets:
                ii = [i for i, x in enumerate(items) if x["set"] == s]
                rec["rows"][f"{s}/{cond}"] = summarize(items, bits, ii)["bits_per_byte"]
        for s in sets:
            r = rec["rows"]
            gain = r[f"{s}/none"] - r[f"{s}/same"]
            rec[f"{s}_kept_back1"] = round((r[f"{s}/none"] - r[f"{s}/back1"]) / gain, 4) if gain > 0 else None
            rec[f"{s}_kept_back2"] = round((r[f"{s}/none"] - r[f"{s}/back2"]) / gain, 4) if gain > 0 else None
        res["runs"].append(rec)
        print(rec["run"], json.dumps(rec["rows"]), {k: v for k, v in rec.items() if "kept" in k},
              f"({time.time() - t0:.0f}s)", flush=True)
        del model
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
