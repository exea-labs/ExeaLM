"""How much a model gains from reading a long document window by window with its state carried from one window to
the next (MemoryContext(stream=...), docs/brain.md), against reading every window alone. ExeaLM carries its TONN
layers' recurrent state and its attention layers' keys and values of the window before; a Llama carries only the keys
and values of the window before.

    python experiments/stream_eval.py runs/finalize/core_s0 runs/finalize/brain_core_s0 runs/finalize/gated_s0 \\
        --windows 4 --out results/finalize/stream.json

For each held-out set it takes every document at least windows x seq + 1 tokens long (its first that many tokens),
and prints bits per byte for each window position: streamed (the windows before it carried in) and alone. Window 0 is
the same both ways. 32-bit, no board, no library.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exealm import MemoryContext  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.train.pretrain import load_model, pick_device  # noqa: E402

SETS = "val_en,val_ja_wiki,val_ja_web,val_code"


def checkpoint(path):
    if os.path.isdir(path):
        for name in ("model.pt", "ckpt.pt"):
            if os.path.exists(os.path.join(path, name)):
                return os.path.join(path, name)
        raise SystemExit(f"{path} holds neither model.pt nor ckpt.pt")
    return path


def long_documents(val, length, limit):
    """(N, length) tokens: the start of every document at least `length` tokens long, at most `limit` of them."""
    ends = np.append(val.offsets[1:], len(val.tokens))
    starts = [int(s) for s, e in zip(val.offsets, ends) if e - s >= length][:limit]
    if not starts:
        return None
    return torch.from_numpy(np.stack([np.asarray(val.tokens[s:s + length], dtype=np.int64) for s in starts]))


@torch.no_grad()
def window_bits(model, rows, windows, seq, batch, dev):
    """Summed bits and target counts per window position, streamed and alone."""
    out = {m: np.zeros(windows) for m in ("streamed", "alone")}
    count = np.zeros(windows)
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch].to(dev)
        state = {}
        for w in range(windows):
            x, y = r[:, w * seq:(w + 1) * seq], r[:, w * seq + 1:(w + 1) * seq + 1]
            for mode, mem in (("streamed", MemoryContext(stream=state)), ("alone", MemoryContext())):
                logits = model(x, mem)[0].float()
                out[mode][w] += float(F.cross_entropy(logits.transpose(1, 2), y, reduction="sum")) / math.log(2)
            count[w] += y.numel()
    return {m: v / count for m, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders or checkpoint files")
    ap.add_argument("--root", default="data/tokpack/bpe32k")
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--docs", type=int, default=128, help="at most this many documents per set")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    sets = {}
    for name in a.sets.split(","):
        val = TokenShards(os.path.join(a.root, name))
        rows = long_documents(val, a.windows * a.seq + 1, a.docs)
        if rows is not None:
            sets[name] = (rows, val.meta["bytes_per_token"])
    print("documents: " + ", ".join(f"{k} {len(v[0])}" for k, v in sets.items()))
    results = []
    for run in a.runs:
        model, ck = load_model(checkpoint(run), dev)
        model.eval()
        rec = dict(run=os.path.basename(run.rstrip("/")), model=ck["kind"], step=ck["step"], sets={})
        for name, (rows, bpt) in sets.items():
            bits = window_bits(model, rows, a.windows, a.seq, a.batch, dev)
            rec["sets"][name] = {m: [round(float(b / bpt), 4) for b in v] for m, v in bits.items()}
        for m in ("streamed", "alone"):
            rec[m] = [round(float(np.mean([s[m][w] for s in rec["sets"].values()])), 4) for w in range(a.windows)]
        results.append(rec)
        gain = [round(al - st, 4) for st, al in zip(rec["streamed"], rec["alone"])]
        print(f"{rec['run']}: streamed {rec['streamed']}  alone {rec['alone']}  gain by window {gain}")
        del model
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(args=vars(a), runs=results), open(a.out, "w"), indent=1)
    return results


if __name__ == "__main__":
    main()
