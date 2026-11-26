"""Bits per byte of trained models on the held-out sets, every row read alone: no board shared with other rows, no
library, 32-bit. It is the number of the same-size comparisons (experiments/tokenizer_lm.py scores its models the
same way, on the same windows), so an ExeaLM trained in swarms and a Llama are scored alike.

    python experiments/eval_text.py runs/exealm_500m_s0 runs/gated_500m_s0 --out results/expanse/text_500m.json

A run folder stands for its model.pt, or for its ckpt.pt while the run is still training. Prints one line per model.
Runs whose names differ only in a seed at the end (..._s0, ..._s1) are one group: the last lines are each group's mean
over its seeds, next to the first group's.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exealm import MemoryContext, count_params  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.progress import Progress  # noqa: E402
from exealm.train.pretrain import load_model, pick_device  # noqa: E402

SETS = "val_en,val_ja_wiki,val_ja_web,val_code"


def mix_sets(path):
    """{source: packed directory} from a mix .json (the val mix fetch.sh writes); a source given as a list or a pattern
    uses its first directory."""
    import glob as _glob
    out = {}
    for name, src in json.load(open(path))["sources"].items():
        p = src["paths"] if isinstance(src["paths"], str) else src["paths"][0]
        found = sorted(_glob.glob(p)) if any(c in p for c in "*?[") else [p]
        if found:
            out[name] = found[0]
    return out


def checkpoint(path):
    if os.path.isdir(path):
        for name in ("model.pt", "ckpt.pt"):
            if os.path.exists(os.path.join(path, name)):
                return os.path.join(path, name)
        raise SystemExit(f"{path} holds neither model.pt nor ckpt.pt")
    return path


@torch.no_grad()
def bits_per_byte(model, val, batch, seq, batches, dev):
    """The same windows for every model: the generator starts from the same seed for every set."""
    rng = np.random.default_rng(10_000)
    nll = 0.0
    for _ in range(batches):
        x, y, _ = val.batch(batch, seq, rng)
        nll += float(model(x.to(dev), MemoryContext(), y.to(dev))[1])
    return round(nll / batches / math.log(2) / val.meta["bytes_per_token"], 4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders or checkpoint files")
    ap.add_argument("--root", default="data/tokpack/bpe32k", help="the folder the held-out packs are in")
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--mix", default=None, help="a mix .json (e.g. data/mix_pretrain_val.json): score every source in it "
                    "instead of --root/--sets; each source must be one packed directory")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="a .json file for the rows")
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    vals = ({n: TokenShards(d) for n, d in mix_sets(a.mix).items()} if a.mix
            else {s: TokenShards(os.path.join(a.root, s)) for s in a.sets.split(",")})
    rows = []
    bar = Progress(total=len(a.runs), desc="held-out text", unit="run", every=0)
    for run in a.runs:
        bar.write(f"scoring {run} on {len(vals)} held-out sets")
        path = checkpoint(run)
        model, ck = load_model(path, dev)
        model.eval()
        bits = {s[4:] if s.startswith("val_") else s: bits_per_byte(model, v, a.batch, a.seq, a.eval_batches, dev)
                for s, v in vals.items()}
        rows.append(dict(run=run, file=os.path.basename(path), model=ck["kind"], step=ck["step"],
                         params_M=round(count_params(model) / 1e6, 2), bits_per_byte=bits,
                         mean=round(sum(bits.values()) / len(bits), 4)))
        print(json.dumps(rows[-1]), flush=True)
        bar.update(1, last=f"{rows[-1]['mean']:.4f}")
        del model, ck
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    groups = {}
    for r in rows:
        groups.setdefault(re.sub(r"_s\d+$", "", os.path.basename(os.path.normpath(r["run"]))), []).append(r["mean"])
    means = {g: round(sum(v) / len(v), 4) for g, v in groups.items()}
    first = next(iter(means))
    for g, v in groups.items():
        against = "" if g == first else f"; {means[g] - means[first]:+.4f} against {first}"
        print(f"{g}: {means[g]:.4f} bits per byte, mean of {len(v)} ({', '.join(f'{x:.4f}' for x in v)}){against}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(runs=rows, means=means), open(a.out, "w"), indent=1)
        print("wrote", a.out)
    return rows, means


if __name__ == "__main__":
    main()
