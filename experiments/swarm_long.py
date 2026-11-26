"""Claim 11 of docs/report_plan.md at small scale: do several small ExeaLM agents sharing a board read long documents
better than one bigger model? Every model reads the same first windows x seq + 1 tokens of the same long held-out
documents, and is scored in bits per byte for each window position.

- A swarm model (an ExeaLM with board slots, post-trained in swarms; experiments/expanse/board_post.sh) gives one
  window to each agent: agent j reads window j, and with board_order "doc" it sees the posts of the agents before it,
  which read the earlier text, and nothing from the agents after it. "board" shares the board; "cut" scores every
  agent alone (swarm size 1), the same weights with the board removed.
- A single model (any ExeaLM, Llama or Qwen3.5-style checkpoint) reads the windows in order: "streamed" carries the
  state of the window before into the next (stream_eval.window_bits), "alone" reads each window by itself.

Window 0 has no earlier text, so the comparison is in the later windows. Every agent runs in parallel: a swarm of
three 0.1B agents spends about 0.38 times the compute per token of one 0.25B model.

    python experiments/swarm_long.py --swarm runs/boardpost_ag_100m --single runs/exealm_250m_4.1x_s0 \\
        runs/qwen_250m_4.1x_s0 runs/boardpost_solo_100m --mix data/mix_pretrain_val.json --out results/expanse/board_100m.json
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exealm import MemoryContext  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.train.pretrain import load_model, pick_device  # noqa: E402
from stream_eval import checkpoint, long_documents, mix_sets, window_bits  # noqa: E402


@torch.no_grad()
def swarm_bits(model, rows, windows, seq, docs_per_batch, dev):
    """Summed bits per target, per window position, with the board shared ("board") and cut ("cut"). rows: (N,
    windows * seq + 1). Each document becomes `windows` consecutive rows, agent j reading window j."""
    out = {m: np.zeros(windows) for m in ("board", "cut")}
    count = np.zeros(windows)
    for i in range(0, len(rows), docs_per_batch):
        r = rows[i:i + docs_per_batch]
        x = torch.stack([r[:, w * seq:(w + 1) * seq] for w in range(windows)], 1).reshape(-1, seq).to(dev)
        y = torch.stack([r[:, w * seq + 1:(w + 1) * seq + 1] for w in range(windows)], 1).reshape(-1, seq).to(dev)
        docs = torch.arange(i, i + len(r)).repeat_interleave(windows).to(dev)
        for mode, swarm in (("board", windows), ("cut", 1)):
            logits = model(x, MemoryContext(doc_ids=docs, swarm_size=swarm))[0].float()
            nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none").sum(1).view(len(r), windows)
            out[mode] += nll.sum(0).double().cpu().numpy() / math.log(2)
        count += len(r) * seq
    return {m: v / count for m, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--swarm", nargs="*", default=[], help="ExeaLM checkpoints with board slots, scored as swarms")
    ap.add_argument("--single", nargs="*", default=[], help="checkpoints scored as one model reading the windows")
    ap.add_argument("--mix", default="data/mix_pretrain_val.json", help="the held-out mix: every source is scored")
    ap.add_argument("--windows", type=int, default=3, help="windows per document: the swarm's agents")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--docs", type=int, default=64, help="at most this many documents per held-out set")
    ap.add_argument("--batch", type=int, default=8, help="documents per forward pass")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    dev = pick_device(a.device)
    sets = {}
    for name, path in mix_sets(a.mix).items():
        val = TokenShards(path)
        rows = long_documents(val, a.windows * a.seq + 1, a.docs)
        if rows is not None:
            sets[name] = (rows, val.meta["bytes_per_token"])
    print("documents: " + ", ".join(f"{k} {len(v[0])}" for k, v in sets.items()), flush=True)
    results = []
    for kind, runs in (("swarm", a.swarm), ("single", a.single)):
        for run in runs:
            model, ck = load_model(checkpoint(run), dev)
            model.eval()
            if kind == "swarm" and model.cfg.swarm_size != a.windows:
                print(f"{run}: trained in swarms of {model.cfg.swarm_size}, scored in swarms of {a.windows}")
            rec = dict(run=os.path.basename(run.rstrip("/")), kind=kind, model=ck["kind"], step=ck.get("step"),
                       params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2), sets={})
            for name, (rows, bpt) in sets.items():
                if kind == "swarm":
                    bits = swarm_bits(model, rows, a.windows, a.seq, a.batch, dev)
                else:
                    bits = window_bits(model, rows, a.windows, a.seq, a.batch, dev)
                rec["sets"][name] = {m: [round(float(b / bpt), 4) for b in v] for m, v in bits.items()}
            modes = ("board", "cut") if kind == "swarm" else ("streamed", "alone")
            for m in modes:
                rec[m] = [round(float(np.mean([s[m][w] for s in rec["sets"].values()])), 4) for w in range(a.windows)]
            later = {m: round(float(np.mean(rec[m][1:])), 4) for m in modes}
            rec["later_windows"] = later
            print(f"{rec['run']} ({kind}, {rec['params_M']}M): " + "  ".join(f"{m} {rec[m]}" for m in modes)
                  + f"  windows 1+: {later}", flush=True)
            results.append(rec)
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(args=vars(a), runs=results), open(a.out, "w"), indent=1)
        print("wrote", a.out)
    return results


if __name__ == "__main__":
    main()
