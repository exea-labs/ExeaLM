"""Give a trained ExeaLM memory slots it was pretrained without: library (RG), board (AG) or weight-alter (WA).

The scaling study pretrains the memory-free hybrid (docs/budget.md). The board, the library and weight edits are
post-training and inference features, so they are added to a finished checkpoint here and taught afterwards
(`python -m exealm.train.pretrain --init <graft> ...`). In every TONN layer the circuits that contribute least (their
measured output size times the size of their output weights) are replaced. A new slot starts with output weights near
zero, so the grafted model predicts almost exactly as before; only the replaced circuits are lost.

    python -m exealm.train.graft --ckpt runs/core/model.pt --add RG:1,AG:1 --out runs/core_rg/model.pt \\
        --set '{"rag_key_dim": 128, "rag_lexical": 128}'

--set changes config fields at the same time (retrieval keys, weight-alter settings, swarm size). The new checkpoint
records what was replaced under "grafted".
"""
import argparse
import json
import os
from dataclasses import asdict, replace

import torch

from ..config import ModelConfig
from ..model import ExeaLM
from ..model.circuits import MEMORY_CIRCUITS


def contribution(layer, k):
    """How much slot k adds to the layer's output: its running output size times the size of its output weights."""
    h = layer.h
    scale = float(layer.scale[k]) or 1.0
    return scale * float(layer.out.weight[:, k * h:(k + 1) * h].norm())


def graft(ck, add, settings=None, out_scale=0.01):
    """ck: a TONN checkpoint dict. add: {"RG": 1, ...} slots per TONN layer. Returns the new checkpoint dict."""
    if ck["kind"] != "tonn":
        raise ValueError("only an ExeaLM (TONN) checkpoint can take memory slots")
    bad = set(add) - set(MEMORY_CIRCUITS)
    if bad:
        raise ValueError(f"graft adds memory circuits ({', '.join(MEMORY_CIRCUITS)}), not {', '.join(sorted(bad))}")
    cfg = ModelConfig.from_dict(ck["cfg"])
    old = ExeaLM(cfg, ck["specs"])
    old.load_state_dict(ck["model"])
    specs = ck["specs"]
    replaced = []
    for i, layer in enumerate(old.layers):
        free = [k for k, s in enumerate(specs[i]) if s["tmpl"] not in MEMORY_CIRCUITS]
        need = sum(add.values())
        if need > len(free):
            raise ValueError(f"TONN layer {i} has {len(free)} circuits to replace, {need} asked for")
        weakest = sorted(free, key=lambda k: contribution(layer, k))[:need]
        kinds = [t for t, n in add.items() for _ in range(n)]
        for k, t in zip(weakest, kinds):
            replaced.append(dict(layer=i, slot=k, was=specs[i][k]["tmpl"], now=t))
            specs[i][k] = dict(tmpl=t, lag=0, lam0=0.5)
    pinned = dict(cfg.pinned)
    for t, n in add.items():
        pinned[t] = pinned.get(t, 0) + n
    menu = tuple(cfg.menu) + tuple(t for t in add if t not in cfg.menu)
    new_cfg = replace(cfg, pinned=pinned, menu=menu, **(settings or {}))
    new = ExeaLM(new_cfg, specs)
    mine = new.state_dict()
    new.load_state_dict({k: v for k, v in ck["model"].items() if k in mine and mine[k].shape == v.shape}, strict=False)
    for r in replaced:                                  # the new slots start fresh, with output weights near zero
        new.layers[r["layer"]].set_spec(r["slot"], specs[r["layer"]][r["slot"]], reinit=True, out_scale=out_scale)
    return dict(model=new.state_dict(), cfg=asdict(new_cfg), kind="tonn", specs=new.specs, step=ck.get("step", 0),
                grafted=dict(added=add, settings=settings or {}, replaced=replaced, source_step=ck.get("step")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a trained ExeaLM model.pt (or ckpt.pt)")
    ap.add_argument("--add", required=True, help="memory slots per TONN layer, e.g. RG:1,AG:1")
    ap.add_argument("--set", default="{}", help="JSON of config fields to change at the same time")
    ap.add_argument("--out", required=True, help="where to write the new model.pt")
    a = ap.parse_args(argv)
    add = {t: int(n) for t, n in (part.split(":") for part in a.add.split(","))}
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    new = graft(ck, add, json.loads(a.set))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(new, a.out)
    print(json.dumps(dict(out=a.out, added=add, replaced=[f"layer {r['layer']} slot {r['slot']}: {r['was']} -> {r['now']}"
                                                          for r in new["grafted"]["replaced"]])))
    return new


if __name__ == "__main__":
    main()
