"""Average the weights of two runs that share the start of their training: one model that reads windows alone as well
as the run trained without streams, and long documents as well as the run trained with them, if the merge works.

The study's ExeaLM and the same ExeaLM trained without streams (exealm_nostream_*) have the same seed and data, so
their first 80% of steps are the same; they part only in the stream phase. Weight averaging of models fine-tuned from
one starting point often keeps what each learned (Wortsman et al. 2022, "Model soups"); this tests it on the stream
phase. Writes a run folder with model.pt (no optimizer state) that every scoring script reads.

    python experiments/expanse/merge.py runs/exealm_100m_4.1x_s0 runs/exealm_nostream_100m_4.1x_s0 \\
        --weight 0.5 --out runs/_merge/exealm_merge_100m_4.1x_s0
"""
import argparse
import json
import os

import torch


def checkpoint(run):
    return os.path.join(run, "model.pt") if os.path.isdir(run) else run


def merge(a, b, weight):
    """weight * a + (1 - weight) * b for every floating-point tensor; the rest (integer buffers) from a."""
    if a["kind"] != b["kind"] or a["cfg"] != b["cfg"] or a.get("specs") != b.get("specs"):
        raise SystemExit("the two runs are different models: same kind, config and circuits needed")
    out = {}
    for k, v in a["model"].items():
        w = b["model"][k]
        out[k] = (weight * v.float() + (1 - weight) * w.float()).to(v.dtype) if v.is_floating_point() else v.clone()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", help="a run folder or model.pt")
    ap.add_argument("b", help="the other run folder or model.pt")
    ap.add_argument("--weight", type=float, default=0.5, help="the share of the first run")
    ap.add_argument("--out", required=True, help="a run folder to write model.pt and merge.json into")
    a = ap.parse_args(argv)
    ca = torch.load(checkpoint(a.a), map_location="cpu", weights_only=False)
    cb = torch.load(checkpoint(a.b), map_location="cpu", weights_only=False)
    os.makedirs(a.out, exist_ok=True)
    ck = {k: v for k, v in ca.items() if k not in ("opt", "model")}
    ck["model"] = merge(ca, cb, a.weight)
    torch.save(ck, os.path.join(a.out, "model.pt"))
    json.dump(dict(a=a.a, b=a.b, weight=a.weight), open(os.path.join(a.out, "merge.json"), "w"), indent=1)
    print(f"{a.out}: {a.weight} x {a.a} + {1 - a.weight:g} x {a.b}")


if __name__ == "__main__":
    main()
