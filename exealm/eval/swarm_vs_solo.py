"""The first swarm experiment (docs/swarms.md): does a split swarm predict its text better than solo agents reading the
same text?

It scores the same windows of held-out text in several ways and prints bits per token for each:

    swarm       the checkpoint, its agents grouped into swarms of cfg.swarm_size that share a board
    solo        the same checkpoint and windows, every agent alone (it can only read its own earlier posts)
    baseline    optional (--baseline): another checkpoint, run solo on the same windows. Train it with the same
                data and steps but swarm_size 1 (--config solo.json with {"swarm_size": 1}) for the fair comparison,
                or pass the Llama baseline

"swarm" minus "solo" is what the board is worth to a model that was trained with it. "swarm" minus "baseline" asks
whether training in swarms was worth it at all. Both are reported for "split" swarms, where agents read consecutive
parts of one stretch of text, and for "random" swarms, the control: unrelated texts, where the board has nothing to
give and the difference should be close to zero. Results are also broken down by agent (agent 0 reads the first part of
a split window, agent 3 the last).

    python -m exealm.eval.swarm_vs_solo --ckpt runs/small_swarm/model.pt --data data/packed/val \\
        --baseline runs/small_solo/model.pt --datastore_data data/packed/train --datastore_chunks 20000

Fairness detail: in a split swarm, the last target of agent i is the first token agent i + 1 reads, and agent i can hear
agent i + 1's first post. Every row's last position is therefore left out of the score in all conditions (--keep_last
puts it back).
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from ..data.loader import SyntheticShards, TokenShards
from ..memory import Datastore, MemoryContext


def fixed_batches(data, n, B, T, swarm, policy, seed):
    """n batches of B windows of T tokens, rows grouped into swarms of `swarm` with the given policy."""
    rng = np.random.default_rng(seed)
    return [data.batch(B, T, rng, swarm, policy) for _ in range(n)]


@torch.no_grad()
def eval_bits(model, batches, swarm, agents, ds=None, keep_last=False, device="cpu"):
    """Mean bits per token over the batches with rows grouped into swarms of `swarm` (1: every row alone), and the
    mean for each agent position 0 .. agents-1 (row % agents)."""
    model.eval()
    tot, cnt = torch.zeros(agents, dtype=torch.float64), torch.zeros(agents, dtype=torch.float64)
    for x, y, d in batches:
        x, y, d = x.to(device), y.to(device), d.to(device)
        logits = model(x, MemoryContext(datastore=ds, doc_ids=d, swarm_size=swarm))[0].float()
        nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none", ignore_index=-100)   # (B, T)
        scored = y != -100                                                            # the loader may hide targets
        if not keep_last:
            nll, scored = nll[:, :-1], scored[:, :-1]
        rows = torch.arange(x.shape[0]) % agents
        tot.index_add_(0, rows, nll.sum(1).double().cpu())
        cnt.index_add_(0, rows, scored.sum(1).double().cpu())
    bits = tot / cnt / math.log(2)
    return round(float(tot.sum() / cnt.sum() / math.log(2)), 4), [round(float(b), 4) for b in bits]


def compare(model, data, baseline=None, ds=None, policies=("split", "random"), batches=16, batch=None, seq=256,
            seed=0, keep_last=False, device="cpu"):
    """Returns a list of result dicts, one per (policy, condition). ds: the datastore for `model`; a TONN baseline gets
    the same chunks re-encoded with its own encoder."""
    sw = model.cfg.swarm_size
    B = batch or 4 * sw
    if B % sw:
        raise ValueError(f"batch {B} must be a multiple of the swarm size {sw}")
    base_ds = None
    if baseline is not None and ds is not None and getattr(baseline, "encoder", None) is not None:
        base_ds = Datastore(ds.tokens, ds.doc_ids)
        base_ds.refresh(baseline.encoder, baseline.emb)
    out = []
    for policy in policies:
        bs = fixed_batches(data, batches, B, seq, sw, policy, seed)
        runs = [("swarm", model, sw, ds), ("solo", model, 1, ds)]
        if baseline is not None:
            runs.append(("baseline", baseline, 1, base_ds))
        res = {}
        for name, m, s, store in runs:
            bits, per_agent = eval_bits(m, bs, s, sw, store, keep_last, device)
            res[name] = bits
            out.append(dict(policy=policy, condition=name, bits_per_token=bits, bits_per_agent=per_agent,
                            swarm_size=s, windows=B * batches, seq=seq))
        out.append(dict(policy=policy, condition="swarm_gain_over_solo",
                        bits_per_token=round(res["solo"] - res["swarm"], 4)))
        if baseline is not None:
            out.append(dict(policy=policy, condition="swarm_gain_over_baseline",
                            bits_per_token=round(res["baseline"] - res["swarm"], 4)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a TONN model.pt trained with swarms (exealm.train.pretrain)")
    ap.add_argument("--baseline", default=None, help="another model.pt, scored solo on the same windows")
    ap.add_argument("--data", default="synthetic", help="held-out packed directory to score, or 'synthetic'")
    ap.add_argument("--datastore_data", default=None, help="packed directory to sample the datastore from")
    ap.add_argument("--datastore_chunks", type=int, default=0, help="0: no retrieval (RG slots output zeros)")
    ap.add_argument("--policies", default="split,random")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=None, help="windows per batch (default: 4 swarms)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="the training run's --seed (the datastore uses seed + 1)")
    ap.add_argument("--keep_last", action="store_true", help="also score each row's last position (see the docstring)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="also append the JSON lines to this file")
    a = ap.parse_args(argv)

    from ..train.pretrain import load_model
    model, ck = load_model(a.ckpt, a.device)
    if ck["kind"] != "tonn":
        raise SystemExit("--ckpt must be a TONN model; pass the Llama baseline as --baseline")
    baseline = load_model(a.baseline, a.device)[0] if a.baseline else None
    data = SyntheticShards() if a.data == "synthetic" else TokenShards(a.data)
    ds = None
    if a.datastore_chunks:
        src = data if a.datastore_data is None else TokenShards(a.datastore_data)
        ds = Datastore.from_shards(src, a.datastore_chunks, model.cfg.rag_chunk_len, seed=a.seed + 1).to(a.device)
        ds.refresh(model.encoder, model.emb)
    recs = compare(model, data, baseline, ds, tuple(a.policies.split(",")), a.batches, a.batch, a.seq, a.seed + 3,
                   a.keep_last, a.device)
    for r in recs:
        print(json.dumps(r))
        if a.out:
            with open(a.out, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(r) + "\n")
    return recs


if __name__ == "__main__":
    main()
