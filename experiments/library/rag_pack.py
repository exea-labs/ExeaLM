"""Training items with the library entry a keyword search picks placed before them: ordinary RAG, for post-training.

For every item of the library's training split (question/answer pairs and tasks), the entry that BM25 ranks first for
the item's prefix (score.bm25_pick) goes first, then a blank line, the prefix and the target: the layout score.py's
"bm25" condition gives a held-out item. The search is the one used at test time, so the model learns to use the entry
it will be shown, wrong ones included (the search picks another entry for some items). Writes a pack TokenShards reads
(tokens.bin, offsets.npy, meta.json) for a post-training mix (experiments/expanse/library_post.sh with RAG=1).

    python experiments/library/rag_pack.py --library data/library/v1 --out data/library/v1/rag/train
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.loader import TokenShards  # noqa: E402
from score import bm25_pick, entry_ids, load_items, pick_device  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True, help="the packed library from experiments/library/build.py pack")
    ap.add_argument("--sets", default="qa,tasks")
    ap.add_argument("--split", default="train", choices=["train", "heldout"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lib = TokenShards(a.library)
    items, eot, sep = load_items(a.library, [s for s in a.sets.split(",") if s], a.split)
    picked, top1, top5 = bm25_pick(lib, items, eot, sep, dev=pick_device(a.device))
    dtype = np.dtype(lib.meta["dtype"])
    os.makedirs(a.out, exist_ok=True)
    offsets, n_tok = [], 0
    with open(os.path.join(a.out, "tokens.bin"), "wb") as f:
        for it, d in zip(items, picked):
            ids = entry_ids(lib, int(d)) + list(sep) + it["prefix"] + it["target"] + [eot]
            f.write(np.array(ids, dtype=dtype).tobytes())
            offsets.append(n_tok)
            n_tok += len(ids)
    np.save(os.path.join(a.out, "offsets.npy"), np.array(offsets, dtype=np.int64))
    meta = dict(dtype=dtype.name, vocab_size=lib.meta["vocab_size"], n_tokens=n_tok, n_docs=len(offsets),
                bytes_per_token=lib.meta["bytes_per_token"], tokenizer=lib.meta.get("tokenizer"), eot_id=eot,
                sep_ids=list(sep), source=dict(kind="library items with a BM25-picked entry before them",
                                               library=a.library, sets=a.sets, split=a.split,
                                               bm25_top1=round(float(top1.mean()), 4),
                                               bm25_top5=round(float(top5.mean()), 4)))
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{a.out}: {len(offsets)} items, {n_tok:,} tokens; the search put the item's own entry first for "
          f"{top1.mean():.3f} of them, in the first 5 for {top5.mean():.3f}")


if __name__ == "__main__":
    main()
