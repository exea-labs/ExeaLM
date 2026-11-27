"""Training items with the library entry a keyword search picks placed before them: ordinary RAG, for post-training.

For every item of the library's training split (question/answer pairs and tasks), the entry that BM25 ranks first for
the item's prefix (score.bm25_pick) goes first, then a blank line, the prefix and the target: the layout score.py's
"bm25" condition gives a held-out item. The search is the one used at test time, so the model learns to use the entry
it will be shown, wrong ones included (the search picks another entry for some items). Writes a pack TokenShards reads
(tokens.bin, offsets.npy, meta.json) for a post-training mix (experiments/post/rag_post.sh).

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
    ap.add_argument("--human", default=None, help="a second tier (index.py index, e.g. the human pool's): its top "
                                                  "passage for the item goes after the entry, as open-book evaluation "
                                                  "gives it (exealm.eval.harness --library)")
    ap.add_argument("--human_tokens", type=int, default=384, help="tokens kept of the second tier's passage")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lib = TokenShards(a.library)
    items, eot, sep = load_items(a.library, [s for s in a.sets.split(",") if s], a.split)
    dev = pick_device(a.device)
    picked, top1, top5 = bm25_pick(lib, items, eot, sep, dev=dev)
    human = None
    if a.human:
        import index as ix                                     # experiments/library/index.py
        idx = ix.Index(a.human.split("+"))
        if dev.type == "cuda":
            idx.to(dev)
            found = []
            for b in range(0, len(items), 256):
                found += [d for d, _ in idx.search_gpu_batch([it["prefix"] for it in items[b:b + 256]], k=1,
                                                             max_df=0.1, skip=(eot, *sep))]
        else:
            found = [idx.search(it["prefix"], k=1, max_df=0.1, skip=(eot, *sep))[0] for it in items]
        human = [idx.passage(int(d[0]))[:a.human_tokens] if len(d) else [] for d in found]
    dtype = np.dtype(lib.meta["dtype"])
    os.makedirs(a.out, exist_ok=True)
    offsets, n_tok = [], 0
    with open(os.path.join(a.out, "tokens.bin"), "wb") as f:
        for i, (it, d) in enumerate(zip(items, picked)):
            second = (human[i] + list(sep)) if human is not None and human[i] else []
            ids = entry_ids(lib, int(d)) + list(sep) + second + it["prefix"] + it["target"] + [eot]
            f.write(np.array(ids, dtype=dtype).tobytes())
            offsets.append(n_tok)
            n_tok += len(ids)
    np.save(os.path.join(a.out, "offsets.npy"), np.array(offsets, dtype=np.int64))
    meta = dict(dtype=dtype.name, vocab_size=lib.meta["vocab_size"], n_tokens=n_tok, n_docs=len(offsets),
                bytes_per_token=lib.meta["bytes_per_token"], tokenizer=lib.meta.get("tokenizer"), eot_id=eot,
                sep_ids=list(sep), source=dict(kind="library items with a BM25-picked entry before them"
                                               + (" and the second tier's top passage" if a.human else ""),
                                               library=a.library, human=a.human, human_tokens=a.human_tokens,
                                               sets=a.sets, split=a.split,
                                               bm25_top1=round(float(top1.mean()), 4),
                                               bm25_top5=round(float(top5.mean()), 4)))
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{a.out}: {len(offsets)} items, {n_tok:,} tokens; the search put the item's own entry first for "
          f"{top1.mean():.3f} of them, in the first 5 for {top5.mean():.3f}")


if __name__ == "__main__":
    main()
