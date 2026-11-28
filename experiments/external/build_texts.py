"""The study's held-out evaluation as plain text, so a released model with its own tokenizer can be scored on exactly
what ExeaLM and the baselines were scored on (experiments/external/score_hf.py), in bits per UTF-8 byte.

Writes one .jsonl with two kinds of rows:
- "window": the 512 windows eval_text.py draws from each held-out set (the same generator and seed), as the text of
  their 513 tokens. "bytes" counts the bytes a model predicts: all of the text but its first token's. With --ckpts,
  each of our checkpoints' bits on the window, so every model is divided by the same exact bytes.
- "item": the library's held-out questions and tasks (experiments/library/score.py), as prefix and target text, with
  the entry placed before them in the "pasted" (the item's own entry) and "bm25" (the search's first pick) conditions.

    python experiments/external/build_texts.py --mix data/mix_pretrain_val.json --library data/library/v1 \\
        --ckpts runs/exealm_250m_4.1x_s0 runs/qwen_250m_4.1x_s0 --out data/external/texts.jsonl
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "library"))
from exealm import MemoryContext  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.train.pretrain import load_model, pick_device  # noqa: E402
from eval_text import checkpoint, mix_sets  # noqa: E402
from score import bm25_pick, entry_ids, load_items  # noqa: E402


def windows(val, batch=32, seq=512, batches=16):
    """eval_text.bits_per_byte's windows: (N, seq + 1) token ids."""
    rng = np.random.default_rng(10_000)
    out = []
    for _ in range(batches):
        x, y, _ = val.batch(batch, seq, rng)
        out.append(torch.cat([x, y[:, -1:]], 1))
    return torch.cat(out)


@torch.no_grad()
def bits_of(model, rows, dev, batch=16):
    """Summed bits on tokens 1 .. end of every row, every row read alone."""
    out = []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch].to(dev)
        logits = model(r[:, :-1], MemoryContext())[0].float()
        nll = F.cross_entropy(logits.transpose(1, 2), r[:, 1:], reduction="none").sum(1)
        out += (nll / math.log(2)).tolist()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mix", default="data/mix_pretrain_val.json")
    ap.add_argument("--library", default="data/library/v1")
    ap.add_argument("--tokenizer", default="data/exealm-pretrain/slice-30b-r2/tokenizer")
    ap.add_argument("--ckpts", nargs="*", default=[], help="our checkpoints to score on the same windows")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="library items of each set (score.py's --limit; 0: all)")
    ap.add_argument("--windows", type=int, default=1, help="0: library items only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    dec = lambda ids: tok.decode([int(t) for t in ids], skip_special_tokens=False)   # noqa: E731
    dev = pick_device(a.device)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    sets = {name: windows(TokenShards(path)) for name, path in mix_sets(a.mix).items()} if a.windows else {}
    for name, w in sets.items():
        for i, ids in enumerate(w.tolist()):
            text = dec(ids)
            rows.append(dict(kind="window", set=name, i=i, text=text,
                             bytes=len(text.encode("utf-8")) - len(dec(ids[:1]).encode("utf-8")), bits={}))
    print(f"windows: {len(rows)} in {len(sets)} sets", flush=True)
    for path in a.ckpts:
        model, _ = load_model(checkpoint(path), dev)
        model.eval()
        name = os.path.basename(path.rstrip("/"))
        k = 0
        for s, w in sets.items():
            for b in bits_of(model, w, dev):
                rows[k]["bits"][name] = round(b, 4)
                k += 1
        print(f"scored {name}", flush=True)
        del model
    lib = TokenShards(a.library)
    items, eot, sep = [], None, None
    for s_ in ("qa", "tasks"):                                # first --limit of each, as score.py --limit takes them
        it, eot, sep = load_items(a.library, [s_], "heldout", a.limit)
        items += it
    picked, top1, _ = bm25_pick(lib, items, eot, sep, dev=dev)
    for it, d, hit in zip(items, picked, top1):
        rows.append(dict(kind="item", set=it["set"], type=it["kind"], prefix=dec(it["prefix"]),
                         target=dec(it["target"]), target_bytes=it["target_bytes"], sep=dec(sep),
                         pasted=dec(entry_ids(lib, it["source_doc"])), bm25=dec(entry_ids(lib, int(d))),
                         bm25_hit=bool(hit)))
    print(f"library items: {len(items)}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
