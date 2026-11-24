"""Tokenizer size: train byte-level BPE tokenizers of several vocabulary sizes on the same sample of English, Japanese
and code, and measure bytes per token on held-out text of each kind (more bytes per token = shorter sequences =
cheaper training and longer effective context). Also reports what each size costs in embedding parameters for the
0.1B model (width 640, tied input and output embeddings).

python experiments/tokenizer_compare.py --train en.parquet ja_wiki.jsonl.gz ... --max_mb 100 \
    --heldout en=val.parquet ja=ja_held.jsonl.gz code=code_held.jsonl.gz --vocabs 32768,49152,65536 --out results/tokenizer
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exealm.data.tokenizer import texts, train  # noqa: E402


def bytes_per_token(tok, path, max_mb):
    nb = nt = 0
    for t in texts([path], 10 ** 9, max_mb):
        nb += len(t.encode("utf-8"))
        nt += len(tok.encode(t).ids)
    return round(nb / max(1, nt), 3), nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--max_mb", type=float, default=100, help="MB of text used from each training file")
    ap.add_argument("--heldout", nargs="+", required=True, help="name=path")
    ap.add_argument("--heldout_mb", type=float, default=5)
    ap.add_argument("--vocabs", default="32768,49152,65536")
    ap.add_argument("--d", type=int, default=640)
    ap.add_argument("--out", default="results/tokenizer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for V in [int(v) for v in a.vocabs.split(",")]:
        t0 = time.time()
        tok = train(a.train, V, os.path.join(a.out, f"bpe{V // 1024}k"), max_mb_per_file=a.max_mb)
        row = dict(vocab=V, train_secs=round(time.time() - t0, 1), embedding_params_M=round(V * a.d / 1e6, 1))
        for spec in a.heldout:
            name, path = spec.split("=", 1)
            row[f"bytes_per_token_{name}"], row[f"heldout_bytes_{name}"] = bytes_per_token(tok, path, a.heldout_mb)
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump(dict(args=vars(a), rows=rows), open(os.path.join(a.out, "compare.json"), "w", encoding="utf-8", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
