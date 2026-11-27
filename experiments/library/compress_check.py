"""What the library's text costs to ship, as token ids or as UTF-8 bytes, compressed with zstd (test T0 of
docs/library_codes_proposal.md).

Samples whole passages at random from every part of the human tier (the same number of tokens from each part), writes
them once as the token ids the library stores (uint16) and once as UTF-8 text decoded with the library's tokenizer, and
compresses both with zstd at each level. Prints and writes the sizes, the bits per UTF-8 byte of each form, and what
each would make of the full 25.0B-token human tier.

    python experiments/library/compress_check.py --pool /expanse/lustre/projects/fli100/abhatia7/library_v2/human_pool \\
        --tokenizer data/exealm-pretrain/slice-30b-r2/tokenizer --out results/expanse/library_compress.json
"""
import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np
from tokenizers import Tokenizer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", required=True, help="the human tier: part_*/ with tokens.bin, offsets.npy, meta.json")
    ap.add_argument("--tokenizer", required=True, help="the folder holding tokenizer.json")
    ap.add_argument("--tokens", type=float, default=2e8, help="tokens sampled in all")
    ap.add_argument("--levels", default="3,19")
    ap.add_argument("--full_tokens", type=float, default=25.0e9, help="the human tier's size, to scale to")
    ap.add_argument("--work", default=None, help="scratch folder (default: next to --out)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), "compress_work")
    os.makedirs(work, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    parts = sorted(glob.glob(os.path.join(a.pool, "part_*")))
    per_part = int(a.tokens / len(parts))
    f_ids, f_txt = os.path.join(work, "sample.u16"), os.path.join(work, "sample.txt")
    n_tok = n_pass = n_bytes = 0
    with open(f_ids, "wb") as fi, open(f_txt, "wb") as ft:
        for p in parts:
            meta = json.load(open(os.path.join(p, "meta.json")))
            toks = np.memmap(os.path.join(p, "tokens.bin"), dtype=meta["dtype"], mode="r")
            offs = np.load(os.path.join(p, "offsets.npy"))
            ends = np.append(offs[1:], len(toks))
            got, chosen = 0, []
            for d in rng.permutation(len(offs)):
                chosen.append(int(d))
                got += int(ends[d] - offs[d])
                if got >= per_part:
                    break
            chosen.sort()                                      # read in file order
            for b0 in range(0, len(chosen), 4000):
                rows = [np.asarray(toks[offs[d]:ends[d]]) for d in chosen[b0:b0 + 4000]]
                for r in rows:
                    fi.write(r.astype(np.uint16).tobytes())
                texts = tok.decode_batch([r.tolist() for r in rows], skip_special_tokens=True)
                blob = "".join(t + "\n" for t in texts).encode("utf-8")
                ft.write(blob)
                n_tok += sum(len(r) for r in rows)
                n_bytes += len(blob)
                n_pass += len(rows)
            print(f"{os.path.basename(p)}: {len(chosen):,} passages, {got:,} tokens ({time.time() - t0:.0f}s)", flush=True)
    res = dict(passages=n_pass, tokens=n_tok, utf8_bytes=n_bytes, bytes_per_token=round(n_bytes / n_tok, 4),
               token_id_bytes=n_tok * 2, forms={})
    for level in [int(x) for x in a.levels.split(",")]:
        for name, f in (("token ids", f_ids), ("UTF-8 text", f_txt)):
            out = f + f".zst{level}"
            t1 = time.time()
            subprocess.run(["zstd", f"-{level}", "-T16", "--long=27", "-q", "-f", f, "-o", out], check=True)
            size = os.path.getsize(out)
            raw = os.path.getsize(f)
            res["forms"][f"{name}, zstd -{level}"] = dict(
                raw_bytes=raw, compressed_bytes=size, ratio=round(raw / size, 3),
                bits_per_utf8_byte=round(8 * size / n_bytes, 4), bytes_per_token=round(size / n_tok, 4),
                full_tier_gb=round(size / n_tok * a.full_tokens / 1e9, 1), seconds=round(time.time() - t1, 1))
            print(f"{name:10s} zstd -{level:<2d}: {raw / 1e6:8.1f} MB -> {size / 1e6:8.1f} MB "
                  f"({8 * size / n_bytes:.3f} bits a UTF-8 byte; full tier {size / n_tok * a.full_tokens / 1e9:.1f} GB)",
                  flush=True)
            os.remove(out)
    res["full_tier"] = dict(tokens=a.full_tokens, token_ids_raw_gb=round(a.full_tokens * 2 / 1e9, 1),
                            utf8_raw_gb=round(a.full_tokens * n_bytes / n_tok / 1e9, 1))
    res["seconds"] = round(time.time() - t0, 1)
    os.remove(f_ids)
    os.remove(f_txt)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
