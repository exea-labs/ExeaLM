"""Tokenize documents and pack them into one flat token file per split, with document boundaries kept.

Output directory:
    tokens.bin    every token of every document, end to end, <|endoftext|> after each (uint16 if the vocabulary fits)
    offsets.npy   where each document starts in tokens.bin (so retrieval can skip the document being read)
    meta.json     dtype, vocabulary size, token and document counts, bytes per token

Input: jsonl.gz files with a "text" field (what download.py and quality.py write) or Parquet files with a "text"
column (Hugging Face datasets). `--tokenizer bytes` packs raw UTF-8 bytes (vocabulary 257: the 256 byte values plus
end-of-text), which needs no trained tokenizer, so experiments can start before the tokenizer exists.

python -m exealm.data.pack data/filtered/*.jsonl.gz --tokenizer tokenizer --out data/packed/train
python -m exealm.data.pack fineweb-edu/000_00000.parquet --tokenizer bytes --max_bytes 200e6 --out data/bytes/train
"""
import argparse
import gzip
import json
import os

import numpy as np

from .files import expand


def iter_texts(files):
    for path in files:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                yield from pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        else:
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)["text"]


class ByteTokenizer:
    """UTF-8 bytes as tokens 0..255, end-of-text 256."""
    EOT = 256

    def get_vocab_size(self):
        return 257

    def token_to_id(self, name):
        return self.EOT if name == "<|endoftext|>" else None

    def encode_batch(self, texts):
        return [type("Enc", (), {"ids": list(t.encode("utf-8"))})() for t in texts]


def pack(files, tokenizer, out, max_docs=None, max_bytes=None, batch=1000):
    if tokenizer == "bytes":
        tok = ByteTokenizer()
    else:
        from .tokenizer import load
        tok = load(tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    V = tok.get_vocab_size()
    dtype = np.uint16 if V <= 65535 else np.uint32
    os.makedirs(out, exist_ok=True)
    offsets, n_tok, n_bytes, buf = [], 0, 0, []
    with open(os.path.join(out, "tokens.bin"), "wb") as fo:
        def flush():
            nonlocal n_tok
            for enc in tok.encode_batch(buf):
                offsets.append(n_tok)
                ids = np.array(enc.ids + [eot], dtype=dtype)
                fo.write(ids.tobytes())
                n_tok += len(ids)
            buf.clear()
        for text in iter_texts(files):
            n_bytes += len(text.encode("utf-8"))
            buf.append(text)
            if len(buf) >= batch:
                flush()
            if (max_docs and len(offsets) + len(buf) >= max_docs) or (max_bytes and n_bytes >= max_bytes):
                break
        if buf:
            flush()
    np.save(os.path.join(out, "offsets.npy"), np.array(offsets, dtype=np.int64))
    meta = dict(dtype=np.dtype(dtype).name, vocab_size=V, n_tokens=n_tok, n_docs=len(offsets),
                bytes_per_token=n_bytes / max(1, n_tok), tokenizer=tokenizer, files=[os.path.basename(f) for f in files])
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8", newline="\n"), indent=2)
    print(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", default="tokenizer", help="a trained tokenizer directory, or 'bytes'")
    ap.add_argument("--out", default="data/packed/train")
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--max_bytes", type=float, default=None)
    a = ap.parse_args()
    a.files = expand(a.files)
    pack(a.files, a.tokenizer, a.out, a.max_docs, a.max_bytes)


if __name__ == "__main__":
    main()
