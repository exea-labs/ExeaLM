"""Train the tokenizer: byte-level BPE on a sample of the mix (Japanese, English, math, code).

At 0.1B parameters the embedding table is about a third of the model (49,152 tokens x 640 = 31M parameters), so the
vocabulary size is a real trade-off: more tokens make Japanese text shorter (cheaper to train on, longer effective
context) but leave fewer parameters for everything else. 49,152 is a placeholder (SmolLM2-135M uses the same size).

  TODO(data): compare 32k / 49k / 64k on bytes per token for Japanese and English, and on a small model's loss.
  TODO(data): check how Japanese is split (no spaces between words: pre-tokenization matters).

python -m exealm.data.tokenizer data/filtered/*.jsonl.gz --vocab 49152 --out tokenizer/
"""
import argparse
import os

from .files import expand

SPECIAL = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>", "<|system|>", "<|user|>", "<|assistant|>",
           "<|tool|>", "<|retrieved|>", "<|board|>"]


def texts(files, max_docs, max_mb_per_file=None):
    """Documents from jsonl.gz or Parquet files, at most max_mb_per_file MB of text from each file."""
    from .pack import iter_texts
    n = 0
    for path in files:
        got = 0
        for t in iter_texts([path]):
            yield t
            n += 1
            got += len(t.encode("utf-8"))
            if n >= max_docs:
                return
            if max_mb_per_file and got >= max_mb_per_file * 1e6:
                break


def train(files, vocab, out, max_docs=2_000_000, max_mb_per_file=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL, min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts(files, max_docs, max_mb_per_file), trainer=tr)
    os.makedirs(out, exist_ok=True)
    tok.save(os.path.join(out, "tokenizer.json"))
    return tok


def load(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--vocab", type=int, default=49152)
    ap.add_argument("--max_docs", type=int, default=2_000_000)
    ap.add_argument("--out", default="tokenizer")
    a = ap.parse_args()
    a.files = expand(a.files)
    tok = train(a.files, a.vocab, a.out, a.max_docs)
    for s in ["こんにちは、世界。今日は良い天気ですね。", "The quick brown fox jumps over the lazy dog."]:
        ids = tok.encode(s).ids
        print(f"{len(s.encode('utf-8')) / len(ids):.2f} bytes/token  {s}")


if __name__ == "__main__":
    main()
