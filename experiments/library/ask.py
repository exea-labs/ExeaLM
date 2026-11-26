"""Ask the 0.1B ExeaLM research questions with its library, the setup of results/expanse/README.md, section 11.

A keyword search (BM25, over token ids) picks the library entry for the question; the entry, a blank line and
"Question: ...\\nAnswer:" go in; the model writes the rest of the line. The library is 18,545 entries on research,
math and code topics, written by a larger model (Qwen3-30B-A3B). On the 6,253 held-out questions (topics the model
never trained on), the released model, ragpost_exealm_100m_s0, scores 0.588 bits per byte on the answers with the
library and 1.635 without it.

    pip install torch numpy safetensors tokenizers huggingface_hub
    python experiments/library/ask.py                              # 5 held-out questions, with and without the library
    python experiments/library/ask.py --question "What does gradient descent move the parameters along?"
    python experiments/library/ask.py --interactive
    python experiments/library/ask.py --score 500                  # bits per byte on 500 held-out questions, both ways

The model comes from huggingface.co/exea-labs/exealm-nairr (about 400 MB) and the library from
huggingface.co/datasets/exea-labs/exealm-library (about 100 MB), each downloaded once. --model and --library take local
folders instead: a released model folder (model.safetensors, config.json) or a run folder (model.pt), and a packed
library (data/library/v1).

It is a 95M-parameter base model, not an assistant: it answers best when the library has an entry on the topic, and
when the search picks the wrong entry it often answers from that entry anyway. Every answer shows which entry it read.
"""
import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message="MPS:")      # Apple GPUs pad one circuit's FFT in a slower way; harmless

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.generate import generate  # noqa: E402
from exealm.hub import build_model  # noqa: E402
from score import (bm25_index, bm25_pick, entry_ids, layout, load_items, pick_device, score,  # noqa: E402
                   summarize)

MODEL = "exea-labs/exealm-nairr/ragpost_exealm_100m_s0"
LIBRARY = "exea-labs/exealm-library/v1/packed"
SOURCE_REF = re.compile(r"\b(the|this) (text|article|paper|passage|document|author|authors|study|excerpt|snippet|code|"
                        r"post|page|section|example|problem)\b", re.I)


def split_hub(spec):
    """'owner/repo/sub/folder' -> ('owner/repo', 'sub/folder')."""
    parts = spec.strip("/").split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])


def load_model(spec, dev):
    """(model in eval mode, path of a tokenizer.json or None). spec: a released folder, a run folder or .pt file, or
    'owner/repo/folder' on Hugging Face."""
    tok, folder = None, None
    if os.path.isfile(spec) or (os.path.isdir(spec) and not os.path.exists(os.path.join(spec, "model.safetensors"))):
        path = spec
        if os.path.isdir(spec):
            path = next((os.path.join(spec, n) for n in ("model.pt", "ckpt.pt") if os.path.exists(os.path.join(spec, n))),
                        None)
            if path is None:
                raise SystemExit(f"{spec} holds no model.safetensors, model.pt or ckpt.pt")
        ck, folder = torch.load(path, map_location="cpu", weights_only=False), os.path.dirname(os.path.abspath(path))
    else:
        from safetensors.torch import load_file
        if os.path.isdir(spec):
            folder = os.path.abspath(spec)
            get = lambda f: os.path.join(folder, f)  # noqa: E731
        else:
            from huggingface_hub import hf_hub_download
            repo, sub = split_hub(spec)
            get = lambda f: hf_hub_download(repo, f"{sub}/{f}")  # noqa: E731
            tok = hf_hub_download(repo, "tokenizer/tokenizer.json")
        c = json.load(open(get("config.json")))
        ck = dict(cfg=c["cfg"], specs=c.get("specs"), kind=c["kind"], model=load_file(get("model.safetensors")))
    if tok is None and folder is not None:
        near = os.path.join(os.path.dirname(folder), "tokenizer", "tokenizer.json")    # a release keeps it beside the models
        tok = near if os.path.exists(near) else None
    return build_model(ck).to(dev).eval(), tok


def load_library(spec):
    if os.path.isdir(spec):
        return spec
    from huggingface_hub import snapshot_download
    repo, sub = split_hub(spec)
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{sub}/*", f"{sub}/**"])
    return os.path.join(root, sub)


def newline_ids(tok, vocab_size):
    """Every token whose text holds a line break: an answer is one line."""
    return [i for i in range(vocab_size) if "\n" in tok.decode([i])]


class Asker:
    def __init__(self, a):
        self.dev = pick_device(a.device)
        self.model, tok_path = load_model(a.model, self.dev)
        self.lib_dir = load_library(a.library)
        self.lib = TokenShards(self.lib_dir)
        from tokenizers import Tokenizer
        tok_path = a.tokenizer or tok_path or os.path.join(self.lib_dir, "tokenizer.json")   # the library keeps a copy
        if not os.path.exists(tok_path):
            raise SystemExit(f"no tokenizer at {tok_path}: pass --tokenizer path/to/tokenizer.json")
        self.tok = Tokenizer.from_file(tok_path)
        meta = json.load(open(os.path.join(self.lib_dir, "qa", "heldout", "meta.json")))
        self.eot, self.sep = meta["eot_id"], meta["sep_ids"]
        index = os.path.join(self.lib_dir, "index.jsonl")
        self.topics = [json.loads(line)["topic"] for line in open(index, encoding="utf-8")] \
            if os.path.exists(index) else None
        print(f"building the keyword index over {len(self.lib.offsets):,} library entries...", flush=True)
        self.index = bm25_index(self.lib)
        self.stop = [self.eot] + newline_ids(self.tok, self.model.cfg.vocab_size)
        self.gen = dict(max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_k=a.top_k or None)
        print(f"{type(self.model).__name__}, {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M parameters, "
              f"on {self.dev}", flush=True)

    def item(self, question):
        return dict(prefix=self.tok.encode(f"Question: {question.strip()}\nAnswer:").ids, target=[], source_doc=-1)

    def pick(self, items):
        return bm25_pick(self.lib, items, self.eot, self.sep, index=self.index)

    def topic(self, d):
        if self.topics is not None:
            return self.topics[d]
        return self.tok.decode(entry_ids(self.lib, d)).split("\n", 1)[0]

    @torch.no_grad()
    def answer(self, it, entry=None):
        """The model's answer to one item: with the library entry `entry` before it, or alone (entry None)."""
        cond = "bm25" if entry is not None else "none"
        ids, _ = layout(dict(it, target=[]), cond, self.model.cfg.chunk, self.eot, self.sep,
                        entry_ids(self.lib, int(entry)) if entry is not None else None)
        new = generate(self.model, [ids], stop=self.stop, **self.gen)[0]
        return self.tok.decode(new).strip()


def show(asker, it, picked, right=None, reference=None, alone=True):
    q = it.get("q") or asker.tok.decode(it["prefix"]).removeprefix("Question: ").removesuffix("\nAnswer:")
    print(f"\nQ: {q}")
    mark = "" if right is None else ("  (the right entry)" if right else "  (not this question's entry)")
    print(f"   entry the search picked: {asker.topic(int(picked))}{mark}")
    print(f"   with the library:    {asker.answer(it, picked)}")
    if alone:
        print(f"   without it:          {asker.answer(it)}")
    if reference is not None:
        print(f"   reference answer:    {reference.strip()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=MODEL, help="a released folder, a run folder or .pt, or owner/repo/folder")
    ap.add_argument("--library", default=LIBRARY, help="a packed library folder, or owner/repo/folder (a dataset)")
    ap.add_argument("--tokenizer", default=None, help="a tokenizer.json (default: the one released with the model)")
    ap.add_argument("--question", default=None, help="ask this and exit")
    ap.add_argument("--interactive", action="store_true", help="type questions")
    ap.add_argument("--heldout", type=int, default=5, help="without --question: this many held-out questions")
    ap.add_argument("--score", type=int, default=None, help="score this many held-out questions (0: all 6,253) with "
                                                            "and without the library, in bits per byte; no generation")
    ap.add_argument("--batch", type=int, default=0, help="questions per forward pass when scoring (default: 8 on CUDA, "
                                                         "1 elsewhere)")
    ap.add_argument("--no_alone", action="store_true", help="skip the answer without the library")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0, help="0: the most likely token every time")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0, help="which held-out questions are shown")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    asker = Asker(a)
    alone = not a.no_alone

    if a.score is not None:
        items, eot, sep = load_items(asker.lib_dir, ["qa"], "heldout", a.score)
        picked, top1, _ = asker.pick(items)
        print(f"{len(items):,} held-out questions; the search picks the question's own entry for {top1.mean():.3f}")
        amp = dict(device_type="cuda", dtype=torch.bfloat16) if asker.dev.type == "cuda" else None
        out = {}
        batch = a.batch or (8 if asker.dev.type == "cuda" else 1)
        for cond in ("none", "bm25"):
            bits, _, _ = score(asker.model, items, cond, eot, sep, asker.lib, batch=batch, dev=asker.dev, amp=amp,
                               picked=picked)
            out[cond] = summarize(items, bits, range(len(items)))["bits_per_byte"]
        print(f"bits per byte on the answers (lower is better): without the library {out['none']:.4f}, "
              f"with it {out['bm25']:.4f}")
        return out

    if a.question or a.interactive:
        while True:
            try:
                q = a.question if a.question else input("\nquestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                if a.question:
                    break
                continue
            it = asker.item(q)
            show(asker, it, asker.pick([it])[0][0], alone=alone)
            if a.question:
                break
        return

    items, _, _ = load_items(asker.lib_dir, ["qa"], "heldout")
    # 6% of the questions point at their source ("What is the DOI of the article?"): no reader could answer them
    # without it, so the examples leave them out. --score keeps every question.
    standalone = [it for it in items if not SOURCE_REF.search(it.get("q") or "")]
    rng = np.random.default_rng(a.seed)
    chosen = [standalone[i] for i in rng.choice(len(standalone), size=min(a.heldout, len(standalone)), replace=False)]
    picked, top1, _ = asker.pick(chosen)
    print(f"{len(chosen)} random held-out questions (their topics were never trained on; questions that point at "
          f"their source, like 'the article', are left out):")
    for it, d, ok in zip(chosen, picked, top1):
        show(asker, it, d, right=bool(ok), reference=it.get("a"), alone=alone)


if __name__ == "__main__":
    main()
