"""The chat template, and a demo: talk to one agent or to a swarm of agents in the terminal.

    python -m exealm.chat --ckpt 17m/exealm_together_gated_seed0            # downloads from Hugging Face
    python -m exealm.chat --ckpt runs/pretrain/model.pt --tokenizer tokenizer --swarm 4
    python -m exealm.chat --ckpt runs/pretrain/model.pt --tokenizer none --raw   # a `tiny` byte-level checkpoint

The template is the format exealm/train/sft.py trains on:

    <|im_start|><|system|>...<|im_end|><|im_start|><|user|>...<|im_end|><|im_start|><|assistant|>...<|im_end|>

With --swarm N, N agents get the same conversation and write their answers in lockstep in one batch, sharing a board
(exealm/generate.py); every agent's answer is shown. --raw skips the template and continues your text.

No released checkpoint has been through post-training yet, so today's checkpoints do not answer questions: with the
template they see tokens they never trained on. Use --raw to see what a base checkpoint writes. The demo is here so
the post-training work has something to plug into.
"""
import argparse
import sys

import torch

from .generate import generate
from .train.sft import ROLE, format_chat

# The same format as a Jinja template (what Hugging Face tokenizers call a chat template). tests/test_chat.py checks
# that rendering it and tokenizing the result gives exactly chat_prompt's tokens.
CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|><|{{ m['role'] }}|>{{ m['content'] }}<|im_end|>{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|><|assistant|>{% endif %}")


def chat_prompt(messages, tok, add_generation_prompt=True):
    """Token ids of a conversation, ending (by default) with the header that asks the assistant to answer."""
    for m in messages:
        if m["role"] not in ROLE:
            raise ValueError(f"unknown role {m['role']!r}: use one of {sorted(ROLE)}")
    ids, _ = format_chat(messages, tok)
    if add_generation_prompt:
        ids = ids + [tok.token_to_id("<|im_start|>"), tok.token_to_id(ROLE["assistant"])]
    return ids


class Bytes:
    """Stand-in tokenizer for byte-level checkpoints (the `tiny` preset on synthetic data): --raw only."""

    def encode(self, text):
        return argparse.Namespace(ids=list(text.encode("utf-8")))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def token_to_id(self, token):
        return None


def answer(model, tok, history, agents=1, raw=False, **gen):
    """One turn. history: one message list per agent (raw: one text per agent). Returns each agent's new text."""
    if raw:
        prompts = [tok.encode(h).ids or [0] for h in history]
        stop = [t for t in [tok.token_to_id("<|endoftext|>")] if t is not None]
    else:
        prompts = [chat_prompt(h, tok) for h in history]
        stop = [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")]
    new = generate(model, prompts, stop=stop, swarm=agents > 1, **gen)
    return [tok.decode(n) for n in new]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a local .pt file, or a name under checkpoints/ on Hugging Face")
    ap.add_argument("--tokenizer", default="hub", help="'hub' (the published 32k tokenizer), a path, or 'none' (bytes)")
    ap.add_argument("--swarm", type=int, default=1, help="number of agents answering together")
    ap.add_argument("--raw", action="store_true", help="no chat template: continue the text you type")
    ap.add_argument("--system", default=None, help="a system message for every agent")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_context", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=None, help="answer this once and exit (default: an interactive session)")
    a = ap.parse_args(argv)
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):    # Windows consoles default to a legacy code page
            stream.reconfigure(encoding="utf-8", errors="replace")
    if a.swarm < 1:
        ap.error("--swarm must be at least 1")
    from .hub import from_pretrained
    from .train.pretrain import pick_device
    model, tok = from_pretrained(a.ckpt, pick_device(a.device),
                                 tokenizer={"hub": True, "none": False}.get(a.tokenizer, a.tokenizer))
    if tok is None:
        if not a.raw:
            ap.error("a byte-level checkpoint has no chat tokens: add --raw")
        tok = Bytes()
    torch.manual_seed(a.seed)
    gen = dict(max_new_tokens=a.max_new_tokens, temperature=a.temperature, top_k=a.top_k, max_context=a.max_context)
    start = [dict(role="system", content=a.system)] if a.system else []
    history = ["" for _ in range(a.swarm)] if a.raw else [list(start) for _ in range(a.swarm)]
    replies = []
    while True:
        try:
            text = a.prompt if a.prompt is not None else input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            continue
        history = [h + text for h in history] if a.raw else [h + [dict(role="user", content=text)] for h in history]
        replies = answer(model, tok, history, a.swarm, a.raw, **gen)
        for i, r in enumerate(replies):
            print(f"agent {i}> {r}" if a.swarm > 1 else f"model> {r}", flush=True)
        history = [h + r for h, r in zip(history, replies)] if a.raw else \
            [h + [dict(role="assistant", content=r)] for h, r in zip(history, replies)]
        if a.prompt is not None:
            break
    return replies


if __name__ == "__main__":
    main()
