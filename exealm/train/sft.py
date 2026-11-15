"""Post-training step 1: supervised fine-tuning (SFT) on conversations, so the base model learns to answer.

The loss counts only the assistant's tokens. The chat format uses the special tokens from exealm/data/tokenizer.py:

    <|im_start|><|system|>...<|im_end|><|im_start|><|user|>...<|im_end|><|im_start|><|assistant|>...<|im_end|>

Data to start from (licenses on the cards): IFM/SFT-Reasoning (instruction-following, Apache 2.0); for Japanese,
LLM-jp's instruction datasets (https://llm-jp.nii.ac.jp/en/release) -- check each license.

Training (after pretraining, with the tokenizer the checkpoint was trained with):
    python -m exealm.train.sft --ckpt runs/pretrain/model.pt --tokenizer tokenizer --data sft.jsonl.gz --steps 2000

Data: one conversation per line of a .jsonl or .jsonl.gz file, either {"messages": [{"role": ..., "content": ...}]}
or the ShareGPT form {"conversations": [{"from": "human" | "gpt" | "system", "value": ...}]}.

Batches: every conversation gets its own row, padded on the right. The model reads left to right, so padding after a
conversation cannot change it, and rows never see each other (the run is solo: no swarm board, no datastore).
Conversations of similar length share a batch, so little of a batch is padding. That is how conversations are kept
apart here; packing several into one row would need every circuit to reset its state at the boundary.

  TODO(post-training): download and format the SFT sources (IFM/SFT-Reasoning, LLM-jp instruction data).
  TODO(post-training): swarm SFT, with the loss on the answering agent and the others' text on the board.
"""
import argparse
import gzip
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

ROLE = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>", "tool": "<|tool|>"}


def format_chat(messages, tok):
    """messages: [{"role": ..., "content": ...}]. Returns token ids and a loss mask (1 on assistant tokens)."""
    ids, mask = [], []
    start, end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    for m in messages:
        head = [start, tok.token_to_id(ROLE[m["role"]])]
        body = tok.encode(m["content"]).ids + [end]
        ids += head + body
        mask += [0] * len(head) + [int(m["role"] == "assistant")] * len(body)
    return ids, mask


def sft_loss(model, ids, mask, mem=None):
    """ids, mask (B, T+1). Mean next-token loss over positions whose TARGET is an assistant token."""
    logits = model(ids[:, :-1], mem)[0]
    tgt = ids[:, 1:].masked_fill(mask[:, 1:] == 0, -100)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)


FROM = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system",
        "tool": "tool"}


def load_conversations(path):
    """Message lists from a .jsonl or .jsonl.gz file (see the module docstring for the two accepted forms).
    Lines without an assistant turn, or with a role or content that cannot be read, are skipped and counted."""
    out, skipped = [], 0
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if "messages" in rec:
                    msgs = [dict(role=m["role"], content=m["content"]) for m in rec["messages"]]
                else:
                    msgs = [dict(role=FROM[m["from"]], content=m["value"]) for m in rec["conversations"]]
                ok = all(m["role"] in ROLE and isinstance(m["content"], str) for m in msgs) and \
                    any(m["role"] == "assistant" and m["content"].strip() for m in msgs)
            except (KeyError, TypeError, ValueError):
                ok = False
            if ok:
                out.append(msgs)
            else:
                skipped += 1
    return out, skipped


def encode(conversations, tok, max_len):
    """[(ids, mask)] as lists, cut to max_len tokens. A conversation whose kept part holds no assistant token is
    dropped (its answer started after the cut)."""
    out = []
    for msgs in conversations:
        ids, mask = format_chat(msgs, tok)
        ids, mask = ids[:max_len], mask[:max_len]
        if len(ids) >= 2 and any(mask[1:]):
            out.append((ids, mask))
    return out


def pad_batch(examples, pad):
    """ids, mask (B, L) for a list of (ids, mask): right-padded, mask 0 on the padding."""
    width = max(len(i) for i, _ in examples)
    ids = torch.full((len(examples), width), pad, dtype=torch.long)
    mask = torch.zeros(len(examples), width, dtype=torch.long)
    for r, (i, m) in enumerate(examples):
        ids[r, :len(i)] = torch.tensor(i)
        mask[r, :len(m)] = torch.tensor(m)
    return ids, mask


def batches(examples, batch, rng, pad, pool=50):
    """Endless (ids, mask) batches. Shuffles, then sorts by length inside pools of `pool` batches, so the rows of a
    batch have similar lengths; the batches of a pool come in random order."""
    while True:
        order = rng.permutation(len(examples))
        for p in range(0, len(order), batch * pool):
            chunk = sorted(order[p:p + batch * pool], key=lambda k: len(examples[k][0]))
            groups = [chunk[b:b + batch] for b in range(0, len(chunk), batch)]
            for g in rng.permutation(len(groups)):
                yield pad_batch([examples[k] for k in groups[g]], pad)


@torch.no_grad()
def evaluate(model, examples, batch, pad, device):
    """Mean loss per assistant token over all the examples (each token counted once)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    order = sorted(range(len(examples)), key=lambda k: len(examples[k][0]))
    for b in range(0, len(order), batch):
        ids, mask = pad_batch([examples[k] for k in order[b:b + batch]], pad)
        n = int(mask[:, 1:].sum())
        total += float(sft_loss(model, ids.to(device), mask.to(device))) * n
        count += n
    model.train(was_training)
    return total / max(1, count)


def main(argv=None):
    from ..hub import build_model
    from ..model import ExeaLM, count_params
    from .pretrain import lr_at, make_optimizer, pick_device, save
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the pretrained checkpoint to start from")
    ap.add_argument("--tokenizer", default="tokenizer", help="the tokenizer the checkpoint was trained with")
    ap.add_argument("--data", required=True, help="conversations, .jsonl or .jsonl.gz")
    ap.add_argument("--val", default=None, help="held-out conversations (default: the last --val_frac of --data)")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024, help="longer conversations are cut to this many tokens")
    ap.add_argument("--lr", type=float, default=3e-4, help="lower than pretraining's: the model already knows text")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--decay_frac", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sft")
    a = ap.parse_args(argv)

    from ..data.tokenizer import load
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = pick_device(a.device)
    tok = load(a.tokenizer)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck)
    if tok.get_vocab_size() != model.cfg.vocab_size:
        raise ValueError(f"the tokenizer has {tok.get_vocab_size()} tokens but the checkpoint was trained with "
                         f"{model.cfg.vocab_size}")
    for t in ("<|im_start|>", "<|im_end|>", "<|pad|>", *ROLE.values()):
        if tok.token_to_id(t) is None:
            raise ValueError(f"the tokenizer has no {t} token (train it with exealm.data.tokenizer)")
    pad = tok.token_to_id("<|pad|>")
    model = model.to(dev).train()
    kind = "tonn" if isinstance(model, ExeaLM) else "llama"

    convs, skipped = load_conversations(a.data)
    train = encode(convs, tok, a.max_len)
    dropped = len(convs) - len(train)
    if a.val:
        val = encode(load_conversations(a.val)[0], tok, a.max_len)
    else:
        n_val = max(1, int(len(train) * a.val_frac))
        train, val = train[:-n_val], train[-n_val:]
    if not train or not val:
        raise ValueError("no usable conversations: each needs an assistant turn inside the first --max_len tokens")

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    emit(dict(event="start", model=kind, params=count_params(model), device=str(dev), conversations=len(train),
              val_conversations=len(val), skipped_lines=skipped, dropped_after_cut=dropped,
              val_loss=round(evaluate(model, val, a.batch, pad, dev), 4)))
    opt = make_optimizer(model, a.lr, a.wd)
    warmup = a.warmup if a.warmup is not None else max(1, min(500, a.steps // 20))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else None
    stream = batches(train, a.batch, rng, pad)
    losses, t0 = [], time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, warmup, a.decay_frac)
        ids, mask = next(stream)
        with torch.autocast(**amp) if amp else torch.enable_grad():
            loss = sft_loss(model, ids.to(dev), mask.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            v = evaluate(model, val, a.batch, pad, dev)
            emit(dict(event="eval", step=step + 1, train_loss=round(sum(losses) / len(losses), 4),
                      val_loss=round(v, 4), val_bits_per_token=round(v / math.log(2), 4),
                      lr=round(opt.param_groups[0]["lr"], 6), secs=round(time.time() - t0, 1)))
            losses = []
    save(os.path.join(a.out, "model.pt"), model, opt, a.steps, model.cfg, kind)
    return model


if __name__ == "__main__":
    main()
