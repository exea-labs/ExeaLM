"""Post-training step 2: preference tuning with DPO (Direct Preference Optimization, Rafailov et al. 2023).

Given a prompt, a preferred answer and a rejected one, raise the model's likelihood of the preferred answer relative to
the rejected one, measured against a frozen copy of the model it started from (the SFT model), with no reward model.

    loss = -log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))])

Each log-probability is summed over the answer's tokens, the closing <|im_end|> included, given the prompt in the
chat format of exealm/train/sft.py.

Training (after SFT, with the tokenizer the checkpoint was trained with):
    python -m exealm.train.dpo --ckpt runs/sft/model.pt --tokenizer tokenizer --data prefs.jsonl.gz --steps 1000

Data: one preference per line of a .jsonl or .jsonl.gz file:
    {"prompt": "...", "chosen": "...", "rejected": "..."}                       a single user turn, or
    {"messages": [{"role": ..., "content": ...}], "chosen": "...", "rejected": "..."}    the conversation so far

Batches: every answer gets its own row, padded on the right, as in SFT, so rows never see each other. A batch of B
preferences is 2B rows (chosen and rejected), read by the model and, without gradients, by the frozen reference.

  TODO(post-training): preference data (English and Japanese; check licenses).
"""
import argparse
import copy
import gzip
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .sft import ROLE, format_chat, pad_batch


def dpo_loss(logp_chosen, logp_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Each argument (N,): summed log-prob of the answer tokens under the policy or the reference model."""
    return -F.logsigmoid(beta * ((logp_chosen - ref_chosen) - (logp_rejected - ref_rejected))).mean()


def answer_logp(model, ids, mask, mem=None):
    """ids, mask (B, T+1). (B,): summed log-prob of the tokens whose mask is 1, each given everything before it."""
    logits = model(ids[:, :-1], mem)[0].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (lp * mask[:, 1:]).sum(1)


def load_preferences(path):
    """[(messages, chosen, rejected)] from a .jsonl or .jsonl.gz file. Lines that cannot be read, or whose two
    answers are the same or empty, are skipped and counted."""
    out, skipped = [], 0
    with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                msgs = ([dict(role=m["role"], content=m["content"]) for m in rec["messages"]] if "messages" in rec
                        else [dict(role="user", content=rec["prompt"])])
                c, r = rec["chosen"], rec["rejected"]
                ok = (bool(msgs) and all(m["role"] in ROLE and isinstance(m["content"], str) for m in msgs)
                      and isinstance(c, str) and isinstance(r, str) and c.strip() and r.strip() and c != r)
            except (KeyError, TypeError, ValueError):
                ok = False
            if ok:
                out.append((msgs, c, r))
            else:
                skipped += 1
    return out, skipped


def encode(prefs, tok, max_len):
    """[((ids, mask) of the chosen answer, (ids, mask) of the rejected one)]: the prompt, then the answer with mask 1.
    The prompt is cut from the LEFT so that prompt and answer fit max_len tokens (the answer must be scored whole);
    a preference whose answer alone does not fit is dropped."""
    out = []
    for msgs, chosen, rejected in prefs:
        prompt = format_chat(msgs, tok)[0]
        pair = []
        for text in (chosen, rejected):
            ids, mask = format_chat([dict(role="assistant", content=text)], tok)
            room = max_len - len(ids)
            if room < 1:
                break
            p = prompt[-room:]
            pair.append((p + ids, [0] * len(p) + mask))
        if len(pair) == 2:
            out.append(tuple(pair))
    return out


def batch_logps(model, examples, pad, device):
    """(chosen, rejected) summed answer log-probs, each (B,), for a list of encoded preferences."""
    ids, mask = pad_batch([e[0] for e in examples] + [e[1] for e in examples], pad)
    lp = answer_logp(model, ids.to(device), mask.to(device))
    return lp[:len(examples)], lp[len(examples):]


@torch.no_grad()
def evaluate(model, ref, examples, batch, pad, device, beta):
    """Mean DPO loss, and how often the model gives the chosen answer the higher likelihood."""
    was_training = model.training
    model.eval()
    total, wins = 0.0, 0
    order = sorted(range(len(examples)), key=lambda k: len(examples[k][0][0]))
    for b in range(0, len(order), batch):
        ex = [examples[k] for k in order[b:b + batch]]
        pc, pr = batch_logps(model, ex, pad, device)
        rc, rr = batch_logps(ref, ex, pad, device)
        total += float(dpo_loss(pc, pr, rc, rr, beta)) * len(ex)
        wins += int((pc > pr).sum())
    model.train(was_training)
    return total / len(examples), wins / len(examples)


def main(argv=None):
    from ..hub import build_model
    from ..model import ExeaLM, count_params
    from .pretrain import lr_at, make_optimizer, pick_device, save
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the SFT checkpoint to start from; a frozen copy is the reference")
    ap.add_argument("--tokenizer", default="tokenizer", help="the tokenizer the checkpoint was trained with")
    ap.add_argument("--data", required=True, help="preferences, .jsonl or .jsonl.gz")
    ap.add_argument("--val", default=None, help="held-out preferences (default: the last --val_frac of --data)")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.1, help="how hard to stay close to the reference model")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8, help="preferences per step (2 rows each)")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--decay_frac", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/dpo")
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
    model = model.to(dev)
    ref = copy.deepcopy(model).eval()                      # eval: its circuit scales stay as the checkpoint has them
    for p in ref.parameters():
        p.requires_grad_(False)
    model.train()
    kind = "tonn" if isinstance(model, ExeaLM) else "llama"

    prefs, skipped = load_preferences(a.data)
    train = encode(prefs, tok, a.max_len)
    dropped = len(prefs) - len(train)
    if a.val:
        val = encode(load_preferences(a.val)[0], tok, a.max_len)
    else:
        n_val = max(1, int(len(train) * a.val_frac))
        train, val = train[:-n_val], train[-n_val:]
    if not train or not val:
        raise ValueError("no usable preferences: each needs a prompt and two different answers that fit --max_len")

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a", encoding="utf-8", newline="\n")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    v, w = evaluate(model, ref, val, a.batch, pad, dev, a.beta)
    emit(dict(event="start", model=kind, params=count_params(model), device=str(dev), preferences=len(train),
              val_preferences=len(val), skipped_lines=skipped, dropped_too_long=dropped, val_loss=round(v, 4),
              val_prefers_chosen=round(w, 4)))
    opt = make_optimizer(model, a.lr, a.wd)
    warmup = a.warmup if a.warmup is not None else max(1, min(500, a.steps // 20))
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else None
    losses, t0 = [], time.time()
    order, at = rng.permutation(len(train)), 0
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, warmup, a.decay_frac)
        if at + a.batch > len(order):
            order, at = rng.permutation(len(train)), 0
        ex = [train[k] for k in order[at:at + a.batch]]
        at += a.batch
        with torch.autocast(**amp) if amp else torch.enable_grad():
            pc, pr = batch_logps(model, ex, pad, dev)
            with torch.no_grad():
                rc, rr = batch_logps(ref, ex, pad, dev)
            loss = dpo_loss(pc, pr, rc, rr, a.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            v, w = evaluate(model, ref, val, a.batch, pad, dev, a.beta)
            emit(dict(event="eval", step=step + 1, train_loss=round(sum(losses) / len(losses), 4),
                      val_loss=round(v, 4), val_prefers_chosen=round(w, 4),
                      lr=round(opt.param_groups[0]["lr"], 6), secs=round(time.time() - t0, 1)))
            losses = []
    save(os.path.join(a.out, "model.pt"), model, opt, a.steps, model.cfg, kind)
    return model


if __name__ == "__main__":
    main()
