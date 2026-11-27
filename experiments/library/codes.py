"""Memory codes: a library entry read as a few learned 32-byte codes in place of its text (docs/library_codes_proposal.md,
tests T1 and T2).

An encoder (a copy of a trained ExeaLM) reads an entry followed by k query positions. Each query position's final hidden
state is projected to 64 numbers, bounded with tanh and rounded to 16 levels (finite scalar quantization, Mentzer et al.,
2023): 64 x 4 bits = 32 bytes, one code. A reader (the same trained ExeaLM) sees the k codes, projected back to its width,
where it would read the entry's tokens, then the item's prefix and target. Encoder and reader train together on the
library's training items with the entry BM25 ranks first for the prefix (score.bm25_pick, the search used at test time),
the loss on the target only. Rounding sits inside training (straight-through gradient), so the reader learns to read
the rounded codes it will be given.

The first code is also the search key. The encoder reads a question with one query position after it; an InfoNCE loss
pulls that key toward the first code of the question's source entry, against the other entries of the batch and a queue
of recent entries' keys.

Held-out items, bits per byte on the target (as score.py):
    none          the reader with no entry
    text          the reader with the BM25 entry's tokens (--mode text trains this way: the control for the codes runs)
    codes         the reader with the BM25 entry's codes, decoded from the stored bytes
    codes_search  the reader with the codes of the entry the code search ranks first
Search, over every entry of the library: how often the item's source entry is ranked first (and in the first 5) by BM25
and by the code search, on the same held-out items.

    python experiments/library/codes.py --library data/library_v2/v2 --init runs/ragpost_exealm_100m_s0 \\
        --mode codes --k 16 --steps 6000 --out results/expanse/codes/k16.json --save runs/codes_k16
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.hub import build_model  # noqa: E402
from score import bm25_index, bm25_pick, checkpoint, entry_ids, load_items, pick_device, summarize  # noqa: E402

DIMS, LEVELS = 64, 16                      # one code: 64 numbers of 4 bits = 32 bytes


class Splice(nn.Module):
    """A model's token embedding, with chosen positions taking given vectors instead of their token's embedding."""

    def __init__(self, emb):
        super().__init__()
        self.inner = emb
        self.mask = None                   # (B, T) bool
        self.vals = None                   # (number of set positions, d), in row-major order of the mask

    @property
    def weight(self):
        return self.inner.weight

    def forward(self, idx):
        x = self.inner(idx)
        if self.mask is not None:
            x = x.masked_scatter(self.mask[..., None].expand_as(x), self.vals.to(x.dtype))
        return x


class Codes(nn.Module):
    """The query positions, the rounding to codes, and the projection of codes into the reader."""

    def __init__(self, d_enc, d_read, k, quant, scale):
        super().__init__()
        self.k, self.quant = k, quant
        self.queries = nn.Parameter(torch.randn(k, d_enc) * scale)
        self.down = nn.Linear(d_enc, DIMS)
        self.up = nn.Linear(DIMS, d_read)
        self.gain = nn.Parameter(torch.tensor(float(scale)))
        self.log_temp = nn.Parameter(torch.tensor(math.log(20.0)))

    def bound(self, h):
        return torch.tanh(self.down(h.float()))

    def round(self, z):
        """z in [-1, 1] -> the nearest of 16 levels, with a straight-through gradient (identity without --quant)."""
        if not self.quant:
            return z
        q = torch.round((z + 1) / 2 * (LEVELS - 1)) / (LEVELS - 1) * 2 - 1
        return z + (q - z).detach()

    def expand(self, z):
        v = self.up(z)
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * self.gain


def to_bytes(z):
    """Rounded codes (..., 64) in [-1, 1] -> (..., 32) uint8: two 4-bit levels a byte. What ships."""
    lv = torch.round((z.float() + 1) / 2 * (LEVELS - 1)).clamp(0, LEVELS - 1).to(torch.uint8)
    return lv[..., 0::2] * 16 + lv[..., 1::2]


def from_bytes(b):
    hi, lo = (b // 16).float(), (b % 16).float()
    lv = torch.stack([hi, lo], -1).flatten(-2)
    return lv / (LEVELS - 1) * 2 - 1


def encode(enc, codes, entries, eot, n_queries, dev, amp):
    """Each entry (a list of token ids), then n_queries query positions -> the bounded vectors (B, n_queries, 64).
    Rows are padded on the left, so the query positions are the last n_queries of every row."""
    T = max(len(e) for e in entries) + 1 + n_queries
    x = torch.full((len(entries), T), eot, dtype=torch.long)
    for r, e in enumerate(entries):
        x[r, T - n_queries - len(e):T - n_queries] = torch.tensor(e, dtype=torch.long)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, T - n_queries:] = True
    enc.emb.mask = mask.to(dev)
    enc.emb.vals = codes.queries[:n_queries].repeat(len(entries), 1)
    try:
        with torch.autocast(**amp):
            h = enc(x.to(dev))[0][:, -n_queries:]           # enc.head is the identity: the final normed hidden state
    finally:
        enc.emb.mask = enc.emb.vals = None
    return codes.bound(h)


def reader_rows(items, idx, cond, eot, sep, k, max_entry, entries_of=None):
    """(x, y, code mask) for the reader. cond: none, text (entries_of[i] gives the entry's tokens) or codes (k code
    positions where the entry would be)."""
    rows = []
    for i in idx:
        it = items[i]
        if cond == "none":
            head = [eot]
        elif cond == "text":
            head = [eot] + entries_of(i)[:max_entry] + list(sep)
        else:
            head = [eot] * (1 + k) + list(sep)
        rows.append((head, it["prefix"] + it["target"], len(head) + len(it["prefix"])))
    T = max(len(h) + len(b) for h, b, _ in rows)
    x = torch.full((len(rows), T), eot, dtype=torch.long)
    y = torch.full((len(rows), T), -100, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.bool)
    for r, (h, b, ts) in enumerate(rows):
        ids = h + b
        x[r, :len(ids)] = torch.tensor(ids)
        y[r, ts - 1:len(ids) - 1] = torch.tensor(ids[ts:])    # position p predicts token p + 1
        if cond == "codes":
            m[r, 1:1 + k] = True
    return x, y, m


def read(reader, codes, x, m, z, dev, amp):
    """Reader logits; z (B, k, 64): the codes placed at the mask's positions (codes rows only)."""
    if z is not None:
        reader.emb.mask = m.to(dev)
        reader.emb.vals = codes.expand(z).reshape(-1, codes.up.out_features)
    try:
        with torch.autocast(**amp):
            return reader(x.to(dev))[0]
    finally:
        reader.emb.mask = reader.emb.vals = None


def target_bits(logits, y, dev):
    sel = (y != -100).to(dev)
    nll = F.cross_entropy(logits[sel].float(), y.to(dev)[sel], reduction="none")
    rows = sel.nonzero()[:, 0]
    return torch.zeros(y.shape[0], device=dev).index_add_(0, rows, nll) / math.log(2), nll


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--library", required=True)
    ap.add_argument("--init", required=True, help="a trained ExeaLM (run folder or .pt): the reader and the encoder")
    ap.add_argument("--mode", choices=["codes", "text"], default="codes")
    ap.add_argument("--k", type=int, default=16, help="codes an entry")
    ap.add_argument("--no_quant", action="store_true", help="keep the bounded codes unrounded (2 bytes a number)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4, help="encoder and reader")
    ap.add_argument("--lr_new", type=float, default=1e-3, help="query positions and code projections")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--search_weight", type=float, default=0.1)
    ap.add_argument("--queue", type=int, default=8192)
    ap.add_argument("--max_entry", type=int, default=448, help="entry tokens read, as text or by the encoder")
    ap.add_argument("--max_item", type=int, default=512, help="items with a longer prefix and target are left out")
    ap.add_argument("--train_qa", type=int, default=200000)
    ap.add_argument("--train_tasks", type=int, default=100000)
    ap.add_argument("--eval_n", type=int, default=3000, help="held-out items of each set")
    ap.add_argument("--search_entries", type=int, default=0, help="0: search all entries (the smoke test uses fewer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None, help="a folder for the trained encoder, reader and code projections")
    a = ap.parse_args(argv)

    t_start = time.time()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = pick_device(a.device)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    lib = TokenShards(a.library)
    n_lib = len(lib.offsets)

    def fits(it):
        return len(it["prefix"]) + len(it["target"]) <= a.max_item
    tr_qa, eot, sep = load_items(a.library, ["qa"], "train", a.train_qa)
    tr_tk, _, _ = load_items(a.library, ["tasks"], "train", a.train_tasks)
    train = [it for it in tr_qa + tr_tk if fits(it)]
    held = []
    for s in ("qa", "tasks"):
        h, _, _ = load_items(a.library, [s], "heldout")
        h = [it for it in h if fits(it)]
        held += [h[i] for i in sorted(rng.choice(len(h), min(a.eval_n, len(h)), replace=False))]
    print(f"items: {len(train):,} training, {len(held):,} held out; library {n_lib:,} entries", flush=True)

    W = bm25_index(lib, dev=dev)
    pick_tr, top1_tr, _ = bm25_pick(lib, train, eot, sep, dev=dev, index=W)
    pick_he, top1_he, top5_he = bm25_pick(lib, held, eot, sep, dev=dev, index=W)
    del W
    print(f"BM25: the source entry first for {top1_tr.mean():.3f} of training items, {top1_he.mean():.3f} held out "
          f"({time.time() - t_start:.0f}s)", flush=True)

    base = build_model(torch.load(checkpoint(a.init), map_location="cpu", weights_only=False))
    reader = base.to(dev)
    reader.emb = Splice(reader.emb)
    scale = float(reader.emb.weight.detach().float().pow(2).mean().sqrt())
    enc = codes = None
    if a.mode == "codes":
        enc = copy.deepcopy(reader)
        enc.head = nn.Identity()
        codes = Codes(reader.cfg.d, reader.cfg.d, a.k, not a.no_quant, scale).to(dev)
    groups = [dict(params=list(reader.parameters()), lr=a.lr)]
    if enc is not None:
        groups += [dict(params=list(enc.parameters()), lr=a.lr), dict(params=list(codes.parameters()), lr=a.lr_new)]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

    def ent(d):
        return entry_ids(lib, int(d))[:a.max_entry]

    queue_k = torch.zeros(0, DIMS, device=dev)
    queue_d = torch.zeros(0, dtype=torch.long, device=dev)
    log = []
    order = rng.permutation(len(train))
    t0, seen_tok = time.time(), 0
    for step in range(1, a.steps + 1):
        frac = step / a.steps
        mult = min(1.0, step / max(1, a.warmup)) * (1.0 - 0.9 * max(0.0, frac - 0.1) / 0.9)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        b0 = ((step - 1) * a.batch) % len(order)
        idx = [int(i) for i in order[b0:b0 + a.batch]]
        if len(idx) < a.batch:
            idx += [int(i) for i in order[:a.batch - len(idx)]]
        picked = [int(pick_tr[i]) for i in idx]
        search_loss = torch.zeros((), device=dev)
        if a.mode == "text":
            x, y, m = reader_rows(train, idx, "text", eot, sep, 0, a.max_entry, lambda i: ent(pick_tr[i]))
            logits = read(reader, None, x, m, None, dev, amp)
        else:
            uniq = sorted(set(picked))
            z = encode(enc, codes, [ent(d) for d in uniq], eot, a.k, dev, amp)      # (U, k, 64)
            zq = codes.round(z)
            pos = {d: j for j, d in enumerate(uniq)}
            zr = zq[[pos[d] for d in picked]]
            x, y, m = reader_rows(train, idx, "codes", eot, sep, a.k, a.max_entry)
            logits = read(reader, codes, x, m, zr, dev, amp)
            # search: questions whose BM25 entry is their source; positives are that entry's first code
            qs = [r for r, i in enumerate(idx) if picked[r] == train[i]["source_doc"]]
            if qs and a.search_weight > 0:
                qz = encode(enc, codes, [train[idx[r]]["prefix"] for r in qs], eot, 1, dev, amp)[:, 0]
                keys = F.normalize(zq[:, 0], dim=-1)
                bank = torch.cat([keys, queue_k])
                bank_d = torch.cat([torch.tensor(uniq, device=dev), queue_d])
                sim = F.normalize(qz, dim=-1) @ bank.T * codes.log_temp.exp()
                want = torch.tensor([pos[picked[r]] for r in qs], device=dev)
                same = bank_d[None] == bank_d[want][:, None]
                same[torch.arange(len(qs)), want] = False                 # the positive stays
                sim = sim.masked_fill(same, float("-inf"))
                search_loss = F.cross_entropy(sim, want)
                queue_k = torch.cat([keys.detach(), queue_k])[:a.queue]
                queue_d = torch.cat([torch.tensor(uniq, device=dev), queue_d])[:a.queue]
        bits, nll = target_bits(logits, y, dev)
        loss = nll.mean() + a.search_weight * search_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        seen_tok += int((y != -100).sum())
        if step % 100 == 0 or step == a.steps:
            row = dict(step=step, target_loss=round(float(nll.mean()), 4), search_loss=round(float(search_loss), 4),
                       lr=round(opt.param_groups[0]["lr"], 7), seconds=round(time.time() - t0, 1))
            log.append(row)
            print(json.dumps(row), flush=True)

    # ------------------------------------------------------------------ held-out reading and search
    reader.eval()
    if enc is not None:
        enc.eval()
    res = dict(args=vars(a), items=dict(train=len(train), heldout=len(held)), train_log=log,
               bm25=dict(train_top1=round(float(top1_tr.mean()), 4)))
    with torch.no_grad():
        table = None
        if a.mode == "codes":
            n_s = a.search_entries or n_lib
            ids = list(range(n_s))
            if a.search_entries:                                  # the smoke test: the held-out sources and picks too
                ids = sorted(set(ids) | {it["source_doc"] for it in held} | {int(d) for d in pick_he})
            by_len = sorted(ids, key=lambda d: int(lib.offsets[d + 1] if d + 1 < n_lib else len(lib.tokens))
                            - int(lib.offsets[d]))
            store = {}
            t1 = time.time()
            for b0 in range(0, len(by_len), 128):
                part = by_len[b0:b0 + 128]
                zq = codes.round(encode(enc, codes, [ent(d) for d in part], eot, a.k, dev, amp))
                stored = to_bytes(zq) if codes.quant else zq.half()
                for d, s in zip(part, stored.cpu()):
                    store[d] = s
            table = torch.stack([store[d] for d in ids])           # (n, k, 32) uint8, or (n, k, 64) float16
            row_of = {d: r for r, d in enumerate(ids)}
            dec = (lambda t: from_bytes(t)) if codes.quant else (lambda t: t.float())
            keys = F.normalize(dec(table[:, 0]).to(dev), dim=-1)
            res["codes_bytes_per_entry"] = int(table[0].numel() * table.element_size())
            res["codes_table_bytes"] = int(table.numel() * table.element_size())
            res["encode_seconds"] = round(time.time() - t1, 1)
            print(f"encoded {len(ids):,} entries in {res['encode_seconds']}s: "
                  f"{res['codes_bytes_per_entry']} bytes an entry", flush=True)
            found = np.zeros((len(held), 5), np.int64)
            for b0 in range(0, len(held), 256):
                part = held[b0:b0 + 256]
                qz = F.normalize(encode(enc, codes, [it["prefix"] for it in part], eot, 1, dev, amp)[:, 0], dim=-1)
                top = (qz @ keys.T).topk(5, dim=-1).indices.cpu().numpy()
                found[b0:b0 + len(part)] = np.array(ids)[top]
            src = np.array([it["source_doc"] for it in held])
            c1, c5 = found[:, 0] == src, (found == src[:, None]).any(1)

        rows = []
        conds = ["none", "text"] + (["codes", "codes_search"] if a.mode == "codes" else [])
        sets = list(dict.fromkeys(it["set"] for it in held))
        for cond in conds:
            bits = np.zeros(len(held))
            order_h = sorted(range(len(held)), key=lambda i: len(held[i]["prefix"]) + len(held[i]["target"]))
            for b0 in range(0, len(order_h), 64):
                idx = order_h[b0:b0 + 64]
                if cond in ("none", "text"):
                    x, y, m = reader_rows(held, idx, cond, eot, sep, 0, a.max_entry, lambda i: ent(pick_he[i]))
                    logits = read(reader, None, x, m, None, dev, amp)
                else:
                    which = pick_he if cond == "codes" else found[:, 0]
                    z = dec(torch.stack([table[row_of[int(which[i])]] for i in idx])).to(dev)
                    x, y, m = reader_rows(held, idx, "codes", eot, sep, a.k, a.max_entry)
                    logits = read(reader, codes, x, m, z, dev, amp)
                bits[idx] = target_bits(logits, y, dev)[0].cpu().numpy()
            for s in sets:
                ii = [i for i, it in enumerate(held) if it["set"] == s]
                r = dict(condition=cond, set=s, **summarize(held, bits, ii))
                if cond == "text":
                    r["bm25_top1"] = round(float(top1_he[ii].mean()), 4)
                if cond == "codes_search":
                    r["search_top1"], r["search_top5"] = round(float(c1[ii].mean()), 4), round(float(c5[ii].mean()), 4)
                rows.append(r)
                print(json.dumps(r), flush=True)
        res["heldout"] = rows
        res["search"] = {s: dict(bm25_top1=round(float(top1_he[ii].mean()), 4),
                                 bm25_top5=round(float(top5_he[ii].mean()), 4),
                                 **(dict(codes_top1=round(float(c1[ii].mean()), 4),
                                         codes_top5=round(float(c5[ii].mean()), 4)) if a.mode == "codes" else {}))
                         for s in sets for ii in [[i for i, it in enumerate(held) if it["set"] == s]]}
    res["seconds"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} ({res['seconds']}s)", flush=True)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        torch.save(dict(reader=reader.state_dict(), encoder=enc.state_dict() if enc is not None else None,
                        codes=codes.state_dict() if codes is not None else None, args=vars(a)),
                   os.path.join(a.save, "codes_model.pt"))


if __name__ == "__main__":
    main()
