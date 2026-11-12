"""Experiment 9: the whole ExeaLM at 17M parameters, with retrieval and the agent board on, trained on real text.

Everything earlier tested one mechanism at a time. This trains one model the way ExeaLM is meant to be trained and
checks what it can do, against a same-size Llama that sees the same token stream:

  training batches (one kind per step, same random sequence for both models)
    text     80%  English, Japanese and code (50/30/20), read in swarms of 4 (half split, half random), seams hidden
    wiki     15%  second halves of Wikipedia articles, with the first halves in the retrieval library
    relay     5%  the relay task (experiments/swarm_relay.py): facts split across 2 agents, only the board connects them

  evaluation (held-out data only)
    language       bits per byte on English, Japanese wiki, Japanese web and code; ExeaLM with the library attached
                   and removed
    swarm          held-out English read as split swarms: board shared vs cut
    knowledge      held-out Wikipedia articles with a fresh library of their first halves: attached vs removed, and
                   how often the search finds the right article
    relay          accuracy with the board shared vs cut (chance is 1 in 64)
    samples        text the model writes, and a retrieval demo: the same prompt with and without the library

python experiments/full_small.py --model exealm --steps 24000 --out results/full_small
python experiments/full_small.py --model llama  --steps 24000 --out results/full_small
python experiments/full_small.py --smoke 1      # a few steps on synthetic data, CPU, to check the code paths
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from exealm import Datastore, ExeaLM, MemoryContext, count_params, get_config  # noqa: E402
from exealm.data.loader import MultiShards, SyntheticShards, TokenShards  # noqa: E402
from exealm.model.baseline import Llama  # noqa: E402
from exealm.train.pretrain import lr_at, make_optimizer, pick_device  # noqa: E402
from rag_wiki_split import TEXT_DOC_OFFSET, hit_rate, split_articles, text_batch  # noqa: E402
from swarm_relay import make_batch as relay_batch  # noqa: E402

VAL = ("val_en", "val_ja_wiki", "val_ja_web", "val_code")
VAL_DOC_OFFSET = 2 * 10 ** 9            # held-out documents never share an id with the library
TEXT_ID_OFFSET = 3 * 10 ** 9            # nor do training documents
KINDS = ("text", "wiki", "relay")


def lib_of(arts):
    tokens = np.concatenate([lib for _, lib, _ in arts]).astype(np.int64)
    docs = np.concatenate([np.full(len(lib), d) for d, lib, _ in arts])
    return Datastore(tokens, docs)


def relay(B, C, rng, base, n_facts=4):
    """The relay task in swarms of 2, its tokens moved to the top of the vocabulary (rare BPE ids)."""
    x, y = relay_batch(B // 2, n_facts, C, rng, agents=2)
    return x + base, torch.where(y >= 0, y + base, y)


class Data:
    def __init__(self, a):
        if a.smoke:
            V = 512
            self.text = SyntheticShards(V, 300_000, 400, seed=1)
            wiki = SyntheticShards(V, 300_000, 1200, seed=2)
            self.vals = {v: SyntheticShards(V, 60_000, 400, seed=10 + i) for i, v in enumerate(VAL)}
            for v in self.vals.values():
                v.meta["bytes_per_token"] = 1.0
        else:
            root = os.path.join(a.root, a.toks)
            self.text = MultiShards([os.path.join(root, t) for t in a.train.split(",")], weights=[float(w) for w in a.mix.split(",")])
            wiki = TokenShards(os.path.join(root, "store_wiki"))
            self.vals = {v: TokenShards(os.path.join(root, v)) for v in VAL}
        self.vocab = int(self.text.meta["vocab_size"])
        self.train_arts, self.eval_arts = split_articles(wiki, a.chunk, a.min_tokens, 0, 0.1)
        self.lib_train, self.lib_eval = lib_of(self.train_arts), lib_of(self.eval_arts)


def batch(kind, data, a, rng):
    """(x, y, doc ids or None, swarm size) for one training or evaluation step."""
    if kind == "text":
        x, y, d = data.text.batch(a.batch, a.seq, rng, 4, "mixed", 0.5, True)
        return x, y, d + TEXT_ID_OFFSET, 4                      # never the id of a library article
    if kind == "wiki":
        x, y, d = text_batch(data.train_arts, a.batch, a.seq, rng)
        return x, y, d, 4
    x, y = relay(a.batch, a.chunk, rng, data.vocab - 256)
    return x, y, None, 2


def mem_for(model_kind, ds, d, swarm, dev, cut=False):
    if model_kind == "llama":
        return MemoryContext()
    return MemoryContext(datastore=ds, doc_ids=None if d is None else d.to(dev), swarm_size=1 if cut else swarm)


@torch.no_grad()
def nll(model, kind, x, y, mem, dev):
    return float(model(x.to(dev), mem, y.to(dev))[1])


@torch.no_grad()
def evaluate(model, a, data, dev, mk):
    model.eval()
    out = {}
    rng = np.random.default_rng(10_000)
    # language: random held-out windows, read alone
    for v, sh in data.vals.items():
        bs = [sh.batch(a.batch, a.seq, rng) for _ in range(a.eval_batches)]
        bpt = sh.meta["bytes_per_token"] * math.log(2)
        rec = {"removed": round(sum(nll(model, "", x, y, mem_for(mk, None, d + VAL_DOC_OFFSET, 1, dev), dev) for x, y, d in bs) / len(bs) / bpt, 4)}
        if mk == "exealm":
            rec["attached"] = round(sum(nll(model, "", x, y, mem_for(mk, data.lib_train, d + VAL_DOC_OFFSET, 1, dev), dev) for x, y, d in bs) / len(bs) / bpt, 4)
        out[v] = rec
    # swarm: held-out English as split swarms, seams hidden; board shared vs cut
    sh = data.vals["val_en"]
    bs = [sh.batch(a.batch, a.seq, rng, 4, "split", 0.5, True) for _ in range(a.eval_batches)]
    bpt = sh.meta["bytes_per_token"] * math.log(2)
    shared = sum(nll(model, "", x, y, mem_for(mk, None, d + VAL_DOC_OFFSET, 4, dev), dev) for x, y, d in bs) / len(bs) / bpt
    cut = sum(nll(model, "", x, y, mem_for(mk, None, d + VAL_DOC_OFFSET, 4, dev, cut=True), dev) for x, y, d in bs) / len(bs) / bpt
    out["swarm"] = dict(shared=round(shared, 4), cut=round(cut, 4), board_gain=round(cut - shared, 4))
    # knowledge: held-out articles, a fresh library of their first halves
    erng = np.random.default_rng(20_000)
    wb = [text_batch(data.eval_arts, a.batch, a.seq, erng) for _ in range(a.eval_batches)]
    rem = sum(nll(model, "", x, y, mem_for(mk, None, d, 4, dev), dev) for x, y, d in wb) / len(wb) / math.log(2)
    out["knowledge"] = {"removed": round(rem, 4)}
    if mk == "exealm":
        data.lib_eval.refresh(model.encoder, model.emb)
        att = sum(nll(model, "", x, y, mem_for(mk, data.lib_eval, d, 4, dev), dev) for x, y, d in wb) / len(wb) / math.log(2)
        out["knowledge"].update(attached=round(att, 4), retrieval_gain=round(rem - att, 4), same_article_hits=hit_rate(model, wb, data.lib_eval, a.chunk, dev))
    # relay: accuracy with the board shared vs cut
    rrng = np.random.default_rng(30_000)
    hits = {"shared": 0, "cut": 0}
    n = 0
    for _ in range(max(2, a.eval_batches // 2)):
        x, y = relay(a.batch, a.chunk, rrng, data.vocab - 256)
        m = y >= 0
        for k in hits:
            logits = model(x.to(dev), mem_for(mk, None, None, 2, dev, cut=(k == "cut")))[0].float()
            hits[k] += int((logits[m.to(dev)].argmax(-1).cpu() == y[m]).sum())
        n += int(m.sum())
    out["relay"] = {k: round(v / n, 4) for k, v in hits.items()}
    model.train()
    return out


@torch.no_grad()
def samples(model, a, data, dev, mk, tok):
    """Text the model writes, and the retrieval demo: the same prompt with and without the library."""
    if tok is None:
        return {}
    model.eval()
    torch.manual_seed(0)
    enc = lambda s: torch.tensor([tok.encode(s).ids], device=dev)
    dec = lambda ids: tok.decode(ids.tolist())
    out = {"free": [], "retrieval": []}
    for p in a.prompts.split("||"):
        ids = enc(p)
        g = model.generate(ids, a.gen_tokens, temperature=0.8, top_k=40, mem=mem_for(mk, data.lib_train, torch.full_like(ids, VAL_DOC_OFFSET).cpu(), 1, dev))
        out["free"].append(dict(prompt=p, text=dec(g[0, ids.shape[1]:])))
    if mk == "exealm":
        data.lib_eval.refresh(model.encoder, model.emb)
    rng = np.random.default_rng(7)
    for i in rng.choice(len(data.eval_arts), size=min(4, len(data.eval_arts)), replace=False):
        d, first, second = data.eval_arts[int(i)]
        ids = torch.tensor(second[None, : a.chunk * 2].astype(np.int64), device=dev)
        rec = dict(article_start=dec(torch.tensor(first[0][:48])), prompt=dec(ids[0]))
        for name, ds in (("with_library", data.lib_eval), ("without_library", None)):
            if mk == "llama" and ds is not None:
                continue
            torch.manual_seed(0)
            docs = torch.full_like(ids, d + TEXT_DOC_OFFSET).cpu()
            g = model.generate(ids, a.gen_tokens, temperature=0.7, top_k=20, mem=mem_for(mk, ds, docs, 1, dev))
            rec[name] = dec(g[0, ids.shape[1]:])
        rec["truth"] = dec(torch.tensor(second[a.chunk * 2: a.chunk * 2 + a.gen_tokens].astype(np.int64)))
        out["retrieval"].append(rec)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exealm", choices=["exealm", "llama"])
    ap.add_argument("--root", default="data/tokpack")
    ap.add_argument("--toks", default="bpe32k")
    ap.add_argument("--train", default="train_en_big,train_ja_big,train_code_big")
    ap.add_argument("--mix", default="0.5,0.3,0.2")
    ap.add_argument("--p_kinds", default="0.8,0.15,0.05", help="share of steps for text, wiki and relay batches")
    ap.add_argument("--d", type=int, default=320)
    ap.add_argument("--slots", type=int, default=10)
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--min_tokens", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--refresh_every", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=3000)
    ap.add_argument("--eval_batches", type=int, default=12)
    ap.add_argument("--save_every", type=int, default=4000)
    ap.add_argument("--key_dim", type=int, default=128)
    ap.add_argument("--lexical", type=int, default=128)
    ap.add_argument("--gen_tokens", type=int, default=60)
    ap.add_argument("--prompts", default="The history of the city begins||Photosynthesis is the process by which||日本の首都は||def fibonacci(n):")
    ap.add_argument("--tokenizer", default="results/tokenizer/bpe32k/tokenizer.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/full_small")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch, a.seq, a.chunk, a.min_tokens, a.d, a.slots = 12, 8, 64, 8, 64, 64, 2
        a.eval_every, a.eval_batches, a.refresh_every, a.save_every, a.gen_tokens, a.key_dim, a.lexical = 6, 2, 5, 0, 4, 32, 32
        a.device, a.out = "cpu", "runs/full_small_smoke"
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    data = Data(a)
    cfg = get_config("small", vocab_size=data.vocab, d=a.d, slots=a.slots, h=a.d // a.slots, chunk=a.chunk, rag_chunk_len=a.chunk,
                     rag_key_dim=a.key_dim, rag_lexical=a.lexical, seed=a.seed, **({"enc_dim": 32, "n_layers": 3, "attn_layers": (1,)} if a.smoke else {}))
    model = (ExeaLM(cfg) if a.model == "exealm" else Llama(cfg)).to(dev)
    opt = make_optimizer(model, a.lr, 0.1)
    tok = None
    if os.path.exists(a.tokenizer) and not a.smoke:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"full_small_{a.model}_seed{a.seed}.json")
    info = dict(model=a.model, params=count_params(model), library_chunks=len(data.lib_train), eval_library_chunks=len(data.lib_eval),
                train_articles=len(data.train_arts), eval_articles=len(data.eval_arts), args=vars(a))
    print(json.dumps(info), flush=True)
    if a.model == "exealm":
        data.lib_train = data.lib_train.to(dev)
        data.lib_eval = data.lib_eval.to(dev)
        data.lib_train.refresh(model.encoder, model.emb, step=0)
    p = np.array([float(x) for x in a.p_kinds.split(",")])
    rng = np.random.default_rng(a.seed)
    curve, t0, tokens = [], time.time(), 0
    counts = dict.fromkeys(KINDS, 0)
    for step in range(a.steps):
        if a.model == "exealm" and step > 0 and step % a.refresh_every == 0:
            data.lib_train.refresh(model.encoder, model.emb, step=step)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup, 0.2)
        kind = KINDS[int(rng.choice(3, p=p / p.sum()))]
        x, y, d, swarm = batch(kind, data, a, rng)
        counts[kind] += 1
        tokens += x.numel()
        loss = model(x.to(dev), mem_for(a.model, data.lib_train if kind != "relay" else None, d, swarm, dev), y.to(dev))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            rec = dict(step=step + 1, secs=round(time.time() - t0, 1), tokens=tokens, train_loss=round(float(loss.detach()), 4), batches=dict(counts))
            rec.update(evaluate(model, a, data, dev, a.model))
            curve.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(dict(info, curve=curve), open(path, "w"), indent=1)
        if a.save_every and (step + 1) % a.save_every == 0:
            torch.save(dict(model=model.state_dict(), cfg=vars(cfg), step=step + 1), os.path.join(a.out, f"{a.model}_ckpt.pt"))
    torch.save(dict(model=model.state_dict(), cfg=vars(cfg), step=a.steps), os.path.join(a.out, f"{a.model}_ckpt.pt"))
    s = samples(model, a, data, dev, a.model, tok)
    json.dump(dict(info, curve=curve, samples=s), open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(samples=s), ensure_ascii=False), flush=True)
    print("WROTE", path)


if __name__ == "__main__":
    main()
