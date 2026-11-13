"""Experiment 9: the whole ExeaLM at 17M parameters, with retrieval and the agent board on, trained on real text.

Everything earlier tested one mechanism at a time. This trains one model the way ExeaLM is meant to be trained and
checks what it can do, against a same-size Llama that sees the same token stream:

  training batches (one kind per step, same random sequence for both models)
    text     80%  English, Japanese and code (50/30/20), read in swarms of 4 (half split, half random), seams hidden
    wiki     15%  second halves of Wikipedia articles, with the first halves in the retrieval library
    relay     5%  the relay task (experiments/swarm_relay.py): facts split across 2 agents, only the board connects them
    lookup    0%  the lookup task (experiments/rag_lookup.py): questions about facts that exist only in a fresh datastore
                  (--p_kinds sets the shares; --warm_steps runs a first phase with --warm_kinds, e.g. mostly relay
                  and lookup, so the board and the search are learned before the text drowns them out)

  evaluation (held-out data only)
    language       bits per byte on English, Japanese wiki, Japanese web and code; ExeaLM with the library attached
                   and removed
    swarm          held-out English read as split swarms: board shared vs cut
    knowledge      held-out Wikipedia articles with a fresh library of their first halves: attached vs removed, and
                   how often the search finds the right article
    relay          accuracy with the board shared vs cut (chance is 1 in 64)
    lookup         accuracy on 1,024 unseen facts with the datastore attached vs removed (chance is 1 in 64)
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
from exealm.memory.lexical import LexicalDatastore  # noqa: E402
from exealm.data.loader import MultiShards, SyntheticShards, TokenShards  # noqa: E402
from exealm.model.baseline import Llama  # noqa: E402
from exealm.model.search import Searcher  # noqa: E402
from exealm.train.pretrain import lr_at, make_optimizer, pick_device  # noqa: E402
from rag_wiki_split import TEXT_DOC_OFFSET, OracleDatastore, hit_rate, split_articles, text_batch  # noqa: E402
from swarm_relay import make_batch as relay_batch  # noqa: E402
from rag_lookup import make_kb, make_kb_dense, make_questions, make_questions_dense  # noqa: E402

VAL = ("val_en", "val_ja_wiki", "val_ja_web", "val_code")
VAL_DOC_OFFSET = 2 * 10 ** 9            # held-out documents never share an id with the library
TEXT_ID_OFFSET = 3 * 10 ** 9            # nor do training documents
KINDS = ("text", "wiki", "relay", "lookup", "pair")


def lib_of(arts):
    tokens = np.concatenate([lib for _, lib, _ in arts]).astype(np.int64)
    docs = np.concatenate([np.full(len(lib), d) for d, lib, _ in arts])
    return Datastore(tokens, docs)


def relay(B, C, rng, base, n_facts=4, tokens=0):
    """The relay task in swarms of 2, its tokens moved to the top of the vocabulary (rare BPE ids). One fact per C
    tokens; rows are (n_facts + n_facts) * C long. tokens: if set, use as many rows as fit that many tokens (so a
    short relay batch costs about what a text batch does) instead of B."""
    if tokens:
        B = max(2, tokens // (2 * n_facts * C))
    x, y = relay_batch(B // 2, n_facts, C, rng, agents=2)
    return x + base, torch.where(y >= 0, y + base, y)


def lookup(B, C, n_facts, rng, base, model, dev, dense=False):
    """The lookup task, its tokens moved below the relay's (vocab-512 .. vocab-256): a fresh datastore of n_facts
    facts and B rows of 4 question/answer pairs about them. dense: the datastore chunks are tables of C/4 facts and
    each pair asks about a whole table (rag_lookup.make_kb_dense). The datastore is None for the Llama (no search)."""
    if dense:
        kb = make_kb_dense(max(1, n_facts // (C // 4)), C, rng)
        x, y = make_questions_dense(B, 4, C, kb, rng)
    else:
        kb = make_kb(n_facts, C, rng)
        x, y = make_questions(B, 4, C, kb, rng)
    ds = None
    if model is not None:
        ds = Datastore(torch.from_numpy(kb[0] + base), torch.arange(len(kb[0]))).to(dev)
        ds.refresh(model.encoder, model.emb)
    return x + base, torch.where(y >= 0, y + base, y), ds


def kinds_p(spec):
    p = np.array([float(v) for v in spec.split(",")] + [0.0] * (len(KINDS) - len(spec.split(","))))
    return p / p.sum()


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


def pair_batch(arts, B, T, rng):
    """Swarms of 2 from one Wikipedia article each: row 0 reads T tokens of the first half and has finished reading
    (its posts are visible to row 1 from row 1's first token: MemoryContext.board_prior); row 1 reads T tokens of the
    second half. Teaches the board to carry what one document says to an agent reading another part of it, the way
    notes written ahead of time are read (experiments/irl_eval.py)."""
    xs, ys, ds = [], [], []
    while len(xs) < B:
        d, first, second = arts[int(rng.integers(len(arts)))]
        f = first.reshape(-1)
        if len(second) < T + 1 or len(f) < 2:
            continue
        s = int(rng.integers(0, max(1, len(f) - T)))
        fw = torch.from_numpy(f[s:s + T + 1].astype(np.int64))
        fx, fy = fw[:-1], fw[1:]
        if len(fx) < T:                                                  # a short first half: pad, targets ignored
            fy = torch.cat([fy, torch.full((T - len(fy),), -100)])
            fx = torch.cat([fx, torch.zeros(T - len(fx), dtype=torch.long)])
        s2 = int(rng.integers(0, len(second) - T))
        w = torch.from_numpy(second[s2:s2 + T + 1].astype(np.int64))
        xs += [fx, w[:-1]]
        ys += [fy, w[1:]]
        ds += [torch.full((T,), d), torch.full((T,), d + TEXT_DOC_OFFSET)]
    return torch.stack(xs[:B]), torch.stack(ys[:B]), torch.stack(ds[:B])


def batch(kind, data, a, rng):
    """(x, y, doc ids or None, swarm size) for one training or evaluation step."""
    if kind == "text":
        x, y, d = data.text.batch(a.batch, a.seq, rng, 4, "mixed", 0.5, True)
        return x, y, d + TEXT_ID_OFFSET, 4                      # never the id of a library article
    if kind == "wiki":
        x, y, d = text_batch(data.train_arts, a.batch, a.seq, rng)
        return x, y, d, 4
    if kind == "pair":
        x, y, d = pair_batch(data.train_arts, a.batch - a.batch % 2, a.seq, rng)
        return x, y, d, 2
    x, y = relay(a.batch, a.relay_chunk or a.chunk, rng, data.vocab - 256, a.relay_facts, a.batch * a.seq)
    return x, y, None, 2


def mem_for(model_kind, ds, d, swarm, dev, cut=False):
    if model_kind == "llama":
        return MemoryContext()
    return MemoryContext(datastore=ds, doc_ids=None if d is None else d.to(dev), swarm_size=1 if cut else swarm)


@torch.no_grad()
def nll(model, kind, x, y, mem, dev):
    return float(model(x.to(dev), mem, y.to(dev))[1])


def selection(model, a, data, dev):
    """The circuit search's batches, drawn the same way every round (fixed seed) from the training mix (--search_mix),
    so a circuit that only helps the relay or the lookup still counts. Lookup batches get fresh datastores keyed by
    the current encoder. Returns [(x, y, mem)] on the device."""
    rng = np.random.default_rng(50_000)
    out = []
    for part in a.search_mix.split(","):
        kind, k = part.split(":")
        for _ in range(int(k)):
            if kind == "lookup":
                x, y, ds = lookup(a.batch, a.chunk, a.facts, rng, data.vocab - 512, model, dev, a.dense)
                mem = mem_for("exealm", ds, torch.full(x.shape, -1), 1, dev)
            else:
                x, y, d, swarm = batch(kind, data, a, rng)
                use_lib = kind == "wiki" or (kind == "text" and a.lib_on == "all")
                mem = mem_for("exealm", data.lib_train if use_lib else None, d, swarm, dev)
            out.append((x.to(dev), y.to(dev), mem))
    return out


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
        x, y = relay(a.batch, a.relay_chunk or a.chunk, rrng, data.vocab - 256, a.relay_facts, a.batch * a.seq)
        m = y >= 0
        for k in hits:
            logits = model(x.to(dev), mem_for(mk, None, None, 2, dev, cut=(k == "cut")))[0].float()
            hits[k] += int((logits[m.to(dev)].argmax(-1).cpu() == y[m]).sum())
        n += int(m.sum())
    out["relay"] = {k: round(v / n, 4) for k, v in hits.items()}
    # lookup: 1,024 facts the model has never seen, datastore attached vs removed
    lrng = np.random.default_rng(40_000)
    hits = {"attached": 0, "removed": 0} if mk == "exealm" else {"removed": 0}
    n = 0
    for _ in range(max(2, a.eval_batches // 2)):
        x, y, ds = lookup(a.batch, a.chunk, a.facts, lrng, data.vocab - 512, model if mk == "exealm" else None, dev, a.dense)
        m = y >= 0
        for k in hits:
            mem = mem_for(mk, ds if k == "attached" else None, torch.full(x.shape, -1), 1, dev)
            logits = model(x.to(dev), mem)[0].float()
            hits[k] += int((logits[m.to(dev)].argmax(-1).cpu() == y[m]).sum())
        n += int(m.sum())
    out["lookup"] = {k: round(v / n, 4) for k, v in hits.items()}
    model.train()
    return out


@torch.no_grad()
def generate(model, ids, n, temperature, top_k, mem):
    """model.generate if it has one (ExeaLM); otherwise plain sampling, one token at a time (the Llama baseline)."""
    if hasattr(model, "generate"):
        return model.generate(ids, n, temperature=temperature, top_k=top_k, mem=mem)
    for _ in range(n):
        logits = model(ids, mem)[0][:, -1].float() / temperature
        v, _ = logits.topk(top_k)
        logits[logits < v[:, -1:]] = float("-inf")
        ids = torch.cat([ids, torch.multinomial(logits.softmax(-1), 1)], 1)
    return ids


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
        g = generate(model, ids, a.gen_tokens, 0.8, 40, mem_for(mk, data.lib_train, torch.full_like(ids, VAL_DOC_OFFSET).cpu(), 1, dev))
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
            g = generate(model, ids, a.gen_tokens, 0.7, 20, mem_for(mk, ds, docs, 1, dev))
            rec[name] = dec(g[0, ids.shape[1]:])
        rec["truth"] = dec(torch.tensor(second[a.chunk * 2: a.chunk * 2 + a.gen_tokens].astype(np.int64)))
        out["retrieval"].append(rec)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="exealm", choices=["exealm", "llama"])
    ap.add_argument("--preset", default="small", choices=["small", "100m"], help="100m: the real size (--d, --slots ignored)")
    ap.add_argument("--search", type=int, default=0, help="1: TONN's circuit search during the first --search_until of training")
    ap.add_argument("--search_every", type=int, default=250)
    ap.add_argument("--search_until", type=float, default=0.5)
    ap.add_argument("--search_mix", default="text:3,wiki:1,relay:1,lookup:1", help="batches the search scores circuits on")
    ap.add_argument("--wiki_oracle", type=float, default=0.0, help="share of Wikipedia batches whose search is perfect "
                    "(it returns chunks of the right article's first half), so the read learns on real retrieved text")
    ap.add_argument("--init_ckpt", default="", help="start from this checkpoint (same architecture arguments)")
    ap.add_argument("--rag_null", type=int, default=0, help="1: RG's read can attend to nothing (cfg.rag_null)")
    ap.add_argument("--attn_gate", type=int, default=0, help="1: gated attention in the attention layers (Qwen3.5)")
    ap.add_argument("--qk_norm", type=int, default=0, help="1: RMSNorm on each head's queries and keys (Qwen3)")
    ap.add_argument("--memory_slots", type=int, default=0, help="with --init_ckpt: turn the last language circuits of "
                    "each TONN layer into RG and AG until each layer has this many of each (no new parameters)")
    ap.add_argument("--lib_search", default="learned", choices=["learned", "lexical"],
                    help="how Wikipedia batches search the library: the model's own embeddings, or BM25 keywords "
                    "(exealm/memory/lexical.py; the read stays learned)")
    ap.add_argument("--root", default="data/tokpack")
    ap.add_argument("--toks", default="bpe32k")
    ap.add_argument("--train", default="train_en_big,train_ja_big,train_code_big")
    ap.add_argument("--mix", default="0.5,0.3,0.2")
    ap.add_argument("--p_kinds", default="0.8,0.15,0.05,0", help="share of steps for text, wiki, relay and lookup batches")
    ap.add_argument("--warm_steps", type=int, default=0, help="first phase with --warm_kinds instead of --p_kinds")
    ap.add_argument("--warm_kinds", default="0.2,0,0.4,0.4")
    ap.add_argument("--schedule", default="", help="phases before --p_kinds, 'until_step:shares;...', e.g. "
                    "'400:0,0,1,0;2000:0.3,0.2,0.25,0.25' (relay alone first, then a warm mix); overrides --warm_steps")
    ap.add_argument("--facts", type=int, default=1024, help="lookup datastore size (at evaluation, and the cap)")
    ap.add_argument("--grow", type=int, default=16, help="lookup datastore starts this small and doubles (0: off)")
    ap.add_argument("--dense", type=int, default=0, help="1: lookup datastore chunks are tables of chunk/4 facts")
    ap.add_argument("--relay_chunk", type=int, default=0, help="relay: one fact per this many tokens (0: chunk)")
    ap.add_argument("--relay_facts", type=int, default=4, help="relay: facts per holder")
    ap.add_argument("--grow_every", type=int, default=150, help="lookup batches between doublings")
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
    ap.add_argument("--lib_on", default="all", choices=["all", "wiki"], help="training batches that get the library")
    ap.add_argument("--board_chunk", type=int, default=0, help="AG posts every this many tokens (0: every chunk)")
    ap.add_argument("--pool", default="mean", choices=["mean", "attn"], help="how a chunk becomes a board post or a retrieval key")
    ap.add_argument("--tag", default="")
    ap.add_argument("--stop_at", type=int, default=0, help="stop after this step, keeping the --steps schedule (quick checks)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/full_small")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch, a.seq, a.chunk, a.min_tokens, a.d, a.slots = 12, 8, 64, 8, 64, 64, 2
        a.eval_every, a.eval_batches, a.refresh_every, a.save_every, a.gen_tokens, a.key_dim, a.lexical = 6, 2, 5, 0, 4, 32, 32
        a.facts, a.grow_every = 64, 2
        a.device, a.out = "cpu", "runs/full_small_smoke"
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    data = Data(a)
    size = dict(d=a.d, slots=a.slots, h=a.d // a.slots) if a.preset == "small" else {}
    cfg = get_config(a.preset, vocab_size=data.vocab, chunk=a.chunk, rag_chunk_len=a.chunk, rag_key_dim=a.key_dim,
                     rag_lexical=a.lexical, chunk_pool=a.pool, board_chunk=a.board_chunk, seed=a.seed, **size,
                     rag_null=bool(a.rag_null), attn_gate=bool(a.attn_gate), qk_norm=bool(a.qk_norm),
                     **({"enc_dim": 32, "n_layers": 3, "attn_layers": (1,)} if a.smoke else {}))
    init = torch.load(a.init_ckpt, map_location="cpu", weights_only=False) if a.init_ckpt else None
    specs = init.get("specs") if init else None
    if specs and a.memory_slots:
        specs = [[dict(s) for s in row] for row in specs]
        for row in specs:
            for t in ("RG", "AG"):
                while sum(s["tmpl"] == t for s in row) < a.memory_slots:
                    k = max(i for i, s in enumerate(row) if s["tmpl"] not in ("RG", "AG"))
                    row[k] = dict(row[k], tmpl=t, lag=0)
    model = ExeaLM(cfg, specs) if a.model == "exealm" and init else (ExeaLM(cfg) if a.model == "exealm" else Llama(cfg))
    if init:
        missing, unexpected = model.load_state_dict(init["model"], strict=False)
        assert not unexpected and all("null" in k for k in missing), (missing, unexpected)
        if a.model == "exealm":
            model.encoder.idf_ready = True                             # keep the checkpoint's word rarity
    model = model.to(dev)
    opt = make_optimizer(model, a.lr, 0.1)
    tok = None
    if os.path.exists(a.tokenizer) and not a.smoke:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
    os.makedirs(a.out, exist_ok=True)
    name = f"{a.model}{('_' + a.tag) if a.tag else ''}"
    path = os.path.join(a.out, f"full_small_{name}_seed{a.seed}.json")
    info = dict(model=a.model, params=count_params(model), library_chunks=len(data.lib_train), eval_library_chunks=len(data.lib_eval),
                train_articles=len(data.train_arts), eval_articles=len(data.eval_arts), args=vars(a))
    print(json.dumps(info), flush=True)
    if a.model == "exealm":
        data.lib_train = data.lib_train.to(dev)
        data.lib_eval = data.lib_eval.to(dev)
        data.lib_train.refresh(model.encoder, model.emb, step=0)
        data.oracle_train = OracleDatastore(data.lib_train.tokens, data.lib_train.doc_ids)
        data.oracle_train.keys = data.lib_train.keys
        data.lib_lex = LexicalDatastore(data.lib_train.tokens, data.lib_train.doc_ids) if a.lib_search == "lexical" else None
    p = kinds_p(a.p_kinds)
    phases = [(int(u), kinds_p(sh)) for u, sh in (ph.split(":") for ph in a.schedule.split(";") if ph)] if a.schedule \
        else [(a.warm_steps, kinds_p(a.warm_kinds))]
    rng = np.random.default_rng(a.seed)
    curve, t0, tokens = [], time.time(), 0
    searcher = Searcher(every=a.search_every, until=a.search_until, seed=a.seed) if a.search and a.model == "exealm" else None
    searches = []
    counts = dict.fromkeys(KINDS, 0)
    last = min(a.steps, a.stop_at) if a.stop_at else a.steps
    for step in range(last):
        if searcher is not None and searcher.due(step, a.steps):
            rec = searcher.maybe_round(model, opt, step, a.steps, selection(model, a, data, dev), a.seq)
            rec["architecture"] = model.architecture()
            searches.append(rec)
            print(json.dumps(dict(search=rec)), flush=True)
        if a.model == "exealm" and step > 0 and step % a.refresh_every == 0:
            data.lib_train.refresh(model.encoder, model.emb, step=step)
            data.oracle_train.keys = data.lib_train.keys
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup, 0.2)
        kind = KINDS[int(rng.choice(len(KINDS), p=next((q for u, q in phases if step < u), p)))]
        if kind == "lookup":
            n_facts = min(a.facts, a.grow * 2 ** (counts["lookup"] // a.grow_every)) if a.grow else a.facts
            x, y, ds = lookup(a.batch, a.chunk, n_facts, rng, data.vocab - 512, model if a.model == "exealm" else None, dev, a.dense)
            mem = mem_for(a.model, ds, torch.full(x.shape, -1), 1, dev)
        else:
            x, y, d, swarm = batch(kind, data, a, rng)
            use_lib = kind == "wiki" or (kind == "text" and a.lib_on == "all")
            lib = data.lib_train if use_lib else None
            if kind == "wiki" and a.model == "exealm" and data.lib_lex is not None:
                lib = data.lib_lex
            if kind == "wiki" and a.wiki_oracle and a.model == "exealm" and rng.random() < a.wiki_oracle:
                lib = data.oracle_train
            mem = mem_for(a.model, lib, d, swarm, dev)
            if kind == "pair":
                mem.board_prior = True
        counts[kind] += 1
        tokens += x.numel()
        loss = model(x.to(dev), mem, y.to(dev))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev.type == "mps" and step % 200 == 0:
            torch.mps.empty_cache()
        if (step + 1) % a.eval_every == 0 or step + 1 == last:
            rec = dict(step=step + 1, secs=round(time.time() - t0, 1), tokens=tokens, train_loss=round(float(loss.detach()), 4), batches=dict(counts))
            rec.update(evaluate(model, a, data, dev, a.model))
            curve.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(dict(info, curve=curve, searches=searches), open(path, "w"), indent=1)
        if a.save_every and (step + 1) % a.save_every == 0:
            torch.save(dict(model=model.state_dict(), cfg=vars(cfg), step=step + 1, specs=getattr(model, "specs", None)), os.path.join(a.out, f"{name}_seed{a.seed}_ckpt.pt"))
    torch.save(dict(model=model.state_dict(), cfg=vars(cfg), step=last, specs=getattr(model, "specs", None)), os.path.join(a.out, f"{name}_seed{a.seed}_ckpt.pt"))
    s = samples(model, a, data, dev, a.model, tok)
    json.dump(dict(info, curve=curve, searches=searches, samples=s,
                   architecture=model.architecture() if a.model == "exealm" else None),
              open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(samples=s), ensure_ascii=False), flush=True)
    print("WROTE", path)


if __name__ == "__main__":
    main()
