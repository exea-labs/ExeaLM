"""Which paragraphs of a paper answer a question: compare ways to pick them on QASPER, against the annotators' evidence.

Section 30 of results/expanse/README.md found search, not reading, to be the weak link: with the paper in hand, the
paragraphs a keyword search (BM25) picks cut the 3B's bits per byte from 1.211 to 0.904, and the annotators' evidence
paragraphs cut them to 0.454. This script measures how often each method's picks hold an evidence paragraph, on the
same 1,372 QASPER test questions, before any model is scored on them:

- bm25: the keyword search qasper_paper.py uses.
- dense:<model>: a released sentence-embedding model (BGE or E5 through transformers), cosine similarity.
- hybrid:<model>: reciprocal-rank fusion of bm25 and dense:<model>.
- upr:<checkpoint>: our own model reranks the hybrid's top --upr_k paragraphs by how well each predicts the question
  (bits of the question after the paragraph; Sachan et al. 2022, "Improving Passage Retrieval with Zero-Shot Question
  Generation"). No outside model.

Metrics: hit@1 and hit@3 (an evidence paragraph among the top 1 or 3), and "kept" (an evidence paragraph inside the
text qasper_paper.py would give: the top 3, at most 2,500 characters). Items whose evidence is not among the paper's
paragraphs (tables and figures) are left out of the rates and counted.

--write <method> writes the items with that method's paragraphs in indep_build.py's format (set "qasper_paper", the
best paragraph last), for indep_build.py --texts and score_hf.py --texts.

    python experiments/external/qasper_select.py --methods bm25,dense:BAAI/bge-large-en-v1.5 \
        --out results/studio/qasper_select.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from indep_build import qasper_items  # noqa: E402
from qasper_paper import bm25_top, words  # noqa: E402,F401


def bm25_rank(query, paras, k1=1.2, b=0.75):
    from collections import Counter
    docs = [words(p) for p in paras]
    n, avg = len(docs), sum(map(len, docs)) / max(1, len(docs))
    df = Counter(w for d in docs for w in set(d))
    q = set(words(query))
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1) /
                (tf[w] + k1 * (1 - b + b * len(d) / avg)) for w in q if w in tf)
        scores.append(s)
    return list(np.argsort(-np.asarray(scores), kind="stable"))


class Dense:
    """A released embedding model through transformers: BGE (CLS, query instruction) or E5 (mean, query:/passage:)."""

    def __init__(self, name, dev):
        from transformers import AutoModel, AutoTokenizer
        self.name, self.dev = name, dev
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if dev.type != "cpu" else torch.float32)
        self.model.to(dev).eval()
        self.e5 = "e5" in name.lower()

    @torch.no_grad()
    def embed(self, texts, query=False, bs=64):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        elif query:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
            h = self.model(**enc).last_hidden_state.float()
            if self.e5:
                m = enc["attention_mask"][..., None].float()
                v = (h * m).sum(1) / m.sum(1)
            else:
                v = h[:, 0]
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out)

    def rank(self, query, paras):
        q = self.embed([query], query=True)
        p = self.embed(paras)
        return list(torch.argsort(-(p @ q[0]), stable=True).numpy())


def rrf(*ranks, k=60):
    score = {}
    for r in ranks:
        for pos, i in enumerate(r):
            score[int(i)] = score.get(int(i), 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


class Upr:
    """Our checkpoint scores each candidate paragraph by the bits of the question read after it."""

    def __init__(self, ckpt, tokenizer, dev, para_tokens=400):
        from exealm.data.tokenizer import load as load_tokenizer
        from exealm.train.pretrain import load_model
        if ckpt.startswith("hf:"):                        # a released model: hf:<run> in exea-labs/exealm-nairr
            from huggingface_hub import snapshot_download
            run = ckpt[3:]
            ckpt = os.path.join(snapshot_download("exea-labs/exealm-nairr", allow_patterns=[f"{run}/config.json",
                                                  f"{run}/model.safetensors"]), run)
        if os.path.exists(os.path.join(ckpt, "model.safetensors")):
            from safetensors.torch import load_file
            from exealm import ExeaLM, Llama, ModelConfig
            c = json.load(open(os.path.join(ckpt, "config.json")))
            cfg = ModelConfig.from_dict(c["cfg"])
            self.model = ExeaLM(cfg, c["specs"]) if c["kind"] == "tonn" else Llama(cfg)
            self.model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")))
            self.model.to(dev)
        else:
            path = os.path.join(ckpt, "model.pt") if os.path.isdir(ckpt) else ckpt
            self.model, _ = load_model(path, dev)
        self.model.eval()
        self.tok = load_tokenizer(tokenizer)
        self.tok.no_truncation()
        self.tok.no_padding()
        self.eot = self.tok.token_to_id("<|endoftext|>")
        self.dev, self.para_tokens = dev, para_tokens

    @torch.no_grad()
    def bits(self, question, paras):
        from exealm import MemoryContext
        q_ids = self.tok.encode("\n\nQuestion: " + question).ids
        lab = self.tok.encode("\n\nQuestion: ").ids
        n_q = len(q_ids) - len(lab)                       # the question's own tokens, scored
        seqs = [[self.eot] + self.tok.encode(p).ids[:self.para_tokens] + q_ids for p in paras]
        L = max(map(len, seqs))
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        x = x.to(self.dev)
        lp = self.model(x, MemoryContext(stream={}))[0].float().log_softmax(-1)
        out = []
        for i, s in enumerate(seqs):
            pos = torch.arange(len(s) - n_q, len(s), device=self.dev)
            out.append(float(-lp[i, pos - 1].gather(1, x[i, pos][:, None]).sum() / math.log(2)))
        return out


def pick(paras, order, k=3, chars=2500):
    kept, used = [], 0
    for i in order[:k]:
        p = paras[i][:max(0, chars - used)]
        if not p:
            break
        kept.append(p)
        used += len(p) + 2
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", default="bm25")
    ap.add_argument("--tokenizer", default="data/exealm-pretrain/slice-30b-r2/tokenizer")
    ap.add_argument("--upr_k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--write", default=None, help="method whose picks are written as indep_build.py texts")
    ap.add_argument("--texts_out", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = torch.device(a.device)
    items = qasper_items()
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} QASPER items ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for it in items:
        paras = [p for p in it["source"].split("\n")[1:] if len(p) > 40]
        q = it["prefix"].split("\n")[1][len("Question: "):]
        ev = [e for e in (it.get("pasted") or "").split("\n") if e]
        gold = {i for i, p in enumerate(paras) if any(e[:200] in p for e in ev)}
        rows.append(dict(it=it, paras=paras, q=q, gold=gold, ev=ev, ranks={}))
    have = [r for r in rows if r["gold"]]
    print(f"{len(have)} items have an evidence paragraph among the paper's paragraphs "
          f"(median {int(np.median([len(r['paras']) for r in rows]))} paragraphs a paper)", flush=True)

    methods = a.methods.split(",")
    dense = {}
    for m in methods:
        kind, _, arg = m.partition(":")
        if kind in ("dense", "hybrid") and arg not in dense:
            dense[arg] = Dense(arg, dev)
    for r in rows:
        r["ranks"]["bm25"] = bm25_rank(r["q"], r["paras"])
    for name, d in dense.items():
        for r in rows:
            r["ranks"][f"dense:{name}"] = d.rank(r["q"], r["paras"]) if r["paras"] else []
            r["ranks"][f"hybrid:{name}"] = rrf(r["ranks"]["bm25"], r["ranks"][f"dense:{name}"])
        print(f"dense {name} ranked ({time.time() - t0:.0f}s)", flush=True)
        del d
    for m in methods:
        kind, _, arg = m.partition(":")
        if kind != "upr":
            continue
        base = next((x for x in methods if x.startswith("hybrid:")), "bm25")
        u = Upr(arg, a.tokenizer, dev)
        for r in rows:
            cand = r["ranks"][base][:a.upr_k]
            if not cand:
                r["ranks"][m] = []
                continue
            b = u.bits(r["q"], [r["paras"][i] for i in cand])
            r["ranks"][m] = [cand[j] for j in np.argsort(b, kind="stable")] + r["ranks"][base][a.upr_k:]
        print(f"upr {arg} over {base}'s top {a.upr_k} ({time.time() - t0:.0f}s)", flush=True)
        del u

    names = list(rows[0]["ranks"])
    res = dict(items=len(rows), with_evidence_paragraph=len(have), methods={})
    for m in names:
        h1 = np.mean([bool(set(r["ranks"][m][:1]) & r["gold"]) for r in have])
        h3 = np.mean([bool(set(r["ranks"][m][:3]) & r["gold"]) for r in have])
        kept = np.mean([any(e[:200] in "".join(pick(r["paras"], r["ranks"][m])) for e in r["ev"]) for r in have])
        res["methods"][m] = dict(hit1=round(float(h1), 4), hit3=round(float(h3), 4), kept=round(float(kept), 4))
        print(f"{m:45s} hit@1 {h1:.3f}  hit@3 {h3:.3f}  kept {kept:.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)

    if a.write:
        with open(a.texts_out, "w", encoding="utf-8") as f:
            for r in rows:
                it = dict(r["it"])
                it.pop("source", None)
                text = "".join(p + "\n\n" for p in reversed(pick(r["paras"], r["ranks"][a.write])))
                it.update(set="qasper_paper", type="qasper_paper", bm25=text, kind="item", sep="",
                          target_bytes=len(it["target"].encode("utf-8")), bits={}, meta={"select": a.write})
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print("wrote", a.texts_out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
