"""Long-context tasks scored as exact match, from the held-out documents: does the long-document lead in bits per byte
(results/expanse/README.md, sections 10, 19, 24) become a lead on tasks that need text from far back?

Every model reads each document window by window with its state carried (MemoryContext(stream=...), as stream_eval.py),
and also every window alone. Under teacher forcing, a span is an exact match when the model's top token is right at
every position of it, which is the same as greedy decoding writing the span exactly. Three tasks:

    code_line    next-line completion in long code files (as RepoBench's next-line task): lines in window 2 or later
                 that use an identifier last seen before the previous window, so neither the window nor the carried
                 keys and values of the window before hold it. The line's tokens after its first, up to the newline.
    recall       long-range copying in any document: a 6-token run whose first 2 tokens are the cue and last 4 the
                 target, that occurred before the previous window and not since.
    passkey      RULER-style retrieval: "The pass key is 48213. Remember it. 48213 is the pass key." placed in held-out
                 prose at a chosen window, then "What is the pass key? The pass key is 48213." in the last window; the
                 target is the number in that last sentence.

Paired: every model is scored on the same targets, and the per-target results are saved (--save_targets) so the
difference between two models can be tested target by target.

    python experiments/longctx_tasks.py runs/exealm_100m_111x_s0 runs/qwen_100m_111x_s0 \\
        --mix data/mix_pretrain_val.json --tokenizer data/exealm-pretrain/slice-30b-r2/tokenizer \\
        --windows 8 --out results/expanse/longctx_100m_111x.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exealm import MemoryContext  # noqa: E402
from exealm.data.loader import TokenShards  # noqa: E402
from exealm.train.pretrain import load_model, pick_device  # noqa: E402
from stream_eval import checkpoint, long_documents, mix_sets  # noqa: E402

CODE_SETS = ("starcoder", "research_python", "code_reasoning", "algebraic_stack")
PROSE_SETS = ("wikipedia_en", "arxiv", "pes2o", "dclm", "txt360_web")
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{2,}$")


def token_classes(tok, V):
    """(token ids holding a newline, token ids that read as an identifier of 3 or more characters)."""
    nl, ident = np.zeros(V, bool), np.zeros(V, bool)
    for i in range(V):
        s = tok.decode([i])
        nl[i] = "\n" in s
        ident[i] = bool(IDENT.match(s.strip().lstrip("_.")) or IDENT.match(s.strip()))
    return nl, ident


def code_line_targets(doc, seq, nl, ident):
    """[(start, end)] token spans of the lines that need memory beyond the window before (see the docstring)."""
    out = []
    n = len(doc) - 1                                         # the last token is only a target
    last_seen = {}
    line_start = 0
    for p in range(n):
        t = int(doc[p])
        if nl[t]:
            a, b = line_start, p                              # the line's tokens: [a, b), the newline at b
            if a >= 2 * seq and 4 <= b - a <= 48 and a // seq == (b - 1) // seq:
                w0 = (a // seq - 1) * seq                     # the previous window's start
                def far(q):                                   # seen before, but not since the window before
                    t_ = int(doc[q])
                    return ident[t_] and 0 <= last_seen.get(t_, -1) < w0 and t_ not in doc[a:q]
                if any(far(q) for q in range(a + 1, b)):
                    out.append((a + 1, b))                    # the first token is given, the rest are targets
            for q in range(line_start, p + 1):
                last_seen[int(doc[q])] = q
            line_start = p + 1
    return out


def recall_targets(doc, seq, ident, cue=2, span=4):
    """[(start, end)] spans: the last `span` tokens of a (cue + span)-token run last seen before the window before."""
    n, k = len(doc) - 1, cue + span
    seen = {}
    out, p = [], 0
    while p + k <= n:
        key = tuple(int(t) for t in doc[p:p + k])
        a = p + cue
        if a >= 2 * seq and a // seq == (p + k - 1) // seq:
            w0 = (a // seq - 1) * seq
            q = seen.get(key)
            if q is not None and q + k <= w0 and any(ident[t] for t in key[cue:]):
                out.append((a, p + k))
                for j in range(p, p + k):                      # this occurrence becomes the last seen
                    seen[tuple(int(t) for t in doc[j:j + k])] = j
                p += k
                continue
        seen[key] = p
        p += 1
    return out


def passkey_rows(tok, filler, windows, seq, depths, per_depth, rng):
    """(rows (N, windows * seq + 1), [(start, end)] target span, depth) for the pass-key task."""
    rows, spans, dep = [], [], []
    for d in depths:
        for j in range(per_depth):
            key = int(rng.integers(10000, 99999))
            needle = tok.encode(f" The pass key is {key}. Remember it. {key} is the pass key.").ids
            query = tok.encode(" What is the pass key? The pass key is").ids
            ans = tok.encode(f" {key}").ids
            dot = tok.encode(".").ids
            r = filler[(j + d * per_depth) % len(filler)].clone()
            L = r.shape[0]
            at = d * seq + int(rng.integers(32, seq - len(needle) - 32))
            r[at:at + len(needle)] = torch.tensor(needle)
            tail = query + ans + dot
            r[L - 1 - len(tail):L - 1] = torch.tensor(tail)
            a = L - 1 - len(dot) - len(ans)
            rows.append(r)
            spans.append((a, a + len(ans)))
            dep.append(d)
    return torch.stack(rows), spans, dep


@torch.no_grad()
def correct(model, rows, windows, seq, batch, dev, modes=("streamed", "alone")):
    """{mode: (N, windows * seq) bool}: whether the top token at position p is token p + 1."""
    out = {m: np.zeros((rows.shape[0], windows * seq), bool) for m in modes}
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch].to(dev)
        state = {}
        for w in range(windows):
            x, y = r[:, w * seq:(w + 1) * seq], r[:, w * seq + 1:(w + 1) * seq + 1]
            for m in modes:
                mem = MemoryContext(stream=state) if m == "streamed" else MemoryContext()
                hit = (model(x, mem)[0].argmax(-1) == y).cpu().numpy()
                out[m][i:i + len(r), w * seq:(w + 1) * seq] = hit
    return out


def exact(hits, spans):
    """For targets [(row, start, end)], whether every position predicting a target token is right. Position p
    predicts token p + 1, so target tokens [a, b) are predicted at positions [a - 1, b - 1)."""
    return np.array([bool(hits[r, a - 1:b - 1].all()) for r, a, b in spans])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--mix", required=True)
    ap.add_argument("--tokenizer", required=True, help="the folder holding tokenizer.json")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--docs", type=int, default=64, help="documents a set")
    ap.add_argument("--passkey", type=int, default=48, help="trials at each depth (0: no pass-key task)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_targets", default=None, help="an .npz for every target's result, for paired tests")
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    t0 = time.time()
    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    dev = pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    L = a.windows * a.seq + 1
    nl, ident = token_classes(tok, tok.get_vocab_size())
    where = mix_sets(a.mix)
    tasks = {}                                   # name -> (rows, [(row, start, end)], extra)
    for name, path in where.items():
        if name not in CODE_SETS + PROSE_SETS:
            continue
        rows = long_documents(TokenShards(path), L, a.docs)
        if rows is None:
            continue
        if name in CODE_SETS:
            spans = [(r, s, e) for r in range(len(rows)) for s, e in code_line_targets(rows[r].numpy(), a.seq, nl, ident)]
            tasks[f"code_line/{name}"] = (rows, spans, None)
        spans = [(r, s, e) for r in range(len(rows)) for s, e in recall_targets(rows[r].numpy(), a.seq, ident)]
        tasks[f"recall/{name}"] = (rows, spans, None)
    if a.passkey and "wikipedia_en" in where:
        filler = long_documents(TokenShards(where["wikipedia_en"]), L, 4 * a.passkey)
        depths = list(range(a.windows))
        rows, spans, dep = passkey_rows(tok, filler, a.windows, a.seq, depths, a.passkey, rng)
        tasks["passkey"] = (rows, [(r, s, e) for r, (s, e) in enumerate(spans)], np.array(dep))
    for k, (rows, spans, _) in tasks.items():
        print(f"{k}: {len(rows)} rows, {len(spans)} targets", flush=True)

    res, saved = dict(args=vars(a), tasks={k: len(v[1]) for k, v in tasks.items()}, runs=[]), {}
    for run in a.runs:
        model, ck = load_model(checkpoint(run), dev)
        model.eval()
        name = os.path.basename(run.rstrip("/"))
        rec = dict(run=name, model=ck["kind"], step=ck["step"], tasks={})
        cache = {}
        for k, (rows, spans, extra) in tasks.items():
            key = id(rows)
            if key not in cache:                      # code_line and recall on one set share the rows
                cache = {key: correct(model, rows, a.windows, a.seq, a.batch, dev)}
            hits = cache[key]
            r = {}
            for m, h in hits.items():
                em = exact(h, spans)
                saved[f"{name}|{k}|{m}"] = em
                r[m] = round(float(em.mean()), 4) if len(em) else None
                if extra is not None:                 # the pass key, by how many windows back the needle is
                    r[f"{m}_by_distance"] = {int(a.windows - 1 - d): round(float(em[extra == d].mean()), 4)
                                             for d in sorted(set(extra.tolist()))}
            r["targets"] = len(spans)
            rec["tasks"][k] = r
            print(f"{name} {k}: streamed {r['streamed']} alone {r['alone']} ({len(spans)} targets, "
                  f"{time.time() - t0:.0f}s)", flush=True)
        for t in ("code_line", "recall"):
            ks = [k for k in rec["tasks"] if k.startswith(t + "/")]
            for m in ("streamed", "alone"):
                ems = np.concatenate([saved[f"{name}|{k}|{m}"] for k in ks]) if ks else np.array([])
                rec[f"{t}_{m}"] = round(float(ems.mean()), 4) if len(ems) else None
        res["runs"].append(rec)
        print(f"{name}: code_line {rec['code_line_streamed']} (alone {rec['code_line_alone']}), recall "
              f"{rec['recall_streamed']} (alone {rec['recall_alone']})", flush=True)
        del model
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.save_targets:
        np.savez_compressed(a.save_targets, **saved)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
