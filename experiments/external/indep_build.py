"""Tests of the library that we did not write: research and code questions written by other people, scored for our
checkpoints here and for released models by score_hf.py on the same text, in bits per byte of the reference answer.

The library's own held-out questions were written by the model that wrote its entries (results/expanse/README.md,
section 28). These sets were not:

    qasper        QASPER test (Dasigi et al., 2021): questions NLP researchers asked about a paper from its title and
                  abstract, answered by others from the full paper. Prefix: the paper's title and the question. Target:
                  the first annotator's answer (yes/no, free-form or extracted spans; unanswerable ones left out).
                  Oracle: that annotator's evidence paragraphs.
    pubmedqa      PubMedQA, expert-labelled (Jin et al., 2019): a research question (a paper's title), answered yes, no
                  or maybe with the abstract's conclusion. Oracle: the abstract without its conclusion. The library's
                  human tier was checked against PubMedQA, so it cannot hold the abstract itself, only other papers.
    bigcodebench  BigCodeBench v0.1.4 (Zhuo et al., 2024): a Python function's signature and docstring, which uses
                  standard and third-party libraries; target: the reference solution.
    ds1000        DS-1000 (Lai et al., 2023): data-science problems from Stack Overflow; target: the reference code.

Conditions: "none" (prefix and target alone), "bm25" (the top passage of each library tier before the prefix, 384
tokens each, searched with the prefix exactly as exealm/eval/harness.py does open book) and, where the set has one,
"pasted" (the oracle evidence). A passage is never given if it shares a 13-token window with the target (the library
must not hand over the answer's text; for code, Stack Overflow and GitHub in the human tier could hold the very
solution), or if it is one of the written entries that overlap a benchmark. For QASPER, "source_found" records whether
the human tier's passage is from the paper itself (a 13-token window shared with its full text).

    python experiments/external/indep_build.py --sets qasper,pubmedqa,bigcodebench,ds1000 --limit 1000 \\
        --library "written=data/library/v1/bm25+data/library_v2/v2/bm25;human=/expanse/.../bm25_pool_p100000" \\
        --exclude written=data/library_v2/written_bench_hits.json \\
        --ckpts runs/ragpost_exealm_500m_v2 runs/ragpost_qwen_500m_v2 --out data/external/indep_texts.jsonl
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from exealm import MemoryContext  # noqa: E402
from exealm.data.decontam import window_hashes  # noqa: E402
from exealm.data.tokenizer import load as load_tokenizer  # noqa: E402
from exealm.eval.harness import library_index_module  # noqa: E402
from exealm.train.pretrain import load_model, pick_device  # noqa: E402

NGRAM = 13


def hf_rows(repo, files, revision=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rows = []
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", revision=revision)
        if p.endswith(".jsonl"):
            rows += [json.loads(line) for line in open(p, encoding="utf-8")]
        else:
            rows += pq.read_table(p).to_pylist()
    return rows


def qasper_items():
    from huggingface_hub import HfApi
    rev = "refs/convert/parquet"
    files = [f for f in HfApi().list_repo_files("allenai/qasper", repo_type="dataset", revision=rev)
             if "test" in f and f.endswith(".parquet")]
    out = []
    for p in hf_rows("allenai/qasper", files, rev):
        paras = [x for sec in p["full_text"]["paragraphs"] for x in sec]
        full = p["title"] + "\n" + p["abstract"] + "\n" + "\n".join(paras)
        qas = p["qas"]
        for qi, q in enumerate(qas["question"]):
            for ans in qas["answers"][qi]["answer"]:
                if ans["unanswerable"]:
                    continue
                if ans["yes_no"] is not None:
                    target = "Yes." if ans["yes_no"] else "No."
                elif ans["free_form_answer"]:
                    target = ans["free_form_answer"].strip()
                elif ans["extractive_spans"]:
                    target = "; ".join(s.strip() for s in ans["extractive_spans"])
                else:
                    continue
                ev = "\n".join(e for e in ans["evidence"] if e and not e.startswith("FLOAT SELECTED"))
                out.append(dict(set="qasper", id=f"{p['id']}/{qi}", prefix=f"Paper: {p['title']}\nQuestion: {q}\n"
                                f"Answer:", target=" " + target, pasted=ev[:2000] or None, source=full))
                break                                               # the first answerable annotator
    return out


def pubmedqa_items():
    out = []
    for r in hf_rows("qiaojin/PubMedQA", ["pqa_labeled/train-00000-of-00001.parquet"]):
        out.append(dict(set="pubmedqa", id=str(r["pubid"]), prefix=f"Question: {r['question']}\nAnswer:",
                        target=f" {r['final_decision'].capitalize()}. {r['long_answer'].strip()}",
                        pasted="\n".join(r["context"]["contexts"])[:2000]))
    return out


def bigcodebench_items():
    return [dict(set="bigcodebench", id=r["task_id"], prefix=r["complete_prompt"], target=r["canonical_solution"],
                 pasted=None) for r in hf_rows("bigcode/bigcodebench", ["data/v0.1.4-00000-of-00001.parquet"])]


def ds1000_items():
    return [dict(set="ds1000", id=str(r["metadata"]["problem_id"]), prefix=r["prompt"], target=r["reference_code"],
                 pasted=None) for r in hf_rows("xlangai/DS-1000", ["test.jsonl"])]


SETS = dict(qasper=qasper_items, pubmedqa=pubmedqa_items, bigcodebench=bigcodebench_items, ds1000=ds1000_items)


def shares_window(a_ids, b_hashes):
    h = window_hashes(np.asarray(a_ids, dtype=np.int64), NGRAM)
    return bool(len(h)) and bool(np.isin(h, b_hashes).any())


def heads(items, tok, tiers, exclude, rag_tokens, dev, max_df=0.1):
    """Each item's library text: the top passage of each tier for its prefix, as the harness picks it, skipping
    excluded passages and any that share a 13-token window with the target."""
    sep = tok.encode("\n\n").ids
    skip = (0, *sep)
    for it in items:
        q = tok.encode(it["prefix"]).ids
        tgt_h = np.unique(window_hashes(np.asarray(tok.encode(it["target"]).ids, dtype=np.int64), NGRAM))
        src_h = (np.unique(window_hashes(np.asarray(tok.encode(it["source"]).ids, dtype=np.int64), NGRAM))
                 if it.get("source") else None)
        parts, meta = [], {}
        for name, idx in tiers.items():
            find = idx.search_gpu if dev.type == "cuda" else idx.search
            docs, _ = find(q, k=20, max_df=max_df, skip=skip)
            for d in (int(x) for x in docs):
                if d in exclude.get(name, ()):
                    continue
                p = list(idx.passage(d)[:rag_tokens])
                if len(tgt_h) and shares_window(p, tgt_h):
                    meta[f"{name}_skipped_answer"] = meta.get(f"{name}_skipped_answer", 0) + 1
                    continue
                parts.append(tok.decode(p, skip_special_tokens=True))
                meta[f"{name}_doc"] = d
                if src_h is not None and name == "human":
                    meta["source_found"] = shares_window(p, src_h)
                break
        it["bm25"] = "".join(x + "\n\n" for x in parts)
        it["meta"] = meta
        it.pop("source", None)


@torch.no_grad()
def bits_ours(model, tok, eot, text, cut, dev, window=512, max_tokens=4096):
    """Bits on the tokens that end after character `cut` of `text`, read after <|endoftext|> as a stream of windows."""
    enc = tok.encode(text)
    ids, offs = [eot] + enc.ids, [(0, 0)] + list(enc.offsets)
    pos = [i for i, (s, e) in enumerate(offs) if e > cut and i > 0]
    if len(ids) > max_tokens:
        drop = len(ids) - max_tokens
        ids, pos = ids[drop:], [p - drop for p in pos if p - drop > 0]
    x = torch.tensor([ids], device=dev)
    state, logits = {}, []
    for s in range(0, len(ids), window):
        logits.append(model(x[:, s:s + window], MemoryContext(stream=state))[0])
    lp = torch.cat(logits, 1)[0].float().log_softmax(-1)
    p = torch.tensor(pos, device=dev)
    return float(-lp[p - 1].gather(1, x[0, p][:, None]).sum() / math.log(2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", default="qasper,pubmedqa,bigcodebench,ds1000")
    ap.add_argument("--limit", type=int, default=1000, help="items of each set (0: all)")
    ap.add_argument("--library", required=True, help='as the harness takes it: "written=a+b;human=c"')
    ap.add_argument("--exclude", default="", help='"tier=bench_hits.json;..."')
    ap.add_argument("--rag_tokens", type=int, default=384)
    ap.add_argument("--tokenizer", default="data/exealm-pretrain/slice-30b-r2/tokenizer")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--texts", default=None, help="read the searched items from this file (a run of this script) "
                    "instead of building them; --out gets them with these checkpoints' bits added")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    dev = pick_device(a.device)
    tok = load_tokenizer(a.tokenizer)
    tok.no_truncation()
    tok.no_padding()
    eot = tok.token_to_id("<|endoftext|>")
    src = a.texts or (a.out if os.path.exists(a.out) else None)
    if src:
        items = [json.loads(line) for line in open(src, encoding="utf-8")]
        print(f"{len(items)} items read from {src}", flush=True)
    else:
        items = []
        for s in a.sets.split(","):
            got = SETS[s]()
            items += got[:a.limit] if a.limit else got
            print(f"{s}: {len(got)} items, {min(len(got), a.limit or len(got))} kept ({time.time() - t0:.0f}s)",
                  flush=True)
        ix = library_index_module()
        tiers = {}
        for part in a.library.split(";"):
            name, paths = part.split("=", 1)
            tiers[name] = ix.Index(paths.split("+"))
            if dev.type == "cuda":
                tiers[name].to(dev)
        exclude = {}
        for part in filter(None, a.exclude.split(";")):
            name, path = part.split("=", 1)
            exclude[name] = {int(k) for k in json.load(open(path))["drop"]}
        heads(items, tok, tiers, exclude, a.rag_tokens, dev)
        del tiers
        for it in items:
            it.update(kind="item", type=it["set"], sep="", target_bytes=len(it["target"].encode("utf-8")), bits={})
        print(f"library searched ({time.time() - t0:.0f}s)", flush=True)
    for ck in a.ckpts:
        name = os.path.basename(ck.rstrip("/"))
        path = os.path.join(ck, "model.pt") if os.path.isdir(ck) else ck
        model, _ = load_model(path, dev)
        model.eval()
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
        with amp:
            for it in items:
                row = {}
                for cond in ("none", "pasted", "bm25"):
                    if cond != "none" and not it.get(cond):
                        continue
                    head = "" if cond == "none" else it[cond] + it["sep"]
                    text = head + it["prefix"] + it["target"]
                    row[cond] = bits_ours(model, tok, eot, text, len(head) + len(it["prefix"]), dev)
                it["bits"][name] = row
        del model
        print(f"{name} scored ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    by = {}
    for it in items:
        for name, row in it["bits"].items():
            for cond, b in row.items():
                k = (it["set"], name, cond)
                by.setdefault(k, [0.0, 0])
                by[k][0] += b
                by[k][1] += it["target_bytes"]
    for (s, name, cond), (b, n) in sorted(by.items()):
        print(f"{s:13s} {name:28s} {cond:7s} {b / n:.4f}", flush=True)
    found = [it["meta"].get("source_found") for it in items if it["set"] == "qasper"]
    if found:
        print(f"qasper: the paper itself is the human tier's passage for {np.mean([bool(x) for x in found]):.3f}")
    print(f"wrote {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
