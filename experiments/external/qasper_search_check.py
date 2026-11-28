"""Is a QASPER paper in the library's human tier, and which query finds it? For the first --n questions: the share whose
paper has a passage among the top k of the human tier (a passage that shares a 13-token window with the paper's full
text), searching with the title and the question (what indep_build.py does), the title alone, or the question alone.

    python experiments/external/qasper_search_check.py --human /expanse/.../bm25_pool_p100000 --n 300
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from indep_build import NGRAM, qasper_items  # noqa: E402
from exealm.data.decontam import window_hashes  # noqa: E402
from exealm.data.tokenizer import load as load_tokenizer  # noqa: E402
from exealm.eval.harness import library_index_module  # noqa: E402
from exealm.train.pretrain import pick_device  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--human", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", default="1,5,20,100")
    ap.add_argument("--tokenizer", default="data/exealm-pretrain/slice-30b-r2/tokenizer")
    ap.add_argument("--out", default="results/expanse/qasper_search_check.json")
    a = ap.parse_args(argv)
    dev = pick_device("auto")
    tok = load_tokenizer(a.tokenizer)
    tok.no_truncation()
    tok.no_padding()
    items = qasper_items()[:a.n]
    idx = library_index_module().Index([a.human])
    if dev.type == "cuda":
        idx.to(dev)
    ks = [int(x) for x in a.k.split(",")]
    sep = tok.encode("\n\n").ids
    res = {}
    cache = {}
    for mode in ("title_and_question", "title", "question"):
        hits = np.zeros((len(items), len(ks)), dtype=bool)
        for i, it in enumerate(items):
            title, q = it["prefix"].split("\n")[0][len("Paper: "):], it["prefix"].split("\n")[1][len("Question: "):]
            text = {"title_and_question": it["prefix"], "title": title, "question": q}[mode]
            src = it["source"]
            if src not in cache:
                cache[src] = np.unique(window_hashes(np.asarray(tok.encode(src).ids, dtype=np.int64), NGRAM))
            sh = cache[src]
            find = idx.search_gpu if dev.type == "cuda" else idx.search
            docs, _ = find(tok.encode(text).ids, k=max(ks), max_df=0.1, skip=(0, *sep))
            first = None
            for r, d in enumerate(int(x) for x in docs):
                h = window_hashes(np.asarray(idx.passage(d), dtype=np.int64), NGRAM)
                if len(h) and np.isin(h, sh).any():
                    first = r
                    break
            hits[i] = [first is not None and first < k for k in ks]
        res[mode] = {f"top{k}": round(float(hits[:, j].mean()), 3) for j, k in enumerate(ks)}
        print(mode, res[mode], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(n=len(items), papers=len(cache), rule="a top-k human-tier passage shares a 13-token window with "
                   "the paper's full text", **res), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
