"""QASPER through the library in two steps, as a research assistant would search it: find the paper by its title in the
human tier (36.8M passages), then, among that document's passages, the one a keyword search ranks first for the
question. That passage (at most --tokens of it) goes right before the question.

indep_build.py searches once with the title and the question together; the paper's passage then comes first for 0.25 of
questions, and for 0.49 when the title alone is searched (results/expanse/qasper_search_check.json). Passages of one
document sit next to each other in the pack and each starts with the document's title, so the document is the run of
passages around the title search's pick with the same first line.

Writes items in indep_build.py's format (set "qasper_lib2", the passage in "bm25"); meta.source_found says whether the
passage shares a 13-token window with the paper's full text.

    python experiments/external/qasper_lib2.py --human /expanse/.../bm25_pool_p100000 \\
        --out /expanse/.../external/indep_qasper_lib2.jsonl
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
from qasper_paper import bm25_top  # noqa: E402
from exealm.data.decontam import window_hashes  # noqa: E402
from exealm.data.tokenizer import load as load_tokenizer  # noqa: E402
from exealm.eval.harness import library_index_module  # noqa: E402
from exealm.train.pretrain import pick_device  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--human", required=True)
    ap.add_argument("--tokens", type=int, default=700)
    ap.add_argument("--span", type=int, default=300, help="passages looked at on each side of the title's pick")
    ap.add_argument("--tokenizer", default="data/exealm-pretrain/slice-30b-r2/tokenizer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dev = pick_device("auto")
    tok = load_tokenizer(a.tokenizer)
    tok.no_truncation()
    tok.no_padding()
    idx = library_index_module().Index([a.human])
    if dev.type == "cuda":
        idx.to(dev)
    find = idx.search_gpu if dev.type == "cuda" else idx.search
    skip = (0, *tok.encode("\n\n").ids)
    n_total = idx.n
    first_line = {}

    def title_of(d):
        if d not in first_line:
            first_line[d] = tok.decode(list(idx.passage(d)[:48]), skip_special_tokens=True).split("\n")[0].strip()
        return first_line[d]

    items = qasper_items()
    found, docs_sizes = 0, []
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            title = it["prefix"].split("\n")[0][len("Paper: "):]
            q = it["prefix"].split("\n")[1][len("Question: "):]
            d0 = int(find(tok.encode(title).ids, k=1, max_df=0.1, skip=skip)[0][0])
            t0 = title_of(d0)
            run = [d0]
            for step in (-1, 1):
                d = d0 + step
                while abs(d - d0) <= a.span and 0 <= d < n_total and title_of(d) == t0:
                    run.append(d)
                    d += step
            run.sort()
            texts = {d: tok.decode(list(idx.passage(d)), skip_special_tokens=True) for d in run}
            best = bm25_top(q, [texts[d] for d in run], k=1)
            pick = run[[texts[d] for d in run].index(best[0])] if best else d0
            ids = list(idx.passage(pick)[:a.tokens])
            src = np.unique(window_hashes(np.asarray(tok.encode(it.pop("source")).ids, dtype=np.int64), NGRAM))
            h = window_hashes(np.asarray(ids, dtype=np.int64), NGRAM)
            ok = bool(len(h)) and bool(np.isin(h, src).any())
            found += ok
            docs_sizes.append(len(run))
            it.update(set="qasper_lib2", type="qasper_lib2", bm25=tok.decode(ids, skip_special_tokens=True) + "\n\n",
                      kind="item", sep="", target_bytes=len(it["target"].encode("utf-8")), bits={},
                      meta=dict(title_pick=d0, passages_in_document=len(run), pick=pick, source_found=ok))
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items; the passage is from the paper for {found / len(items):.3f}; "
          f"median passages per found document {int(np.median(docs_sizes))}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
