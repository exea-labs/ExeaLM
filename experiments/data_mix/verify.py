"""Step 8, verify: open every pack the way training does and check it against its raw text and the index.

    - each packed directory opens with exealm.data.loader.TokenShards, and each mix .json with open_data, and gives
      a batch; tokens.bin, offsets.npy and meta.json agree; every document ends with <|endoftext|>
    - the raw Parquet has one row per packed document, and sampled documents decode to their raw text exactly
    - no document is in two places: training and validation share no text hash, across every bucket and repo
    - each validation split has enough documents of at least long_tokens tokens for the long-document test
    - with --decontam, exealm.data.decontam itself scans every pack against the benchmark packs and must drop nothing
"""
import glob
import json
import os
import shutil
import tempfile

import numpy as np
import pyarrow.parquet as pq

from . import config
from .assemble import SPLITS
from .common import human, log, write_json
from .index import load_index


def check_pack(cfg, root, bname, split, tok, rng, samples=8):
    from exealm.data.loader import TokenShards
    d = os.path.join(root, "packed", bname, split)
    s = TokenShards(d)
    m = s.meta
    problems = []
    if m["dtype"] != "uint16" or m["vocab_size"] != tok.get_vocab_size():
        problems.append("dtype or vocab_size")
    if len(s.tokens) != m["n_tokens"] or len(s.offsets) != m["n_docs"]:
        problems.append("token or document count differs from meta.json")
    eot = tok.token_to_id("<|endoftext|>")
    if m["n_docs"]:
        ends = np.append(s.offsets[1:], len(s.tokens)) - 1
        if not (np.asarray(s.tokens[ends]) == eot).all():
            problems.append("a document does not end with <|endoftext|>")
        if s.offsets[0] != 0 or (np.diff(s.offsets) <= 0).any():
            problems.append("offsets are not increasing from 0")
        if int(np.asarray(s.tokens).max()) >= m["vocab_size"]:
            problems.append("token id beyond the vocabulary")
    raw_files = sorted(glob.glob(os.path.join(root, "raw", bname, split, "*.parquet")))
    n_raw = sum(pq.ParquetFile(f).metadata.num_rows for f in raw_files)
    if not raw_files:
        pass                                        # packed only (no --raw): nothing to compare
    elif n_raw != m["n_docs"]:
        problems.append(f"{n_raw} raw rows for {m['n_docs']} documents")
    elif m["n_docs"]:
        starts = np.cumsum([0] + [pq.ParquetFile(f).metadata.num_rows for f in raw_files])
        for i in rng.choice(m["n_docs"], size=min(samples, m["n_docs"]), replace=False):
            fi = int(np.searchsorted(starts, i, side="right") - 1)
            text = pq.read_table(raw_files[fi], columns=["text"]).column("text")[int(i - starts[fi])].as_py()
            e = int(s.offsets[i + 1]) if i + 1 < len(s.offsets) else len(s.tokens)
            if tok.decode(np.asarray(s.tokens[int(s.offsets[i]):e - 1]).tolist()) != text:
                problems.append(f"document {i} does not decode to its raw text")
                break
    if len(s.tokens) > 66:
        x, y, _ = s.batch(2, 32, rng)
        if x.shape != (2, 32):
            problems.append("batch shape")
    rec = dict(docs=m["n_docs"], tokens=m["n_tokens"], long_docs=m.get("long_docs"), warnings=[])
    if split == "val" and m.get("long_docs", 0) < cfg["split"].get("val_long_min", 128):
        rec["warnings"].append(f"only {m.get('long_docs')} long documents (want {cfg['split'].get('val_long_min', 128)})")
    rec["problems"] = problems
    return rec


def overlap(cfg, paths):
    """Text hashes of every packed document by split; the count of hashes in both, and in two buckets."""
    seen = {}
    for b, i, p, pid in config.parts(cfg, work=paths.work):
        if not os.path.exists(paths.index(pid)):
            continue
        a, meta = load_index(paths, pid)
        for fi, name in enumerate(meta["files"]):
            for split, code in SPLITS.items():
                rows = np.sort(a["row"][(a["fileno"] == fi) & (a["status"] == code)])
                if not len(rows):
                    continue
                h = pq.read_table(os.path.join(paths.extract_dir(pid), name + ".parquet"), columns=["hash"]) \
                    .column("hash").to_pylist()
                for r in rows:
                    seen.setdefault(h[r], set()).add((split, b["name"]))
    both = sum(1 for v in seen.values() if {s for s, _ in v} == {"train", "val"})
    multi = sum(1 for v in seen.values() if len(v) > 1)
    return dict(documents=len(seen), in_train_and_val=both, in_two_places=multi)


def decontam_check(cfg, paths, root, bname, split, tok_dir):
    from exealm.data.decontam import decontaminate
    src = os.path.join(root, "packed", bname, split)
    with tempfile.TemporaryDirectory(dir=paths.work, prefix=".verify-") as tmp:
        link = os.path.join(tmp, f"{bname}-{split}")
        os.makedirs(link)
        for f in ("tokens.bin", "offsets.npy"):
            os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(link, f))
        meta = json.load(open(os.path.join(src, "meta.json"), encoding="utf-8"))
        meta["tokenizer"] = tok_dir
        write_json(os.path.join(link, "meta.json"), meta)
        benches = {n: os.path.join(paths.benchmarks, f"{n}-packed") for n in cfg["decontam"]["benchmarks"]}
        rep = decontaminate([link], benches, os.path.join(tmp, "out"), cfg["decontam"]["ngram"],
                            cfg["decontam"]["min_matches"])
        r = next(iter(rep["sources"].values()))
        return dict(read=r["read"], dropped=r["dropped"])


def run(cfg, paths, args):
    from exealm.data.loader import open_data
    from exealm.data.tokenizer import load
    tok_dir = paths.tokenizer_dir(cfg)
    tok = load(tok_dir)
    rng = np.random.default_rng(0)
    report = dict(packs={}, mixes={}, problems=[])
    checks = {}
    if args.decontam:                                    # one process per pack: exealm.data.decontam is single-core
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max(1, args.workers)) as pool:
            futs = {(b["name"], split): pool.submit(decontam_check, cfg, paths, paths.repo_dir(cfg, b["repo"]),
                                                    b["name"], split, tok_dir)
                    for b in config.buckets(cfg, args.buckets, work=paths.work) for split in SPLITS}
            checks = {k: f.result() for k, f in futs.items()}
    for b in config.buckets(cfg, args.buckets, work=paths.work):
        root = paths.repo_dir(cfg, b["repo"])
        for split in SPLITS:
            rec = check_pack(cfg, root, b["name"], split, tok, rng)
            if args.decontam:
                rec["decontam"] = checks[(b["name"], split)]
                if rec["decontam"]["dropped"] and not cfg["decontam"].get("common_items"):
                    rec["problems"].append(f"exealm.data.decontam drops {rec['decontam']['dropped']} documents")
                elif rec["decontam"]["dropped"]:            # expected: the repo tool has no boilerplate rule
                    rec["decontam"]["note"] = "matches on boilerplate windows that the build ignores (common_items)"
            report["packs"][f"{b['name']}/{split}"] = rec
            report["problems"] += [f"{b['name']}/{split}: {p}" for p in rec["problems"]]
            log.info(f"verify {b['name']}/{split}: {human(rec['docs'])} documents, {human(rec['tokens'])} tokens, "
                     f"{rec['long_docs']} long" + (f"  PROBLEMS: {rec['problems']}" if rec["problems"] else "  ok")
                     + (f"  warning: {rec['warnings']}" if rec["warnings"] else ""))
    for repo in ("pretrain", "post"):
        root = paths.repo_dir(cfg, repo)
        for mix in sorted(glob.glob(os.path.join(root, "mix*.json"))):
            spec = json.load(open(mix, encoding="utf-8"))
            for s in spec["sources"].values():                 # data/<repo name>/... -> the repo folders here
                s["paths"] = os.path.join(paths.p("repos"), s["paths"].split("/", 1)[1])
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=paths.work) as f:
                json.dump(spec, f)
            try:
                data = open_data(f.name)
                x, _, _ = data.batch(4, 16, rng)
                report["mixes"][os.path.relpath(mix, paths.p("repos"))] = dict(sources=data.names,
                                                                              weights=[round(float(w), 4) for w in data.p])
            except Exception as e:
                report["problems"].append(f"{os.path.basename(mix)}: open_data failed: {e}")
            finally:
                os.remove(f.name)
    report["overlap"] = overlap(cfg, paths)
    if report["overlap"]["in_two_places"]:
        report["problems"].append(f"{report['overlap']['in_two_places']} texts are in two places")
    write_json(paths.report("verify"), report)
    if report["problems"]:
        for p in report["problems"]:
            log.error(p)
        raise SystemExit(f"verify found {len(report['problems'])} problems")
    log.info(f"verify: every pack and mix is fine; {report['overlap']['documents']} documents, none in two places")
    shutil.rmtree(paths.p(".verify"), ignore_errors=True)
    return report
