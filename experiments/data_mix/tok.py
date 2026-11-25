"""Steps 3 to 5: train the tokenizer, pack the benchmarks, and tokenize every selected document with decontamination.

tokenizer (step 3)
    A balanced sample of the pretraining buckets (tokenizer.sample_mb in config.py: more Japanese and code than their
    share of the mix), drawn from documents that are neither duplicates nor validation, by whole row groups in an
    order fixed by the seed. Trained by exealm.data.tokenizer.train, so the model, pre-tokenizer, decoder and special
    tokens are the repo's: byte-level BPE, <|endoftext|> is id 0. Then a second sample of every part measures bytes
    per token (which select uses) and checks that decoding gives back every document exactly.

benchmarks (step 4)
    Exports the held-out benchmark text with exealm.data.benchmarks (needs `datasets` and the network), packs each
    with the new tokenizer, and builds one index of every benchmark 13-token window.

tokenize (step 5)
    Selects the training documents (index.run_select), then tokenizes validation and training documents in chunks of
    row groups, in a process pool. A document with any 13-token window equal to a window of one benchmark is dropped
    and recorded, which is exealm.data.decontam's rule with its defaults (n=13, min_matches=1): windows never cross a
    document's end or include its end-of-text token, a hash finds candidates and an exact comparison decides.
    `verify --decontam` runs exealm.data.decontam itself over the finished packs and expects it to drop nothing.
"""
import glob
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .common import Progress, gb, human, log, read_json, write_json
from .extract import row_groups
from .index import TRAIN, UNUSED, VAL, load_index, run_select, select_more

ROUNDTRIP = {
    "en": "The quick brown fox jumps over the lazy dog. Results: 3.14% of 2,048 runs (p < 0.05).",
    "ja": "こんにちは、世界。今日は良い天気ですね。東京大学の研究チームは2024年に新しい手法を発表した。",
    "code": "def f(x: int) -> list[int]:\n    \"\"\"Squares.\"\"\"\n    return [i ** 2 for i in range(x)]  # ok\n\tif (a != b) { return {'k': [1, 2]}; }\n",
}


def file_digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def tokenizer_file(cfg, paths):
    path = os.path.join(paths.tokenizer_dir(cfg), "tokenizer.json")
    if not os.path.exists(path):
        raise SystemExit(f"no tokenizer at {path}: run the tokenizer step first")
    return path


# ---------------------------------------------------------------------------------------------------------------------
# step 3: the tokenizer

def sample_docs(cfg, paths, b, pid, share, sample_bytes, measure_bytes):
    """Two disjoint lists of texts from one part: the training sample and the measuring sample."""
    a, meta = load_index(paths, pid)
    ok = (a["status"] == UNUSED) | (a["status"] == TRAIN)
    allowed = {}
    for f, r in zip(a["fileno"][ok], a["row"][ok]):
        allowed.setdefault(int(f), set()).add(int(r))
    # row groups of the part's first files (which every build has), shuffled; then later files' if more is needed
    rng = random.Random(f"{cfg['seed']}:tokenizer:{pid}")
    first = max(1, cfg["split"].get("val_files", 1))
    head, tail = [], []
    for fi, name in enumerate(meta["files"]):
        sizes = row_groups(os.path.join(paths.extract_dir(pid), name + ".parquet"))
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int) if sizes else []
        these = [(fi, g, int(st)) for g, st in enumerate(starts)]
        if fi < first:
            head += these
        else:
            rng.shuffle(these)
            tail += these
    rng.shuffle(head)
    groups = head + tail
    sample, measure, got_s, got_m = [], [], 0, 0
    for fi, g, start in groups:
        if got_s >= sample_bytes and got_m >= measure_bytes:
            break
        t = pq.ParquetFile(os.path.join(paths.extract_dir(pid), meta["files"][fi] + ".parquet")).read_row_group(
            g, columns=["text", "n_bytes"])
        rows = allowed.get(fi, set())
        for i, (text, nb) in enumerate(zip(t.column("text").to_pylist(), t.column("n_bytes").to_pylist())):
            if start + i not in rows:
                continue
            if got_s < sample_bytes:
                sample.append(text)
                got_s += nb
            elif got_m < measure_bytes:
                measure.append(text)
                got_m += nb
            else:
                break
    return sample, measure, got_s, got_m


def run_tokenizer(cfg, paths, args):
    from exealm.data.tokenizer import SPECIAL, load, train
    tcfg = cfg["tokenizer"]
    vocab = args.vocab or tcfg["vocab_size"]
    if vocab > 65535:
        raise SystemExit("vocab_size must fit uint16 packing")
    scale = args.sample_scale
    sample_dir = paths.p("tokenizer_sample")
    os.makedirs(sample_dir, exist_ok=True)
    measure_mb = tcfg.get("measure_mb", 20)
    files, sample_bytes, measure_sets = [], {}, {}
    out = paths.tokenizer_dir(cfg)
    reuse = os.path.exists(os.path.join(out, "tokenizer.json")) and not args.retrain
    previous = read_json(paths.report("tokenizer"), {})
    t0 = time.time()
    want_mb = config.sample_mb(cfg)
    for b in config.buckets(cfg, work=paths.work):
        mb = want_mb.get(b["name"], 0) if (b.get("in_tokenizer_sample", True) and not reuse) else 0
        texts, nbytes = [], 0
        for i, p in enumerate(b["parts"]):
            pid = config.part_id(b, i)
            s, m, gs, _ = sample_docs(cfg, paths, b, pid, p["share"], mb * 1e6 * p["share"] * scale,
                                      measure_mb * 1e6 * p["share"] * scale)
            texts += s
            nbytes += gs
            if not m and s:                       # a part too small to spare documents: measure on its sample
                log.warning(f"{pid}: no documents left to measure bytes per token; using its training sample")
                m = s[-max(1, len(s) // 5):]
            measure_sets[pid] = m
        if texts:
            path = os.path.join(sample_dir, f"{b['name']}.parquet")
            pq.write_table(pa.table({"text": pa.array(texts, type=pa.large_string())}), path, compression="zstd")
            files.append(path)
            sample_bytes[b["name"]] = nbytes
            log.info(f"tokenizer sample: {b['name']}: {len(texts)} documents, {nbytes / 1e6:.1f} MB")
    total = sum(sample_bytes.values())
    if reuse:
        # a later, larger build keeps the tokenizer of the first one, so its packs extend the earlier packs
        log.info(f"keeping the tokenizer in {out} (pass --retrain to train a new one); measuring bytes per token only")
        sample_bytes, total = previous.get("sample_bytes", {}), previous.get("sample_total_bytes", 0)
        train_secs = previous.get("train_seconds", 0.0)
    else:
        log.info(f"tokenizer sample: {total / 1e6:.0f} MB in {time.time() - t0:.0f} s; training a {vocab}-token BPE")
        t1 = time.time()
        train(files, vocab, out, max_docs=10 ** 12)
        train_secs = time.time() - t1
        log.info(f"tokenizer trained in {train_secs:.0f} s ({total / 1e6 / max(train_secs, 1e-9):.2f} MB/s)")
    tok = load(out)
    V = tok.get_vocab_size()
    if V > 65535 or V != vocab:
        raise SystemExit(f"tokenizer has {V} tokens; expected {vocab} (at most 65535)")
    ids = {s: tok.token_to_id(s) for s in SPECIAL}
    if ids["<|endoftext|>"] is None or None in ids.values():
        raise SystemExit("a special token is missing from the tokenizer")
    report = dict(vocab_size=V, special_ids=ids, sample_bytes=sample_bytes, sample_total_bytes=total,
                  sample_share={k: round(v / max(1, total), 4) for k, v in sample_bytes.items()},
                  train_seconds=round(train_secs, 1), train_mb_per_s=round(total / 1e6 / max(train_secs, 1e-9), 3),
                  tokenizer_sha256=file_digest(os.path.join(out, "tokenizer.json")), roundtrip={}, parts={},
                  buckets={})
    for name, s in ROUNDTRIP.items():
        enc = tok.encode(s).ids
        report["roundtrip"][name] = dict(ok=tok.decode(enc) == s, bytes_per_token=round(len(s.encode()) / len(enc), 3))
    bad = [k for k, v in report["roundtrip"].items() if not v["ok"]]
    t2 = time.time()
    n_tok_all = n_bytes_all = 0
    for b in config.buckets(cfg, work=paths.work):
        nb_b = nt_b = 0
        for i, p in enumerate(b["parts"]):
            pid = config.part_id(b, i)
            texts = measure_sets.get(pid, [])
            nt = nb = fails = 0
            for k in range(0, len(texts), 256):
                batch = texts[k:k + 256]
                encs = tok.encode_batch(batch)
                for t, e in zip(batch, encs):
                    nt += len(e.ids) + 1                           # + end-of-text, as packed
                    nb += len(t.encode("utf-8"))
                    if tok.decode(e.ids) != t:
                        fails += 1
            report["parts"][pid] = dict(docs=len(texts), bytes=nb, tokens=nt,
                                        bytes_per_token=round(nb / nt, 4) if nt else None, roundtrip_failures=fails)
            if fails:
                bad.append(pid)
            nb_b += nb
            nt_b += nt
        report["buckets"][b["name"]] = dict(bytes_per_token=round(nb_b / nt_b, 4) if nt_b else None, bytes=nb_b)
        n_tok_all += nt_b
        n_bytes_all += nb_b
    report["measure_seconds"] = round(time.time() - t2, 1)
    report["measure_tokens_per_s_one_process"] = round(n_tok_all / max(time.time() - t2, 1e-9))
    for name, r in report["buckets"].items():
        log.info(f"tokenizer: {name}: {r['bytes_per_token']} bytes per token")
    write_json(paths.report("tokenizer"), report)
    if bad:
        raise SystemExit(f"tokenizer round trip failed for {bad}")
    return report


def measured_bpt(paths):
    rep = read_json(paths.report("tokenizer"))
    if not rep:
        raise SystemExit("run the tokenizer step first")
    return {pid: r["bytes_per_token"] for pid, r in rep["parts"].items() if r.get("bytes_per_token")}


# ---------------------------------------------------------------------------------------------------------------------
# step 4: benchmarks

def run_benchmarks(cfg, paths, args):
    from exealm.data.decontam import PackedDocuments, window_hashes
    from exealm.data.pack import pack
    tok_file = tokenizer_file(cfg, paths)
    tok_sha = file_digest(tok_file)
    n = cfg["decontam"]["ngram"]
    os.makedirs(paths.benchmarks, exist_ok=True)
    report = dict(ngram=n, tokenizer_sha256=tok_sha, benchmarks={})
    hashes, bench_ids, positions, flat, items = [], [], [], [], []
    base = 0
    from exealm.data.benchmarks import BENCHMARKS
    from .bench_extra import EXTRA, export as export_extra
    failed = {}
    for bi, name in enumerate(cfg["decontam"]["benchmarks"]):
        export = os.path.join(paths.benchmarks, name)
        if not os.path.exists(os.path.join(export, "text.jsonl.gz")):
            try:
                if name in BENCHMARKS:
                    cmd = [sys.executable, "-m", "exealm.data.benchmarks", "--benchmark", name, "--out", export]
                    if args.bench_limit:
                        cmd += ["--limit", str(args.bench_limit)]
                    log.info("exporting " + " ".join(cmd[2:]))
                    subprocess.run(cmd, check=True)
                elif name in EXTRA:
                    log.info(f"exporting {name} (experiments/data_mix/bench_extra.py)")
                    export_extra(name, export, limit=args.bench_limit)
                else:
                    raise ValueError(f"no exporter for benchmark {name}")
            except Exception as e:                              # loud, recorded in the report and the card
                log.error(f"benchmark {name}: export FAILED ({type(e).__name__}: {e}); it is NOT decontaminated. "
                          "Fix and rerun benchmarks, then tokenize (only changed chunks are redone)")
                failed[name] = f"{type(e).__name__}: {e}"
                continue
        manifest = read_json(os.path.join(export, "manifest.json"), {})
        fields = (cfg["decontam"].get("fields") or {}).get(name)
        text_file = os.path.join(export, "text.jsonl.gz")
        if fields:                       # only these segments: a code benchmark's problem statement, not its solutions
            text_file = os.path.join(paths.benchmarks, f"{name}-{'_'.join(fields)}.jsonl.gz")
            with gzip.open(os.path.join(export, "text.jsonl.gz"), "rt", encoding="utf-8") as fi, \
                    gzip.open(text_file, "wt", encoding="utf-8") as fo:
                for line in fi:
                    if json.loads(line).get("field", "").split("_")[0] in fields:
                        fo.write(line)
        packed = os.path.join(paths.benchmarks, f"{name}-packed")
        done = read_json(os.path.join(packed, "meta.json"))
        if not done or done.get("tokenizer_sha256") != tok_sha or done.get("fields") != fields:
            if os.path.exists(packed):
                import shutil
                shutil.rmtree(packed)
            meta = pack([text_file], os.path.dirname(tok_file), packed)
            meta.update(tokenizer_sha256=tok_sha, fields=fields)
            write_json(os.path.join(packed, "meta.json"), meta)
        docs = PackedDocuments(packed)
        with gzip.open(text_file, "rt", encoding="utf-8") as f:                    # same order as packed
            item_ids = [hash((name, json.loads(line).get("benchmark_id"))) & 0x7FFFFFFFFFFF for line in f]
        windows = 0
        for doc, item in zip(docs, item_ids):
            d = np.asarray(doc[:-1], dtype=np.uint32)
            h = window_hashes(d, n)
            hashes.append(h)
            items.append(np.full(len(h), item, dtype=np.int64))
            bench_ids.append(np.full(len(h), bi, dtype=np.int16))
            positions.append(base + np.arange(len(h), dtype=np.int64))
            flat.append(d)
            base += len(d)
            windows += len(h)
        report["benchmarks"][name] = dict(rows=manifest.get("rows"), segments=manifest.get("text_segments"),
                                          revision=manifest.get("revision"), repo=manifest.get("repo"),
                                          split=manifest.get("split"), config=manifest.get("config"),
                                          sample=manifest.get("sample"), license=manifest.get("license"),
                                          tokens=int(docs.meta["n_tokens"]), windows=windows, fields=fields or "all")
        log.info(f"benchmark {name}: {windows} windows of {n} tokens")
    H = np.concatenate(hashes) if hashes else np.empty(0, np.uint64)
    common = cfg["decontam"].get("common_items", 0)
    keep = np.ones(len(H), dtype=bool)
    if common and len(H):
        # boilerplate: a window found in `common` or more different benchmark items (questions, problems) is generic
        # text (`if __name__ == "__main__":`, `#include <bits/stdc++.h>`, "1, 2, 3, 4, 5, 6, 7"), not a sign that a
        # document copies a benchmark. Such windows are left out of the index.
        I = np.concatenate(items)
        pairs = np.unique(np.stack([H.view(np.int64), I], axis=1), axis=0)
        uh, cnt = np.unique(pairs[:, 0], return_counts=True)
        generic = uh[cnt >= common].view(np.uint64)
        keep = ~np.isin(H, generic)
        report["boilerplate_windows"] = int(len(generic))
        examples = np.flatnonzero(~keep)[:2000]
        from exealm.data.tokenizer import load
        tk = load(tok_file)
        flat_all = np.concatenate(flat)
        P = np.concatenate(positions)
        seen = {}
        for i in examples:
            w = tk.decode(flat_all[P[i]:P[i] + n].tolist())
            seen[w] = seen.get(w, 0) + 1
        report["boilerplate_examples"] = [w for w, _ in sorted(seen.items(), key=lambda x: -x[1])[:25]]
        log.info(f"benchmarks: {len(generic)} window hashes found in {common}+ different items left out as boilerplate")
    min_words = cfg["decontam"].get("min_words", 0)
    if min_words and len(H):
        # a window must say something: at least min_words distinct words of 3+ letters (a CJK character counts as
        # half a word), so lists of numbers and bare signatures such as "(a: int, b: int) -> int:" do not count
        from exealm.data.tokenizer import load
        tk = load(tok_file)
        flat_all = np.concatenate(flat)
        P = np.concatenate(positions)
        idx = np.flatnonzero(keep)
        thin = np.zeros(len(H), dtype=bool)
        for k in range(0, len(idx), 200_000):
            part = idx[k:k + 200_000]
            texts = tk.decode_batch([flat_all[P[i]:P[i] + n].tolist() for i in part])
            for i, t in zip(part, texts):
                thin[i] = len(set(WORDS.findall(t))) + len(CJK.findall(t)) / 2 < min_words
        keep &= ~thin
        report["thin_windows"] = int(thin.sum())
        log.info(f"benchmarks: {int(thin.sum())} windows with fewer than {min_words} words left out")
    order = np.flatnonzero(keep)[np.argsort(H[keep], kind="stable")]
    np.savez(os.path.join(paths.benchmarks, "index.npz"), hash=H[order],
             bench=np.concatenate(bench_ids)[order] if hashes else np.empty(0, np.int16),
             pos=np.concatenate(positions)[order] if hashes else np.empty(0, np.int64),
             tokens=np.concatenate(flat) if flat else np.empty(0, np.uint32),
             names=np.array(cfg["decontam"]["benchmarks"]), ngram=np.array(n))
    report["index_sha256"] = file_digest(os.path.join(paths.benchmarks, "index.npz"))
    report["failed"] = failed
    write_json(paths.report("benchmarks"), report)
    return report


# ---------------------------------------------------------------------------------------------------------------------
# step 5: tokenize, with decontamination

_W = {}
TABLE_BITS = 27
WORDS = re.compile(r"[A-Za-z]{3,}")
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def _init(tok_path, bench_path, min_matches):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"               # one thread per process: the pool is the parallelism
    os.environ["RAYON_NUM_THREADS"] = "1"
    from exealm.data.tokenizer import load
    _W["tok"] = load(tok_path)
    _W["eot"] = _W["tok"].token_to_id("<|endoftext|>")
    z = np.load(bench_path)
    _W["bench"] = {k: z[k] for k in z.files}
    _W["n"] = int(z["ngram"])
    _W["min_matches"] = min_matches
    table = np.zeros(1 << TABLE_BITS, dtype=bool)                   # as in exealm.data.decontam: one lookup rules
    table[(_W["bench"]["hash"] & np.uint64((1 << TABLE_BITS) - 1)).astype(np.int64)] = True   # out most windows
    _W["table"] = table


def contaminated(flat, starts, lengths):
    """For each document of a batch (flat: their tokens, each followed by end-of-text), the benchmark it matches or -1,
    and where its first matching window starts in flat."""
    from exealm.data.decontam import window_hashes
    B, n, need = _W["bench"], _W["n"], _W["min_matches"]
    out = np.full(len(starts), -1, dtype=np.int16)
    first = {}
    if not len(B["hash"]) or len(flat) < n:
        return out, first
    h = window_hashes(flat, n)
    near = np.flatnonzero(_W["table"][(h & np.uint64((1 << TABLE_BITS) - 1)).astype(np.int64)])
    at = np.minimum(np.searchsorted(B["hash"], h[near]), len(B["hash"]) - 1)
    cand = near[B["hash"][at] == h[near]]
    if not len(cand):
        return out, first
    doc = np.searchsorted(starts, cand, side="right") - 1
    inside = cand + n <= starts[doc] + lengths[doc]                  # the window ends before the document's end-of-text
    hits = {}
    for c, d in zip(cand[inside], doc[inside]):
        if out[d] >= 0:                                              # already decided: skip its other windows
            continue
        lo = np.searchsorted(B["hash"], h[c], side="left")
        hi = np.searchsorted(B["hash"], h[c], side="right")
        w = flat[c:c + n]
        matched = set()
        for j in range(lo, hi):
            p = B["pos"][j]
            if np.array_equal(B["tokens"][p:p + n], w):
                matched.add(int(B["bench"][j]))
        for bi in matched:                                           # one window occurrence counts once per benchmark
            key = (int(d), bi)
            hits[key] = hits.get(key, 0) + 1
            if hits[key] >= need and out[d] < 0:
                out[d] = bi
                first[int(d)] = int(c)
    return out, first


def tokenize_chunk(job):
    """Tokenize one chunk (consecutive row groups of one extract file). Writes <out>.bin, .npz and .json."""
    t0 = time.time()
    tok, eot = _W["tok"], _W["eot"]
    pf = pq.ParquetFile(job["extract"])
    sel = dict(zip(job["rows"], job["status"]))
    rows_out, split_out, ntok_out, nbytes_out, bench_out, examples = [], [], [], [], [], []
    tmp = job["out"] + ".bin.tmp"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    n_tokens = n_bytes = 0
    with open(tmp, "wb") as fo:
        for g, start in zip(job["groups"], job["group_starts"]):
            t = pf.read_row_group(g, columns=["text"])
            texts = t.column("text").to_pylist()
            pick = [i for i in range(len(texts)) if start + i in sel]
            for k in range(0, len(pick), job["batch"]):
                idx = pick[k:k + job["batch"]]
                batch = [texts[i] for i in idx]
                encs = tok.encode_batch(batch)
                lengths = np.array([len(e.ids) for e in encs], dtype=np.int64)
                starts = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]]).astype(np.int64)
                flat = np.empty(int((lengths + 1).sum()), dtype=np.uint16)
                for s, e in zip(starts, encs):
                    flat[s:s + len(e.ids)] = e.ids
                    flat[s + len(e.ids)] = eot
                bench, first = contaminated(flat, starts, lengths)
                for d, c in first.items():
                    if len(examples) < 100:
                        examples.append([int(bench[d]), flat[c:c + _W["n"]].tolist()])
                keep = bench < 0
                if keep.all():
                    fo.write(flat.tobytes())
                else:
                    for s, ln, kp in zip(starts, lengths, keep):
                        if kp:
                            fo.write(flat[s:s + ln + 1].tobytes())
                for i, ln, bi, text in zip(idx, lengths, bench, batch):
                    rows_out.append(start + i)
                    split_out.append(sel[start + i])
                    ntok_out.append(ln + 1)
                    nb = len(text.encode("utf-8"))
                    nbytes_out.append(nb)
                    bench_out.append(bi)
                    if bi < 0:
                        n_tokens += ln + 1
                        n_bytes += nb
    os.replace(tmp, job["out"] + ".bin")
    np.savez(job["out"] + ".npz", row=np.array(rows_out, dtype=np.int64), status=np.array(split_out, dtype=np.int8),
             n_tokens=np.array(ntok_out, dtype=np.int64), n_bytes=np.array(nbytes_out, dtype=np.int64),
             bench=np.array(bench_out, dtype=np.int16))
    info = dict(digest=job["digest"], docs=len(rows_out), tokens_written=int(n_tokens), bytes_written=int(n_bytes),
                contaminated=int(sum(b >= 0 for b in bench_out)), seconds=round(time.time() - t0, 2),
                examples=examples)
    write_json(job["out"] + ".json", info)
    return info


def plan_chunks(cfg, paths, pid, chunk_bytes, batch, tok_sha, bench_sha, rnd=0):
    """Jobs for one part and selection round: validation documents go with round 0, training documents with the
    round that selected them. Each job is a run of row groups of one extract file with about chunk_bytes of text."""
    a, meta = load_index(paths, pid)
    jobs = []
    rounds = a.get("round", np.where(a["status"] == TRAIN, 0, -1))
    use = ((a["status"] == TRAIN) & (rounds == rnd)) | ((a["status"] == VAL) & (rnd == 0))
    for fi, name in enumerate(meta["files"]):
        m = use & (a["fileno"] == fi)
        if not m.any():
            continue
        rows, status, nb = a["row"][m], a["status"][m], a["n_bytes"][m]
        o = np.argsort(rows)
        rows, status, nb = rows[o], status[o], nb[o]
        path = os.path.join(paths.extract_dir(pid), name + ".parquet")
        sizes = row_groups(path)
        gstart = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        grp_of = np.searchsorted(gstart, rows, side="right") - 1
        per_group = np.bincount(grp_of, weights=nb, minlength=len(sizes))
        chunk, acc, k = [], 0.0, 0
        groups = [g for g in range(len(sizes)) if per_group[g] > 0]
        for j, g in enumerate(groups):
            chunk.append(g)
            acc += per_group[g]
            if acc >= chunk_bytes or j == len(groups) - 1:
                m2 = np.isin(grp_of, chunk)
                r, s = rows[m2], status[m2]
                digest = hashlib.sha1(r.tobytes() + s.tobytes() + tok_sha.encode() + bench_sha.encode()
                                      + json.dumps(cfg["decontam"], sort_keys=True).encode()).hexdigest()
                jobs.append(dict(pid=pid, extract=path, rows=r.tolist(), status=s.tolist(), groups=list(chunk),
                                 group_starts=[int(gstart[g]) for g in chunk], batch=batch, digest=digest,
                                 out=os.path.join(paths.tok_dir(pid), f"{name}.r{rnd}.{k:04d}"), bytes=float(acc)))
                chunk, acc = [], 0.0
                k += 1
    return jobs


def kept_train_tokens(paths, pid):
    """Training tokens of a part that survived decontamination, over every tokenized chunk."""
    total = 0
    for j in glob.glob(os.path.join(paths.tok_dir(pid), "*.json")):
        z = np.load(j[: -len(".json")] + ".npz")
        total += int(z["n_tokens"][(z["status"] == TRAIN) & (z["bench"] < 0)].sum())
    return total


def run_jobs(jobs, tok_file, bench_index, cfg, args, stats, what):
    total = sum(j["bytes"] for j in jobs)
    log.info(f"{what}: {len(jobs)} chunks to do, {gb(total)} of text, {args.workers} processes")
    if not jobs:
        return
    prog = Progress(what, total=len(jobs), unit="tokens")
    jobs.sort(key=lambda j: -j["bytes"])                             # largest first, so the pool ends together
    with ProcessPoolExecutor(max(1, args.workers), initializer=_init,
                             initargs=(tok_file, bench_index, cfg["decontam"]["min_matches"])) as pool:
        for fut in as_completed([pool.submit(tokenize_chunk, j) for j in jobs]):
            info = fut.result()
            for k, v in (("docs", "docs"), ("tokens", "tokens_written"), ("bytes", "bytes_written"),
                         ("contaminated", "contaminated")):
                stats[k] += info[v]
            stats["chunks"] += 1
            prog.update(items=1, n=info["tokens_written"], size=info["bytes_written"])
    stats["seconds"] += prog.done()


def run_tokenize(cfg, paths, args):
    """Round 0 tokenizes validation and the first training selection. If decontamination leaves a part short of its
    target, the next unused documents (in key order) are selected and tokenized as round 1, and so on (at most
    max_rounds). Every round's chunks are kept, so a rerun redoes nothing."""
    tok_file = tokenizer_file(cfg, paths)
    bench_index = os.path.join(paths.benchmarks, "index.npz")
    if not os.path.exists(bench_index):
        raise SystemExit("run the benchmarks step first (it builds benchmarks/index.npz)")
    run_select(cfg, paths, measured_bpt(paths), args.buckets)
    tok_sha, bench_sha = file_digest(tok_file), file_digest(bench_index)
    parts = [(b, p, pid) for b, _, p, pid in config.parts(cfg, args.buckets, work=paths.work)]
    stats = dict(chunks=0, docs=0, tokens=0, bytes=0, contaminated=0, seconds=0.0, rounds={})
    planned = set()
    max_rounds = cfg["select"].get("max_rounds", 4)
    todo = parts
    for rnd in range(max_rounds):
        jobs = []
        for b, p, pid in todo:
            for j in plan_chunks(cfg, paths, pid, args.chunk_mb * 1e6, args.batch, tok_sha, bench_sha, rnd):
                planned.add(os.path.basename(j["out"]))
                done = read_json(j["out"] + ".json")
                if not (done and done.get("digest") == j["digest"] and os.path.exists(j["out"] + ".bin")):
                    jobs.append(j)
        run_jobs(jobs, tok_file, bench_index, cfg, args, stats, f"tokenize round {rnd}")
        short = []
        for b, p, pid in todo:
            missing = config.unique_target(b) * p["share"] - kept_train_tokens(paths, pid)
            if missing > 0 and rnd + 1 < max_rounds and select_more(cfg, paths, pid, missing, rnd + 1):
                short.append((b, p, pid))
                log.info(f"{pid}: {human(missing)} tokens short after decontamination: selecting more (round {rnd + 1})")
        stats["rounds"][rnd] = [pid for _, _, pid in short]
        if not short:
            break
        todo = short
    for b, p, pid in parts:                                          # chunks of an older selection
        d = paths.tok_dir(pid)
        for f in os.listdir(d) if os.path.isdir(d) else []:
            if f.rsplit(".", 1)[0] not in planned:
                os.remove(os.path.join(d, f))
    secs = max(stats["seconds"], 1e-9)
    stats.update(workers=args.workers, tokens_per_s=round(stats["tokens"] / secs),
                 docs_per_s=round(stats["docs"] / secs), mb_per_s=round(stats["bytes"] / 1e6 / secs, 2))
    log.info(f"tokenize: {human(stats['docs'])} documents, {human(stats['tokens'])} tokens in {secs:.0f} s, "
             f"{stats['contaminated']} contaminated documents dropped")
    write_json(paths.report("tokenize"), stats)
    write_json(paths.report("decontam"), decontam_report(cfg, paths, tok_file, args.buckets))
    return stats


def decontam_report(cfg, paths, tok_file, names=None):
    """Documents dropped per bucket and benchmark, and the benchmark windows that matched most often (decoded), from
    every tokenized chunk. Short generic windows (number lists, alphabets) show that n is too small for that text."""
    from collections import Counter
    from exealm.data.tokenizer import load
    tok = load(tok_file)
    bench_names = cfg["decontam"]["benchmarks"]
    per, windows = {}, Counter()
    for b, _, p, pid in config.parts(cfg, names, work=paths.work):
        docs = per.setdefault(b["name"], dict(selected=0, dropped={}))
        for j in glob.glob(os.path.join(paths.tok_dir(pid), "*.json")):
            info = read_json(j)
            docs["selected"] += info["docs"]
            for bi, ids in info.get("examples", []):
                windows[(bench_names[bi], tok.decode(ids))] += 1
            z = np.load(j[: -len(".json")] + ".npz")
            for bi, cnt in zip(*np.unique(z["bench"][z["bench"] >= 0], return_counts=True)):
                docs["dropped"][bench_names[int(bi)]] = docs["dropped"].get(bench_names[int(bi)], 0) + int(cnt)
    for d in per.values():
        d["share_dropped"] = round(sum(d["dropped"].values()) / max(1, d["selected"]), 4)
    return dict(ngram=cfg["decontam"]["ngram"], min_matches=cfg["decontam"]["min_matches"], buckets=per,
                top_windows=[dict(benchmark=k[0], window=k[1], documents=v) for k, v in windows.most_common(40)])
