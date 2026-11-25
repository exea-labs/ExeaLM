"""Steps 2 and 4, dedup and select: which documents are duplicates, which are validation, which are training.

They read only the small columns of the extracts (hash, key, n_bytes), so they take minutes even on the full data,
and they are recomputed from scratch on every run. The result for each part is <work>/index/<part>.npz:

    fileno, row where the document is: its extract file (index into the .json's "files") and row there
    key         its sampling key, n_bytes its size
    status      0 dropped as an exact duplicate, 1 unused, 2 validation, 3 training

dedup (step 2): exact duplicates by text hash (whitespace and case ignored), across every part of every bucket. The
copy kept is the one every build reaches first: smallest file position (fetch.py), then recipe order, then the part's
own document order. Then each part holds out validation documents from its first val_files files (which every build
has), smallest keys first: long documents first (at least long_tokens tokens, judged by bytes before the tokenizer
exists, with a safety factor of 1.6), then any documents until the part's share of val_tokens is reached.

select (inside `tokenize`): training documents in the part's document order (file order, then key within a file)
until the part's share of the bucket target is covered at the tokenizer's measured bytes per token plus a small
margin; more rounds if decontamination leaves it short. assemble trims to the exact count in the same order. So the
training documents of a small build are the first ones of any larger build.
"""
import glob
import math
import os

import numpy as np
import pyarrow.parquet as pq

from . import config
from .common import gb, human, log, read_json, write_json

DUP, UNUSED, VAL, TRAIN = 0, 1, 2, 3
LONG_SAFETY = 1.6          # a document counts as long if it has long_tokens x bytes-per-token x 1.6 bytes


def hash_pairs(col):
    """(n, 2) uint64 from a column of 16-byte hashes, without making a Python object per row."""
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    if not len(arr):
        return np.empty((0, 2), dtype="<u8")
    data = np.frombuffer(arr.buffers()[1], dtype=np.uint8)[arr.offset * 16:(arr.offset + len(arr)) * 16]
    return data.view("<u8").reshape(-1, 2)


def extract_files(paths, pid):
    files = sorted(glob.glob(os.path.join(paths.extract_dir(pid), "*.json")))
    return [os.path.basename(f)[: -len(".json")] for f in files]


def load_index(paths, pid):
    z = np.load(paths.index(pid))
    meta = read_json(paths.index(pid)[: -len(".npz")] + ".json")
    return {k: z[k] for k in z.files}, meta


def save_index(paths, pid, arrays, meta):
    os.makedirs(os.path.dirname(paths.index(pid)), exist_ok=True)
    tmp = paths.index(pid) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, paths.index(pid))
    write_json(paths.index(pid)[: -len(".npz")] + ".json", meta)


def run_dedup(cfg, paths, args):
    jobs = [(b, p, pid) for b, _, p, pid in config.parts(cfg, work=paths.work)]   # every bucket: dedup is global
    cols, owner, names = [], [], []
    for j, (b, p, pid) in enumerate(jobs):
        files = extract_files(paths, pid)
        if not files:
            log.warning(f"{pid}: nothing extracted yet (run fetch)")
        for fi, name in enumerate(files):
            t = pq.read_table(os.path.join(paths.extract_dir(pid), name + ".parquet"),
                              columns=["source_row", "hash", "key", "n_bytes"])
            info = read_json(os.path.join(paths.extract_dir(pid), name + ".json"))
            h = hash_pairs(t.column("hash"))
            cols.append(dict(h1=h[:, 0].copy(), h2=h[:, 1].copy(), key=t.column("key").to_numpy(),
                             n_bytes=t.column("n_bytes").to_numpy(), row=np.arange(t.num_rows, dtype=np.int64),
                             fileno=np.full(t.num_rows, fi, dtype=np.int32),
                             position=np.full(t.num_rows, float(info.get("position", 0.0)), dtype=np.float64)))
            owner.append(np.full(t.num_rows, j, dtype=np.int32))
        names.append(files)
    if not cols:
        raise SystemExit("nothing extracted: run fetch first")
    cat = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
    part_of = np.concatenate(owner)
    n = len(part_of)
    log.info(f"dedup: {human(n)} documents, {gb(cat['n_bytes'].sum())} of text, in {len(jobs)} parts")
    # the copy that every build of any size reaches first wins: smallest file position, then recipe order, then the
    # part's own document order (file, key, row)
    priority = np.empty(n, dtype=np.int64)
    priority[np.lexsort((cat["row"], cat["key"], cat["fileno"], part_of, cat["position"]))] = np.arange(n)
    order = np.lexsort((priority, cat["h2"], cat["h1"]))               # equal hashes together, earliest copy first
    same = (cat["h1"][order][1:] == cat["h1"][order][:-1]) & (cat["h2"][order][1:] == cat["h2"][order][:-1])
    group = np.concatenate([[0], np.cumsum(~same)])
    first = order[np.flatnonzero(np.concatenate([[True], ~same]))]
    keeper = np.empty(n, dtype=np.int64)
    keeper[order] = first[group]                                        # for every document, the copy that is kept
    keep = np.zeros(n, dtype=bool)
    keep[first] = True
    log.info(f"dedup: {human(int((~keep).sum()))} exact duplicates")

    report = dict(documents=int(n), duplicates=int((~keep).sum()), parts={})
    split_cfg = cfg["split"]
    for j, (b, p, pid) in enumerate(jobs):
        idx = np.flatnonzero(part_of == j)
        status = np.where(keep[idx], UNUSED, DUP).astype(np.int8)
        dup_of = {}
        for other, cnt in zip(*np.unique(part_of[keeper[idx[~keep[idx]]]], return_counts=True)):
            dup_of[jobs[other][2]] = int(cnt)
        arrays = dict(fileno=cat["fileno"][idx], row=cat["row"][idx], key=cat["key"][idx], n_bytes=cat["n_bytes"][idx],
                      status=status)
        val = choose_val(arrays, b, p, split_cfg)
        arrays.pop("position", None)
        save_index(paths, pid, arrays, dict(part=pid, bucket=b["name"], files=names[j], val=val))
        report["parts"][pid] = dict(documents=int(len(idx)), duplicates=int((status == DUP).sum()),
                                    duplicates_of=dup_of, text_bytes=int(arrays["n_bytes"].sum()),
                                    unique_text_bytes=int(arrays["n_bytes"][status != DUP].sum()), val=val)
        log.info(f"{pid}: {len(idx)} documents, {int((status == DUP).sum())} duplicates "
                 f"({', '.join(f'{k}: {v}' for k, v in dup_of.items()) or 'none'}); validation {val['docs']} documents "
                 f"({val['long_docs']} long), {gb(val['bytes'])}")
    write_json(paths.report("dedup"), report)
    return report


def choose_val(a, b, p, s):
    """Mark validation documents in a's status (in place). Returns what was chosen.

    The recipe's rule: val_fraction of every source's documents, by key (a hash of the text), so a document is
    validation in every build that reads its file; plus whole long documents for the long-document test, from the
    part's first val_files files (which every build has), smallest keys first: val_long_docs of them (at least
    val_long_min). With val_fraction 0, val_tokens per bucket are taken by key instead."""
    want_long = math.ceil(s["val_long_docs"] * p["share"])
    min_long = math.ceil(s.get("val_long_min", s["val_long_docs"] // 2) * p["share"])
    long_bytes = s["long_tokens"] * b["bpt"] * LONG_SAFETY
    frac = s.get("val_fraction", 0)
    budget = (s.get("val_tokens", 0) * p["share"] * b["bpt"]) if not frac else float("inf")
    first = np.flatnonzero((a["status"] == UNUSED) & (a["fileno"] < s.get("val_files", 1)))
    first = first[np.argsort(a["key"][first], kind="stable")]       # the first files, by key: the same in every build
    used = 0
    n_long = 0
    for i in first[a["n_bytes"][first] >= long_bytes]:
        if n_long >= want_long or (n_long >= min_long and not frac and used + a["n_bytes"][i] > budget):
            break
        a["status"][i] = VAL
        used += int(a["n_bytes"][i])
        n_long += 1
    if frac:
        sel = (a["status"] == UNUSED) & (a["key"] < np.uint64(min(frac, 1.0) * float(2 ** 64 - 1)))
        a["status"][sel] = VAL
        used += int(a["n_bytes"][sel].sum())
    else:
        rest = first[a["status"][first] == UNUSED]
        if used < budget and len(rest):
            csum = np.cumsum(a["n_bytes"][rest])
            k = int(np.searchsorted(csum, budget - used)) + 1
            a["status"][rest[:k]] = VAL
            used += int(csum[min(k, len(csum)) - 1])
    return dict(docs=int((a["status"] == VAL).sum()), long_docs=n_long, bytes=used, long_bytes_threshold=int(long_bytes),
                fraction=frac)


def canonical(a, idx):
    """idx in the part's document order: file order first (the fixed shuffled order), then key within a file. A build
    takes training documents from the start of this order, so a larger build extends a smaller one."""
    return idx[np.lexsort((a["row"][idx], a["key"][idx], a["fileno"][idx]))]


def mark_train(a, budget, rnd):
    """Mark unused documents as training, in document order, until `budget` bytes. Returns (bytes, documents)."""
    pool = canonical(a, np.flatnonzero(a["status"] == UNUSED))
    if not len(pool) or budget <= 0:
        return 0, 0
    csum = np.cumsum(a["n_bytes"][pool])
    k = min(len(pool), int(np.searchsorted(csum, budget)) + 1)
    a["status"][pool[:k]] = TRAIN
    a["round"][pool[:k]] = rnd
    return int(csum[k - 1]), int(k)


def run_select(cfg, paths, bpt, names=None):
    """Round 0 of training selection: mark training documents from scratch. bpt: measured bytes per token, by part id
    (falls back to the bucket's guess). Later rounds (select_more) top up parts that decontamination left short."""
    report = {}
    margin = cfg["select"]["margin"]
    for b, _, p, pid in config.parts(cfg, names, work=paths.work):
        a, meta = load_index(paths, pid)
        a["status"][a["status"] == TRAIN] = UNUSED
        a["round"] = np.full(len(a["status"]), -1, dtype=np.int8)
        rate = bpt.get(pid) or b["bpt"]
        budget = config.unique_target(b) * p["share"] * rate * (1 + margin)
        got, k = mark_train(a, budget, 0)
        meta["select"] = dict(bytes_per_token=rate, budget_bytes=int(budget), bytes=got, docs=k,
                              short_tokens=max(0, int((budget - got) / rate)) if got < budget else 0, rounds=[k])
        save_index(paths, pid, a, meta)
        report[pid] = meta["select"]
        if got < budget:
            log.warning(f"{pid}: only {gb(got)} of text for {gb(budget)} wanted, about "
                        f"{human(meta['select']['short_tokens'])} tokens short: rerun fetch with a larger --margin")
    write_json(paths.report("select"), report)
    return report


def select_more(cfg, paths, pid, short_tokens, rnd):
    """Top up one part by about short_tokens tokens (plus the select margin) of unused documents, as round `rnd`."""
    a, meta = load_index(paths, pid)
    if "round" not in a:
        a["round"] = np.full(len(a["status"]), -1, dtype=np.int8)
    rate = meta["select"]["bytes_per_token"]
    got, k = mark_train(a, short_tokens * rate * (1 + cfg["select"]["margin"]), rnd)
    meta["select"]["rounds"] = meta["select"].get("rounds", []) + [k]
    meta["select"]["bytes"] += got
    meta["select"]["docs"] += k
    save_index(paths, pid, a, meta)
    return k
