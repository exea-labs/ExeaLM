"""Step 6, assemble: one packed directory per bucket and split, the raw text beside it, and the mix .json files.

For each bucket, in each repo folder under <work>/repos/<repo name>/:
    packed/<bucket>/train, packed/<bucket>/val     exealm.data.pack's format: tokens.bin (uint16), offsets.npy
                                                   (int64 document starts), meta.json
    raw/<bucket>/train/*.parquet, raw/<bucket>/val/*.parquet
                                                   the same documents as text, in the same order: row i of the
                                                   concatenated files is document i of the pack

Training documents are trimmed to the target exactly: each part keeps its smallest-key documents until its share of
the target is reached (the document that crosses it is kept). Validation keeps everything selected for it.
Documents that matched a benchmark were dropped by tokenize and are only counted here.

Then mix.json and mix_val.json, which exealm.data.loader.open_data reads, with paths as fetch.sh lays the repos out
under the ExeaLM checkout (data/<repo name>/...) and weights equal to the targets in billions of tokens.
"""
import glob
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import config
from .common import gb, human, log, read_json, write_json
from .extract import read_rows
from .index import TRAIN, VAL, load_index

SPLITS = {"train": TRAIN, "val": VAL}


def chunks_of(paths, pid):
    """The tokenized chunks of a part, in file and chunk order: (base path, arrays)."""
    out = []
    for j in sorted(glob.glob(os.path.join(paths.tok_dir(pid), "*.json"))):
        base = j[: -len(".json")]
        z = np.load(base + ".npz")
        out.append((base, {k: z[k] for k in z.files}))
    return out


class RawWriter:
    """Parquet files of about mb_per_file MB of text each: raw/<bucket>/<split>/<bucket>-<split>-00000.parquet."""

    SCHEMA = pa.schema([("text", pa.large_string()), ("n_tokens", pa.int32()), ("source", pa.string()),
                        ("source_file", pa.string()), ("source_row", pa.int64()), ("meta", pa.large_string())])

    def __init__(self, folder, stem, mb_per_file):
        self.folder, self.stem, self.limit = folder, stem, mb_per_file * 1e6
        os.makedirs(folder, exist_ok=True)
        for old in glob.glob(os.path.join(folder, "*.parquet")):
            os.remove(old)
        self.n_files, self.w, self.size, self.rows = 0, None, 0, 0
        self.files = []

    def write(self, table, n_bytes):
        if self.w is None or self.size >= self.limit:
            self.close()
            path = os.path.join(self.folder, f"{self.stem}-{self.n_files:05d}.parquet")
            self.w = pq.ParquetWriter(path, self.SCHEMA, compression="zstd", compression_level=6)
            self.files.append(path)
            self.n_files += 1
            self.size = 0
        self.w.write_table(table, row_group_size=10_000)
        self.size += n_bytes
        self.rows += table.num_rows

    def close(self):
        if self.w is not None:
            self.w.close()
            self.w = None


def data_path(cfg, repo, *rest):
    """Where fetch.sh puts a repo file, relative to the ExeaLM checkout: data/<repo name>/<release>/..."""
    parts = ["data", cfg["repos"][repo]["repo_id"].split("/")[-1]] + ([cfg["release"]] if cfg.get("release") else [])
    return "/".join(parts + list(rest))


def tokenizer_ref(cfg):
    """meta.json's tokenizer: where fetch.sh puts it (exealm.data.decontam loads it from there)."""
    return data_path(cfg, "pretrain", "tokenizer")


def source_name(p):
    if p["source"] == "llmjp":
        return f"llm-jp-corpus-v4/{p['path']}"
    if p["source"] == "urls":
        return f"{p['repo']}:{p['list_file']}:{p['contains']}"
    return f"{p['repo']}/{p.get('label') or ','.join(p.get('patterns') or [])}"


def assemble_bucket(cfg, work, bname, tok_sha, vocab_size, write_raw=False):
    from .common import Paths
    paths = Paths(work)
    b = next(x for x in cfg["buckets"] if x["name"] == bname)
    repo = cfg["repos"][b["repo"]]
    repo_name = repo["repo_id"].split("/")[-1]
    root = paths.repo_dir(cfg, b["repo"])
    long_tokens = cfg["split"]["long_tokens"]
    t0 = time.time()

    # which training documents survive the trim, per part
    plans = []
    for i, p in enumerate(b["parts"]):
        pid = config.part_id(b, i)
        a, meta = load_index(paths, pid)
        loc = (a["fileno"].astype(np.int64) << 40) | a["row"]
        o = np.argsort(loc, kind="stable")
        loc_sorted, key_sorted = loc[o], a["key"][o]
        name_to_file = {n: fi for fi, n in enumerate(meta["files"])}
        recs = []
        for base, z in chunks_of(paths, pid):
            fi = name_to_file[os.path.basename(base).rsplit(".", 2)[0]]        # <extract name>.r<round>.<chunk>
            keys = key_sorted[np.searchsorted(loc_sorted, (np.int64(fi) << 40) | z["row"])]
            recs.append((base, fi, z, keys))
        target = config.unique_target(b) * p["share"]
        # training documents that survive the trim: the first ones in the part's document order (file, key, row)
        # until the target is reached (the one that crosses it is kept)
        if recs:
            fno = np.concatenate([np.full(len(z["row"]), fi, dtype=np.int64) for _, fi, z, _ in recs])
            rows = np.concatenate([z["row"] for _, _, z, _ in recs])
            ks = np.concatenate([keys for _, _, _, keys in recs])
            nts = np.concatenate([z["n_tokens"] for _, _, z, _ in recs])
            ok = np.concatenate([(z["status"] == TRAIN) & (z["bench"] < 0) for _, _, z, _ in recs])
            idx = np.flatnonzero(ok)
            idx = idx[np.lexsort((rows[idx], ks[idx], fno[idx]))]
            csum = np.cumsum(nts[idx])
            k = min(len(idx), int(np.searchsorted(csum, target)) + 1)
            keep = np.zeros(len(ok), dtype=bool)
            keep[idx[:k]] = True
            bounds = np.cumsum([0] + [len(z["row"]) for _, _, z, _ in recs])
            recs = [(base, fi, z, keys, keep[bounds[r]:bounds[r + 1]]) for r, (base, fi, z, keys) in enumerate(recs)]
        plans.append(dict(pid=pid, part=p, meta=meta, recs=recs, target=target))

    stats = dict(bucket=bname, repo=repo["repo_id"], splits={}, parts={})
    for split, code in SPLITS.items():
        pdir = os.path.join(root, "packed", bname, split)
        os.makedirs(pdir, exist_ok=True)
        raw = RawWriter(os.path.join(root, "raw", bname, split), f"{bname}-{split}", cfg["raw"]["mb_per_file"]) \
            if write_raw else None
        offsets, n_tok, n_bytes, n_long = [], 0, 0, 0
        files_used = []
        dropped = {}
        trimmed_docs = trimmed_tokens = 0
        tmp = os.path.join(pdir, "tokens.bin.tmp")
        with open(tmp, "wb") as fo:
            for plan in plans:
                p, meta = plan["part"], plan["meta"]
                per_file = {}
                for base, fi, z, keys, in_target in plan["recs"]:
                    kept = z["bench"] < 0
                    pos = np.concatenate([[0], np.cumsum(z["n_tokens"][kept])]).astype(np.int64)
                    where = np.full(len(kept), -1, dtype=np.int64)
                    where[kept] = np.arange(int(kept.sum()))
                    want = (z["status"] == code) & kept
                    if code == TRAIN:
                        over = want & ~in_target
                        trimmed_docs += int(over.sum())
                        trimmed_tokens += int(z["n_tokens"][over].sum())
                        want &= ~over
                    for bi in z["bench"][(z["status"] == code) & ~kept]:
                        name = cfg["decontam"]["benchmarks"][int(bi)]
                        dropped[name] = dropped.get(name, 0) + 1
                    idx = np.flatnonzero(want)
                    if not len(idx):
                        continue
                    toks = np.memmap(base + ".bin", dtype=np.uint16, mode="r") if os.path.getsize(base + ".bin") \
                        else np.empty(0, np.uint16)
                    # copy runs of consecutive documents at once
                    w = where[idx]
                    breaks = np.flatnonzero(np.diff(w) != 1) + 1
                    for run in np.split(np.arange(len(idx)), breaks):
                        s, e = pos[w[run[0]]], pos[w[run[-1]] + 1]
                        fo.write(np.asarray(toks[s:e]).tobytes())
                    lens = z["n_tokens"][idx]
                    offsets.append(n_tok + np.concatenate([[0], np.cumsum(lens)[:-1]]))
                    n_tok += int(lens.sum())
                    n_bytes += int(z["n_bytes"][idx].sum())
                    n_long += int((lens - 1 >= long_tokens).sum())
                    rows = z["row"][idx]
                    src_file = meta["files"][fi].split("__", 1)[1]
                    prov = read_json(os.path.join(paths.extract_dir(plan["pid"]), meta["files"][fi] + ".json"))
                    if raw is not None:
                        t = read_rows(os.path.join(paths.extract_dir(plan["pid"]), meta["files"][fi] + ".parquet"),
                                      rows, ["text", "source_row", "meta"])
                        table = pa.table({
                            "text": t.column("text"),
                            "n_tokens": pa.array(lens.astype(np.int32)),
                            "source": pa.array([source_name(p)] * len(idx)),
                            "source_file": pa.array([prov["provenance"]["path"]] * len(idx)),
                            "source_row": t.column("source_row"),
                            "meta": t.column("meta"),
                        }, schema=RawWriter.SCHEMA)
                        raw.write(table, int(z["n_bytes"][idx].sum()))
                    pv = prov["provenance"]
                    f = per_file.setdefault(src_file, dict(path=pv["path"], repo=pv.get("repo") or pv.get("project"),
                                                           revision=pv.get("revision"), size=pv.get("size"),
                                                           sha256=pv.get("sha256"), docs=0, tokens=0))
                    f["docs"] += len(idx)
                    f["tokens"] += int(lens.sum())
                files_used += [dict(part=plan["pid"], **v) for v in per_file.values()]
        if raw is not None:
            raw.close()
        os.replace(tmp, os.path.join(pdir, "tokens.bin"))
        off = np.concatenate(offsets).astype(np.int64) if offsets else np.empty(0, np.int64)
        np.save(os.path.join(pdir, "offsets.npy"), off)
        pmeta = dict(dtype="uint16", vocab_size=vocab_size, n_tokens=n_tok, n_docs=int(len(off)),
                     bytes_per_token=n_bytes / max(1, n_tok), tokenizer=tokenizer_ref(cfg),
                     tokenizer_sha256=tok_sha, files=[f["path"] for f in files_used], dataset=repo["repo_id"],
                     bucket=bname, split=split, license=b["license"], long_docs=n_long, long_tokens=long_tokens,
                     sources=[source_name(p) for p in b["parts"]])
        write_json(os.path.join(pdir, "meta.json"), pmeta)
        if raw is not None and raw.rows != len(off):
            raise RuntimeError(f"{bname}/{split}: {raw.rows} raw rows but {len(off)} packed documents")
        stats["splits"][split] = dict(docs=int(len(off)), tokens=n_tok, text_bytes=n_bytes,
                                      bytes_per_token=round(n_bytes / max(1, n_tok), 4), long_docs=n_long,
                                      contaminated_dropped=dropped, files=files_used,
                                      raw_files=[os.path.relpath(f, root) for f in raw.files] if raw else [],
                                      packed=os.path.relpath(pdir, root),
                                      packed_bytes=os.path.getsize(os.path.join(pdir, "tokens.bin")))
        if split == "train":
            stats["splits"][split].update(target=config.unique_target(b), epochs=b.get("epochs", 1.0),
                                          trimmed_docs=trimmed_docs, trimmed_tokens=trimmed_tokens,
                                          short=max(0, config.unique_target(b) - n_tok))
        log.info(f"{bname}/{split}: {human(len(off))} documents, {human(n_tok)} tokens"
                 + (f" (target {human(b['target'])})" if split == "train" else f", {n_long} long documents"))
    stats["seconds"] = round(time.time() - t0, 1)
    stats["repo_name"] = repo_name
    write_json(paths.p("reports", "assemble", f"{bname}.json"), stats)
    return stats


def mix_json(cfg, paths, split, include, anneal=False):
    """open_data's mix format, with paths as fetch.sh lays the repos out under the ExeaLM checkout. Weights are the
    targets in billions of tokens, so each source is read at its recipe share; the anneal mix multiplies each by its
    anneal_weight (the recipe's reweighting for the last 10% of steps)."""
    sources = {}
    for b in config.buckets(cfg, work=paths.work):
        if b["repo"] not in include:
            continue
        w = b["target"] / 1e9 * (b.get("anneal_weight", 1.0) if anneal else 1.0)
        sources[b["name"]] = dict(paths=data_path(cfg, b["repo"], "packed", b["name"], split), weight=round(w, 9))
    out = dict(sources=sources)
    if anneal:
        out["note"] = (f"for the last {cfg.get('recipe', {}).get('anneal', {}).get('share_of_steps', 0.1):.0%} of "
                       "training steps: weights are share x anneal_weight")
    return out


def write_mixes(cfg, paths):
    """In the pretraining repo: mix.json, mix_val.json and mix_anneal.json, with The Stack's repo as one more source,
    and mix_public*.json without it (for readers without access to the gated repo). In the post-training repo:
    mix.json and mix_val.json."""
    out = {}
    for repo in ("pretrain", "post"):
        if not config.buckets(cfg, repo=repo, work=paths.work):
            continue
        root = paths.repo_dir(cfg, repo)
        os.makedirs(root, exist_ok=True)
        include = [r for r in ("pretrain", "pretrain_stack") if config.buckets(cfg, repo=r, work=paths.work)] \
            if repo == "pretrain" else [repo]
        kinds = [("train", "", False), ("val", "_val", False)] + ([("train", "_anneal", True)] if repo == "pretrain" else [])
        for split, suffix, anneal in kinds:
            write_json(os.path.join(root, f"mix{suffix}.json"), mix_json(cfg, paths, split, include, anneal))
            out[f"{repo}/mix{suffix}.json"] = include
            if repo == "pretrain" and "pretrain_stack" in include:
                write_json(os.path.join(root, f"mix_public{suffix}.json"),
                           mix_json(cfg, paths, split, ["pretrain"], anneal))
    return out


def write_sources(cfg, paths, results):
    """sources.json in each repo folder: every source file used, per bucket (repo, commit, path, size, SHA-256 when
    the host gives one, documents and tokens used), and the build settings. With packed tokens and no raw text, this
    is what says exactly where every document came from."""
    for repo in ("pretrain", "pretrain_stack", "post"):
        bs = [b for b in config.buckets(cfg, repo=repo, work=paths.work) if b["name"] in results]
        if not bs:
            continue
        out = dict(recipe=cfg.get("recipe", {}).get("name"), build=cfg.get("build"), release=cfg.get("release"),
                   manifest=cfg.get("manifest"), buckets={})
        for b in bs:
            r = results[b["name"]]
            files = {}
            for split in ("train", "val"):
                for f in r["splits"][split]["files"]:
                    g = files.setdefault(f["path"], {k: v for k, v in f.items() if k not in ("docs", "tokens", "part")})
                    g[f"{split}_docs"] = g.get(f"{split}_docs", 0) + f["docs"]
                    g[f"{split}_tokens"] = g.get(f"{split}_tokens", 0) + f["tokens"]
            out["buckets"][b["name"]] = dict(source=b.get("source"), license=b["license"],
                                             tokens=r["splits"]["train"]["tokens"], target=b["target"],
                                             epochs=b.get("epochs", 1.0), files=list(files.values()))
        write_json(os.path.join(paths.repo_dir(cfg, repo), "sources.json"), out)


def run(cfg, paths, args):
    from .tok import file_digest, tokenizer_file
    tok_file = tokenizer_file(cfg, paths)
    from exealm.data.tokenizer import load
    vocab = load(tok_file).get_vocab_size()
    tok_sha = file_digest(tok_file)
    names = [b["name"] for b in config.buckets(cfg, args.buckets, work=paths.work)]
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max(1, min(args.workers, len(names)))) as pool:
        futs = {pool.submit(assemble_bucket, cfg, paths.work, n, tok_sha, vocab, args.raw): n for n in names}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    write_mixes(cfg, paths)
    write_sources(cfg, paths, results)
    dst = os.path.join(paths.repo_dir(cfg, "pretrain"), "tokenizer")          # the tokenizer ships with the packs
    os.makedirs(dst, exist_ok=True)
    shutil.copy(tok_file, os.path.join(dst, "tokenizer.json"))
    tot = sum(r["splits"]["train"]["tokens"] for r in results.values())
    log.info(f"assemble: {len(results)} buckets, {human(tot)} training tokens in {time.time() - t0:.0f} s")
    short = {n: r["splits"]["train"]["short"] for n, r in results.items() if r["splits"]["train"]["short"] > 0}
    if short:
        log.warning("buckets short of their target (tokens): " + ", ".join(f"{k} {human(v)}" for k, v in short.items()))
    return results


def disk_usage(folder):
    total = 0
    for d, _, fs in os.walk(folder):
        for f in fs:
            total += os.path.getsize(os.path.join(d, f))
    return total


__all__ = ["run", "assemble_bucket", "write_mixes", "mix_json", "disk_usage", "gb"]
