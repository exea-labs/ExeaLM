"""Step 1, fetch: download only as many source files as each part needs, and extract the documents its filter keeps.

Each part's files are put once in an order fixed by the seed (files already on this machine first, when --reuse
points at them) and the order is saved in <work>/manifest/order/. A build reads each part's files from the start of
that order until the text kept covers the part's share of (target + validation) tokens times (1 + margin), at the
bucket's guessed bytes per token. A larger build later reads further down the same order, so a small build (the first
slice) is the start of every larger one.

Each file also gets a position: the tokens, at full scale, that its part has before it, divided by the part's
full-scale target. Positions interleave the parts as a build of any size reaches them, and dedup keeps the copy with
the smallest position, so a file's documents never lose to a file that only a larger build reads.

Parts run in parallel threads, extraction in a process pool. A finished file is never downloaded again. Downloads
are deleted once extracted (--keep_raw keeps them); files reused from --reuse folders are never deleted.
"""
import glob
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from . import config
from .common import gb, log, read_json, slug, write_json
from .extract import extract
from .sources import HfSource, LlmjpSource, UrlSource

TEXT_PER_FILE_BYTE = {".parquet": 2.8, ".gz": 3.2, ".zst": 3.0, ".zstd": 3.0}   # guesses, for planning only


def text_ratio(path):
    for ext, r in TEXT_PER_FILE_BYTE.items():
        if path.endswith(ext):
            return r
    return 3.0


def extracted(paths, pid):
    """Finished extracts of a part, in file order: their .json contents."""
    return [read_json(j) for j in sorted(glob.glob(os.path.join(paths.extract_dir(pid), "*.json")))]


def needed_bytes(cfg, b, p, margin):
    s = cfg["split"]
    tokens = config.unique_target(b) * (1 + s.get("val_fraction", 0)) * p["share"] + s.get("val_tokens", 0) * p["share"]
    return tokens * b["bpt"] * (1 + margin)


class Sources:
    def __init__(self, cfg, paths, args):
        reuse = list(args.reuse or [])
        self.hf = HfSource(paths.manifest, paths.p("raw", "hf"), local=args.local_hf, reuse=reuse)
        self.urls = UrlSource(self.hf, paths.p("raw", "urls"), local=args.local_hf)
        self.lj = LlmjpSource(cfg, paths.manifest, paths.p("raw", "llmjp"), local=args.local_llmjp)
        self.local = bool(args.local_hf or args.local_llmjp)

    def listing(self, p):
        if p["source"] == "manifest":
            return [dict(f, sha256=None) for f in p["files"]]
        if p["source"] == "hf":
            return [dict(f) for f in self.hf.list(p["repo"], p["patterns"])]
        if p["source"] == "urls":
            return [dict(f) for f in self.urls.list(p["repo"], p["list_file"], p["contains"])]
        return [dict(path=f, size=None) for f in self.lj.list(p["path"])]

    def size(self, p, f):
        if p["source"] == "manifest":
            return f.get("size") or 0
        if f.get("size") is None:
            f["size"] = self.urls.size(f) if p["source"] == "urls" else self.lj.info(f["path"])["size"]
        return f["size"]

    def manifest_revision(self, repo):
        """The repo's current commit, recorded once (download_slice.py read main); "unresolved" when offline."""
        from .sources import LOCK
        path = os.path.join(self.hf.manifest_dir, "hf_refs.json")
        with LOCK:                                                      # parallel parts share this file
            refs = read_json(path, {})
            if repo not in refs:
                try:
                    from huggingface_hub import HfApi
                    sha = "local" if self.local else HfApi().dataset_info(repo).sha
                except Exception:
                    return "unresolved"
                refs = read_json(path, {})
                refs[repo] = sha
                write_json(path, refs)
            return refs[repo]

    def is_local(self, p, f):
        if p["source"] == "manifest":
            return True
        return p["source"] == "hf" and self.hf.reused(p["repo"], f["path"], f.get("size")) is not None

    def fetch(self, p, f):
        """(local path, provenance, whether we downloaded it)."""
        if p["source"] == "manifest":
            local = os.path.join(p["root"], p["repo"].split("/")[-1], f["path"])
            if not os.path.exists(local) or (f.get("size") and os.path.getsize(local) != f["size"]):
                raise Missing(local)
            rev = self.manifest_revision(p["repo"])
            return local, dict(source="hf", repo=p["repo"], revision=rev, path=f["path"], size=os.path.getsize(local),
                               sha256=None, url=f"https://huggingface.co/datasets/{p['repo']}/blob/{rev}/{f['path']}",
                               local=local), False
        if p["source"] == "hf":
            local, dl = self.hf.fetch(p["repo"], f["path"], f["size"], f["sha256"])
            return local, self.hf.provenance(p["repo"], f), dl
        if p["source"] == "urls":
            local, dl = self.urls.fetch(f)
            return local, self.urls.provenance(p["repo"], p["list_file"], f, local), dl
        local, dl = self.lj.fetch(f["path"], attempts=4 if p.get("optional") else 10)
        return local, self.lj.provenance(f["path"]), dl


class Missing(Exception):
    """A file the slice manifest lists is not on disk (yet)."""


def file_order(cfg, paths, srcs, b, p, pid, prefer_local):
    """The part's files in their fixed order, each with its rank and position. Computed once, then read back. A
    manifest part keeps the manifest's order (download_slice.py's fixed shuffle)."""
    path = os.path.join(paths.manifest, "order", f"{pid}.json")
    saved = read_json(path)
    if saved and (p["source"] != "manifest" or len(saved["files"]) >= len(p["files"])):
        return saved["files"]
    files = srcs.listing(p)
    if p["source"] != "manifest":
        random.Random(f"{cfg['seed']}:{pid}").shuffle(files)
    local = [f for f in files if prefer_local and srcs.is_local(p, f)]
    if local:
        log.info(f"{pid}: {len(local)} of {len(files)} files are already here: they come first")
        files = local + [f for f in files if f not in local]
    full = max(1.0, b["full_target"] * p["share"])
    acc = 0.0
    sized = 0
    for k, f in enumerate(files):
        f["rank"], f["position"] = k, acc / full
        if f.get("size") is None and sized < 64:              # GitLab and URL sizes cost a request each: the first few
            srcs.size(p, f)
            sized += 1
        acc += (f.get("size") or 0) * text_ratio(f["path"]) / b["bpt"]
    write_json(path, dict(part=pid, seed=cfg["seed"], prefer_local=bool(local), files=files))
    return files


def fetch_part(cfg, paths, b, p, pid, srcs, pool, args):
    margin = cfg["fetch"]["margin"] if args.margin is None else args.margin
    need = needed_bytes(cfg, b, p, margin)
    min_files = max(cfg["fetch"].get("min_files", 1), cfg["split"].get("val_files", 1))
    files = file_order(cfg, paths, srcs, b, p, pid, prefer_local=bool(args.reuse))
    have, used, planned_download, waiting = 0, [], 0, None
    out_dir = paths.extract_dir(pid)
    # files already on this machine are extracted several at a time; downloads go one at a time
    batch = max(1, args.part_parallel) if (p["source"] == "manifest" or srcs.local or args.reuse) else 1
    k = 0
    while k < len(files):
        if (args.max_files and k >= args.max_files) or (have >= need and k >= min_files):
            break
        group = list(enumerate(files))[k:k + batch]
        if args.max_files:
            group = [(i, f) for i, f in group if i < args.max_files]
        infos, futs = {}, {}
        for i, f in group:
            dst = os.path.join(out_dir, f"{i:05d}__{slug(f['path'])}.parquet")
            info = read_json(dst[: -len(".parquet")] + ".json")
            if info is not None:
                infos[i] = info
                continue
            if args.dry_run:
                size = srcs.size(p, f)
                local = srcs.is_local(p, f)
                planned_download += 0 if local else size
                infos[i] = dict(estimated=True, size=size, reused=local, text_bytes_kept=size * text_ratio(f["path"]))
                continue
            srcs.size(p, f)
            try:
                local, prov, downloaded = srcs.fetch(p, f)
            except Missing as e:
                log.warning(f"{pid}: {e} is not downloaded yet: stopping this source here (rerun fetch later)")
                waiting = str(e)
                break
            log.info(f"{pid}: extracting {f['path']}")
            futs[i] = (pool.submit(extract, local, dst, prov, p, cfg["seed"], pid, f["position"], i), local, downloaded)
        for i, (fut, local, downloaded) in futs.items():
            infos[i] = fut.result()
            if downloaded and not args.keep_raw and not srcs.local and os.path.exists(local):
                os.remove(local)
            log.info(f"{pid}: {files[i]['path']}: kept {infos[i]['rows_kept']} of {infos[i]['rows_read']} documents, "
                     f"{gb(infos[i]['text_bytes_kept'])} of text in {infos[i]['seconds']} s")
        for i, f in group:
            if i not in infos:                       # after a missing file: counted on the next run
                break
            info = infos[i]
            have += info["text_bytes_kept"]
            if info.get("estimated"):
                used.append(dict(path=f["path"], size=info["size"], estimated=True, reused=info["reused"]))
            else:
                used.append(dict(path=f["path"], size=info["provenance"]["size"], kept_bytes=info["text_bytes_kept"]))
            k = i + 1
            if have >= need and k >= min_files:
                break
        if waiting:
            break
    rec = dict(part=pid, bucket=b["name"], needed_text_bytes=int(need), text_bytes=int(have), files=used,
               enough=have >= need, files_available=len(files), waiting_for=waiting)
    if args.dry_run:
        rec["download_bytes"] = planned_download
    level = log.info if rec["enough"] else log.warning
    level(f"{pid}: {len(used)} of {len(files)} files, {gb(have)} of text for {gb(need)} needed"
          + ("" if rec["enough"] else "  NOT ENOUGH: every file is used" if len(used) == len(files) else ""))
    return rec


def run(cfg, paths, args):
    srcs = Sources(cfg, paths, args)
    jobs = list(config.parts(cfg, args.buckets))
    t0 = time.time()
    results, skipped = [], {}
    with ProcessPoolExecutor(max(1, args.workers)) as pool, ThreadPoolExecutor(max(1, args.download_workers)) as th:
        futs = {th.submit(fetch_part, cfg, paths, b, p, pid, srcs, pool, args): (b, p, pid) for b, _, p, pid in jobs}
        for fut in as_completed(futs):
            b, p, pid = futs[fut]
            try:
                results.append(fut.result())
            except (Exception, SystemExit) as e:                # one failing part must not stop the others
                msg = f"{type(e).__name__}: {e}"
                if b.get("optional"):
                    log.warning(f"{pid}: optional bucket {b['name']} skipped: {msg}")
                    skipped[b["name"]] = msg
                else:
                    log.error(f"{pid}: FAILED: {msg}")
                results.append(dict(part=pid, bucket=b["name"], error=msg, optional=bool(b.get("optional")),
                                    enough=False, files=[], text_bytes=0, download_bytes=0))
    order = [pid for _, _, _, pid in jobs]
    results.sort(key=lambda r: order.index(r["part"]))
    report = dict(dry_run=bool(args.dry_run), seconds=round(time.time() - t0, 1), build=cfg.get("build"),
                  parts=results, skipped=skipped)
    if args.dry_run:
        report["download_bytes"] = sum(r.get("download_bytes", 0) for r in results)
        n = sum(len([u for u in r["files"] if u.get("estimated") and not u.get("reused")]) for r in results)
        log.info(f"dry run: would download {gb(report['download_bytes'])} in {n} files")
        write_json(paths.report("fetch_plan"), report)
    else:
        write_json(paths.report("fetch"), report)
    short = [r["part"] for r in results if not r["enough"] and not r.get("error")]
    if short:
        log.warning(f"parts with less text than needed: {short}")
    waiting = [r["part"] for r in results if r.get("waiting_for") and not r["enough"]]
    if waiting:
        log.warning(f"parts waiting for files still downloading: {waiting}: rerun fetch when the download is done")
    failed = [r["part"] for r in results if r.get("error") and not r.get("optional")]
    if failed:
        raise SystemExit(f"fetch failed for {failed}: rerun the same command to retry (finished files are kept)")
    return report
