"""Upload the repo folders that the pipeline made (<work>/repos/<name>) to Hugging Face, under exea-labs.

    python -m experiments.data_mix.upload --work /data/mix --dry-run          # print what would be uploaded
    python -m experiments.data_mix.upload --work /data/mix --repos pretrain   # upload (the box is logged in)

The repos already exist (exea-labs created them); this never creates or reconfigures one. It refuses a repo that
does not exist, and refuses to send The Stack's data to a repo that is not gated. With a release (build.json's
"release", e.g. slice-30b) only that folder is uploaded, as <release>/ in the repo, so the repo's own card and other
releases are left alone. upload_large_folder uploads in resumable chunks and keeps its state in
<work>/repos/<name>/.cache/huggingface: if it stops, run the same command again.
"""
import argparse
import os

from . import config
from .common import Paths, gb, log, read_json, setup_logging


def files_of(folder):
    out = []
    for d, dirs, fs in os.walk(folder):
        dirs[:] = sorted(x for x in dirs if x != ".cache")
        for f in sorted(fs):
            full = os.path.join(d, f)
            if f.endswith(".tmp") or f.endswith(".part"):
                continue
            out.append((os.path.relpath(full, folder).replace(os.sep, "/"), os.path.getsize(full)))
    return out


def check_folder(cfg, repo, folder, paths=None):
    """Refuse a folder that is not finished: no card, or a bucket with no pack."""
    problems = []
    if not os.path.exists(os.path.join(folder, "README.md")):
        problems.append("no README.md (run the cards step)")
    for b in config.buckets(cfg, repo=repo, work=paths.work if paths else None):
        for split in ("train", "val"):
            if not os.path.exists(os.path.join(folder, "packed", b["name"], split, "meta.json")):
                problems.append(f"no pack for {b['name']}/{split} (run assemble)")
    if cfg["repos"][repo].get("tokenizer") and not os.path.exists(os.path.join(folder, "tokenizer", "tokenizer.json")):
        problems.append("no tokenizer/tokenizer.json")
    if cfg["repos"][repo].get("gated") and os.path.exists(os.path.join(folder, "README.md")):
        card = open(os.path.join(folder, "README.md"), encoding="utf-8").read()
        if "could not be fetched" in card or "extra_gated_prompt" not in card:
            problems.append("the card lacks the Terms of Use for The Stack: rerun cards with network access")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--work", default="data/data_mix")
    ap.add_argument("--config", default=None)
    ap.add_argument("--repos", nargs="*", default=None, help="config repo keys: pretrain pretrain_stack post library")
    ap.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true",
                    help="list the files and the actions; touch nothing on the Hub")
    ap.add_argument("--num_workers", type=int, default=None, help="upload_large_folder workers (default: its own)")
    a = ap.parse_args(argv)
    paths = Paths(a.work)
    setup_logging(paths.work, "upload")
    cfg = config.load(a.config)
    build = read_json(os.path.join(paths.work, "build.json"), {})
    cfg["release"] = build.get("release") or ""
    if build.get("slice_manifest"):
        config.use_manifest(cfg, build["slice_manifest"], build.get("data_root") or os.path.dirname(build["slice_manifest"]))
    config.apply_scale(cfg, build.get("scale") or 1.0, build.get("tokens"), build.get("post_tokens"))
    keys = a.repos or list(cfg["repos"])
    plan = []
    for key in keys:
        if key not in cfg["repos"]:
            raise SystemExit(f"unknown repo {key}; one of {list(cfg['repos'])}")
        r = cfg["repos"][key]
        folder = paths.repo_dir(cfg, key)
        if not os.path.isdir(folder):
            if a.repos:
                raise SystemExit(f"{folder} does not exist: run the pipeline (cards at least) first")
            continue
        problems = check_folder(cfg, key, folder, paths)
        root = paths.p("repos", r["repo_id"].split("/")[-1])            # upload_large_folder's folder: the repo root
        files = [(f"{cfg['release']}/{rel}" if cfg["release"] else rel, size) for rel, size in files_of(folder)]
        plan.append((key, r, root, files, problems))

    for key, r, folder, files, problems in plan:
        total = sum(s for _, s in files)
        print(f"\n{r['repo_id']}  (dataset{', gated: users must accept The Stack terms' if r.get('gated') else ''})")
        print(f"  from {folder}: {len(files)} files, {gb(total)}")
        print(f"  {'would check' if a.dry_run else 'check'}: the repo exists"
              + (" and is gated" if r.get("gated") else "") + " (it is never created or changed here)")
        pat = f", allow_patterns=['{cfg['release']}/**']" if cfg["release"] else ""
        print(f"  {'would upload' if a.dry_run else 'upload'}: upload_large_folder('{r['repo_id']}', '{folder}', "
              f"repo_type='dataset'{pat})")
        for rel, size in files:
            print(f"    {size:>15,d}  {rel}")
        for p in problems:
            print(f"  PROBLEM: {p}")
    bad = [k for k, *_, p in plan if p]
    if a.dry_run:
        print(f"\ndry run: nothing was uploaded. {'Fix the problems above first: ' + str(bad) if bad else ''}")
        return plan
    if bad:
        raise SystemExit(f"not uploading: unfinished repo folders {bad} (see PROBLEM lines)")

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    log.info(f"logged in as {who.get('name')}")
    for key, r, folder, files, _ in plan:
        if not api.repo_exists(r["repo_id"], repo_type="dataset"):
            raise SystemExit(f"{r['repo_id']} does not exist: it should have been created already; not creating it")
        if r.get("gated") and not api.dataset_info(r["repo_id"]).gated:
            raise SystemExit(f"{r['repo_id']} is not gated: refusing to upload The Stack's files to it")
        log.info(f"{r['repo_id']}: uploading {len(files)} files, {gb(sum(s for _, s in files))}")
        kw = dict(repo_type="dataset")
        if a.num_workers:
            kw["num_workers"] = a.num_workers
        if cfg["release"]:
            kw["allow_patterns"] = [f"{cfg['release']}/**"]
        api.upload_large_folder(r["repo_id"], folder, **kw)
        log.info(f"{r['repo_id']}: done: https://huggingface.co/datasets/{r['repo_id']}")
    return plan


if __name__ == "__main__":
    main()
