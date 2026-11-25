"""Make the scaling-study datasets: run the steps in order. Every step can be stopped and rerun; it continues.

    python -m experiments.data_mix.pipeline fetch      --work /data/mix    # download, filter, extract
    python -m experiments.data_mix.pipeline dedup      --work /data/mix    # exact duplicates; validation split
    python -m experiments.data_mix.pipeline tokenizer  --work /data/mix    # the 32k BPE, bytes per token
    python -m experiments.data_mix.pipeline benchmarks --work /data/mix    # held-out benchmark windows
    python -m experiments.data_mix.pipeline tokenize   --work /data/mix    # select training docs, tokenize, decontaminate
    python -m experiments.data_mix.pipeline assemble   --work /data/mix    # packs, raw Parquet, mix .json
    python -m experiments.data_mix.pipeline cards      --work /data/mix    # README dataset cards
    python -m experiments.data_mix.pipeline verify     --work /data/mix    # open and check everything
    python -m experiments.data_mix.pipeline all        --work /data/mix    # all of the above, in order

Size: --scale s (0.5 to 1.5) multiplies every recipe target (200B at s = 1). --tokens T builds only the first T
pretraining tokens of that, every source at its share; run again with a larger T to extend the same build. Both are
saved in <work>/build.json, so later steps need not repeat them.

The repo folders to upload end up in <work>/repos/. experiments/data_mix/README.md has times and disk sizes.
"""
import argparse
import os
import sys
import time

from . import config
from .common import Paths, log, read_json, setup_logging, write_json

STEPS = ["fetch", "dedup", "tokenizer", "benchmarks", "tokenize", "assemble", "cards", "verify"]


def run_step(step, cfg, paths, args):
    t0 = time.time()
    log.info(f"== {step} ==")
    if step == "fetch":
        from .fetch import run
    elif step == "dedup":
        from .index import run_dedup as run
    elif step == "tokenizer":
        from .tok import run_tokenizer as run
    elif step == "benchmarks":
        from .tok import run_benchmarks as run
    elif step == "tokenize":
        from .tok import run_tokenize as run
    elif step == "assemble":
        from .assemble import run
    elif step == "cards":
        from .cards import run
    elif step == "verify":
        from .verify import run
    else:
        raise SystemExit(f"unknown step {step}")
    out = run(cfg, paths, args)
    log.info(f"== {step} done in {(time.time() - t0) / 60:.1f} min ==")
    return out


def build_settings(paths, args):
    """The build's scale and size: from the command line when given (and then saved), else from <work>/build.json, so
    every step of one build uses the same targets."""
    path = os.path.join(paths.work, "build.json")
    saved = read_json(path, dict(scale=1.0, tokens=None, post_tokens=None, slice_manifest=None, data_root=None,
                                 release=""))
    new = dict(saved)
    for k in ("scale", "tokens", "post_tokens", "slice_manifest", "data_root", "release"):
        if getattr(args, k) is not None:
            new[k] = getattr(args, k)
    if new.get("slice_manifest"):
        new["slice_manifest"] = os.path.abspath(new["slice_manifest"])
    if new != saved or not os.path.exists(path):
        if os.path.exists(path):
            log.info(f"build settings change from {saved} to {new}")
        write_json(path, new)
    return new


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--work", default="data/data_mix", help="work directory (needs ~400 GB for the full run)")
    ap.add_argument("--config", default=None, help="a .json config instead of config.py's CONFIG (tests use this)")
    ap.add_argument("--buckets", nargs="*", default=None, help="only these buckets (fetch, tokenize, assemble, verify)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes (default: every core)")
    ap.add_argument("--scale", type=float, default=None, help="multiply every recipe target (0.5 to 1.5; default 1)")
    ap.add_argument("--tokens", type=float, default=None,
                    help="build only the first slice: this many pretraining tokens (e.g. 24e9), every source at its "
                         "share; a later run with more extends it. Remembered in <work>/build.json")
    ap.add_argument("--post_tokens", type=float, default=None, help="post-training tokens (default: the same fraction)")
    ap.add_argument("--slice_manifest", default=None,
                    help="read sources from the files experiments/recipe/download_slice.py downloaded (its "
                         "manifest_slice.json); sources it lacks are left out. Remembered in <work>/build.json")
    ap.add_argument("--data_root", default=None, help="where those files are (default: the manifest's folder)")
    ap.add_argument("--release", default=None,
                    help="folder inside each repo for this build, e.g. slice-30b (remembered in <work>/build.json)")
    # fetch
    ap.add_argument("--download_workers", type=int, default=6, help="fetch: parts downloaded at the same time")
    ap.add_argument("--margin", type=float, default=None, help="fetch: override config fetch.margin")
    ap.add_argument("--max_files", type=int, default=None, help="fetch: at most this many files per part (testing)")
    ap.add_argument("--keep_raw", action="store_true", help="fetch: keep downloaded files after extracting them")
    ap.add_argument("--part_parallel", type=int, default=8,
                    help="fetch: local files of one source extracted at the same time (default 8)")
    ap.add_argument("--reuse", nargs="*", default=None,
                    help="fetch: folders that already hold source files, read in place (e.g. /work/data_raw)")
    ap.add_argument("--dry_run", action="store_true", help="fetch: list what would be downloaded, download nothing")
    ap.add_argument("--local_hf", default=None, help="read Hugging Face sources from this folder (<repo>/<subset>/)")
    ap.add_argument("--local_llmjp", default=None, help="read LLM-jp sources from this folder (<path>/*.jsonl.gz)")
    # tokenizer
    ap.add_argument("--vocab", type=int, default=None, help="tokenizer: override config tokenizer.vocab_size")
    ap.add_argument("--sample_scale", type=float, default=1.0, help="tokenizer: scale every sample size")
    ap.add_argument("--retrain", action="store_true", help="tokenizer: train a new one even if one exists")
    # benchmarks
    ap.add_argument("--bench_limit", type=int, default=None, help="benchmarks: export only this many rows (smoke)")
    ap.add_argument("--raw", action="store_true", help="assemble: also write the raw text as Parquet beside the packs")
    # tokenize
    ap.add_argument("--chunk_mb", type=float, default=256, help="tokenize: MB of text per job")
    ap.add_argument("--batch", type=int, default=512, help="tokenize: documents per encode_batch call")
    # verify
    ap.add_argument("--decontam", action="store_true", help="verify: rescan every pack with exealm.data.decontam")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    paths = Paths(args.work)
    os.makedirs(paths.work, exist_ok=True)
    setup_logging(paths.work, args.step)
    cfg = config.load(args.config)
    build = build_settings(paths, args)
    cfg["release"] = build.get("release") or ""
    if build.get("slice_manifest"):
        config.use_manifest(cfg, build["slice_manifest"],
                            build.get("data_root") or os.path.dirname(build["slice_manifest"]))
    config.apply_scale(cfg, build["scale"], build["tokens"], build["post_tokens"])
    config.check(cfg)
    b = cfg["build"]
    log.info(f"{' '.join(sys.argv)}  (work {paths.work}, {args.workers} workers)")
    log.info(f"build: scale {b['scale']}, {b['fraction']:.4f} of every pretraining target: "
             f"{b['pretrain_tokens'] / 1e9:.2f}B pretraining, {b['post_tokens'] / 1e9:.2f}B post-training tokens")
    steps = STEPS if args.step == "all" else [args.step]
    for s in steps:
        run_step(s, cfg, paths, args)


if __name__ == "__main__":
    main()
