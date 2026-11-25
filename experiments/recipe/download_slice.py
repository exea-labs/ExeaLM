"""Download the raw files for a slice of the data recipe from Hugging Face (experiments/recipe/exealm_recipe_v2.json).

    python experiments/recipe/download_slice.py --tokens 30e9 --out /work/data_raw     # the first slice, ~30B tokens

Each source gets its share of --tokens. Files are taken in a fixed shuffled order (seed 0), so a bigger slice later
starts with the same files and only adds new ones; files already on disk are skipped. Sizes are estimated from the
compressed file size (BYTES_PER_TOKEN). Sources that need filtering (research_python) are fetched at OVERSAMPLE times
their share. Gated or optional sources, and ones without a Hugging Face layout here, are skipped and listed.
Writes <out>/manifest_slice.json: every file per source, with its size.
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download

BYTES_PER_TOKEN = 1.2                    # compressed bytes per token, roughly, for json.gz, jsonl.zst and parquet
OVERSAMPLE = {"research_python": 8.0, "starcoder": 1.0}
# Where each source's files live: repo, folder prefixes, file suffixes
LAYOUT = {
    "research_python": ("allenai/olmo-mix-1124", ["data/starcoder/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "starcoder": ("allenai/olmo-mix-1124", ["data/starcoder/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "code_reasoning": ("IFM/Code-Reasoning", ["code-no-thinking/", "code-no-thinking-hard/", "code-task-synthesis/", "codeio/"], (".parquet",)),
    "algebraic_stack": ("allenai/olmo-mix-1124", ["data/algebraic-stack/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "pretrain_behaviors": ("IFM/Pretrain-Behaviors", ["data-science/", "planning/", "reasoning/"], (".parquet",)),
    "stackexchange": ("allenai/dolmino-mix-1124", ["data/stackexchange/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "pes2o": ("allenai/peS2o", ["data/v2/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "arxiv": ("allenai/olmo-mix-1124", ["data/arxiv/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "wikipedia_en": ("allenai/dolmino-mix-1124", ["data/wiki/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "megamath": ("IFM/MegaMath", ["megamath-web/"], (".parquet",)),   # web-pro, synthetic and QA parts: rewritten by Llama 3.3
    "open_web_math": ("allenai/olmo-mix-1124", ["data/open-web-math/"], (".json.gz", ".jsonl.gz", ".zst", ".parquet")),
    "math_reasoning": ("IFM/Math-Reasoning", ["math-rewrite/", "math-dialogue/"], (".parquet",)),
    "dclm": ("mlfoundations/dclm-baseline-1.0", ["global-shard_01_of_10/"], (".zst", ".jsonl", ".parquet")),
    "txt360_web": ("IFM/TxT360-v2", ["web-high-medium/"], (".parquet",)),
    "fineweb2_ja": ("HuggingFaceFW/fineweb-2", ["data/jpn_Jpan/train/"], (".parquet",)),
    "wikipedia_ja": ("wikimedia/wikipedia", ["20231101.ja/"], (".parquet",)),
    "txt360_qa": ("IFM/TxT360-v2", ["txt360-qa/", "web-high-nltk-qa/"], (".parquet",)),
    "sft_pretrain": ("IFM/SFT-Reasoning", ["3efforts-pretrain/"], (".parquet",)),
}


def files_of(api, repo, prefixes, suffixes):
    out = []
    for pre in prefixes:
        for f in api.list_repo_tree(repo, path_in_repo=pre.rstrip("/"), recursive=True, repo_type="dataset"):
            if getattr(f, "size", None) and f.path.endswith(suffixes):
                out.append((f.path, f.size))
    random.Random(0).shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="experiments/recipe/exealm_recipe_v2.json")
    ap.add_argument("--tokens", type=float, default=30e9)
    ap.add_argument("--out", default="/work/data_raw")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    recipe = json.load(open(a.recipe))
    total = sum(s["tokens_b"] for s in recipe["pretrain"])
    api = HfApi()
    plan, skipped = [], []
    manifest = {}
    for s in recipe["pretrain"]:
        name = s["name"]
        if s.get("gated") or s.get("optional") or name not in LAYOUT:
            skipped.append(name)
            continue
        repo, prefixes, suffixes = LAYOUT[name]
        want = a.tokens * s["tokens_b"] / total * BYTES_PER_TOKEN * OVERSAMPLE.get(name, 1.0)
        got, chosen = 0, []
        for path, size in files_of(api, repo, prefixes, suffixes):
            if got >= want:
                break
            chosen.append((path, size))
            got += size
        manifest[name] = dict(repo=repo, files=[dict(path=p, bytes=b) for p, b in chosen], target_bytes=int(want),
                              bytes=got)
        plan += [(repo, p) for p, _ in chosen]
        print(f"{name}: {len(chosen)} files, {got / 1e9:.1f} GB of {want / 1e9:.1f} GB wanted", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(tokens=a.tokens, skipped=skipped, sources=manifest), open(os.path.join(a.out, "manifest_slice.json"), "w"), indent=1)
    print("skipped:", skipped, flush=True)

    def get(item):
        repo, path = item
        local = os.path.join(a.out, repo.split("/")[1], path)
        if os.path.exists(local):
            return
        hf_hub_download(repo, path, repo_type="dataset", local_dir=os.path.join(a.out, repo.split("/")[1]))
        print("got", repo, path, flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(get, sorted(set(plan))))
    print("done", flush=True)


if __name__ == "__main__":
    main()
