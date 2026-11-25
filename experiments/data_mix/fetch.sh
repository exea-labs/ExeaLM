#!/bin/bash
# Download the scaling-study training data from Hugging Face into data/ and write mix files with local paths.
# For the SDSC Expanse login node (or any machine). Run it from the ExeaLM repo root:
#
#     bash experiments/data_mix/fetch.sh --release slice-30b --no-stack   # the first slice (no Stack in it)
#     bash experiments/data_mix/fetch.sh                  # pretraining packs + tokenizer + The Stack (needs HF_TOKEN)
#     bash experiments/data_mix/fetch.sh --no-stack       # without the gated Stack bucket (no token needed)
#     bash experiments/data_mix/fetch.sh --post           # also the post-training packs
#     bash experiments/data_mix/fetch.sh --dry-run        # list what would be downloaded
#
# Options: --dest DIR (default data), --revision REV (default main; pass a commit to pin the data), --release NAME
# (a build published under NAME/ in each repo, e.g. slice-30b).
#
# What it downloads: packed/ (tokens.bin, offsets.npy, meta.json per bucket and split), tokenizer/, the mix .json
# files, sources.json and the cards. Not raw text, which training does not read. Each packed token is 2 bytes: the
# 30B slice is about 60 GB (Expanse home is 100 GB). Rerunning skips finished files.
#
# The Stack's repo (exea-labs/exealm-pretrain-stack) is gated: open its page on huggingface.co, accept the Terms of
# Use for The Stack, then `export HF_TOKEN=hf_...` (a read token of that account) before running this.
#
# Then it writes, with paths that exist here:
#     <dest>/mix_pretrain.json, mix_pretrain_val.json, mix_pretrain_anneal.json   every pretraining bucket, The
#                                                     Stack included (not with --no-stack); anneal = the last 10%
#     <dest>/mix_post.json, <dest>/mix_post_val.json  with --post
# and checks that each one opens with exealm.data.loader.open_data. Train with, for example:
#     python -m exealm.train.pretrain --data data/mix_pretrain.json --val data/mix_pretrain_val.json ...
set -euo pipefail

DEST=data
REV=main
STACK=1
POST=0
DRY=0
REL=""
ORG=exea-labs
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --revision) REV="$2"; shift 2 ;;
    --no-stack) STACK=0; shift ;;
    --post) POST=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --release) REL="$2"; shift 2 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
done

if [ ! -f exealm/data/loader.py ]; then
  echo "run this from the ExeaLM repo root"; exit 2
fi
if [ "$STACK" = 1 ] && [ -z "${HF_TOKEN:-}" ] && [ "$DRY" = 0 ]; then
  echo "The Stack's repo $ORG/exealm-pretrain-stack is gated. Accept its terms on huggingface.co, then"
  echo "export HF_TOKEN=hf_... and run this again, or pass --no-stack to train without it."
  exit 2
fi

# hf (huggingface_hub >= 0.34) takes --include once per pattern; huggingface-cli takes them all after one --include
if command -v hf >/dev/null 2>&1; then
  CLI=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  CLI=huggingface-cli
else
  python -m pip install -q huggingface_hub
  CLI=$(command -v hf >/dev/null 2>&1 && echo hf || echo huggingface-cli)
fi

PRE=""
[ -n "$REL" ] && PRE="$REL/"
get() {   # get <repo name> <pattern>...  (patterns inside the release folder)
  local name="$1"; shift
  local args=(download "$ORG/$name" --repo-type dataset --revision "$REV" --local-dir "$DEST/$name")
  local pats=()
  for p in "$@"; do pats+=("$PRE$p"); done
  set -- "${pats[@]}"
  if [ "$CLI" = hf ]; then
    for p in "$@"; do args+=(--include "$p"); done
    [ "$DRY" = 1 ] && args+=(--dry-run)
  else
    args+=(--include "$@")
    if [ "$DRY" = 1 ]; then echo "would download $ORG/$name: $*"; return; fi
  fi
  echo "== $ORG/$name -> $DEST/$name"
  "$CLI" "${args[@]}"
}

mkdir -p "$DEST"
get exealm-pretrain 'packed/*' 'tokenizer/*' 'mix*.json' 'sources.json' 'README.md'
[ "$STACK" = 1 ] && get exealm-pretrain-stack 'packed/*' 'sources.json' 'README.md'
[ "$POST" = 1 ] && get exealm-post 'packed/*' 'mix*.json' 'sources.json' 'README.md'
[ "$DRY" = 1 ] && { echo "dry run: nothing downloaded"; exit 0; }

python - "$DEST" "$STACK" "$POST" "$REL" <<'PY'
import json, os, sys
dest, stack, post, rel = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", sys.argv[4]
root = os.getcwd()
base = os.path.relpath(os.path.abspath(dest), root)           # data/... when dest is inside the repo
if base.startswith(".."):
    base = os.path.abspath(dest)                              # elsewhere: absolute paths


def localize(src, out):
    spec = json.load(open(src, encoding="utf-8"))
    for name, s in spec["sources"].items():
        repo_rel = s["paths"].split("/", 1)[1]                # data/<repo name>/[<release>/]packed/... -> without data/
        s["paths"] = os.path.join(base, repo_rel)
        if not os.path.exists(os.path.join(s["paths"], "meta.json")):
            raise SystemExit(f"{out}: {name}: no pack at {s['paths']}")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    from exealm.data.loader import open_data
    mix = open_data(out)
    print(f"wrote {out}: {len(mix.names)} sources, {mix.meta['n_tokens'] / 1e9:.2f}B tokens: {', '.join(mix.names)}")


pre = os.path.join(dest, "exealm-pretrain", rel)
for suffix in ("", "_val", "_anneal"):
    src = os.path.join(pre, f"mix{suffix}.json" if stack else f"mix_public{suffix}.json")
    if not os.path.exists(src):                               # a release without The Stack has only mix*.json
        src = os.path.join(pre, f"mix{suffix}.json")
    localize(src, os.path.join(dest, f"mix_pretrain{suffix}.json"))
    if post and suffix != "_anneal":
        localize(os.path.join(dest, "exealm-post", rel, f"mix{suffix}.json"), os.path.join(dest, f"mix_post{suffix}.json"))
print(f"tokenizer: {os.path.join(base, 'exealm-pretrain', rel, 'tokenizer', 'tokenizer.json')}")
PY
