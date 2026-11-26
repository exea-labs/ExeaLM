#!/bin/bash
# The library as ordinary RAG, trained for: post-training where every library item has the entry a BM25 search picks
# placed before it (experiments/library/rag_pack.py), then scoring with the same search (score.py's bm25). From the
# repo root on the login node:
#
#     bash experiments/post/rag_post.sh runs/exealm_midstream_100m_4.1x_s2 runs/qwen_100m_4.1x_s2
#
# Two runs read the same mix as library_post.sh's (the pretraining mix, half; the library's training items, half),
# except that half of the library share is the items with their searched entry in front:
#   ragpost_exealm_<tag>   ExeaLM
#   ragpost_qwen_<tag>     the Qwen3.5-style model
# Writes results/expanse/library_rag_<tag>.jsonl: none, pasted and bm25 on the held-out topics. Compare with
# library_bm25_<tag>.jsonl, where post-training never showed an entry. STEPS (600), LR (3e-4), TAG (100m).
set -euo pipefail
source experiments/expanse/env.sh
EXEALM=${1:?an ExeaLM run folder (with model.pt)}
QWEN=${2:?a Qwen3.5-style run folder (with model.pt)}
STEPS=${STEPS:-600}
LR=${LR:-3e-4}
TAG=${TAG:-100m}
LIB=$PWD/data/library/v1
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-fli100}
mkdir -p logs/expanse results/expanse

if [ -f "$LIB/rag/train/meta.json" ]; then           # built by an earlier run of this script: the same for every seed
  jr="" && echo "RAG pack: $LIB/rag/train (already built)"
else
  jr=$(sbatch --parsable -p nairr-gpu-shared -N 1 --gpus=h100:1 --ntasks-per-node=1 --cpus-per-task=8 -t 00:45:00 \
    -J "exealm-ragpack" -o "logs/expanse/%x-%j.out" --wrap "source experiments/expanse/env.sh; export PYTHONUNBUFFERED=1; \
    python experiments/library/rag_pack.py --library $LIB --out $LIB/rag/train")
  echo "RAG pack: $jr -> $LIB/rag/train"
fi

python - "$LIB" <<'PY'
import json, sys
lib = sys.argv[1]
mix = json.load(open("data/mix_pretrain.json"))
total = sum(s["weight"] for s in mix["sources"].values())
tok = {k: json.load(open(f"{lib}/{k}/train/meta.json"))["n_tokens"] for k in ("qa", "tasks")}
for k, n in tok.items():
    mix["sources"][f"library_{k}"] = {"paths": f"{lib}/{k}/train", "weight": round(total / 2 * n / sum(tok.values()), 3)}
mix["sources"]["library_rag"] = {"paths": f"{lib}/rag/train", "weight": round(total / 2, 3)}
mix["note"] = ("post-training for the RAG test (experiments/post/rag_post.sh): the pretraining mix, half; the "
               "library's training items, a quarter; the same items with a BM25-picked entry before them, a quarter")
json.dump(mix, open("data/mix_library_rag.json", "w"), indent=1)
print("data/mix_library_rag.json:", {k: v["weight"] for k, v in mix["sources"].items() if k.startswith("library")})
PY

COMMON="--data data/mix_library_rag.json --val data/mix_pretrain_val.json --seq 512 --batch 64 --accum 2 \
--steps $STEPS --warmup $((STEPS / 10)) --lr $LR --seed 0 --eval_every 200 --eval_batches 8"
submit() {          # submit <run> <args>: one GPU of the shared partition, through train.sb, after the pack
  RUN=$1 ARGS="$2 $COMMON" SAVE_MIN=0 SBATCH_JOB_NAME=exealm-$1 SBATCH_TIMELIMIT=02:00:00 \
    SBATCH_PARTITION=nairr-gpu-shared SBATCH_GPUS=h100:1 SBATCH_EXTRA="--ntasks-per-node=1" \
    sbatch --parsable ${jr:+--dependency=afterok:$jr} --ntasks-per-node=1 experiments/expanse/train.sb
}
j1=$(submit "ragpost_exealm_$TAG" "--search 0 --init $PWD/$EXEALM/model.pt")
j2=$(submit "ragpost_qwen_$TAG" "--model llama --init $PWD/$QWEN/model.pt")
echo "post-training: ragpost_exealm_$TAG $j1, ragpost_qwen_$TAG $j2"

j3=$(sbatch --parsable -p nairr-gpu-shared -N 1 --gpus=h100:1 --ntasks-per-node=1 --cpus-per-task=8 -t 01:30:00 \
  -J "exealm-ragscore-$TAG" -o "logs/expanse/%x-%j.out" --dependency="afterok:$j1:$j2" \
  --wrap "source experiments/expanse/env.sh; export PYTHONUNBUFFERED=1; python experiments/library/score.py \
    --library $LIB runs/ragpost_exealm_$TAG runs/ragpost_qwen_$TAG --conditions none,pasted,bm25 \
    --out results/expanse/library_rag_$TAG.jsonl")
echo "scoring: exealm-ragscore-$TAG $j3 (after the two runs) -> results/expanse/library_rag_$TAG.jsonl"
