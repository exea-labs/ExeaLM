#!/bin/bash
# Claim 10 of docs/report_plan.md, at small scale: does the library distilled from Qwen3-30B-A3B (data/library/v1,
# experiments/expanse/library.sb) make ExeaLM better on research and code it was never trained on? From the repo root
# on the login node:
#
#     bash experiments/expanse/library_post.sh runs/exealm_100m_4.1x_s0 runs/qwen_100m_4.1x_s0
#
# Three post-training runs read the same data: the library's training questions and tasks (half of every step's
# tokens) and the pretraining mix (the other half, so nothing is forgotten). Their topics are disjoint from the
# held-out ones.
#   libpost_rg_<tag>      ExeaLM with a library slot (RG) grafted into its TONN layer, searching the library
#   libpost_exealm_<tag>  ExeaLM as it was, no library
#   libpost_qwen_<tag>    the Qwen3.5-style model, no library
# Then experiments/library/score.py scores the held-out topics' questions and tasks, with the library attached
# (library), with nothing (none) and with the right entry pasted before the question (pasted: retrieval that is
# never wrong). The base checkpoints are scored too, before any post-training. Writes
# results/expanse/library_<tag>.jsonl. Each run is 1 GPU for about 15 minutes; the scoring about 20.
#
# STEPS (600), LR (3e-4) and TAG (100m) can be set in the environment.
set -euo pipefail
source experiments/expanse/env.sh
EXEALM=${1:?an ExeaLM run folder (with model.pt)}
QWEN=${2:?a Qwen3.5-style run folder (with model.pt)}
STEPS=${STEPS:-600}
LR=${LR:-3e-4}
TAG=${TAG:-100m}
LIB=$PWD/data/library/v1
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-fli100}
mkdir -p runs/_graft logs/expanse results/expanse

# The post-training mix: the pretraining mix's sources and weights, and the library's two training packs with as much
# weight again, split between them by tokens. Library paths are absolute: training jobs copy data/ to the node's disk
# without data/library*.
python - "$LIB" <<'PY'
import json, sys
lib = sys.argv[1]
mix = json.load(open("data/mix_pretrain.json"))
total = sum(s["weight"] for s in mix["sources"].values())
tok = {k: json.load(open(f"{lib}/{k}/train/meta.json"))["n_tokens"] for k in ("qa", "tasks")}
for k, n in tok.items():
    mix["sources"][f"library_{k}"] = {"paths": f"{lib}/{k}/train", "weight": round(total * n / sum(tok.values()), 3)}
mix["note"] = ("post-training for the library test (experiments/expanse/library_post.sh): the pretraining mix and "
               "the library's training questions and tasks, half and half")
json.dump(mix, open("data/mix_library_post.json", "w"), indent=1)
print("data/mix_library_post.json:", {k: v["weight"] for k, v in mix["sources"].items() if k.startswith("library")},
      "against", round(total, 2), "for the pretraining sources")
PY

GRAFT=$PWD/runs/_graft/exealm_${TAG}_rg.pt
python -m exealm.train.graft --ckpt "$EXEALM/model.pt" --add RG:1 --set '{"rag_key_dim": 128, "rag_lexical": 128}' \
  --out "$GRAFT"

COMMON="--data data/mix_library_post.json --val data/mix_pretrain_val.json --seq 512 --batch 64 --accum 2 \
--steps $STEPS --warmup $((STEPS / 10)) --lr $LR --seed 0 --eval_every 200 --eval_batches 8"
submit() {          # submit <run> <args>: one GPU of the shared partition, through train.sb
  RUN=$1 ARGS="$2 $COMMON" SAVE_MIN=0 SBATCH_JOB_NAME=exealm-$1 SBATCH_TIMELIMIT=02:00:00 \
    SBATCH_PARTITION=nairr-gpu-shared SBATCH_GPUS=h100:1 SBATCH_EXTRA="--ntasks-per-node=1" \
    sbatch --parsable --ntasks-per-node=1 experiments/expanse/train.sb
}
j1=$(submit "libpost_rg_$TAG" "--search 0 --init $GRAFT --datastore_data $LIB --datastore_chunks 130000")
j2=$(submit "libpost_exealm_$TAG" "--search 0 --init $PWD/$EXEALM/model.pt")
j3=$(submit "libpost_qwen_$TAG" "--model llama --init $PWD/$QWEN/model.pt")
echo "post-training: libpost_rg_$TAG $j1, libpost_exealm_$TAG $j2, libpost_qwen_$TAG $j3"

j4=$(sbatch --parsable -p nairr-gpu-shared --gpus=h100:1 --ntasks-per-node=1 --cpus-per-task=8 -t 02:00:00 \
  -J "exealm-libscore-$TAG" -o "logs/expanse/%x-%j.out" --dependency="afterok:$j1:$j2:$j3" \
  --wrap "source experiments/expanse/env.sh; export PYTHONUNBUFFERED=1; python experiments/library/score.py \
    --library $LIB runs/libpost_rg_$TAG runs/libpost_exealm_$TAG runs/libpost_qwen_$TAG $EXEALM $QWEN \
    --out results/expanse/library_$TAG.jsonl")
echo "scoring: exealm-libscore-$TAG $j4 (after the three runs) -> results/expanse/library_$TAG.jsonl"
