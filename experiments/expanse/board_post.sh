#!/bin/bash
# Claim 11 of docs/report_plan.md, at small scale: three 0.1B ExeaLM agents sharing a board against one 0.25B model,
# on long held-out documents. From the repo root on the login node, once the curve's runs below are finished:
#
#     bash experiments/expanse/board_post.sh runs/exealm_100m_4.1x_s0 runs/exealm_250m_4.1x_s0 runs/qwen_250m_4.1x_s0
#
# Four post-training runs read the same pretraining mix for the same steps, so no model gets more training than
# another after pretraining:
#   boardpost_ag_<tag>      the 0.1B ExeaLM with board slots (AG) grafted into its TONN layer, trained in swarms of 3:
#                           agent j reads window j of 3 consecutive windows; board_order "doc" lets it see only the
#                           agents before it, which read the earlier text
#   boardpost_solo_<tag>    the same 0.1B ExeaLM without the board, every row alone
#   boardpost_exealm_big    the 0.25B ExeaLM, every row alone
#   boardpost_qwen_big      the 0.25B Qwen3.5-style model, every row alone
# Then experiments/swarm_long.py scores them (and the four starting checkpoints) on the first 3 x 512 tokens of long
# held-out documents: the swarm with the board and with it cut, the single models streamed and window by window.
# Writes results/expanse/board_<tag>.json. Each run is 1 GPU for 15 to 40 minutes; the scoring about 30.
#
# STEPS (600), LR (3e-4), AG (slots, 2) and TAG (100m) can be set in the environment.
set -euo pipefail
source experiments/expanse/env.sh
SMALL=${1:?the 0.1B ExeaLM run folder}
BIG=${2:?the 0.25B ExeaLM run folder}
QWEN=${3:?the 0.25B Qwen3.5-style run folder}
STEPS=${STEPS:-600}
LR=${LR:-3e-4}
AG=${AG:-2}
TAG=${TAG:-100m}
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-fli100}
mkdir -p runs/_graft logs/expanse results/expanse

GRAFT=$PWD/runs/_graft/exealm_${TAG}_ag.pt
# The graft runs on a compute node: the login node caps a process's memory (it could not hold a 0.1B model)
jg=$(sbatch --parsable -p nairr-gpu-shared --gpus=h100:1 --ntasks-per-node=1 --cpus-per-task=4 -t 00:20:00 \
  -J "exealm-graft-$TAG" -o "logs/expanse/%x-%j.out" --wrap "source experiments/expanse/env.sh; \
  python -m exealm.train.graft --ckpt $SMALL/model.pt --add AG:$AG --set '{\"swarm_size\": 3, \"board_order\": \"doc\"}' --out $GRAFT")
echo "graft: $jg -> $GRAFT"

COMMON="--data data/mix_pretrain.json --val data/mix_pretrain_val.json --seq 512 --accum 2 --steps $STEPS \
--warmup $((STEPS / 10)) --lr $LR --seed 0 --eval_every 200 --eval_batches 8"
submit() {          # submit <run> <args>: one GPU of the shared partition, through train.sb
  RUN=$1 ARGS="$2 $COMMON" SAVE_MIN=0 SBATCH_JOB_NAME=exealm-$1 SBATCH_TIMELIMIT=02:00:00 \
    SBATCH_PARTITION=nairr-gpu-shared SBATCH_GPUS=h100:1 SBATCH_EXTRA="--ntasks-per-node=1" \
    sbatch --parsable ${DEP:+--dependency=$DEP} --ntasks-per-node=1 experiments/expanse/train.sb
}
# --batch 63: a multiple of the swarm size, or the loader and the board fall back to swarms of 1 without a word
j1=$(DEP=afterok:$jg submit "boardpost_ag_$TAG" "--search 0 --init $GRAFT --swarm_policy split --batch 63")
j2=$(submit "boardpost_solo_$TAG" "--search 0 --init $PWD/$SMALL/model.pt --batch 64")
j3=$(submit "boardpost_exealm_big" "--search 0 --init $PWD/$BIG/model.pt --batch 64")
j4=$(submit "boardpost_qwen_big" "--model llama --init $PWD/$QWEN/model.pt --batch 64")
echo "post-training: ag $j1, solo $j2, exealm 0.25B $j3, qwen 0.25B $j4"

j5=$(sbatch --parsable -p nairr-gpu-shared --gpus=h100:1 --ntasks-per-node=1 --cpus-per-task=8 -t 02:00:00 \
  -J "exealm-boardscore-$TAG" -o "logs/expanse/%x-%j.out" --dependency="afterok:$j1:$j2:$j3:$j4" \
  --wrap "source experiments/expanse/env.sh; export PYTHONUNBUFFERED=1; python experiments/swarm_long.py \
    --swarm runs/boardpost_ag_$TAG --single runs/boardpost_solo_$TAG runs/boardpost_exealm_big runs/boardpost_qwen_big \
    $SMALL $BIG $QWEN --mix data/mix_pretrain_val.json --out results/expanse/board_$TAG.json")
echo "scoring: exealm-boardscore-$TAG $j5 (after the four runs) -> results/expanse/board_$TAG.json"
