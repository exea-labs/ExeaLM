#!/bin/bash
# Run the scaling study on SDSC Expanse, phase by phase, from the repo root on a login node (docs/runbook.md):
#
#     export SBATCH_ACCOUNT=<project>  KEEP=<where runs are kept: a folder or s3://bucket/prefix>
#     bash experiments/expanse/go.sh setup      # Python environment (once)
#     bash experiments/expanse/go.sh data       # the 30B slice from Hugging Face into data/ (~60 GB)
#     bash experiments/expanse/go.sh first      # the first job: correctness, speeds, 4-GPU speed-up, lead check
#     bash experiments/expanse/go.sh lr         # learning-rate checks at every size, every model
#     bash experiments/expanse/go.sh pick       # pick each model's rate from the checks -> lr_choice.json
#     bash experiments/expanse/go.sh curve      # 0.1B to 0.5B: ExeaLM, the base Qwen3.5-style model and the plain Llama
#     bash experiments/expanse/go.sh 3b         # the 3B pair: ExeaLM and the base Qwen3.5-style model
#     bash experiments/expanse/go.sh eval       # score every finished run, upload results to Hugging Face
#     bash experiments/expanse/go.sh status     # what is queued, running and finished
#     bash experiments/expanse/go.sh all        # every phase after data, in order, waiting for each (run it in tmux)
#
# DRY=1 prints the sbatch lines instead of submitting them. RELEASE picks the data build (default slice-30b-r2).
# Every training job is experiments/expanse/train.sb: it checkpoints to KEEP every SAVE_MIN minutes, stops cleanly
# before the 48-hour limit, resubmits itself, and uploads the finished model to Hugging Face (HF_REPO).
set -uo pipefail
cd "$(dirname "$0")/../.."
RELEASE=${RELEASE:-slice-30b-r2}
KEEP=${KEEP:-$PWD/runs}
export KEEP
DRY=${DRY:-0}
PLAN="python experiments/expanse/plan.py"
log() { echo "$(date '+%F %T') $*"; }
need_account() { [ -n "${SBATCH_ACCOUNT:-}" ] || [ "$DRY" = 1 ] || { echo "export SBATCH_ACCOUNT=<your project> first"; exit 1; }; }
submit() {                                       # run plan.py's lines (RUN=... ARGS=... sbatch ...), or print them
  if [ "$DRY" = 1 ]; then cat; else bash; fi
}
queued() { squeue -h -u "$USER" -o "%j" 2>/dev/null | grep -c "^exealm" ; }
wait_queue() {                                   # until no exealm job of ours is queued or running
  while [ "$(queued)" -gt 0 ]; do log "waiting: $(queued) exealm job(s) in the queue"; sleep 900; done
}
logs_to() {                                      # the runs' logs in a local folder, for pick and status
  if [[ $KEEP == s3://* ]]; then
    mkdir -p runs_logs && s3cmd sync --no-progress --exclude '*' --include '*/log.jsonl' "$KEEP/" runs_logs/ > /dev/null
    echo runs_logs
  else
    echo "$KEEP"
  fi
}

phase() {
  case $1 in
    setup) bash experiments/expanse/setup.sh ;;
    data)
      bash experiments/data_mix/fetch.sh --release "$RELEASE" --no-stack
      ls data/mix_pretrain.json data/mix_pretrain_val.json data/mix_pretrain_anneal.json ;;
    first) need_account
      if [ "$DRY" = 1 ]; then echo "sbatch experiments/expanse/first_contact.sb"; else sbatch experiments/expanse/first_contact.sb; fi ;;
    lr) need_account
      for size in 100m 250m 500m; do $PLAN --lr_check "$size" | grep -v "^#" | submit; done
      $PLAN --lr_check 3b --lrs 3e-4,6e-4 | grep -v "^#" | submit ;;           # two rates at 3B, as budgeted
    pick) python experiments/expanse/pick_lr.py --keep "$(logs_to)" ;;
    curve|3b) need_account
      part=$([ "$1" = 3b ] && echo 3B || echo curve)
      for run in $($PLAN --list "$part"); do $PLAN --run "$run" --lr auto | grep -v "^#" | submit; done ;;
    eval) need_account
      if [ "$DRY" = 1 ]; then echo "sbatch experiments/expanse/eval.sb"; else sbatch experiments/expanse/eval.sb; fi ;;
    status)
      squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null
      dir=$(logs_to)
      for f in "$dir"/*/log.jsonl; do
        [ -f "$f" ] || continue
        printf "%-24s " "$(basename "$(dirname "$f")")"
        grep '"event": "eval"' "$f" | tail -1 | python -c "import sys,json; l=sys.stdin.read().strip(); r=json.loads(l) if l else {}; print('step', r.get('step'), 'bits/byte', r.get('val_bits_per_byte'), 'tok/s', r.get('tokens_per_sec'))"
      done ;;
    all)
      for p in first lr; do phase $p; [ "$DRY" = 1 ] || wait_queue; done
      phase pick
      for p in curve 3b; do phase $p; [ "$DRY" = 1 ] || wait_queue; done
      phase eval ;;
    *) sed -n '2,20p' "$0"; exit 1 ;;
  esac
}
phase "${1:-help}"
