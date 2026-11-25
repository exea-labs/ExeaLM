#!/bin/bash
# Run the scaling study on SDSC Expanse, phase by phase, from the repo root on a login node (docs/runbook.md):
#
#     export SBATCH_ACCOUNT=<project>  KEEP=<where runs are kept: a folder or s3://bucket/prefix>
#     bash experiments/expanse/go.sh setup      # Python environment (once)
#     bash experiments/expanse/go.sh data       # the 30B slice from Hugging Face into data/ (~60 GB)
#     bash experiments/expanse/go.sh first      # the first job: correctness, speeds, 4-GPU speed-up, lead check
#     bash experiments/expanse/go.sh lr         # learning-rate checks at every size, every model
#     bash experiments/expanse/go.sh pick       # pick each model's rate from the checks -> lr_choice.json
#     bash experiments/expanse/go.sh curve      # 0.1B to 0.5B: ExeaLM, the base Qwen3.5-style model and the plain Llama;
#                                               # and the 0.1B pair at the 9B run's 111 tokens per parameter
#     bash experiments/expanse/go.sh 3b         # the 3B pair: ExeaLM and the base Qwen3.5-style model
#     bash experiments/expanse/go.sh eval       # score every finished run, write the report, upload to Hugging Face
#     bash experiments/expanse/go.sh report     # the report again from results/expanse (report.md, scaling.png)
#     bash experiments/expanse/go.sh nine_b     # the 9B model's speed and memory on 1, 2 and 4 nodes
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
mkdir -p logs/expanse results/expanse              # Slurm writes job output to logs/expanse (git ignores logs/)
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
      parts=$([ "$1" = 3b ] && echo 3B || echo "curve long")
      for part in $parts; do
        for run in $($PLAN --list "$part"); do $PLAN --run "$run" --lr auto | grep -v "^#" | submit; done
      done ;;
    eval) need_account
      if [ "$DRY" = 1 ]; then echo "sbatch experiments/expanse/eval.sb"; else sbatch experiments/expanse/eval.sb; fi ;;
    report) python experiments/expanse/report.py --results results/expanse \
              $([ -f results/expanse/throughput_h100.json ] && echo "--rates results/expanse/throughput_h100.json") ;;
    nine_b) need_account                           # one node first; two and four only if it wrote a row
      if [ "$DRY" = 1 ]; then
        echo "sbatch --parsable --nodes=1 experiments/expanse/nine_b.sb"
        for n in 2 4; do echo "sbatch --dependency=afterok:<that job> --nodes=$n experiments/expanse/nine_b.sb"; done
      else
        jid=$(sbatch --parsable --nodes=1 experiments/expanse/nine_b.sb) || exit 1
        for n in 2 4; do sbatch --dependency=afterok:"$jid" --nodes=$n experiments/expanse/nine_b.sb; done
      fi ;;
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
      phase eval ;;                                # nine_b is separate: it needs FSDP checked on the cluster first
    *) sed -n '2,20p' "$0"; exit 1 ;;
  esac
}
phase "${1:-help}"
