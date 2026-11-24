#!/bin/bash
# Start the jobs of a job file (TAG <tab> OUTPUT FILE <tab> COMMAND, as make_jobs.py writes them) in order, each as
# soon as fewer than MAX training processes are running on the machine (any queue's), skipping jobs that finished or
# are running already. Then run the evaluation file with experiments/gpu_session/runner.py. From the repo root:
#
#     bash experiments/finalize/launch.sh experiments/finalize/jobs/stage5_train.txt experiments/finalize/jobs/stage5_eval.txt 8
train=$1
evals=$2
max=${3:-8}
mkdir -p logs/gpu_session
running() { pgrep -fc "^python -m exealm.train.pretrain"; }
training() {                                  # is a run writing to folder $1? (names hold +, so no regular expressions)
  ps -eo args= | awk -v d="--out $1" 'index($0, "python -m exealm.train.pretrain") == 1 &&
    substr($0, length($0) - length(d) + 1) == d { found = 1 } END { exit !found }'
}
while IFS=$'\t' read -r tag out cmd; do
  if [ -f "logs/gpu_session/$tag.ok" ] || [ -f "$out" ]; then continue; fi
  if training "${out%/model.pt}"; then continue; fi                                # started by another queue
  while [ "$(running)" -ge "$max" ]; do sleep 20; done
  echo "$(date '+%F %T') start $tag"
  ( bash -c "$cmd" > "logs/gpu_session/$tag.log" 2>&1 && [ -f "$out" ] && echo "$out" > "logs/gpu_session/$tag.ok" ) &
  sleep 15
done < "$train"
wait
busy() { for r in $(grep -o 'runs/finalize/[^ /]*' "$evals" | sort -u); do training "$r" && return 0; done; return 1; }
while busy; do sleep 30; done                                                     # runs it scores still training
echo "$(date '+%F %T') evaluating"
python experiments/gpu_session/runner.py "$evals" 2
echo "$(date '+%F %T') done"
