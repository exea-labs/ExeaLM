#!/bin/bash
# Everything the MI300X box does overnight before the cluster runs, on its own (from /work inside the container):
#   1. the last test that decides how the cluster trains: streaming only in the last 20% of training (--stream_from
#      0.8) against streaming all the way (stage 7) and not at all (stage 1), for the hybrid and the gated Llama;
#   2. waits for every queue (stages 6y, 7, 8, 9), then scores every finished run: eval_text (every row alone) and
#      stream_eval (long documents, 4 and 8 windows);
#   3. meanwhile downloads the IFM shards the data mix needs into /work/data_raw (no packing: that needs the pipeline);
#   4. every hour, and at the end, uploads results/finalize, the logs and every model.pt to the private Hugging Face
#      repo exea-labs/exealm-nairr-mi300x.
#     nohup bash experiments/finalize/overnight.sh > logs/overnight.log 2>&1 &
set -u
cd /work
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
REPO=exea-labs/exealm-nairr-mi300x
log() { echo "$(date '+%F %T') $*"; }

RECIPE="--preset 100m --data experiments/finalize/mix.json --val data/tokpack/bpe32k/val_en --seq 512 --warmup 200 --search 0 --eval_every 500 --steps 3000 --lr 0.001 --optimizer adamw"
mkdir -p experiments/finalize/jobs results/finalize logs
: > experiments/finalize/jobs/stage9_train.txt
for seed in 0 1; do
  for arm in "late_core:experiments/finalize/configs/core.json:" "late_gated:experiments/finalize/configs/gated.json:--model llama"; do
    IFS=: read -r name cfg extra <<< "$arm"
    [ "$name" = late_gated ] && [ "$seed" = 1 ] && continue
    out="runs/finalize/${name}_s${seed}"
    printf 'fin_%s_s%s\t%s/model.pt\tpython -m exealm.train.pretrain %s --seed %s --config %s %s --stream 4 --stream_from 0.8 --batch 8 --out %s\n' \
      "$name" "$seed" "$out" "$RECIPE" "$seed" "$cfg" "$extra" "$out" >> experiments/finalize/jobs/stage9_train.txt
  done
done
echo -e "fin9_none\tresults/finalize/.none\ttrue" > experiments/finalize/jobs/stage9_eval.txt
log "stage 9: $(wc -l < experiments/finalize/jobs/stage9_train.txt) runs"
nohup bash experiments/finalize/launch.sh experiments/finalize/jobs/stage9_train.txt experiments/finalize/jobs/stage9_eval.txt 7 \
  > logs/stage9.log 2>&1 &

prefetch() {
  python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
want = {"IFM/TxT360-v2": {"web-high-medium": 5, "txt360-qa": 1, "web-high-nltk-qa": 1},
        "IFM/Code-Reasoning": {"code-no-thinking": 2, "code-task-synthesis": 1, "codeio": 1, "code-no-thinking-hard": 1},
        "IFM/Math-Reasoning": {"math-rewrite": 2, "math-dialogue": 1},
        "IFM/SFT-Reasoning": {"instruction-following": 1}}
for repo, subsets in want.items():
    files = api.list_repo_files(repo, repo_type="dataset")
    for sub, n in subsets.items():
        for f in sorted(x for x in files if x.startswith(sub + "/") and x.endswith(".parquet"))[:n]:
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=f"/work/data_raw/{repo.split('/')[1]}")
            print("got", p, flush=True)
PY
}
prefetch > logs/prefetch.log 2>&1 &

upload() {
  python - "$REPO" <<'PY'
import glob, os, sys
from huggingface_hub import HfApi
repo = sys.argv[1]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path="/work/results/finalize", path_in_repo="results/finalize", repo_id=repo, repo_type="model")
api.upload_folder(folder_path="/work/logs", path_in_repo="logs", repo_id=repo, repo_type="model",
                  allow_patterns=["*.log", "gpu_session/*.log"])
api.upload_folder(folder_path="/work/experiments/finalize", path_in_repo="experiments/finalize", repo_id=repo,
                  repo_type="model", allow_patterns=["jobs/*", "configs/*"])
api.upload_folder(folder_path="/work/runs/finalize", path_in_repo="runs/finalize", repo_id=repo, repo_type="model",
                  allow_patterns=["*/model.pt", "*/log.jsonl"])
print("uploaded to", repo, flush=True)
PY
}

busy() { pgrep -f "^bash experiments/finalize/launch.sh" > /dev/null; }
sleep 120
last=0
while busy; do
  if [ $(( $(date +%s) - last )) -ge 3600 ]; then log "hourly upload"; upload >> logs/upload.log 2>&1; last=$(date +%s); fi
  sleep 300
done
log "all queues done; scoring every finished run"
RUNS=$(ls -d runs/finalize/*/ | sed 's:/$::' | while read r; do [ -f "$r/model.pt" ] && echo "$r"; done | tr '\n' ' ')
python experiments/eval_text.py $RUNS --out results/finalize/final_text.json > logs/final_text.log 2>&1
python experiments/stream_eval.py $RUNS --windows 4 --docs 128 --out results/finalize/final_stream4.json > logs/final_stream4.log 2>&1
python experiments/stream_eval.py $RUNS --windows 8 --docs 64 --out results/finalize/final_stream8.json > logs/final_stream8.log 2>&1
upload >> logs/upload.log 2>&1
log "done"
