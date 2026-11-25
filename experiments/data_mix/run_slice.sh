#!/bin/bash
# Build the first 30B-token slice of recipe v2 on the MI300X box and publish it to exea-labs/exealm-pretrain under
# slice-30b/ (inside the container, from /work). Stops at the first failing step; rerun to continue: every step
# skips work already done. Log: logs/run_slice.log, one line per step.
#     nohup bash experiments/data_mix/run_slice.sh > logs/run_slice.log 2>&1 &
set -u
cd /work
export WORK=/work/data_mix
M="python -m experiments.data_mix.pipeline"
log() { echo "$(date '+%F %T') $*"; }
step() { log "start $1"; shift; "$@" || { log "FAILED: $*"; exit 1; }; log "done"; }

while pgrep -f "download_slice.py" > /dev/null; do sleep 60; done                       # the first download pass
for d in megamath-web-pro megamath-code megamath-text-code-block; do                    # Llama-rewritten or no text
  rm -rf "/work/data_raw/MegaMath/$d"
done
step "download (MegaMath web only, anything missing)" python experiments/recipe/download_slice.py --tokens 30e9 --out /work/data_raw
step "dependencies" pip install -q --break-system-packages -e ".[data]" pyarrow zstandard huggingface_hub
step "pipeline tests" python -m pytest -q tests/test_data_mix.py
step fetch $M fetch --work $WORK --slice_manifest /work/data_raw/manifest_slice.json --release slice-30b --tokens 30e9 --download_workers 20 --part_parallel 4
step dedup $M dedup --work $WORK
step tokenizer $M tokenizer --work $WORK
step benchmarks $M benchmarks --work $WORK
step tokenize $M tokenize --work $WORK
step assemble $M assemble --work $WORK
step cards $M cards --work $WORK
step verify $M verify --work $WORK
step "upload dry run" python -m experiments.data_mix.upload --work $WORK --repos pretrain --dry-run
step upload python -m experiments.data_mix.upload --work $WORK --repos pretrain
log "slice-30b is on https://huggingface.co/datasets/exea-labs/exealm-pretrain"
