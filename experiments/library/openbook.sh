#!/bin/bash
# Benchmarks closed-book and open-book, side by side, for each run given (docs/decisions.md, 26 November): the same
# library and search for every model; written entries that share a 13-token window with any of the 14 checked
# benchmarks are never given (--rag_exclude); results only on those benchmarks. The written tier is library v1 and v2
# searched as one (v2's entry numbers follow v1's 18,545), the human tier the public pool.
#
#     TAG=500m_v2 bash experiments/library/openbook.sh runs/ragpost_exealm_500m_v2 runs/exealm_500m_4.1x_s0
set -uo pipefail
source experiments/expanse/env.sh
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
TASKS=${TASKS:-sciq,arc_easy,arc_challenge,mmlu,pubmedqa}
HUMAN=${HUMAN:-/expanse/lustre/projects/fli100/abhatia7/library_v2/bm25_pool_p100000}
OUT=results/expanse/openbook
mkdir -p "$OUT"
EX=data/library_v2/written_bench_hits.json        # v1's and v2's overlapping entries, in the combined numbering
python - <<'EOF'
import json
v1 = json.load(open("data/library/v1/bench_hits.json"))["drop"]
v2 = json.load(open("data/library_v2/v2/bench_hits.json"))["drop"]
base = json.load(open("data/library/v1/meta.json"))["n_docs"]
drop = {**{str(int(k)): v for k, v in v1.items()}, **{str(int(k) + base): v for k, v in v2.items()}}
json.dump({"library": "data/library/v1 + data/library_v2/v2", "base_v2": base, "drop": drop},
          open("data/library_v2/written_bench_hits.json", "w"), indent=1)
print(f"excluded written entries: {len(v1)} of v1, {len(v2)} of v2 (v2 numbered from {base})")
EOF
LIB="written=data/library/v1/bm25+data/library_v2/v2/bm25;human=$HUMAN"
for r in "$@"; do
  name=$(basename "$r")
  for m in closed open; do
    f="$OUT/${name}_$m.json"
    [ -s "$f" ] && continue
    L=""
    [ $m = open ] && L="--library $LIB --rag_exclude written=$EX"
    python -m exealm.eval.harness --ckpt "$r/model.pt" --tokenizer data/exealm-pretrain/slice-30b-r2/tokenizer \
      --tasks "$TASKS" --device cuda --max_length 2048 --stream_window 512 $L --out "$f" || echo "$m failed for $r"
  done
done
