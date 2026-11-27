#!/bin/bash
# Does the library work when ExeaLM reads it inside its TONN layer? Three post-training runs from the same 0.1B model
# (the seed-2 curve model), the same steps and the same library items, differing only in how the library reaches it:
#
#   rg     the RG circuit grafted into the TONN layer (runs/_graft/exealm_100m_rg.pt), its candidates from BM25 over
#          the chunk just read (--datastore_lexical); the entries never enter the context
#   text   ordinary RAG: half the items have the BM25 entry pasted before them (data/library/v1/rag/train)
#   none   the items alone
#
# then scores each on the held-out topics: rg with its library attached (and cut), text with the BM25 entry pasted
# (and without), none alone. Section 7 of results/expanse/README.md found the RG circuit's own learned search returned
# the right entry for 0.1% of questions; this run gives it the search that works (BM25) and asks whether its read does.
#
#     DEVICE=mps STEPS=600 bash experiments/post/rg_lexical_test.sh
set -euo pipefail
DEVICE=${DEVICE:-cuda}
STEPS=${STEPS:-600}
LIB=${LIB:-data/library/v1}
BASE=${BASE:-runs/exealm_100m_4.1x_s2/model.pt}
GRAFT=${GRAFT:-runs/_graft/exealm_100m_rg.pt}
OUT=${OUT:-runs/rgtest}
PY=${PY:-python}
mkdir -p "$OUT" results/rgtest
$PY - "$LIB" <<'EOF'
import json, sys
lib = sys.argv[1]
n = {k: json.load(open(f"{lib}/{k}/train/meta.json"))["n_tokens"] for k in ("qa", "tasks")}
items = {f"library_{k}": {"paths": f"{lib}/{k}/train", "weight": round(n[k] / sum(n.values()), 4)} for k in n}
json.dump({"sources": items, "note": "rg_lexical_test.sh: the library's training items"},
          open("data/mix_rgtest_items.json", "w"), indent=1)
rag = dict(items)
rag = {k: {"paths": v["paths"], "weight": round(v["weight"] / 2, 4)} for k, v in items.items()}
rag["library_rag"] = {"paths": f"{lib}/rag/train", "weight": 0.5}
json.dump({"sources": rag, "note": "rg_lexical_test.sh: the items, half with the BM25 entry pasted"},
          open("data/mix_rgtest_text.json", "w"), indent=1)
print("mixes written")
EOF
COMMON="--val data/mix_pretrain_val.json --seq 512 --batch ${BATCH:-16} --accum ${ACCUM:-4} --steps $STEPS \
--warmup $((STEPS / 10)) --lr 3e-4 --seed 0 --eval_every $((STEPS / 3)) --eval_batches 4 --device $DEVICE --search 0"
for arm in ${ARMS:-rg text none}; do
  case $arm in
    rg)   args="--init $GRAFT --data data/mix_rgtest_items.json --datastore_data $LIB --datastore_chunks 130000 --datastore_lexical" ;;
    text) args="--init $BASE --data data/mix_rgtest_text.json" ;;
    none) args="--init $BASE --data data/mix_rgtest_items.json" ;;
  esac
  [ -f "$OUT/$arm/model.pt" ] || $PY -m exealm.train.pretrain $args $COMMON --out "$OUT/$arm"
done
$PY experiments/library/score.py --library $LIB "$OUT/rg" --conditions none,library --lexical --device $DEVICE \
  --out results/rgtest/rg.jsonl
$PY experiments/library/score.py --library $LIB "$OUT/text" "$OUT/none" --conditions none,bm25 --device $DEVICE \
  --out results/rgtest/text_none.jsonl
