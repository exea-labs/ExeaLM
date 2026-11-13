source ~/exea_work/env.sh >/dev/null 2>&1
cd ~/exealm_work/ExeaLM
until grep -q DONE-DL2 data/tok2/get.log && grep -q "END eqt_100m_hybrid_lr1e3_s1" results/sweep/runner.log; do sleep 30; done
echo "$(date "+%F %T") packing"
TOK=results/tokenizer/bpe32k
O=data/tokpack/bpe32k
SRC=~/lm_work/src/sample/10BT
$PY -m exealm.data.pack $SRC/000_00000.parquet --tokenizer $TOK --max_bytes 1e9 --out $O/train_en_big
$PY -m exealm.data.pack data/tok2/llmjp__ja__ja_fineweb-2__0000.first300MB.jsonl.gz data/tok2/llmjp__ja__ja_wiki__0000.first150MB.jsonl.gz --tokenizer $TOK --out $O/train_ja_big
$PY -m exealm.data.pack data/tok2/llmjp__code__code_stack__0000.first220MB.jsonl.gz --tokenizer $TOK --out $O/train_code_big
for d in train_en_big train_ja_big train_code_big; do cat $O/$d/meta.json | head -5; done
B="experiments/full_small.py --steps 24000 --out results/full_small"
printf "%s\n" "full_small_exealm|$B --model exealm" "full_small_llama|$B --model llama" >> results/sweep/queue.txt
echo "$(date "+%F %T") queued"; cat results/sweep/queue.txt | cut -c1-80
