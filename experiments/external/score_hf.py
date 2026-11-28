"""Score a released Hugging Face model on the study's held-out evaluation (experiments/external/build_texts.py), in bits
per UTF-8 byte, beside our own checkpoints on the same text and the same bytes. Runs in an environment with vLLM (on
Expanse: `conda activate vllm`), which returns the log-probability of every prompt token.

- Windows: the released model reads each window's text alone and predicts every token but its first; bits per byte
  are its bits over the bytes it predicts. Our models' bits (from build_texts.py) are divided the same way, so every
  row of the table is the same text and the same arithmetic. Bits per byte do not depend on the tokenizer.
- Library items: bits on the target's tokens (every token that ends after the prefix) over the target's bytes, with
  nothing, the item's own entry ("pasted") or the BM25 search's pick ("bm25") placed before the prefix, as
  experiments/library/score.py lays them out.

    python experiments/external/score_hf.py --texts data/external/texts.jsonl --model /path/to/Qwen3.5-9B-Base \\
        --out results/expanse/external_qwen35_9b.json
"""
import argparse
import json
import math
import os
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--texts", required=True)
    ap.add_argument("--model", required=True, help="a local folder or Hugging Face id")
    ap.add_argument("--name", default=None, help="the model's name in the output (default: the folder's name)")
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    name = a.name or os.path.basename(a.model.rstrip("/"))
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    rows = [json.loads(line) for line in open(a.texts, encoding="utf-8")]

    jobs = []                                                       # (prompt ids, positions to score, where to add)
    win = [r for r in rows if r["kind"] == "window"]
    for r in win:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"][:a.max_len]
        r["hf_bytes"] = len(r["text"].encode("utf-8")) - len(tok.decode(ids[:1]).encode("utf-8"))
        jobs.append((ids, list(range(1, len(ids))), ("window", r)))
    items = [r for r in rows if r["kind"] == "item"]
    for r in items:
        for cond in ("none", "pasted", "bm25"):
            head = "" if cond == "none" else r[cond] + r["sep"]
            full = head + r["prefix"] + r["target"]
            enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            cut = len(head) + len(r["prefix"])                      # the target's first character
            pos = [i for i, (s, e) in enumerate(offs) if e > cut and i > 0]
            if len(ids) > a.max_len:                                # keep the end: the target and what precedes it
                drop = len(ids) - a.max_len
                ids, pos = ids[drop:], [p - drop for p in pos if p - drop > 0]
            jobs.append((ids, pos, ("item", r, cond)))
    print(f"{len(jobs)} prompts", flush=True)
    outs = llm.generate([{"prompt_token_ids": ids} for ids, _, _ in jobs], sp)
    for (ids, pos, where), o in zip(jobs, outs):
        lp = o.prompt_logprobs
        bits = -sum(lp[p][ids[p]].logprob for p in pos) / math.log(2)
        if where[0] == "window":
            where[1]["hf_bits"] = bits
        else:
            where[1].setdefault("hf", {})[where[2]] = bits

    res = dict(model=name, sets={}, library={})
    ours = sorted({k for r in win for k in r["bits"]})
    by_set = defaultdict(list)
    for r in win:
        by_set[r["set"]].append(r)
    for s, rs in by_set.items():
        row = {name: round(sum(r["hf_bits"] for r in rs) / sum(r["hf_bytes"] for r in rs), 4)}
        for k in ours:
            row[k] = round(sum(r["bits"][k] for r in rs) / sum(r["bytes"] for r in rs), 4)
        res["sets"][s] = row
    if res["sets"]:                                          # none when the texts hold library items only
        res["mean_of_sets"] = {k: round(sum(v[k] for v in res["sets"].values()) / len(res["sets"]), 4)
                               for k in [name] + ours}
    groups = defaultdict(list)
    for r in items:
        groups[(r["set"], "all")].append(r)
        groups[(r["set"], r["type"])].append(r)
    for (s, t), rs in sorted(groups.items()):
        res["library"][f"{s}/{t}"] = {c: round(sum(r["hf"][c] for r in rs) / sum(r["target_bytes"] for r in rs), 4)
                                      for c in ("none", "pasted", "bm25")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res.get("mean_of_sets", res["library"])), flush=True)
    for k, v in res["library"].items():
        print(k, v, flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
