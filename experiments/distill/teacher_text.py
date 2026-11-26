"""Distillation into the weights through text: a teacher model continues openings drawn from our training data, and
the student pretrains on those continuations beside the real data (sequence-level distillation; the teacher's
tokenizer does not need to match ours). Three steps, each its own command:

    # 1. openings, in the exealm environment: N openings of K tokens from the training mix, as text
    python experiments/distill/teacher_text.py prompts --mix data/mix_pretrain.json --n 200000 --tokens 128 \\
        --out /path/prompts.jsonl
    # 2. continuations, in the vllm environment: the teacher writes --max_new tokens after each opening
    python experiments/distill/teacher_text.py generate --prompts /path/prompts.jsonl --model /path/teacher \\
        --max_new 512 --out /path/teacher.jsonl
    # 3. a pack TokenShards reads, in the exealm environment: opening and continuation as one document
    python experiments/distill/teacher_text.py pack --texts /path/teacher.jsonl --out data/distill/qwen35_9b/train

Openings come only from the training packs (`data/mix_pretrain.json`), never from the held-out sets, so the student's
held-out score cannot see the teacher's text of them.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
TOKENIZER = "data/exealm-pretrain/slice-30b-r2/tokenizer"


def prompts(a):
    import numpy as np
    from tokenizers import Tokenizer
    from exealm.data.loader import TokenShards, open_data
    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    data = open_data(a.mix)
    rng = np.random.default_rng(a.seed)
    out, tries = [], 0
    with open(a.out, "w", encoding="utf-8") as f:
        while len(out) < a.n and tries < 4 * a.n:
            tries += 1
            x, _, _ = data.batch(64, a.tokens, rng)
            for row in x.tolist():
                if eot in row:                                  # an opening that crosses a document boundary
                    row = row[row.index(eot) + 1:]
                if len(row) < a.tokens // 2:
                    continue
                text = tok.decode(row)
                if len(text.strip()) < 40:
                    continue
                out.append(text)
                f.write(json.dumps(dict(i=len(out) - 1, prompt=text), ensure_ascii=False) + "\n")
                if len(out) >= a.n:
                    break
    print(f"{a.out}: {len(out)} openings of {a.tokens} tokens from {a.mix}")


def generate(a):
    from vllm import LLM, SamplingParams
    rows = [json.loads(line) for line in open(a.prompts, encoding="utf-8")]
    done = set()
    if os.path.exists(a.out):                                   # resume: skip openings already continued
        done = {json.loads(line)["i"] for line in open(a.out, encoding="utf-8")}
    rows = [r for r in rows if r["i"] not in done]
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_new + 1024, gpu_memory_utilization=0.9,
              tensor_parallel_size=a.tp)
    sp = SamplingParams(max_tokens=a.max_new, temperature=a.temperature, top_p=a.top_p, seed=a.seed)
    with open(a.out, "a", encoding="utf-8") as f:
        for k in range(0, len(rows), a.chunk):
            part = rows[k:k + a.chunk]
            outs = llm.generate([r["prompt"] for r in part], sp)
            for r, o in zip(part, outs):
                f.write(json.dumps(dict(i=r["i"], prompt=r["prompt"], completion=o.outputs[0].text),
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{k + len(part)} of {len(rows)} continued", flush=True)


def pack(a):
    import numpy as np
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    os.makedirs(a.out, exist_ok=True)
    offsets, n_tok, n_bytes = [], 0, 0
    with open(a.out + "/tokens.bin", "wb") as f:
        for line in open(a.texts, encoding="utf-8"):
            r = json.loads(line)
            text = r["prompt"] + r["completion"]
            ids = tok.encode(text).ids + [eot]
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            offsets.append(n_tok)
            n_tok += len(ids)
            n_bytes += len(text.encode("utf-8"))
    np.save(a.out + "/offsets.npy", np.array(offsets, dtype=np.int64))
    meta = dict(dtype="uint16", vocab_size=tok.get_vocab_size(), n_tokens=n_tok, n_docs=len(offsets),
                bytes_per_token=n_bytes / max(1, n_tok), tokenizer=a.tokenizer,
                source=dict(kind="teacher continuations of training openings", texts=a.texts))
    json.dump(meta, open(a.out + "/meta.json", "w"), indent=2)
    print(f"{a.out}: {len(offsets)} documents, {n_tok:,} tokens")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--mix", default="data/mix_pretrain.json")
    p.add_argument("--n", type=int, default=200000)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--out", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--max_new", type=int, default=512)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs")
    g.add_argument("--chunk", type=int, default=5000, help="openings per vLLM call (progress is saved after each)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", required=True)
    k = sub.add_parser("pack")
    k.add_argument("--texts", required=True)
    k.add_argument("--tokenizer", default=TOKENIZER)
    k.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dict(prompts=prompts, generate=generate, pack=pack)[a.cmd](a)


if __name__ == "__main__":
    main()
