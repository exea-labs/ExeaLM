"""The architecture leaderboard: every TONN-vs-Llama run (experiments/tokenizer_lm.py) in one markdown table, ranked by
average bits per byte over English, Japanese (wiki and web) and code, with the gap to the Llama baseline (lr 2e-3,
averaged over its seeds). Lower is better; a negative gap means better than Llama.

python experiments/leaderboard.py results/arch results/arch2 results/sweep
"""
import glob
import json
import os
import sys

DOMS = ("en", "ja_wiki", "ja_web", "code")


def rows(dirs):
    out = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "lm_compare*.json"))):
            for r in json.load(open(f, encoding="utf-8"))["rows"]:
                r = dict(r)
                r.setdefault("tag", "")
                r.setdefault("cfg", "{}")
                r.setdefault("lr", json.load(open(f, encoding="utf-8"))["args"].get("lr", 2e-3))
                r["mean"] = sum(r["bits_per_byte"][k] for k in DOMS) / len(DOMS)
                r["source"] = os.path.basename(d.rstrip("/"))
                out.append(r)
    return out


def main(dirs):
    rs = rows(dirs)
    base = [r for r in rs if r["model"] == "llama" and abs(r["lr"] - 2e-3) < 1e-9]
    ref = sum(r["mean"] for r in base) / len(base) if base else None
    rs.sort(key=lambda r: r["mean"])
    print("| Rank | Run | Model | Config | lr | Seed | English | Japanese wiki | Japanese web | Code | Mean | vs Llama |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rs, 1):
        b = r["bits_per_byte"]
        gap = "" if ref is None else f"{r['mean'] - ref:+.3f}"
        cfg = r["cfg"] if r["cfg"] not in ("{}", "") else ""
        print(f"| {i} | {r['tag'] or r['source']} | {r['model']} | {cfg} | {r['lr']:g} | {r['seed']} | "
              + " | ".join(f"{b[k]:.3f}" for k in DOMS) + f" | {r['mean']:.3f} | {gap} |")
    if ref is not None:
        print(f"\nLlama baseline (lr 2e-3, {len(base)} seeds): mean {ref:.3f} bits per byte.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/arch", "results/arch2", "results/sweep"])
