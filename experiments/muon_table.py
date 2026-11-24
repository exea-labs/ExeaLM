"""Muon against AdamW (#21): python experiments/muon_table.py results/muon

Reads every lm_compare_*.json in the folder (rows written by experiments/tokenizer_lm.py with --optimizer), prints the
learning-rate sweep (bits per byte: mean of English, Japanese wiki, Japanese web and code), picks each optimizer's best
learning rate on seed 0, and prints the paired difference between the two at those rates on every seed both have.
Seed 0 chose the rates, so the other seeds are the clean check. Writes <folder>/summary.json.
"""
import glob
import json
import os
import statistics as st
import sys

DOMS = ("en", "ja_wiki", "ja_web", "code")


def load(folder):
    runs = {}                                                  # (optimizer, lr) -> {seed: row}
    for f in sorted(glob.glob(os.path.join(folder, "lm_compare_*.json"))):
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            b = r["bits_per_byte"]
            runs.setdefault((r.get("optimizer", "adamw"), r["lr"]), {})[r["seed"]] = dict(
                mean=sum(b[d] for d in DOMS) / len(DOMS), secs=r["secs"], steps=r["steps"], **b)
    return runs


def best_lr(runs, opt):
    swept = {lr: seeds[0]["mean"] for (o, lr), seeds in runs.items() if o == opt and 0 in seeds}
    return min(swept, key=swept.get) if swept else None


def summarize(runs):
    seeds = sorted({s for v in runs.values() for s in v})
    lines = ["| Optimizer | Learning rate | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean | Seconds |",
             "|---|---|" + "---|" * len(seeds) + "---|---|"]
    for (opt, lr), v in sorted(runs.items()):
        cells = [f"{v[s]['mean']:.4f}" if s in v else "" for s in seeds]
        lines.append(f"| {opt} | {lr:g} | " + " | ".join(cells)
                     + f" | {st.mean(x['mean'] for x in v.values()):.4f} | {st.mean(x['secs'] for x in v.values()):.0f} |")
    out = dict(runs={f"{o}_lr{lr:g}": {str(s): x["mean"] for s, x in sorted(v.items())} for (o, lr), v in runs.items()})
    best = {o: best_lr(runs, o) for o in ("adamw", "muon")}
    if None not in best.values():
        a, m = runs[("adamw", best["adamw"])], runs[("muon", best["muon"])]
        shared = sorted(set(a) & set(m))
        diffs = [a[s]["mean"] - m[s]["mean"] for s in shared]           # positive: Muon is better
        sd = st.stdev(diffs) if len(diffs) > 1 else float("nan")
        lines += ["", f"Best learning rate on seed 0: AdamW {best['adamw']:g}, Muon {best['muon']:g}.",
                  f"- Muon against AdamW at those rates: ahead by {st.mean(diffs):+.4f} bits per byte (std {sd:.4f}), "
                  f"better on {sum(d > 0 for d in diffs)} of {len(diffs)} seeds; per seed "
                  + ", ".join(f"{s}: {d:+.4f}" for s, d in zip(shared, diffs))]
        out.update(best_lr=best, seeds=shared, diffs=diffs, mean_diff=st.mean(diffs))
    return "\n".join(lines), out


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/muon"
    text, out = summarize(load(folder))
    print(text)
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
