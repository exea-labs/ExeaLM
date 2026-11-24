"""Consolidate the same-size comparisons into one table: python experiments/funding_table.py results/funding

Reads every lm_compare_*.json in the folder (rows written by experiments/tokenizer_lm.py), groups the runs by model
and size, and prints for each group the mean and standard deviation over seeds of bits per byte (mean of English,
Japanese wiki, Japanese web and code), plus the paired difference against every other model on the seeds they share.
"""
import glob
import json
import os
import statistics as st
import sys

NAMES = {"hybrid_gate": "ExeaLM hybrid (TONN + gated attention)", "qwen35": "Qwen3.5-style",
         "llama_gate": "Llama + gated attention + QK-norm", "llama": "Llama"}
DOMS = ("en", "ja_wiki", "ja_web", "code")


def load(folder):
    runs = {}
    for f in sorted(glob.glob(os.path.join(folder, "lm_compare_*.json"))):
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            tag = r["tag"]                                   # f13_<model>_s<seed> or f48_<model>_lr.._s<seed>
            size, rest = tag.split("_", 1)
            model, seed = rest.rsplit("_s", 1)
            b = r["bits_per_byte"]
            runs.setdefault((size, model), {})[int(seed)] = dict(
                mean=sum(b[d] for d in DOMS) / 4, params=r["params_M"], steps=r["steps"], secs=r["secs"], **b)
    return runs


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/funding"
    runs = load(folder)
    out = {}
    for size in sorted({s for s, _ in runs}):
        group = {m: v for (s, m), v in runs.items() if s == size}
        print(f"\n## {size}\n")
        print("| Model | Parameters | Seeds | English | Japanese wiki | Japanese web | Code | Mean | Std over seeds | Seconds |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for m, seeds in sorted(group.items(), key=lambda kv: st.mean(x["mean"] for x in kv[1].values())):
            vals = list(seeds.values())
            mean = st.mean(x["mean"] for x in vals)
            sd = st.stdev(x["mean"] for x in vals) if len(vals) > 1 else float("nan")
            base = m.split("_lr")[0]
            name = NAMES.get(base, base) + (f" (lr {m.split('_lr')[1]})" if "_lr" in m else "")
            print(f"| {name} | {vals[0]['params']}M | {len(vals)} | " + " | ".join(f"{st.mean(x[d] for x in vals):.3f}" for d in DOMS)
                  + f" | **{mean:.4f}** | {sd:.4f} | {st.mean(x['secs'] for x in vals):.0f} |")
            out[f"{size}/{m}"] = dict(seeds=sorted(seeds), mean=mean, std=sd, per_seed={s: seeds[s]["mean"] for s in sorted(seeds)})
        ours = [m for m in group if m.startswith("hybrid_gate")]
        for o in ours:
            print()
            for m, seeds in sorted(group.items()):
                if m == o:
                    continue
                shared = sorted(set(seeds) & set(group[o]))
                if not shared:
                    continue
                diffs = [seeds[s]["mean"] - group[o][s]["mean"] for s in shared]      # positive: ExeaLM is better
                wins = sum(d > 0 for d in diffs)
                sd = st.stdev(diffs) if len(diffs) > 1 else float("nan")
                print(f"- {o} against {m}: ahead by {st.mean(diffs):+.4f} bits per byte (std {sd:.4f}), "
                      f"better on {wins} of {len(diffs)} seeds; per seed {[round(d, 4) for d in diffs]}")
                out[f"{size}/{o}_vs_{m}"] = dict(seeds=shared, diffs=diffs, mean=st.mean(diffs), wins=wins)
    json.dump(out, open(os.path.join(folder, "summary.json"), "w", encoding="utf-8", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
