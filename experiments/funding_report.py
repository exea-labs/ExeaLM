"""Write results/funding/README.md from the result files of the 15-16 November 2026 MI300X session.

python experiments/funding_report.py            # reads results/funding, results/weight_alter*, results/combined
"""
import glob
import json
import os
import statistics as st

R = os.path.join(os.path.dirname(__file__), "..", "results")
DOMS = ("en", "ja_wiki", "ja_web", "code")
NAMES = {"hybrid_gate": "ExeaLM hybrid", "hybrid_gate_norp_fix": "ExeaLM hybrid, no rotary attention in the starting menu (leak fix on)",
         "hybrid_gate_fix": "ExeaLM hybrid with the leak fix (#47)", "hybrid_gate_search_fix": "ExeaLM hybrid, circuit search on (leak fix on)",
         "hybrid_gate_wa": "ExeaLM hybrid + one weight-alter slot per TONN layer", "qwen35": "Qwen3.5-style",
         "llama_gate": "Llama + gated attention + QK-norm", "llama": "Llama", "hybrid_gate_fix_lr1e3": "ExeaLM hybrid (leak fix on)",
         "llama_lr1e3": "Llama", "hybrid_gate_norp_fix_lr1e3": "ExeaLM hybrid, no rotary attention in the starting menu (leak fix on)",
         "hybrid_gate_wahkey_fix": "ExeaLM hybrid + one weight-alter slot per TONN layer, key from the edited layer (leak fix on)",
         "hybrid_gate_wahkey_in_fix": "ExeaLM hybrid + one weight-alter slot, key from the edited layer, input matrix (leak fix on)", "llama_gate_lr1e3": "Llama + gated attention + QK-norm", "qwen35_lr1e3": "Qwen3.5-style"}


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(R, "funding", "lm_compare_*.json"))):
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            size, rest = r["tag"].split("_", 1)
            model, seed = rest.rsplit("_s", 1)
            b = r["bits_per_byte"]
            runs.setdefault((size, model), {})[int(seed)] = dict(mean=sum(b[d] for d in DOMS) / 4, params=r["params_M"], **b)
    return runs


def table(runs, size, order):
    out = ["| Model | Parameters | Seeds | English | Japanese wiki | Japanese web | Code | Mean | Median | Min to max | Per seed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        s = runs.get((size, m))
        if not s:
            continue
        v = [s[k] for k in sorted(s)]
        means = [x["mean"] for x in v]
        out.append(f"| {NAMES[m]} | {v[0]['params']}M | {len(v)} | " + " | ".join(f"{st.mean(x[d] for x in v):.3f}" for d in DOMS)
                   + f" | **{st.mean(means):.3f}** | {st.median(means):.3f} | {min(means):.3f} to {max(means):.3f} | "
                   + ", ".join(f"{x:.3f}" for x in means) + " |")
    return "\n".join(out)


def paired(runs, size, a, others):
    out = []
    for b in others:
        A, B = runs.get((size, a)), runs.get((size, b))
        if not A or not B or not set(A) & set(B):
            continue
        sh = sorted(set(A) & set(B))
        d = [B[k]["mean"] - A[k]["mean"] for k in sh]
        out.append(f"- {NAMES[a]} against {NAMES[b]}: {st.mean(d):+.3f} bits per byte on average (positive: ExeaLM is better), "
                   f"better on {sum(x > 0 for x in d)} of {len(d)} seeds (per seed {', '.join(f'{x:+.3f}' for x in d)})")
    return "\n".join(out)


def relay_rows():
    rows = [("Board", "seed 0 (from its log)", 0.999, 0.021, ""), ("Weight-alter", "seed 0 (from its log)", 0.5107, 0.0166, "")]
    for folder, label in (("weight_alter", ""), ("weight_alter_in", ", edits to the MLP's input matrix"), ("weight_alter_diag", ", rerun scored by question"),
                          ("weight_alter_hkey", ", key from the edited layer")):
        for f in sorted(glob.glob(os.path.join(R, folder, "relay*.json"))):
            d = json.load(open(f, encoding="utf-8"))
            a = d["args"]
            for r in d["runs"]:
                if r["arm"] != "swarm":
                    continue
                name = "Board" if a["memory"] == "AG" else f"Weight-alter{label}" + (f", {a['wa_slots']} slots per layer" if a.get("wa_slots", 1) != 1 else "")
                if folder == "weight_alter_hkey" and a.get("wa_matrix") == "in":
                    name += ", edits to the MLP's input matrix"
                if a.get("facts", 4) != 4:
                    name += f", {a['facts']} facts"
                by = ", ".join(f"{100 * x:.0f}%" for x in r["final"]["shared"].get("by_question", []))
                rows.append((name, f"seed {a['seed']}", r["final"]["shared"]["acc"], r["final"]["cut"]["acc"], by))
    out = ["| Path | Run | Facts passed | With the path cut | By question, in the order asked |", "|---|---|---|---|---|"]
    return "\n".join(out + [f"| {n} | {s} | {100 * x:.1f}% | {100 * c:.1f}% | {b} |" for n, s, x, c, b in sorted(rows)])


def combined():
    out = ["| Model | Step | English | Mean of 4 sets | Relay (shared / cut) | Relay, board only | Relay, weight-alter only | Lookup (attached / removed) | Wikipedia gain from the library | Search hits |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for n, label in (("exealm_all3_seed0", "ExeaLM, all three circuits (94.7M), seed 0, lr 2e-3"),
                     ("exealm_all3_hkey_seed0", "ExeaLM, weight-alter key from the edited layer, seed 0, lr 2e-3"),
                     ("exealm_all3_s1_seed1", "ExeaLM, all three circuits, seed 1, lr 2e-3 (did not train)"),
                     ("exealm_all3_lr1e3_seed0", "ExeaLM, all three circuits, seed 0, lr 1e-3"),
                     ("exealm_all3_lr1e3_s1_seed1", "ExeaLM, all three circuits, seed 1, lr 1e-3"),
                     ("llama_base_seed0", "Llama (89.7M), seed 0, lr 2e-3"), ("llama_base_s1_seed1", "Llama, seed 1, lr 2e-3"),
                     ("llama_base_lr1e3_seed0", "Llama, seed 0, lr 1e-3"), ("llama_base_lr1e3_s1_seed1", "Llama, seed 1, lr 1e-3")):
        p = os.path.join(R, "combined", f"full_small_{n}.json")
        if not os.path.exists(p):
            continue
        curve = json.load(open(p, encoding="utf-8"))["curve"]
        for e in (curve if n in ("exealm_all3_seed0", "llama_base_seed0") else curve[-1:]):
            bpb = sum(e[v]["removed"] for v in ("val_en", "val_ja_wiki", "val_ja_web", "val_code")) / 4
            r, lk, k = e["relay"], e["lookup"], e.get("knowledge", {})
            pc = lambda x: f"{100 * x:.1f}%"
            out.append(f"| {label} | {e['step']:,} | {e['val_en']['removed']:.3f} | {bpb:.3f} | {pc(r['shared'])} / {pc(r['cut'])} | "
                       f"{pc(r['board_only']) if 'board_only' in r else ''} | {pc(r['weight_alter_only']) if 'weight_alter_only' in r else ''} | "
                       f"{pc(lk['attached']) + ' / ' if 'attached' in lk else ''}{pc(lk['removed'])} | "
                       f"{k.get('retrieval_gain', '')} | {pc(k['same_article_hits']) if 'same_article_hits' in k else ''} |")
    return "\n".join(out)


def throughput():
    p = os.path.join(R, "funding", "throughput_mi300x.json")
    if not os.path.exists(p):
        return ""
    out = ["| Size | Model | Parameters | Precision | Seconds per step | Tokens per second | Effective TFLOP/s |", "|---|---|---|---|---|---|---|"]
    for r in json.load(open(p, encoding="utf-8")):
        out.append(f"| {r['size']} | {'ExeaLM hybrid' if r['model'] == 'hybrid' else 'Llama'} | {r['params_M']}M | {r['precision']} | {r['s_per_step']} | "
                   f"{r['tokens_per_s']:,} | {r['effective_TFLOPs']} |")
    return "\n".join(out)


def main():
    runs = load()
    small = ["hybrid_gate", "hybrid_gate_norp_fix", "hybrid_gate_fix", "hybrid_gate_search_fix", "hybrid_gate_wa", "hybrid_gate_wahkey_fix", "hybrid_gate_wahkey_in_fix", "qwen35", "llama_gate", "llama"]
    big = ["hybrid_gate_norp_fix_lr1e3", "hybrid_gate_fix_lr1e3", "qwen35_lr1e3", "llama_gate_lr1e3", "llama_lr1e3"]
    text = open(os.path.join(R, "funding", "README.in.md"), encoding="utf-8").read()
    text = (text.replace("{{TABLE_13M}}", table(runs, "f13", small))
            .replace("{{PAIRS_13M}}", paired(runs, "f13", "hybrid_gate", ["qwen35", "llama_gate", "llama"]) + "\n"
                     + paired(runs, "f13", "hybrid_gate_norp_fix", ["qwen35", "llama_gate", "llama"]))
            .replace("{{PAIRS_FIX}}", paired(runs, "f13", "hybrid_gate_fix", ["hybrid_gate"]).replace("ExeaLM is better", "the fix is better"))
            .replace("{{TABLE_100M}}", table(runs, "f100", big))
            .replace("{{PAIRS_100M}}", paired(runs, "f100", "hybrid_gate_fix_lr1e3", ["qwen35_lr1e3", "llama_gate_lr1e3", "llama_lr1e3"]) + "\n"
                     + paired(runs, "f100", "hybrid_gate_norp_fix_lr1e3", ["qwen35_lr1e3", "llama_gate_lr1e3", "llama_lr1e3"]))
            .replace("{{RELAY}}", relay_rows()).replace("{{COMBINED}}", combined()).replace("{{THROUGHPUT}}", throughput()))
    open(os.path.join(R, "funding", "README.md"), "w", encoding="utf-8", newline="\n").write(text)
    print(text)


if __name__ == "__main__":
    main()
