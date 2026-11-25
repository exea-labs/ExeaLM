"""What the funding application cites, from the scores of the study's finished runs (docs/budget.md, "Reading the
results"). Run it after eval.sb, which also runs it:

    python experiments/expanse/report.py                    # reads results/expanse/, writes report.md, report.json
    python experiments/expanse/report.py --rates results/expanse/throughput_h100.json

It reads text.json (eval_text.py: bits per byte on every held-out set, every row read alone) and, when they are there,
stream8.json (stream_eval.py), bench_<run>.json (the lm-eval harness) and nine_b.json (nine_b.py). It writes:

  - the curve: for every size and model, bits per byte (mean over seeds, and each seed), the gap to ExeaLM, and on how
    many seeds ExeaLM is ahead;
  - the compute multiplier: a power law, bits per byte = A x C^-k, fitted to each baseline's curve (C is training
    compute, 6 x parameters x tokens); for every ExeaLM point, the compute the baseline would need to reach ExeaLM's
    score, divided by ExeaLM's own. In FLOPs, and in GPU-hours at each model's measured speed. A pure power law with no
    floor, fitted to three or four points: a guide, not a law;
  - the long pair: the gap at the 9B run's 111 tokens per parameter against the gap at 4.1;
  - every held-out set (code, research text, ...) at every size;
  - long documents read as a stream: bits per byte by window;
  - downstream benchmarks;
  - the price of the 9B run at the speed measured for the 9B model (or, before that, for ExeaLM at 3B);
  - scaling.png: bits per byte against parameters, every seed and the means.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan  # noqa: E402

NAME = re.compile(r"^(exealm|llama|qwen)_([0-9a-z]+?)_([\d.]+)x_s(\d+)$")
LABEL = {"exealm": "ExeaLM", "qwen": "Qwen3.5-style", "llama": "Llama"}
BASELINES = ("qwen", "llama")


def parse(run):
    """(model, preset, tokens per parameter, seed) from a run's folder name, or None for runs outside the plan."""
    m = NAME.match(os.path.basename(os.path.normpath(run)))
    return (m[1], m[2], float(m[3]), int(m[4])) if m else None


def tokens_of(preset, tpp):
    return plan.steps_of(preset, tpp) * plan.STEP_TOKENS


def collect(text):
    """{(model, preset, tpp): {"seeds": {seed: mean}, "sets": {set: [scores]}, "params": M}} from text.json."""
    out = {}
    for r in text["runs"]:
        key = parse(r["run"])
        if key is None or key[1] not in plan.SIZES:
            continue
        model, preset, tpp, seed = key
        g = out.setdefault((model, preset, tpp), dict(seeds={}, sets={}, params=plan.SIZES[preset][
            "core" if model == "exealm" else model]))
        g["seeds"][seed] = r["mean"]
        for s, v in r["bits_per_byte"].items():
            g["sets"].setdefault(s, []).append(v)
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def fit(points):
    """A and k of bits per byte = A x C^-k, least squares in log-log, from (compute, bits per byte) points."""
    c, l = np.log([p[0] for p in points]), np.log([p[1] for p in points])
    slope, intercept = np.polyfit(c, l, 1)
    return math.exp(intercept), -slope


def multipliers(groups, rates):
    """For every ExeaLM point of the curve and every baseline with at least two points: the compute the baseline
    would need for ExeaLM's score, over ExeaLM's compute."""
    rows = []
    curve = sorted({(p, t) for (m, p, t) in groups if m == "exealm" and t == plan.TPP},
                   key=lambda pt: plan.SIZES[pt[0]]["core"])
    for base in BASELINES:
        pts = [(6 * groups[(base, p, t)]["params"] * 1e6 * tokens_of(p, t), mean(groups[(base, p, t)]["seeds"].values()))
               for p, t in curve if (base, p, t) in groups]
        if len(pts) < 2:
            continue
        a_, k = fit(pts)
        if k <= 0:
            rows.append(dict(baseline=base, error="the baseline's curve does not fall with compute: no fit"))
            continue
        top = max(c for c, _ in pts)
        for p, t in curve:
            e = groups[("exealm", p, t)]
            score = mean(e["seeds"].values())
            c_e = 6 * e["params"] * 1e6 * tokens_of(p, t)
            c_b = (score / a_) ** (-1 / k)
            flops = c_b / c_e
            hours = flops * plan.rate_of(rates, p, "exealm") / plan.rate_of(rates, p, base)
            rows.append(dict(baseline=base, size=p, exealm_bits_per_byte=round(score, 4), flops_multiplier=round(flops, 2),
                             gpu_hour_multiplier=round(hours, 2), extrapolated=bool(c_b > top), A=a_, k=k))
    return rows


def stream_rows(path):
    if not os.path.exists(path):
        return []
    groups = {}
    for r in json.load(open(path))["runs"]:
        key = parse(r["run"])
        if key:
            groups.setdefault(key[:3], []).append(r)
    return [dict(model=m, size=p, tpp=t, streamed=[round(mean(x), 4) for x in zip(*[r["streamed"] for r in rs])],
                 alone=[round(mean(x), 4) for x in zip(*[r["alone"] for r in rs])])
            for (m, p, t), rs in sorted(groups.items())]


def bench_rows(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "bench_*.json"))):
        res = json.load(open(f)).get("results", {})
        scores = {}
        for task, r in res.items():
            for k in ("acc_norm,none", "acc,none"):
                if isinstance(r.get(k), (int, float)):
                    scores[task] = round(100 * r[k], 1)
                    break
        rows.append(dict(run=os.path.basename(f)[6:-5], scores=scores))
    return rows


def nine_b_rate(folder):
    """Effective TFLOP/s per GPU of the 9B model at the most nodes measured, best batch, or None."""
    path = os.path.join(folder, "nine_b.json")
    if not os.path.exists(path):
        return None, None
    rows = [r for r in json.load(open(path)) if not r.get("error")]
    key = next((k for k in ("effective_TFLOPs_per_gpu", "tflops_per_gpu", "effective_TFLOPs") if rows and k in rows[0]),
               None)
    if not key:
        return None, None
    most = max(r.get("nodes", 1) for r in rows)
    best = max((r for r in rows if r.get("nodes", 1) == most), key=lambda r: r[key])
    return best[key], most


def figure(groups, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, colour in (("exealm", "#c2410c"), ("qwen", "#1d4ed8"), ("llama", "#6b7280")):
        pts = sorted((g["params"], g) for (m, p, t), g in groups.items() if m == model and t == plan.TPP)
        if not pts:
            continue
        for n, g in pts:
            ax.scatter([n] * len(g["seeds"]), list(g["seeds"].values()), color=colour, s=12, alpha=0.5)
        ax.plot([n for n, _ in pts], [mean(g["seeds"].values()) for _, g in pts], color=colour, marker="o",
                label=LABEL[model])
    ax.set_xscale("log")
    ax.set_xlabel("parameters (millions)")
    ax.set_ylabel("bits per byte, held-out text (lower is better)")
    ax.set_title(f"Same data, same tokens per parameter ({plan.TPP:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/expanse")
    ap.add_argument("--rates", default=None, help="a throughput_<gpu>.json; default: the MI300X rates in plan.py")
    a = ap.parse_args(argv)
    rates, _ = plan.measured(json.load(open(a.rates)) if a.rates else [])
    text = json.load(open(os.path.join(a.results, "text.json")))
    groups = collect(text)
    lines, report = ["# The scaling study: results", ""], {}

    sizes = sorted({p for (_, p, _) in groups}, key=lambda p: plan.SIZES[p]["core"])
    lines += [f"## Bits per byte at {plan.TPP:g} tokens per parameter (lower is better)", "",
              "| Size | ExeaLM | Qwen3.5-style | Llama | ExeaLM ahead of Qwen3.5-style | ExeaLM ahead of Llama |",
              "|---|---|---|---|---|---|"]
    curve = []
    for p in sizes:
        row = dict(size=p)
        for m in ("exealm", "qwen", "llama"):
            g = groups.get((m, p, plan.TPP))
            row[m] = dict(mean=round(mean(g["seeds"].values()), 4), seeds=g["seeds"]) if g else None
        if not row["exealm"]:
            continue
        cells = []
        for m in BASELINES:
            if row[m]:
                gap = row[m]["mean"] - row["exealm"]["mean"]
                both = sorted(set(row[m]["seeds"]) & set(row["exealm"]["seeds"]))
                ahead = sum(row[m]["seeds"][s] > row["exealm"]["seeds"][s] for s in both)
                row[f"gap_{m}"], row[f"ahead_{m}"] = round(gap, 4), f"{ahead} of {len(both)}"
                cells.append(f"{gap:+.3f}, {ahead} of {len(both)} seeds")
            else:
                cells.append("not trained")
        fmt = lambda m: f"{row[m]['mean']:.3f} ({len(row[m]['seeds'])})" if row[m] else "-"      # noqa: E731
        lines.append(f"| {p} | {fmt('exealm')} | {fmt('qwen')} | {fmt('llama')} | {cells[0]} | {cells[1]} |")
        curve.append(row)
    report["curve"] = curve
    lines += ["", "In brackets: seeds. A positive gap means ExeaLM is better.", ""]
    gaps = [r["gap_qwen"] for r in curve if "gap_qwen" in r]
    if len(gaps) >= 2:
        lines.append(f"Gap to the Qwen3.5-style model from the smallest to the largest size: {gaps[0]:+.3f} to "
                     f"{gaps[-1]:+.3f} ({'not shrinking' if gaps[-1] >= gaps[0] else 'shrinking'}).")
        report["gap_trend_qwen"] = dict(first=gaps[0], last=gaps[-1], shrinking=gaps[-1] < gaps[0])

    mult = multipliers(groups, rates)
    report["multiplier"] = mult
    if mult:
        lines += ["", "## Compute multiplier", "",
                  "The compute a baseline would need to reach ExeaLM's score, over ExeaLM's compute, from a power law "
                  "fitted to the baseline's curve. Above 1, ExeaLM needs less.", "",
                  "| Baseline | Size | In FLOPs | In GPU-hours | |", "|---|---|---|---|---|"]
        for r in mult:
            if "error" in r:
                lines.append(f"| {LABEL[r['baseline']]} | | | | {r['error']} |")
            else:
                lines.append(f"| {LABEL[r['baseline']]} | {r['size']} | {r['flops_multiplier']:.2f}x | "
                             f"{r['gpu_hour_multiplier']:.2f}x | {'beyond its largest run' if r['extrapolated'] else ''} |")

    longs = [(p, t) for (m, p, t) in groups if m == "exealm" and t != plan.TPP]
    if longs:
        lines += ["", "## Trained to the 9B run's tokens per parameter", ""]
        report["long"] = []
        for p, t in sorted(longs):
            e, q = groups[("exealm", p, t)], groups.get(("qwen", p, t))
            short = next((r for r in curve if r["size"] == p and "gap_qwen" in r), None)
            if q:
                gap = mean(q["seeds"].values()) - mean(e["seeds"].values())
                report["long"].append(dict(size=p, tpp=t, gap_qwen=round(gap, 4),
                                           gap_qwen_at_curve=short["gap_qwen"] if short else None))
                lines.append(f"At {p}, {t:g} tokens per parameter: ExeaLM {mean(e['seeds'].values()):.3f}, Qwen3.5-style "
                             f"{mean(q['seeds'].values()):.3f}, gap {gap:+.3f}"
                             + (f" (at {plan.TPP:g}: {short['gap_qwen']:+.3f})." if short else "."))

    sets = sorted({s for g in groups.values() for s in g["sets"]})
    if sets:
        lines += ["", "## Every held-out set", "", "| Size | Model | " + " | ".join(sets) + " |",
                  "|---|---|" + "---|" * len(sets)]
        for p in sizes:
            for m in ("exealm", "qwen", "llama"):
                g = groups.get((m, p, plan.TPP))
                if g:
                    lines.append(f"| {p} | {LABEL[m]} | " + " | ".join(
                        f"{mean(g['sets'][s]):.3f}" if s in g["sets"] else "-" for s in sets) + " |")

    stream = stream_rows(os.path.join(a.results, "stream8.json"))
    report["stream"] = stream
    if stream:
        lines += ["", "## Long documents read as a stream: bits per byte by window", "",
                  "| Size | Model | First window | Last window | Last window, read alone |", "|---|---|---|---|---|"]
        for r in stream:
            lines.append(f"| {r['size']} ({r['tpp']:g}x) | {LABEL[r['model']]} | {r['streamed'][0]:.3f} | "
                         f"{r['streamed'][-1]:.3f} | {r['alone'][-1]:.3f} |")

    bench = bench_rows(a.results)
    report["benchmarks"] = bench
    if bench:
        tasks = sorted({t for r in bench for t in r["scores"]})
        lines += ["", "## Benchmarks (accuracy, percent, zero-shot)", "", "| Run | " + " | ".join(tasks) + " |",
                  "|---|" + "---|" * len(tasks)]
        for r in bench:
            lines.append(f"| `{r['run']}` | " + " | ".join(f"{r['scores'].get(t, '-')}" for t in tasks) + " |")

    rate, nodes = nine_b_rate(a.results)
    source = f"the 9B model on {nodes} node(s)" if rate else "ExeaLM at 3B"
    rate = rate or plan.rate_of(rates, "3b", "exealm")
    price = plan.gpu_hours(plan.NINE_B_PARAMS / 1e6, plan.TARGET[1], rate)
    report["nine_b"] = dict(rate_TFLOPs=rate, rate_from=source, gpu_hours=round(price))
    lines += ["", "## The 9B run", "",
              f"ExeaLM at 9B on 1T tokens at {rate:g} effective TFLOP/s per GPU ({source}): about {price:,.0f} GPU-hours."]

    if figure(groups, os.path.join(a.results, "scaling.png")):
        lines += ["", "![Bits per byte against parameters](scaling.png)"]
    open(os.path.join(a.results, "report.md"), "w").write("\n".join(lines) + "\n")
    json.dump(report, open(os.path.join(a.results, "report.json"), "w"), indent=1, default=str)
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    main()
