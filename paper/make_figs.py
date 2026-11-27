"""Figures and generated tables for the ExeaLM paper, drawn from the result files in results/expanse.

Run from anywhere:  python paper/make_figs.py
Reads only JSON; loads no model. Writes paper/figs/*.pdf and paper/figs/tab_sets.tex, and prints the numbers the
text quotes so they can be checked against results/expanse/README.md.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "expanse"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

# Reference categorical palette (validated: slots 1-3, light surface). Text stays in ink.
BLUE = "#2a78d6"     # slot 1
ORANGE = "#eb6834"   # slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
PEND = "#f1f0ec"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


# ----------------------------------------------------------------------------------------------- reading the files
def load(name):
    return json.loads((RES / name).read_text())


def by_group(name):
    """{group: {seed: run}} for a results file; a group is a run name without its _s<seed> ending."""
    out = {}
    for r in load(name)["runs"]:
        m = re.match(r"(.*)_s(\d+)$", Path(r["run"]).name)
        out.setdefault(m.group(1), {})[int(m.group(2))] = r
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def alone(name, group):
    """Windows read alone: per-seed means of the 18 held-out sets, and their mean."""
    runs = by_group(name)[group]
    per = {s: r["mean"] for s, r in sorted(runs.items())}
    return per, mean(per.values())


def streamed(name, group, first=1, last=None):
    """Long documents read as a stream: per-seed means of windows first..last, and their mean."""
    runs = by_group(name)[group]
    per = {}
    for s, r in sorted(runs.items()):
        w = r["streamed"]
        hi = len(w) - 1 if last is None else last
        per[s] = mean(w[first:hi + 1])
    return per, mean(per.values())


def stream_curve(name, group):
    """Per-window streamed bits per byte, averaged over seeds, and the number of seeds."""
    runs = by_group(name)[group]
    n = len(next(iter(runs.values()))["streamed"])
    return [mean(r["streamed"][w] for r in runs.values()) for w in range(n)], len(runs)


def pct(ours, theirs):
    return 100.0 * (theirs - ours) / theirs


def lead_pct_by_seed(per_ours, per_theirs):
    return [pct(per_ours[s], per_theirs[s]) for s in per_ours if s in per_theirs]


def pairs_ahead(name, ours, theirs):
    """Set-and-seed pairs where `ours` scores lower on windows read alone, out of all pairs; and the range of leads."""
    g = by_group(name)
    a, b = g[ours], g[theirs]
    d = [b[s]["bits_per_byte"][k] - a[s]["bits_per_byte"][k] for s in a if s in b for k in a[s]["bits_per_byte"]]
    return sum(x > 0 for x in d), len(d), min(d), max(d)


def pairs_ahead_streamed(name, ours, theirs):
    g = by_group(name)
    a, b = g[ours], g[theirs]
    d = [mean(b[s]["sets"][k]["streamed"][1:]) - mean(a[s]["sets"][k]["streamed"][1:])
         for s in a if s in b for k in a[s]["sets"]]
    return sum(x > 0 for x in d), len(d), min(d), max(d)


# ----------------------------------------------------------------------------------------------- the numbers
SIZES = [
    # label, text file, ExeaLM group, Qwen3.5-style group, Llama group, stream file
    ("0.1B", "text_100m.json", "exealm_midstream_100m_4.1x", "qwen_100m_4.1x", "llama_100m_4.1x", "stream8_100m.json"),
    ("0.25B", "text_250m.json", "exealm_250m_4.1x", "qwen_250m_4.1x", "llama_250m_4.1x", "stream8_250m.json"),
    ("0.5B", "text_500m_all.json", "exealm_500m_4.1x", "qwen_500m_4.1x", "llama_500m_4.1x", "stream8_500m_all.json"),
]
LONG = ("110", "text_111x.json", "exealm_100m_111x", "qwen_100m_111x", None, "stream8_111x.json")

rows = {}
print("== Leads over the Qwen3.5-style model (bits per byte; percent of the Qwen3.5-style score)")
for label, tfile, ex, qw, ll, sfile in SIZES + [LONG]:
    ea, eam = alone(tfile, ex)
    qa, qam = alone(tfile, qw)
    es, esm = streamed(sfile, ex)
    qs, qsm = streamed(sfile, qw)
    rows[label] = dict(alone=(eam, qam), long=(esm, qsm), alone_seeds=lead_pct_by_seed(ea, qa),
                       long_seeds=lead_pct_by_seed(es, qs), seeds=len(ea))
    p = pairs_ahead(tfile, ex, qw)
    ps = pairs_ahead_streamed(sfile, ex, qw)
    print(f"{label:>6}: windows alone {eam:.4f} vs {qam:.4f}, lead {qam - eam:.4f} ({pct(eam, qam):.2f}%), "
          f"pairs ahead {p[0]}/{p[1]} ({p[2]:+.4f} to {p[3]:+.4f})")
    print(f"        long documents {esm:.4f} vs {qsm:.4f}, lead {qsm - esm:.4f} ({pct(esm, qsm):.2f}%), "
          f"pairs ahead {ps[0]}/{ps[1]} ({ps[2]:+.4f} to {ps[3]:+.4f})")
    if ll:
        la, lam = alone(tfile, ll)
        ls, lsm = streamed(sfile, ll)
        pl = pairs_ahead(tfile, ex, ll)
        print(f"        plain Llama: windows alone {lam:.4f}, streamed {lsm:.4f}; ExeaLM ahead on {pl[0]}/{pl[1]}")

print("\n== Controls at 0.1B (3 seeds each)")
CONTROLS = [
    # label, text file, group, stream file, group in the stream file
    ("no stream phase", "text_100m.json", "exealm_mid_100m_4.1x", "stream8_100m.json", "exealm_mid_100m_4.1x"),
    ("mixed, last 20%", "text_mix20distill.json", "exealm_mix20_100m_4.1x", "stream8_mix20distill.json",
     "exealm_mix20_100m_4.1x"),
    ("mixed, last 40%", "text_mix100m.json", "exealm_mix40_100m_4.1x", "stream8_mix100m.json", "exealm_mix40_100m_4.1x"),
    ("streams, last 20%", "text_100m.json", "exealm_midstream_100m_4.1x", "stream8_100m.json",
     "exealm_midstream_100m_4.1x"),
    ("attention-only control", "text_100m_strong.json", "strongllama_100m_4.1x", "stream8_100m_strong.json",
     "strongllama_100m_4.1x"),
    ("weights averaged", "text_merge100m.json", "exealm_merge_100m_4.1x", "stream8_merge100m.json",
     "exealm_merge_100m_4.1x"),
]
recipes = {}
for label, tfile, g, sfile, sg in CONTROLS:
    a, am = alone(tfile, g)
    s, sm = streamed(sfile, sg)
    recipes[label] = dict(alone=a, alone_mean=am, long=s, long_mean=sm)
    print(f"{label:>24}: windows alone {am:.4f} {[round(v, 4) for v in a.values()]}, "
          f"4,096 streamed {sm:.4f} {[round(v, 4) for v in s.values()]}")
p = pairs_ahead("text_100m_strong.json", "exealm_100m_4.1x", "strongllama_100m_4.1x")
ps = pairs_ahead_streamed("stream8_100m_strong.json", "exealm_100m_4.1x", "strongllama_100m_4.1x")
print(f"ExeaLM against the attention-only control: windows alone {p[0]}/{p[1]} pairs, streamed {ps[0]}/{ps[1]}")
dm, _ = alone("text_mix20distill.json", "distill_exealm_100m")
print(f"distillation (30% teacher text), seed 0: {dm} against the study's seed 0 "
      f"{alone('text_100m.json', 'exealm_midstream_100m_4.1x')[0][0]}")

print("\n== 16,384-token documents (windows 1-31, 1-7, 24-31)")
for g in ("exealm_100m_4.1x", "strongllama_100m_4.1x", "qwen_100m_4.1x", "exealm_250m_4.1x", "qwen_250m_4.1x"):
    c, n = stream_curve("stream32_100m.json", g)
    print(f"{g:>24} ({n} seeds): {mean(c[1:32]):.4f}, {mean(c[1:8]):.4f}, {mean(c[24:32]):.4f}")

print("\n== Benchmarks at 0.5B (seed 0; acc_norm where the task has it, else acc)")
TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "sciq", "lambada_openai"]
for f in ("bench_exealm_500m_4.1x_s0.json", "bench_qwen_500m_4.1x_s0.json"):
    res = load(f)["results"]
    vals = [res[t].get("acc_norm,none", res[t]["acc,none"]) for t in TASKS]
    print(f"{f}: {[round(v, 3) for v in vals]}, mean {mean(vals):.4f}, mean of rounded {mean(round(v, 3) for v in vals):.4f}")

print("\n== Library v1, RAG post-training, 3 seeds (held-out questions: none, BM25, pasted)")
lib = {}
for s in range(3):
    for line in (RES / f"library_rag_100m_s{s}.jsonl").read_text().splitlines():
        r = json.loads(line)
        model = "exealm" if "exealm" in r["ckpt"] else "qwen"
        lib.setdefault((model, r["set"], r["condition"]), []).append(r)
for model in ("exealm", "qwen"):
    q = [mean(x["bits_per_byte"] for x in lib[(model, "qa", c)]) for c in ("none", "bm25", "pasted")]
    t = [mean(x["by_kind"]["text"] for x in lib[(model, "tasks", c)]) for c in ("none", "bm25")]
    k = [mean(x["by_kind"]["code"] for x in lib[(model, "tasks", c)]) for c in ("none", "bm25")]
    print(f"{model}: questions {[round(v, 3) for v in q]}, text tasks {[round(v, 3) for v in t]}, "
          f"code tasks {[round(v, 3) for v in k]}")
print("per-seed BM25 lead on questions:",
      [round(b["bits_per_byte"] - a["bits_per_byte"], 4)
       for a, b in zip(lib[("exealm", "qa", "bm25")], lib[("qwen", "qa", "bm25")])])
ext = load("external_qwen35_9b_base.json")
print("Qwen3.5-9B-Base questions:", ext["library"]["qa/qa"], "; mean of 18 sets:", ext["mean_of_sets"])


# ----------------------------------------------------------------------------------------------- Figure 1: the model
def box(ax, x, y, w, h, fill, hatch=None, lw=0.7):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK, linewidth=lw, hatch=hatch))


def arrow(ax, a, b, color=INK, lw=0.8, style="-|>", ms=7, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, linewidth=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(6.75, 2.15))
ax.set_xlim(0, 100)
ax.set_ylim(0, 35)
ax.axis("off")
plt.rcParams["hatch.linewidth"] = 0.5

# (a) layer stacks at 0.1B, layer 0 at the bottom
ax.text(0.5, 34, "(a) Layers at 0.1B (14 layers)", fontsize=8.5, color=INK, va="top", weight="bold")
stacks = [
    ("ExeaLM", [7], []),
    ("Attention-only\ncontrol", [], []),
    ("Qwen3.5-\nstyle", [], [i for i in range(14) if i % 4 != 3]),
]
for j, (name, tonn, delta) in enumerate(stacks):
    x0 = 2.0 + j * 10.5
    for i in range(14):
        y = 7.0 + i * 1.75
        if i in tonn:
            box(ax, x0, y, 6.0, 1.45, INK)
            ax.text(x0 + 3.0, y + 0.72, "TONN", ha="center", va="center", fontsize=6, color="white")
        elif i in delta:
            box(ax, x0, y, 6.0, 1.45, "white", hatch="//////")
        else:
            box(ax, x0, y, 6.0, 1.45, "white")
    ax.text(x0 + 3.0, 6.2, name, ha="center", va="top", fontsize=7.5, color=INK)
# key
ky = 0.4
for x, fill, hatch, label in ((0.5, "white", None, "attention"), (10.6, INK, None, "TONN"),
                              (17.6, "white", "//////", "Gated DeltaNet")):
    box(ax, x, ky, 1.6, 1.2, fill, hatch=hatch)
    ax.text(x + 2.0, ky + 0.6, label, va="center", fontsize=7, color=INK2)

# (b) the training rows
ax.text(35.5, 34, "(b) Training rows", fontsize=8.5, color=INK, va="top", weight="bold")
ty = 27.0
box(ax, 36, ty, 49.6, 2.4, "white")
box(ax, 85.6, ty, 12.4, 2.4, PEND)
ax.text(60.8, ty + 1.2, "first 80% of steps: one 512-token window per row", ha="center", va="center", fontsize=7.5,
        color=INK)
ax.text(91.8, ty + 1.2, "stream phase", ha="center", va="center", fontsize=7.5, color=INK)
ax.text(36, ty - 1.3, "0", ha="center", va="top", fontsize=7, color=INK2)
ax.text(85.6, ty - 1.3, "80%", ha="center", va="top", fontsize=7, color=INK2)
ax.text(98, ty - 1.3, "100%", ha="center", va="top", fontsize=7, color=INK2)

# one stream row: four windows of one document
ry = 15.0
ax.text(36, ry + 7.6, "A stream row: four consecutive 512-token windows of one document", fontsize=7.5, color=INK,
        va="center")
for w in range(4):
    x = 36 + w * 15.5
    box(ax, x, ry, 11.0, 4.2, "white")
    ax.text(x + 5.5, ry + 2.1, f"window {w + 1}", ha="center", va="center", fontsize=7.5, color=INK)
    if w < 3:
        arrow(ax, (x + 11.0, ry + 2.9), (x + 15.5, ry + 2.9), lw=0.8)
        arrow(ax, (x + 11.0, ry + 1.3), (x + 15.5, ry + 1.3), lw=0.8, color=INK2)
ax.text(36, ry - 1.6, "Carried to the next window: the keys and values of every attention layer (upper arrow) and",
        fontsize=7, color=INK2, va="top")
ax.text(36, ry - 4.0, "the state of the TONN layer (lower arrow). The gradient flows through all four windows.",
        fontsize=7, color=INK2, va="top")
ax.text(36, ry - 7.6, "Mixed stream phase (the recipe from 26 November): half of each step's micro-batches are",
        fontsize=7, color=INK2, va="top")
ax.text(36, ry - 10.0, "stream rows and half are single windows, at the same tokens per step.",
        fontsize=7, color=INK2, va="top")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_model.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------------------------- Figure 2: the two trends
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.45), sharey=True, gridspec_kw=dict(width_ratios=[1.45, 1]))
for ax in (ax1, ax2):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def trend(ax, xs, labels_):
    for key, color, marker, ls, name, dy in (("long", BLUE, "o", "-", "Long documents, read as a stream", 7),
                                             ("alone", ORANGE, "s", (0, (4, 2)), "512-token windows, read alone", 7)):
        ys = [pct(*rows[k][key]) for k in labels_]
        for x, k in zip(xs, labels_):
            seeds = rows[k][key + "_seeds"]
            ax.scatter([x] * len(seeds), seeds, s=9, color=color, alpha=0.45, linewidths=0, zorder=2)
        ax.plot(xs, ys, color=color, linewidth=2, linestyle=ls, marker=marker, markersize=5, label=name, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, dy), ha="center", color=INK,
                        fontsize=7.5)


trend(ax1, [0, 1, 2], ["0.1B", "0.25B", "0.5B"])
ax1.axvspan(2.55, 3.45, color=PEND, zorder=0)
ax1.text(3.0, 3.2, "3B pair\nin training;\nfinishes\n29 Nov", ha="center", va="center", color=INK2, fontsize=7)
ax1.set_xticks([0, 1, 2, 3])
ax1.set_xticklabels(["0.1B\n3 seeds", "0.25B\n3 seeds", "0.5B\n2 seeds", "3B\n1 seed"])
ax1.set_xlim(-0.4, 3.5)
ax1.set_ylim(0, 6.9)
ax1.set_ylabel("ExeaLM's lead over the\nQwen3.5-style model (%)")
ax1.set_title("(a) By model size, 4 tokens per parameter", fontsize=8, color=INK, loc="left")
ax1.legend(loc="upper right", fontsize=7, handlelength=2.6)

trend(ax2, [0, 1], ["0.1B", "110"])
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["4 tokens per parameter\n0.38B tokens, 3 seeds", "110 tokens per parameter\n10.4B tokens, 2 seeds"])
ax2.set_xlim(-0.45, 1.45)
ax2.set_title("(b) By training length, 0.1B", fontsize=8, color=INK, loc="left")
fig.tight_layout(pad=0.3, w_pad=1.2)
fig.savefig(OUT / "fig_trends.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------------------------- Figure 3: by position
fig, ax = plt.subplots(figsize=(3.3, 2.25))
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
print("\n== Lead by window, 16,384-token documents (percent)")
for ours, theirs, color, marker, ls, label in [
    ("exealm_100m_4.1x", "qwen_100m_4.1x", BLUE, "o", "-", "0.1B, 3 seeds"),
    ("exealm_250m_4.1x", "qwen_250m_4.1x", ORANGE, "s", (0, (4, 2)), "0.25B, 1 seed"),
]:
    a, _ = stream_curve("stream32_100m.json", ours)
    b, _ = stream_curve("stream32_100m.json", theirs)
    lead = [pct(x, y) for x, y in zip(a, b)]
    ends = [(w + 1) * 512 / 1000 for w in range(len(lead))]
    ax.plot(ends, lead, color=color, linewidth=1.6, linestyle=ls, marker=marker, markersize=2.8, label=label)
    print(label, [round(v, 2) for v in lead])
ax.axvline(2.048, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
ax.text(2.4, 0.5, "length of the streams\nread in training", color=INK2, fontsize=7, va="bottom")
ax.set_xlim(0, 16.8)
ax.set_ylim(0, 6.6)
ax.set_xlabel("Position in the document (thousands of tokens)")
ax.set_ylabel("ExeaLM's lead over the\nQwen3.5-style model (%)")
ax.legend(loc="lower right", fontsize=7, handlelength=2.6)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_position.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------------------------- Figure 4: stream recipes
order = ["no stream phase", "mixed, last 20%", "mixed, last 40%", "streams, last 20%", "attention-only control"]
names = {
    "no stream phase": "No stream phase",
    "mixed, last 20%": "Mixed, last 20% (recipe)",
    "mixed, last 40%": "Mixed, last 40%",
    "streams, last 20%": "Streams, last 20%",
    "attention-only control": "Attention-only control",
}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.3, 1.9), sharey=True)
for ax, key, title, lim in ((ax1, "alone", "512-token windows,\nread alone", (1.1475, 1.1595)),
                            (ax2, "long", "4,096-token documents,\nread as a stream", (1.090, 1.137))):
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for i, label in enumerate(order):
        r = recipes[label]
        y = len(order) - 1 - i
        color = BLUE if label == "mixed, last 20%" else INK
        vals = list(r[key].values())
        ax.scatter(vals, [y] * len(vals), s=10, color=color, alpha=0.4, linewidths=0)
        ax.scatter([r[key + "_mean"]], [y], s=22, color=color, zorder=3,
                   marker="D" if label == "attention-only control" else "o")
    ax.set_xlim(*lim)
    ax.set_title(title, fontsize=7.5, color=INK)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("bits per byte", fontsize=7)
ax1.set_yticks(range(len(order)))
ax1.set_yticklabels([names[k] for k in reversed(order)], fontsize=7)
ax1.tick_params(axis="y", length=0)
ax2.tick_params(axis="y", length=0)
ax1.set_ylim(-0.6, len(order) - 0.4)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(OUT / "fig_recipes.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------------------------- Table: every held-out set
SET_NAMES = {
    "research_python": "Research Python", "code_reasoning": "Worked code solutions",
    "algebraic_stack": "Math code (AlgebraicStack)", "pretrain_behaviors": "Data science and planning",
    "starcoder": "GitHub code (StarCoder)", "stackexchange": "Stack Exchange", "pes2o": "Papers (peS2o)",
    "arxiv": "arXiv", "wikipedia_en": "English Wikipedia", "megamath": "MegaMath", "open_web_math": "OpenWebMath",
    "math_reasoning": "Math reasoning", "dclm": "Web (DCLM)", "txt360_web": "Web (TxT360)",
    "fineweb2_ja": "Japanese web (FineWeb-2)", "wikipedia_ja": "Japanese Wikipedia",
    "txt360_qa": "Question-answer text (TxT360)", "sft_pretrain": "Instruction-style text",
}


def per_set(tfile, ours, theirs):
    g = by_group(tfile)
    a, b = g[ours], g[theirs]
    return {k: (mean(a[s]["bits_per_byte"][k] for s in a), mean(b[s]["bits_per_byte"][k] for s in b))
            for k in next(iter(a.values()))["bits_per_byte"]}


cols = [per_set(t, e, q) for _, t, e, q, _, _ in SIZES + [LONG]]
lines = []
for k, name in SET_NAMES.items():
    e5, q5 = cols[2][k]
    leads = " & ".join(f"{c[k][1] - c[k][0]:.3f}" for c in cols)
    tt = k.replace("_", "\\_")
    lines.append(f"\\texttt{{{tt}}} & {name} & {e5:.3f} & {q5:.3f} & {leads} \\\\")
means = " & ".join(f"{mean(c[k][1] - c[k][0] for k in c):.4f}" for c in cols)
e5m, q5m = mean(cols[2][k][0] for k in cols[2]), mean(cols[2][k][1] for k in cols[2])
lines.append("\\midrule")
lines.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Mean of the 18 sets}} & {e5m:.3f} & {q5m:.3f} & {means} \\\\")
lines.append("\\bottomrule")
(OUT / "tab_sets.tex").write_text("% Written by paper/make_figs.py from results/expanse/text_*.json. Do not edit.\n"
                                  + "\n".join(lines) + "\n")
print("\nwrote", ", ".join(p.name for p in sorted(OUT.glob("*.pdf"))), "and tab_sets.tex")
