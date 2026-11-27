"""Figures for the funding report, drawn from the result files in results/expanse.

Run from the repository root:  python docs/funding_report/figs/make_figs.py
Reads only JSON; loads no model.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "expanse"
OUT = Path(__file__).resolve().parent

BLUE = "#2a78d6"    # long documents read as a stream
ORANGE = "#eb6834"  # windows read alone
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RES / name).read_text())


def stream_means(name, first=1, last=None):
    """Per group of seeds: per-window streamed bits per byte, averaged over seeds."""
    d = load(name)
    groups = {}
    for r in d["runs"]:
        g = Path(r["run"]).name.rsplit("_s", 1)[0]
        groups.setdefault(g, []).append(r)
    out = {}
    for g, rs in groups.items():
        n = len(rs[0]["streamed"])
        curve = [sum(r["streamed"][w] for r in rs) / len(rs) for w in range(n)]
        hi = n - 1 if last is None else last
        out[g] = {"curve": curve, "mean": sum(curve[first:hi + 1]) / (hi - first + 1), "seeds": len(rs)}
    return out


def pct(ours, theirs):
    return 100.0 * (theirs - ours) / theirs


# Figure 1: the lead over the Qwen3.5-style model by size.
alone = {
    0.1: (load("text_100m.json")["means"]["exealm_midstream_100m_4.1x"], load("text_100m.json")["means"]["qwen_100m_4.1x"]),
    0.25: (load("text_250m.json")["means"]["exealm_250m_4.1x"], load("text_250m.json")["means"]["qwen_250m_4.1x"]),
    0.5: (load("text_500m_all.json")["means"]["exealm_500m_4.1x"], load("text_500m_all.json")["means"]["qwen_500m_4.1x"]),
}
s100 = stream_means("stream8_100m.json")
s250 = stream_means("stream8_250m.json")
s500 = stream_means("stream8_500m_all.json")
longdoc = {
    0.1: (s100["exealm_midstream_100m_4.1x"]["mean"], s100["qwen_100m_4.1x"]["mean"]),
    0.25: (s250["exealm_250m_4.1x"]["mean"], s250["qwen_250m_4.1x"]["mean"]),
    0.5: (s500["exealm_500m_4.1x"]["mean"], s500["qwen_500m_4.1x"]["mean"]),
}
sizes = [0.1, 0.25, 0.5]
xpos = [0, 1, 2]
la = [pct(*alone[s]) for s in sizes]
ll = [pct(*longdoc[s]) for s in sizes]
print("windows alone, lead %:", [round(v, 2) for v in la], "bits:", [round(alone[s][1] - alone[s][0], 4) for s in sizes])
print("long documents, lead %:", [round(v, 2) for v in ll], "bits:", [round(longdoc[s][1] - longdoc[s][0], 4) for s in sizes])

fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.plot(xpos, ll, color=BLUE, linewidth=2, marker="o", markersize=6, label="Long documents, read as a stream")
ax.plot(xpos, la, color=ORANGE, linewidth=2, marker="s", markersize=6, label="Windows of 512 tokens, read alone")
for x, v in zip(xpos, ll):
    ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points", xytext=(0, 6), ha="center", color=INK, fontsize=8)
for x, v in zip(xpos, la):
    ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points", xytext=(0, -12), ha="center", color=INK, fontsize=8)
ax.axvspan(2.6, 3.4, color="#f1f0ec", zorder=0)
ax.text(3.0, 3.0, "3B pair\ntraining;\ndue\n29 Nov", ha="center", va="center", color=INK2, fontsize=8)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["0.1B\n3 seeds", "0.25B\n3 seeds", "0.5B\n1 seed", "3B"])
ax.set_xlim(-0.4, 3.45)
ax.set_ylim(0, 6.6)
ax.set_ylabel("ExeaLM's lead over the\nQwen3.5-style model (%)")
ax.legend(loc="upper right", bbox_to_anchor=(0.76, 1.0), frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "lead_by_size.pdf")
plt.close(fig)

# Figure 2: the lead by position in 16,384-token documents.
s32 = stream_means("stream32_100m.json")
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for ours, theirs, color, marker, label in [
    ("exealm_100m_4.1x", "qwen_100m_4.1x", BLUE, "o", "0.1B, 3 seeds"),
    ("exealm_250m_4.1x", "qwen_250m_4.1x", ORANGE, "s", "0.25B, 1 seed"),
]:
    a, b = s32[ours]["curve"], s32[theirs]["curve"]
    lead = [pct(x, y) for x, y in zip(a, b)]
    end_tokens = [(w + 1) * 512 / 1000 for w in range(len(lead))]
    ax.plot(end_tokens, lead, color=color, linewidth=2, marker=marker, markersize=3.5, label=label)
    print(label, "lead by window %:", [round(v, 2) for v in lead])
ax.axvline(2.048, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
ax.text(2.35, 1.2, "length of the streams\nread in training (2,048 tokens)", color=INK2, fontsize=8, va="bottom")
ax.set_xlim(0, 16.6)
ax.set_ylim(0, 6.5)
ax.set_xlabel("Position in the document (thousands of tokens, end of each 512-token window)")
ax.set_ylabel("ExeaLM's lead over the\nQwen3.5-style model (%)")
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "lead_by_position.pdf")
plt.close(fig)
print("wrote", OUT / "lead_by_size.pdf", OUT / "lead_by_position.pdf")
