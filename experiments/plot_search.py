"""What did the search choose? For every TONN layer, how many slots hold each circuit type, after every search round.

Each search round's record in pretrain's log.jsonl carries the architecture it leaves behind (exealm/model/search.py).
This script draws one stacked area chart per TONN layer, from the starting circuits to the end of training, and
prints the same counts as a markdown table. Give several logs (for example one per seed) to draw them side by side,
one column per run.

    python -m exealm.train.pretrain --preset tiny --data synthetic --steps 2000 --out runs/search
    python experiments/plot_search.py runs/search/log.jsonl --out results/search_arch/search_arch.png

Band colours follow the circuit families on the project blog (https://blog.exealabs.org/architecture/); the code
inside each band names the circuit type and its count. Reading a log needs nothing extra; drawing needs matplotlib
(pip install matplotlib).
"""
import argparse
import json
import os

# The families of the circuit menu (exealm/config.py MENU), as the blog groups them, in stacking order. Every type
# in MENU is in exactly one.
FAMILIES = (
    ("Memory", ("RG", "AG")),
    ("Short range", ("D", "C")),
    ("Running state", ("I", "GI", "MI", "MIM", "EP", "SM", "OS")),
    ("Gates and products", ("G", "M")),
    ("Attention", ("RP",)),
)
# One colour per family, in the order above: a validated categorical palette (adjacent pairs stay apart under colour
# blindness). The labels inside the bands carry the type, so identity never rests on colour alone.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
ORDER = tuple(t for _, ts in FAMILIES for t in ts)
FAMILY_OF = {t: i for i, (_, ts) in enumerate(FAMILIES) for t in ts}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def read_log(path):
    """The architecture over one training run.

    Returns dict(points=[(step, [counts per TONN layer])], layers=[model layer number of each TONN layer],
    slots=slots per layer, rounds=[steps of the search rounds], replaced=[slots each round replaced], end=last step
    trained, path=path). The first point is the starting architecture (step 0), then one point per search round, then
    the end of training. If the log holds several runs (the same --out used twice), the last one is read. A resumed
    run keeps one record per step: rounds that were run again after a resume replace the ones logged before it."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    starts = [i for i, r in enumerate(recs) if r.get("event") == "start"]
    if not starts:
        raise ValueError(f"{path}: no 'start' record; is this a pretrain log.jsonl?")
    recs = recs[starts[-1]:]
    start = recs[0]
    if not start.get("architecture"):
        raise ValueError(f"{path}: not a TONN run (a {start.get('model')} model has no circuits to plot)")
    cfg = start["cfg"]
    rounds, end_step, end_arch = {}, 0, None
    for r in recs[1:]:
        ev = r.get("event")
        if ev == "resume":                                  # anything logged at or after this step is done again
            rounds = {s: v for s, v in rounds.items() if s < r["step"]}
            end_step = min(end_step, r["step"])
        elif ev == "search":
            if "architecture" not in r:
                raise ValueError(f"{path}: its search records have no 'architecture' (the log is older than that "
                                 "field); train again to plot it")
            rounds[r["step"]] = (r["architecture"], r["replaced"])
        elif ev in ("eval", "stop"):
            end_step = max(end_step, r["step"])
        elif ev == "end":
            end_arch = r["architecture"]
    steps = sorted(rounds)
    points = [(0, start["architecture"])] + [(s, rounds[s][0]) for s in steps]
    last = end_arch if end_arch is not None else points[-1][1]
    points.append((max(end_step, points[-1][0]), last))
    layers = [i for i in range(cfg["n_layers"]) if i not in cfg.get("attn_layers", ())]
    return dict(points=points, layers=layers, slots=cfg["slots"], rounds=steps, replaced=[rounds[s][1] for s in steps],
                end=points[-1][0], path=path)


def types_used(run):
    """Circuit types that appear anywhere in the run, in stacking order."""
    seen = {t for _, arch in run["points"] for row in arch for t in row}
    unknown = seen - set(ORDER)
    if unknown:
        raise ValueError(f"circuit types {sorted(unknown)} have no family: add them to FAMILIES")
    return [t for t in ORDER if t in seen]


def _counts(row, ts):
    return ", ".join(t if row[t] == 1 else f"{t} ×{row[t]}" for t in ts if row.get(t))


def table(run):
    """The architecture as a markdown table: the start, every search round that replaced a slot, and the end. A
    round can replace a slot with a circuit of the same type, so its counts may not change."""
    ts = types_used(run)
    head = "| step | slots replaced | " + " | ".join(f"layer {li}" for li in run["layers"]) + " |"
    lines = [head, "|" + "---|" * (len(run["layers"]) + 2)]
    n = len(run["points"])
    for j, (step, arch) in enumerate(run["points"]):
        if 0 < j < n - 1 and run["replaced"][j - 1] == 0:
            continue
        name = "start" if j == 0 else ("end" if j == n - 1 else str(step))
        rep = "" if j in (0, n - 1) else str(run["replaced"][j - 1])
        lines.append(f"| {name} | {rep} | " + " | ".join(_counts(row, ts) for row in arch) + " |")
    return "\n".join(lines)


def _stretches(xs, band):
    """Stretches of x where a band keeps the same bottom and top: [(x0, x1, lo, hi)], widest first."""
    out, j = [], 0
    while j < len(xs) - 1:
        k = j
        while k + 1 < len(xs) - 1 and band[k + 1] == band[j]:
            k += 1
        lo, hi = band[j]
        if hi > lo:
            out.append((xs[j], xs[k + 1], lo, hi))
        j = k + 1
    return sorted(out, key=lambda r: r[0] - r[1])


def _label_bands(fig, jobs):
    """Write each band's type code (and count) once, in its widest stretch, in the largest font that fits inside the
    band. A stretch that lasts past the last search round is labelled in the middle of that frozen part, so the labels
    line up. Bands too small for any label are left to the table that main() prints."""
    renderer = fig.canvas.get_renderer()
    for ax, xs, stacks, frozen in jobs:
        for t, band in stacks.items():
            for x0, x1, lo, hi in _stretches(xs, band):
                if x1 > frozen:
                    x0 = max(x0, frozen)
                (px0, py0), (px1, py1) = ax.transData.transform([(x0, lo), (x1, hi)])
                n = int(round(hi - lo))
                placed = None
                for size in (7.5, 6.5, 5.5):
                    txt = ax.text((x0 + x1) / 2, (lo + hi) / 2, t if n == 1 else f"{t} ×{n}", ha="center",
                                  va="center", fontsize=size, color=INK)
                    bb = txt.get_window_extent(renderer)
                    if bb.width <= (px1 - px0) - 4 and bb.height <= (py1 - py0) - 2:
                        placed = txt
                        break
                    txt.remove()
                if placed is not None:
                    break


def plot(runs, out, labels):
    """runs: read_log() results, labels: a column title per run. One column per run, one row per TONN layer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as e:
        raise SystemExit("drawing needs matplotlib: pip install matplotlib") from e
    nrow = max(len(r["layers"]) for r in runs)
    fig, axes = plt.subplots(nrow, len(runs), figsize=(4.2 * len(runs), 2.3 * nrow + 0.9), squeeze=False,
                             facecolor=SURFACE)
    families, jobs = set(), []
    for c, (run, lab) in enumerate(zip(runs, labels)):
        ts = types_used(run)
        xs = [s for s, _ in run["points"]]
        for r, li in enumerate(run["layers"]):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            base = [0.0] * len(xs)
            stacks = {}
            for t in ts:
                ys = [arch[r].get(t, 0) for _, arch in run["points"]]
                top = [b + y for b, y in zip(base, ys)]
                if any(ys):
                    stacks[t] = list(zip(base, top))
                    families.add(FAMILY_OF[t])
                    ax.fill_between(xs, base, top, step="post", color=COLORS[FAMILY_OF[t]], edgecolor=SURFACE,
                                    linewidth=1.0)
                base = top
            jobs.append((ax, xs, stacks, run["rounds"][-1] if run["rounds"] else 0))
            for s in run["rounds"]:                                   # a tick for every search round
                ax.axvline(s, ymin=0.97, ymax=1.0, color=MUTED, linewidth=0.8)
            if run["rounds"]:
                ax.axvline(run["rounds"][-1], color=MUTED, linewidth=0.6, linestyle=(0, (2, 3)))
            ax.set_xlim(0, run["end"])
            ax.set_ylim(0, run["slots"])
            ax.set_yticks(range(0, run["slots"] + 1, max(1, run["slots"] // 4)))
            ax.set_title(f"{lab} · layer {li}", fontsize=9, color=INK, loc="left")
            ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#c3c2b7")
            if c == 0:
                ax.set_ylabel("slots", fontsize=8, color=MUTED)
            if r == len(run["layers"]) - 1:
                ax.set_xlabel("training step", fontsize=8, color=MUTED)
        for r in range(len(run["layers"]), nrow):
            axes[r][c].set_visible(False)
    handles = [Patch(facecolor=COLORS[i], label=FAMILIES[i][0]) for i in sorted(families)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle("Circuits in each TONN layer after every search round (ticks: rounds; dotted line: the last round)",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    _label_bands(fig, jobs)                                               # after the layout fixes each axis' size
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", help="pretrain log.jsonl files, one per run")
    ap.add_argument("--labels", nargs="*", default=None, help="a column title per log (default: the log's folder)")
    ap.add_argument("--out", default="results/search_arch/search_arch.png")
    ap.add_argument("--table_only", action="store_true", help="print the tables and skip the figure")
    a = ap.parse_args(argv)
    if a.labels and len(a.labels) != len(a.logs):
        ap.error("give one label per log")
    runs = [read_log(p) for p in a.logs]
    labels = a.labels or [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in a.logs]
    for run, lab in zip(runs, labels):
        print(f"## {lab}: {len(run['rounds'])} search rounds, {sum(run['replaced'])} slots replaced "
              f"({run['path']})\n")
        print(table(run) + "\n")
    print("Rounds that replaced nothing are left out of the tables.\n")
    if not a.table_only:
        print("wrote", plot(runs, a.out, labels))
    return runs


if __name__ == "__main__":
    main()
