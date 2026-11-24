"""Tables of one stage of the runs that pick the ExeaLM of the scaling study (README.md in this folder).

    python experiments/finalize/summarize.py 1          # after jobs/stage1_eval.txt has run
    python experiments/finalize/summarize.py 4          # the library on text

Table 1 is the number of the scaling study: bits per byte on the four held-out sets, every row read alone, so a model
trained in swarms and the Llama are scored alike. "Against gated" and "against core" are paired by seed (positive:
this model is better). Table 2 is what the board is worth on English text when it is used: four windows of one text
read as a swarm, against the same model with every window alone ("over solo") and against the Llama of the same seed
reading the same windows alone ("over the Llama"), in bits per byte, by window (window 0 comes first in the text). The
random rows are the control: four unrelated windows, where a board has nothing to give.
"""
import json
import os
import re
import sys
from collections import defaultdict

ROOT = "results/finalize"


def group(name):
    return re.sub(r"_s\d+$", "", os.path.basename(name.rstrip("/")))


def seed(name):
    return int(re.search(r"_s(\d+)$", name.rstrip("/")).group(1))


def text_table(stage):
    rows = json.load(open(f"{ROOT}/stage{stage}_text.json"))["runs"]
    by = defaultdict(dict)
    for r in rows:
        by[group(r["run"])][seed(r["run"])] = r["mean"]
    ref = {g: by[g] for g in ("gated", "long_gated", "core", "long_core") if g in by}
    gated = ref.get("gated") or ref.get("long_gated") or {}
    core = ref.get("core") or ref.get("long_core") or {}

    def paired(a, b):
        common = sorted(set(a) & set(b))
        if not common:
            return "", ""
        d = [b[s] - a[s] for s in common]
        return f"{sum(d) / len(d):+.4f}", f"{sum(x > 0 for x in d)} of {len(d)}"

    print("| Model | Seeds | Mean | Per seed | Against gated | Better on | Against core | Better on |")
    print("|---|---|---|---|---|---|---|---|")
    order = sorted(by, key=lambda g: sum(by[g].values()) / len(by[g]))
    for g in order:
        v = by[g]
        mean = sum(v.values()) / len(v)
        vg, ng = paired(v, gated) if g not in ("gated", "long_gated") else ("", "")
        vc, nc = paired(v, core) if g not in ("core", "long_core") else ("", "")
        print(f"| {g} | {len(v)} | {mean:.4f} | {', '.join(f'{v[s]:.4f}' for s in sorted(v))} | {vg} | {ng} | {vc} | {nc} |")
    return by


def swarm_table(stage, bytes_per_token):
    files = sorted(f for f in os.listdir(ROOT) if f.startswith(f"stage{stage}_swarm_") and f.endswith(".jsonl"))
    if not files:
        return
    by = defaultdict(list)
    for f in files:
        recs = [json.loads(l) for l in open(os.path.join(ROOT, f))]
        get = lambda p, c: next((r for r in recs if r["policy"] == p and r["condition"] == c), None)  # noqa: E731
        sw, solo, base = get("split", "swarm"), get("split", "solo"), get("split", "baseline")
        rs, rsolo = get("random", "swarm"), get("random", "solo")
        if not (sw and solo):
            continue
        per = [b - a for a, b in zip(sw["bits_per_agent"], solo["bits_per_agent"])]
        by[group(f[len(f"stage{stage}_swarm_"):-len(".jsonl")])].append(dict(
            over_solo=solo["bits_per_token"] - sw["bits_per_token"], per=per,
            over_llama=(base["bits_per_token"] - sw["bits_per_token"]) if base else float("nan"),
            random=(rsolo["bits_per_token"] - rs["bits_per_token"]) if (rs and rsolo) else float("nan")))
    k = 1 / bytes_per_token
    print("\n| Model | Seeds | Split: over solo | By window (0, 1, 2, 3) | Split: over the Llama | Random: over solo |")
    print("|---|---|---|---|---|---|")
    for g, v in sorted(by.items()):
        m = lambda key: sum(x[key] for x in v) / len(v) * k                    # noqa: E731
        per = [sum(x["per"][i] for x in v) / len(v) * k for i in range(len(v[0]["per"]))]
        print(f"| {g} | {len(v)} | {m('over_solo'):+.4f} | {', '.join(f'{p:+.4f}' for p in per)} | "
              f"{m('over_llama'):+.4f} | {m('random'):+.4f} |")


def library_table():
    """Stage 4: held-out bits per byte of each source with the library attached and removed. "lib" trained with it,
    "nolib" is the same model trained without one. Gains are positive when the library (or the model) is better."""
    get = lambda recs, c: next(r["bits_per_token"] for r in recs if r["condition"] == c)        # noqa: E731
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted(os.listdir(ROOT)):
        m = re.match(r"stage4_lib_(lib|nolib)_(\w+?)_s(\d+)_attached\.jsonl$", f)
        if not m or not os.path.exists(os.path.join(ROOT, f.replace("attached", "removed"))):
            continue
        att = [json.loads(l) for l in open(os.path.join(ROOT, f))]
        rem = [json.loads(l) for l in open(os.path.join(ROOT, f.replace("attached", "removed")))]
        by[m.group(2)][m.group(1)].append(dict(seed=int(m.group(3)), att=get(att, "solo"), rem=get(rem, "solo"),
                                                llama=get(rem, "baseline")))
    print("| Source | Model | Seeds | Library attached | Removed | Library gain | Against nolib, removed | Against the Llama |")
    print("|---|---|---|---|---|---|---|---|")
    for src, models in sorted(by.items()):
        try:
            k = 1 / json.load(open(f"data/tokpack/bpe32k/val_{src}/meta.json"))["bytes_per_token"]
        except (OSError, KeyError):
            k = 1.0
        nolib = {r["seed"]: r["rem"] for r in models.get("nolib", [])}
        for name in ("lib", "nolib"):
            v = models.get(name, [])
            if not v:
                continue
            mean = lambda key: sum(r[key] for r in v) / len(v) * k                            # noqa: E731
            pair = [nolib[r["seed"]] - r["att"] for r in v if r["seed"] in nolib]
            vs_nolib = f"{sum(pair) / len(pair) * k:+.4f}" if (pair and name == "lib") else ""
            print(f"| {src} | {name} | {len(v)} | {mean('att'):.4f} | {mean('rem'):.4f} | "
                  f"{mean('rem') - mean('att'):+.4f} | {vs_nolib} | {mean('llama') - mean('att'):+.4f} |")


def main(argv=None):
    stage = int((argv or sys.argv[1:] or ["1"])[0])
    if stage == 4:
        return library_table()
    by = text_table(stage)
    try:
        bpt = json.load(open("data/tokpack/bpe32k/val_en/meta.json"))["bytes_per_token"]
    except (OSError, KeyError):
        bpt = 1.0
        print("\n(val_en/meta.json not found: table 2 is in bits per token)")
    swarm_table(stage, bpt)
    return by


if __name__ == "__main__":
    main()
