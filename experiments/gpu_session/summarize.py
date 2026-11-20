"""Print what the GPU session has produced so far. From the repo root:  python experiments/gpu_session/summarize.py"""
import glob
import json
import os
import statistics as st

R = "results/gpu_session"


def lm(folder, prefix):
    runs = {}
    for f in sorted(glob.glob(os.path.join(R, folder, "lm_compare_*.json"))):
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            name, seed = r["tag"][len(prefix):].rsplit("_s", 1)
            runs.setdefault(name, {})[int(seed)] = (sum(r["bits_per_byte"].values()) / 4, r["steps"], r["secs"], r["params_M"])
    return runs


def table(title, runs):
    if not runs:
        return
    print(f"\n{title}\n| Model | Parameters | Seeds | Mean bits per byte | Per seed | Steps | Seconds |\n|---|---|---|---|---|---|---|")
    for name, s in sorted(runs.items(), key=lambda kv: st.mean(v[0] for v in kv[1].values())):
        v = [s[k] for k in sorted(s)]
        print(f"| {name} | {v[0][3]}M | {len(v)} | {st.mean(x[0] for x in v):.3f} | {', '.join(f'{x[0]:.3f}' for x in v)} | "
              f"{', '.join(str(x[1]) for x in v)} | {', '.join(f'{x[2]:.0f}' for x in v)} |")


def main():
    table("Equal training time at 0.1B (bf16)", lm("eqtime", "eqt100_"))
    table("Longer runs at 0.1B (bf16, equal steps)", lm("long", "long100_"))
    table("13M, one circuit type left out of the starting menu (none: the default)", lm("leave_one_out", "f13_leave_"))
    rows = []
    for f in sorted(glob.glob(os.path.join(R, "combined", "full_small_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        if d["curve"]:
            e = d["curve"][-1]
            rows.append(f"| {os.path.basename(f)[11:-5]} | {e['step']} | {e['val_en']['removed']:.3f} | {100 * e['relay']['shared']:.1f}% | "
                        f"{100 * e['lookup'].get('attached', 0):.1f}% |")
    if rows:
        print("\nCombined 0.1B recipe\n| Run | Step | English bits per byte | Relay | Lookup |\n|---|---|---|---|---|\n" + "\n".join(rows))
    cap = {}
    for f in sorted(glob.glob(os.path.join(R, "relay_capacity", "relay*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        a, fin = d["args"], d["runs"][0]["final"]
        cap.setdefault((a["facts"], "board" if a["memory"] == "AG" else "weight edits"), []).append(fin["shared"]["acc"])
    if cap:
        print("\nRelay: facts passed, by number of facts\n| Facts | Path | Seeds | Passed |\n|---|---|---|---|")
        for (n, path), v in sorted(cap.items()):
            print(f"| {n} | {path} | {len(v)} | {', '.join(f'{100 * x:.1f}%' for x in v)} |")
    ok = len(glob.glob("logs/gpu_session/*.ok"))
    print(f"\n{ok} jobs finished" + (", ALL DONE" if os.path.exists(os.path.join(R, "ALL_DONE")) else ""))


if __name__ == "__main__":
    main()
