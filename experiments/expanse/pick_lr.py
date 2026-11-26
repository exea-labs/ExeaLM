"""Pick each model's learning rate at each size from the short learning-rate checks (plan.py --lr_check), and write
experiments/expanse/lr_choice.json, which `plan.py --run <name> --lr auto` reads.

    python experiments/expanse/pick_lr.py --keep runs

A check run is a folder lr_<model>_<size>_<rate> under --keep with a log.jsonl (only the logs are needed, so a KEEP
on S3 can be synced down without the checkpoints). Its score is the last validation bits per byte it logged; a run is
finished when its log has the "end" event, and one that diverged (no finite score) loses. Prints a table
of every check and the pick. A size whose checks are not all finished is left out, and said so.
"""
import argparse
import glob
import json
import math
import os
import re


def score(folder):
    """(last validation bits per byte, finished) of one check run, or (inf, False)."""
    path = os.path.join(folder, "log.jsonl")
    if not os.path.exists(path):
        return math.inf, False
    best, done = math.inf, False
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        done = done or r.get("event") == "end"                     # pretrain writes it once the run is complete
        v = r.get("val_bits_per_byte")
        if r.get("event") == "eval" and v is not None and math.isfinite(v):
            best = v
    return best, done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default="runs", help="the folder the runs are kept in (train.sb's KEEP)")
    ap.add_argument("--out", default="experiments/expanse/lr_choice.json")
    ap.add_argument("--rates", default="100m:3,250m:3,500m:3,3b:2",
                    help="how many rates each size's checks try (plan.py's LR_RATES): a pick waits for all of them, "
                         "not only for the checks whose folders exist (a queued check has none yet)")
    a = ap.parse_args(argv)
    want = {k: int(v) for k, v in (x.split(":") for x in a.rates.split(",") if x)}
    groups = {}
    for folder in sorted(glob.glob(os.path.join(a.keep, "lr_*"))):
        m = re.match(r"lr_(.+)_([0-9a-z]+)_([0-9.e+-]+)$", os.path.basename(folder))
        if m:
            groups.setdefault(f"{m.group(1)}_{m.group(2)}", []).append((float(m.group(3)), *score(folder)))
    chosen = json.load(open(a.out)) if os.path.exists(a.out) else {}
    print("| Model and size | Rate | Bits per byte | Finished |\n|---|---|---|---|")
    for key, rows in sorted(groups.items()):
        for lr, s, done in sorted(rows):
            print(f"| {key} | {lr:g} | {s:.4f} | {'yes' if done else 'no'} |")
        size = key.rsplit("_", 1)[1]
        if not all(done for _, _, done in rows) or len(rows) < want.get(size, 0):
            print(f"| {key} | (not all {want.get(size, len(rows))} checks finished: no pick yet) | | |")
            chosen.pop(key, None)
            continue
        lr, s, _ = min(rows, key=lambda r: r[1])
        if not math.isfinite(s):
            print(f"| {key} | (every rate diverged) | | |")
            continue
        chosen[key] = lr
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(chosen, open(a.out, "w"), indent=1)
    print(f"\npicked: {json.dumps(chosen)}\nwritten to {a.out}")
    return chosen


if __name__ == "__main__":
    main()
