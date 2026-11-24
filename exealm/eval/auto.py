"""Evaluate every checkpoint of a training run, as the run writes them.

    python -m exealm.eval.auto --run runs/pretrain --tokenizer tokenizer --tasks hellaswag,arc_easy --limit 200 \
        --watch 300

It looks in the run's folder for checkpoints (ckpt.pt, which pretraining rewrites every --save_every steps, and the
final model.pt), runs the lm-evaluation-harness tasks on each one it has not scored yet (exealm/eval/harness.py), and
adds a line to <run>/evals.jsonl and a row to the table in <run>/evals.md. A checkpoint counts as scored when a line
with the same step, tasks and --limit exists, so the command can be stopped and started again.

Without --watch it scores what is there and exits. With --watch N it checks again every N seconds and exits once the
final model.pt has been scored. It runs beside training as its own process (on a CPU or a second GPU), so training
never waits for it. Because ckpt.pt is rewritten in place, a checkpoint that is replaced before the watcher looks is
never scored: keep N below the time between saves. Each checkpoint is copied before it is scored, so a save in the
middle of an evaluation cannot change it.

Needs pip install -e '.[eval]'. Solo, greedy, no chat template: the limits in docs/evaluation.md apply.
"""
import argparse
import json
import os
import shutil
import tempfile
import time

import torch

MAIN = ("acc", "exact_match", "f1", "bleu", "word_perplexity", "perplexity")     # the column shown for a task


def numbers(result):
    """{task: {metric: value}} from a harness result: numeric metrics only, without the harness's ",none" suffix."""
    out = {}
    for task, metrics in result["results"].items():
        out[task] = {k.split(",")[0]: float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return out


def read_log(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()] if os.path.exists(path) else []


def table(records):
    """A Markdown table: one row per step (the latest record wins), one column per task, showing each task's main
    metric (accuracy when it has one)."""
    by_step, tasks = {}, []
    for r in records:
        by_step[r["step"]] = r
        tasks += [t for t in r["metrics"] if t not in tasks]

    def cell(r, task):
        m = r["metrics"].get(task)
        if not m:
            return ""
        key = next((k for k in MAIN if k in m), None) or next((k for k in m if "stderr" not in k), None)
        return "" if key is None else f"{m[key]:.4f} ({key})"

    lines = ["| step | " + " | ".join(tasks) + " |", "|---:|" + "---|" * len(tasks)]
    for step in sorted(by_step):
        lines.append(f"| {step} | " + " | ".join(cell(by_step[step], t) for t in tasks) + " |")
    return "\n".join(lines) + "\n"


def pending(run, records, tasks, limit, names=("ckpt.pt", "model.pt")):
    """[(file name, step)] of the run's checkpoints that have no record yet for these tasks and this limit."""
    done = {(r["step"], tuple(r["tasks"]), r["limit"]) for r in records}
    out = []
    for name in names:
        path = os.path.join(run, name)
        if not os.path.exists(path):
            continue
        try:
            step = int(torch.load(path, map_location="cpu", weights_only=False)["step"])
        except Exception:                                  # half-written or not a checkpoint: look again next time
            continue
        if (step, tuple(tasks), limit) not in done and step not in [s for _, s in out]:
            out.append((name, step))
    return out


def score(run, name, step, tokenizer, tasks, limit, **kw):
    """Copy the checkpoint, run the tasks on the copy and return the record for evals.jsonl."""
    from .harness import evaluate_checkpoint, file_hash
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        copy = os.path.join(tmp, name)
        shutil.copyfile(os.path.join(run, name), copy)
        if int(torch.load(copy, map_location="cpu", weights_only=False)["step"]) != step:
            return None                                    # replaced between the look and the copy: next round
        result = evaluate_checkpoint(copy, tokenizer, tasks, limit, **kw)
        sha = file_hash(copy)
    return dict(step=step, file=name, tasks=list(tasks), limit=limit, metrics=numbers(result),
                checkpoint_sha256=sha, secs=round(time.time() - t0, 1))


def run_once(run, tokenizer, tasks, limit, **kw):
    """Score every checkpoint that needs it. Returns the new records."""
    log = os.path.join(run, "evals.jsonl")
    records, new = read_log(log), []
    for name, step in pending(run, records, tasks, limit):
        rec = score(run, name, step, tokenizer, tasks, limit, **kw)
        if rec is None:
            continue
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
        records.append(rec)
        new.append(rec)
        with open(os.path.join(run, "evals.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(table(records))
        print(json.dumps(rec), flush=True)
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the training run's --out folder")
    ap.add_argument("--tokenizer", default="tokenizer", help="the tokenizer the run was trained with")
    ap.add_argument("--tasks", default="hellaswag", help="comma-separated harness task names")
    ap.add_argument("--include_path", default=None, help="directory of custom harness task YAML files")
    ap.add_argument("--limit", type=int, default=None, help="examples per task (default: all)")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--watch", type=float, default=0, help="check again every this many seconds (0: once)")
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",")
    kw = dict(include_path=a.include_path, max_length=a.max_length, device=a.device)
    while True:
        run_once(a.run, a.tokenizer, tasks, a.limit, **kw)
        final = os.path.join(a.run, "model.pt")
        if not a.watch or (os.path.exists(final) and not pending(a.run, read_log(os.path.join(a.run, "evals.jsonl")),
                                                                 tasks, a.limit, names=("model.pt",))):
            break
        time.sleep(a.watch)
    return read_log(os.path.join(a.run, "evals.jsonl"))


if __name__ == "__main__":
    main()
