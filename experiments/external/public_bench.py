"""Released models and their training checkpoints on SciQ, ARC-Easy, ARC-Challenge (and MMLU), zero-shot, in one Python
process: lm-eval's task index is read once (reading it from Lustre takes minutes), then each model in turn. Writes
results/expanse/public_bench_<name>.json and openbook_samples/public_<name>_closed.jsonl (each question's result, the
format experiments/library/openbook_paired.py reads), and skips a model whose results exist.

    python experiments/external/public_bench.py /expanse/lustre/projects/fli100/abhatia7/hf/pythia-410m-step1000 ...

Run in the lm-eval venv over the vllm environment (experiments/external/public_bench.sb).
"""
import argparse
import gc
import json
import os
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", help="local model folders; the folder's name names the results")
    ap.add_argument("--tasks", default="sciq,arc_easy,arc_challenge")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results/expanse")
    a = ap.parse_args(argv)
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    t0 = time.time()
    tm = TaskManager()
    print(f"task index read in {time.time() - t0:.0f}s", flush=True)
    tasks = a.tasks.split(",")
    for path in a.models:
        name = os.path.basename(path.rstrip("/"))
        res_path = os.path.join(a.out, f"public_bench_{name}.json")
        if os.path.exists(res_path):
            print(f"{name}: done before", flush=True)
            continue
        t1 = time.time()
        lm = HFLM(pretrained=path, dtype="bfloat16", batch_size=a.batch, device="cuda")
        r = simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, log_samples=True, bootstrap_iters=0,
                            task_manager=tm)
        os.makedirs(os.path.join(a.out, "openbook_samples"), exist_ok=True)
        with open(os.path.join(a.out, "openbook_samples", f"public_{name}_closed.jsonl"), "w", encoding="utf-8") as f:
            for task, rows in r.get("samples", {}).items():
                for s in rows:
                    line = dict(task=task, doc_id=s["doc_id"], question=s.get("doc", {}).get("question", ""))
                    for k in ("acc", "acc_norm"):
                        if k in s:
                            line[k] = float(s[k])
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        keep = {k: r[k] for k in ("results", "n-shot", "versions", "lm_eval_version", "date") if k in r}
        keep["model_path"] = path
        json.dump(keep, open(res_path, "w"), indent=1, default=str)
        acc = {t: round(r["results"][t].get("acc_norm,none", r["results"][t].get("acc,none")), 4) for t in tasks}
        print(f"{name}: {acc} ({time.time() - t1:.0f}s)", flush=True)
        del lm, r
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
