"""Run a file of jobs, a few at a time, skipping the ones that already finished.

    python experiments/gpu_session/runner.py jobs/eqtime.txt 1

Each line of the file is  TAG <tab> OUTPUT FILE <tab> SHELL COMMAND.  A job that exits cleanly and wrote its output
file gets a marker, logs/gpu_session/TAG.ok, and is skipped from then on, so the same call resumes a queue after a
crash or a reboot. A job that fails is listed at the end; run the same call again to retry it. The command's own
output goes to logs/gpu_session/TAG.log.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def run(job):
    tag, out, cmd = job
    os.makedirs("logs/gpu_session", exist_ok=True)
    marker = f"logs/gpu_session/{tag}.ok"
    if os.path.exists(marker):
        return tag, "skipped (already done)"
    t0 = time.time()
    with open(f"logs/gpu_session/{tag}.log", "w", encoding="utf-8", newline="\n") as log:
        code = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT).returncode
    ok = code == 0 and os.path.exists(out)
    if ok:
        open(marker, "w", encoding="utf-8", newline="\n").write(out + "\n")
    print(f"{time.strftime('%H:%M:%S')} {'done  ' if ok else 'FAILED'} {tag} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return tag, "done" if ok else f"FAILED (exit {code}; see logs/gpu_session/{tag}.log)"


def main():
    path, parallel = sys.argv[1], int(sys.argv[2])
    jobs = [line.rstrip("\n").split("\t", 2) for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#")]
    with ThreadPoolExecutor(parallel) as ex:
        results = list(ex.map(run, jobs))
    failed = [f"{t}: {r}" for t, r in results if r.startswith("FAILED")]
    print(f"{path}: {sum(r == 'done' for _, r in results)} done, {sum(r.startswith('skipped') for _, r in results)} skipped, {len(failed)} failed")
    for f in failed:
        print("  " + f)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
