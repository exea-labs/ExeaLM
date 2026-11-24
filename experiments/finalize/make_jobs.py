"""Write the job files that pick the ExeaLM the scaling study trains (README.md in this folder), for
experiments/gpu_session/runner.py. Run from the repo root:

    python experiments/finalize/make_jobs.py                       # stage 1: the board on text, 7 models x 3 seeds
    python experiments/finalize/make_jobs.py --stage 2 --base doc_split     # stage 2: around the stage-1 pick
    python experiments/finalize/make_jobs.py --stage 3 --base doc_split --opt muon --lr 2e-3   # stage 3: confirm
    python experiments/finalize/make_jobs.py --stage 4                     # the library on text, English and code
    python experiments/finalize/make_jobs.py --stage 5                     # stage 1 trimmed, plus the fixes for the cost
                                                                           # of split texts
    python experiments/finalize/make_jobs.py --stage 6                     # changes from recent papers, on the hybrid
                                                                           # and the gated Llama
    python experiments/finalize/make_jobs.py --stage 7                     # the brain: reading 4 windows with the
                                                                           # state carried (docs/brain.md)
    python experiments/finalize/make_jobs.py --stage 8                     # the brain with weight changes carried
                                                                           # across windows (wa_persist)

Each stage writes jobs/stage<N>_train.txt and jobs/stage<N>_eval.txt. Run the training file first, then the
evaluation file, then summarize.py. Every line is TAG <tab> OUTPUT FILE <tab> COMMAND.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from exealm.config import MENU  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.relpath(HERE)
NO_LIBRARY = [m for m in MENU if m != "RG"]
# The plain models these runs compare against, frozen here: the cluster's configs in experiments/expanse now carry the
# changes these runs chose (value residual and XSA), so they are no longer the plain models.
GATED = "experiments/finalize/configs/gated.json"
CORE = "experiments/finalize/configs/core.json"

# name: (config overrides or a config file, extra pretrain arguments, what it tests)
ARMS = {
    "gated": (GATED, "--model llama", "the baseline: a Llama with gated attention and QK-norm"),
    "core": (CORE, "", "the memory-free hybrid, every row alone: the model of the 0.1B lead"),
    "plan": (None, "", "the plan's ExeaLM: board and library slots, mixed swarms, time-order board"),
    "board_random": (dict(pinned={"AG": 1}, menu=NO_LIBRARY), "--swarm_policy random",
                     "board only, unrelated texts in every swarm: the cost of the slot and of swarm batches"),
    "time_mixed": (dict(pinned={"AG": 1}, menu=NO_LIBRARY), "--swarm_policy mixed",
                   "board only, mixed swarms, time-order board: the plan without the library"),
    "doc_mixed": (dict(pinned={"AG": 1}, menu=NO_LIBRARY, board_order="doc"), "--swarm_policy mixed",
                  "board only, mixed swarms, doc-order board"),
    "doc_split": (dict(pinned={"AG": 1}, menu=NO_LIBRARY, board_order="doc"), "--swarm_policy split",
                  "board only, every swarm one text in four windows, doc-order board"),
}
BOARD_ARMS = ("plan", "board_random", "time_mixed", "doc_mixed", "doc_split")
STAGE2 = {                                                         # changes to the stage-1 pick
    "chunk16": (dict(board_chunk=16), "", "a post every 16 tokens instead of 64"),
    "chunk32": (dict(board_chunk=32), "", "a post every 32 tokens"),
    "pool": (dict(chunk_pool="attn"), "", "posts weigh the chunk's positions (learned) instead of averaging them"),
    "ag2": ("AG2", "", "two board slots per TONN layer"),
    "swarm2": (dict(swarm_size=2), "", "swarms of 2"),
    "swarm8": (dict(swarm_size=8), "", "swarms of 8"),
}
NO_BOARD = [m for m in MENU if m != "AG"]
LIBRARY = dict(pinned={"RG": 1}, menu=NO_BOARD, swarm_size=1, rag_key_dim=128, rag_lexical=128)
LIB_CHUNKS = 200_000                                               # 12.8M tokens of the training pack
RECIPE = ("--preset 100m --data experiments/finalize/mix.json --val data/tokpack/bpe32k/val_en --seq 512 --batch 32 "
          "--warmup 200 --search 0 --eval_every 500")


def config_file(name, over):
    if over is None or isinstance(over, str):
        return over
    os.makedirs(os.path.join(HERE, "configs"), exist_ok=True)
    path = os.path.join(REL, "configs", f"{name}.json")
    json.dump(over, open(path, "w"), indent=1)
    return path


def train(tag, cfg, extra, seed, steps=3000, lr=1e-3, opt="adamw"):
    out = f"runs/finalize/{tag}_s{seed}"
    cmd = (f"python -m exealm.train.pretrain {RECIPE} --steps {steps} --lr {lr:g} --optimizer {opt} --seed {seed} "
           + (f"--config {cfg} " if cfg else "") + f"{extra} --out {out}").replace("  ", " ")
    return f"fin_{tag}_s{seed}", f"{out}/model.pt", cmd


def evals(stage, runs, board, baseline):
    """eval_text over every run, and swarm_vs_solo for every board run with the same seed's Llama as its baseline."""
    jobs = [(f"fin{stage}_text", f"results/finalize/stage{stage}_text.json",
             f"python experiments/eval_text.py {' '.join(runs)} --out results/finalize/stage{stage}_text.json")]
    for run in board:
        seed = run.rsplit("_s", 1)[1]
        out = f"results/finalize/stage{stage}_swarm_{os.path.basename(run)}.jsonl"
        jobs.append((f"fin{stage}_swarm_{os.path.basename(run)}", out,
                     f"rm -f {out} && python -m exealm.eval.swarm_vs_solo --ckpt {run}/model.pt "
                     f"--baseline {baseline.format(seed=seed)}/model.pt --data data/tokpack/bpe32k/val_en --seq 512 "
                     f"--batches 64 --device cuda --out {out}"))
    return jobs


def base_config(name):
    over, extra, _ = ARMS[name]
    if isinstance(over, str):
        over = json.load(open(over))
    return dict(over or {}), extra


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--base", default="doc_split", help="the stage-1 arm stages 2 and 3 start from")
    ap.add_argument("--change", default="", help="stage 3: stage-2 changes to add to the base, e.g. chunk32,pool")
    ap.add_argument("--opt", default="adamw", help="stage 3: the optimizer stage 2 picked")
    ap.add_argument("--lr", type=float, default=1e-3, help="stage 3: the ExeaLM learning rate stage 2 picked")
    ap.add_argument("--gated_lr", type=float, default=None, help="stage 3: the Llama's (default: --lr)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    os.makedirs("results/finalize", exist_ok=True)
    jobs, runs, board = [], [], []
    if a.stage == 1:
        for seed in (0, 1, 2):
            for name, (over, extra, _) in ARMS.items():
                jobs.append(train(name, config_file(name, over), extra, seed))
                runs.append(f"runs/finalize/{name}_s{seed}")
                if name in BOARD_ARMS:
                    board.append(runs[-1])
        baseline = "runs/finalize/gated_s{seed}"
    elif a.stage == 2:
        over, extra = base_config(a.base)
        for seed in (0, 1):
            for name, (change, more, _) in STAGE2.items():
                cfg = dict(over, **change) if isinstance(change, dict) else dict(over, pinned={**over.get("pinned", {}),
                                                                                                "AG": 2})
                tag = f"{a.base}+{name}"
                jobs.append(train(tag, config_file(tag, cfg), f"{extra} {more}", seed))
                runs.append(f"runs/finalize/{tag}_s{seed}")
                board.append(runs[-1])
        for model, (over_m, extra_m) in (("gated", (GATED, "--model llama")), (a.base, (None, extra))):
            cfg = over_m if model == "gated" else config_file(a.base, over or None)
            for opt in ("adamw", "muon"):
                for lr in (1e-3, 2e-3, 4e-3):
                    if opt == "adamw" and lr == 1e-3:
                        continue                                       # stage 1 ran it
                    tag = f"{model}_{opt}_{lr:g}"
                    jobs.append(train(tag, cfg, extra_m, 0, lr=lr, opt=opt))
                    runs.append(f"runs/finalize/{tag}_s0")
        baseline = "runs/finalize/gated_s{seed}"
    elif a.stage == 8:
        # The brain that changes its own weights (docs/brain.md): one weight-alter slot per TONN layer, key from the
        # edited layer, edits to the MLP's input matrix (the cheapest setting on text so far), each window's changes
        # carried into the next with 0.9 of them kept. core_wa is the same model trained one window at a time.
        core = json.load(open(CORE))
        wa = config_file("wa", dict(core, pinned={"WA": 1}, menu=list(core["menu"]) + ["WA"], wa_key="hidden",
                                    wa_matrix="in", wa_persist=0.9))
        for seed in (0, 1):
            jobs.append(train("brain_wa", wa, "--stream 4 --batch 8", seed))
            jobs.append(train("core_wa", wa, "", seed))
        runs = [f"runs/finalize/{n}_s{s_}" for s_ in (0, 1) for n in ("core", "brain_core", "brain_gated", "brain_gd",
                                                                      "core_gd", "brain_wa", "core_wa")]
        runs = [r for r in runs if r != "runs/finalize/core_s1"] + ["runs/finalize/core_s1", "runs/finalize/gated_s0",
                                                                    "runs/finalize/gated_s1"]
        ev = [("fin8_text", "results/finalize/stage8_text.json",
               f"python experiments/eval_text.py {' '.join(runs)} --out results/finalize/stage8_text.json")]
        for w, docs in ((4, 128), (8, 64)):
            out = f"results/finalize/stage8_stream{w}.json"
            ev.append((f"fin8_stream{w}", out, f"python experiments/stream_eval.py {' '.join(runs)} --windows {w} "
                                                f"--docs {docs} --out {out}"))
        with open(os.path.join(HERE, "jobs", "stage8_train.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in jobs)
        with open(os.path.join(HERE, "jobs", "stage8_eval.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in ev)
        print(f"stage 8: {len(jobs)} training runs, {len(ev)} evaluation jobs in {REL}/jobs/")
        return
    elif a.stage == 7:
        # The brain (docs/brain.md): the memory-free hybrid and the gated Llama, each trained on rows of 4 windows read
        # in order with the state carried (8 rows x 4 windows x 512 tokens: the same tokens per step). Scored on long
        # held-out documents, streamed against every window alone, next to the same models trained one window at a time.
        # gd: two gated delta-rule slots per TONN layer, a matrix memory written at every token that, read in windows,
        # is carried through the whole document; core_gd is the same model trained one window at a time.
        core = json.load(open(CORE))
        gd = config_file("gd", dict(core, pinned={"GD": 2}, menu=list(core["menu"]) + ["GD"]))
        arms = (("brain_core", CORE, "--stream 4 --batch 8"), ("brain_gated", GATED, "--model llama --stream 4 --batch 8"),
                ("brain_gd", gd, "--stream 4 --batch 8"), ("core_gd", gd, ""))
        for seed in (0, 1):
            for name, cfg, extra in arms:
                jobs.append(train(name, cfg, extra, seed))
        runs = ["runs/finalize/core_s0", "runs/finalize/gated_s0"] + [f"runs/finalize/{n}_s{s_}" for s_ in (0, 1)
                                                                      for n, _, _ in arms]
        ev = [("fin7_text", "results/finalize/stage7_text.json",
               f"python experiments/eval_text.py {' '.join(runs)} --out results/finalize/stage7_text.json")]
        for w, docs in ((4, 128), (8, 64)):
            out = f"results/finalize/stage7_stream{w}.json"
            ev.append((f"fin7_stream{w}", out, f"python experiments/stream_eval.py {' '.join(runs)} --windows {w} "
                                                f"--docs {docs} --out {out}"))
        with open(os.path.join(HERE, "jobs", "stage7_train.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in jobs)
        with open(os.path.join(HERE, "jobs", "stage7_eval.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in ev)
        print(f"stage 7: {len(jobs)} training runs, {len(ev)} evaluation jobs in {REL}/jobs/")
        return
    elif a.stage == 6:
        # Changes from recent papers (exealm/model/baseline.py), each alone on the memory-free hybrid, the three that
        # cost no compute together, and Canon layers. The together set and Canon also on the gated Llama: a change that
        # helps both is fine for the Llama baseline to have too. Seeds 0 and 1; core_s0 and gated_s0 are stage 1's.
        core, gated = json.load(open(CORE)), json.load(open(GATED))
        three = dict(value_residual=True, xsa=True, peri_ln=True)
        arms = [("core", core, ""), ("gated", gated, "--model llama")]
        arms += [(f"core+{k}", dict(core, **v), "") for k, v in (("vr", dict(value_residual=True)), ("xsa", dict(xsa=True)),
                                                                   ("peri", dict(peri_ln=True)), ("canon", dict(canon=True)),
                                                                   ("three", three),
                                                                   ("two", dict(value_residual=True, xsa=True)))]
        arms += [("gated+three", dict(gated, **three), "--model llama"), ("gated+canon", dict(gated, canon=True), "--model llama")]
        for seed in (0, 1):
            for name, over, extra in arms:
                runs.append(f"runs/finalize/{name}_s{seed}")
                if name in ("core", "gated") and seed == 0:
                    continue                                           # stage 1 trained them
                cfg = over if name in ("core", "gated") else config_file(name, over)
                jobs.append(train(name, CORE if name == "core" else GATED if name == "gated" else cfg, extra, seed))
        baseline = "runs/finalize/gated_s{seed}"
    elif a.stage == 5:
        # After the first wave of stage 1: the plan's ExeaLM and the time-order board in mixed swarms were 0.10 bits
        # per byte behind the memory-free hybrid at step 1,000, far beyond the seed spread, so their other seeds are
        # dropped. In their place, three ways to win back the cost of reading one text in four windows: the doc-order
        # board (already in stage 1), swarms of 2, and steps four times as large (--accum 4, the same 49M tokens), which
        # is closer to the 512-row steps of the Expanse runs. Jobs stage 1 already finished are skipped by the runner.
        order = []
        for seed in (0, 1, 2):
            keep = ["doc_split", "doc_mixed", "core", "gated", "board_random"] + (["plan", "time_mixed"] if seed == 0 else [])
            order += [(name, seed) for name in keep]
        over_doc = dict(ARMS["doc_split"][0])
        for seed in (0, 1, 2):
            for name, (over, extra, _) in ARMS.items():
                if (name, seed) in order:
                    jobs.append(train(name, config_file(name, over), extra, seed))
                    runs.append(f"runs/finalize/{name}_s{seed}")
                    if name in BOARD_ARMS:
                        board.append(runs[-1])
            if seed == 2:
                continue
            tag = "doc_split+swarm2"
            jobs.append(train(tag, config_file(tag, dict(over_doc, swarm_size=2)), "--swarm_policy split", seed))
            runs.append(f"runs/finalize/{tag}_s{seed}")
            board.append(runs[-1])
            for name in ("core", "gated", "time_mixed", "doc_split", "doc_mixed"):
                over, extra, _ = ARMS[name]
                tag = f"big_{name}"
                jobs.append(train(tag, config_file(name, over), f"{extra} --accum 4 --warmup 50", seed, steps=750,
                                  lr=2e-3))
                runs.append(f"runs/finalize/{tag}_s{seed}")
                if name in BOARD_ARMS:
                    board.append(runs[-1])
        jobs.sort(key=lambda j: (int(j[0].rsplit("_s", 1)[1]), 0 if "doc" in j[0] else 1))   # seed by seed, doc first
        baseline = "runs/finalize/gated_s{seed}"
        ev = evals(5, runs, [b for b in board if "big_" not in b], baseline)
        ev += evals(5, [], [b for b in board if "big_" in b], "runs/finalize/big_gated_s{seed}")[1:]
        with open(os.path.join(HERE, "jobs", "stage5_train.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in jobs)
        with open(os.path.join(HERE, "jobs", "stage5_eval.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in ev)
        print(f"stage 5: {len(jobs)} training runs, {len(ev)} evaluation jobs in {REL}/jobs/")
        return
    elif a.stage == 4:
        # The library on text. One source at a time, so the library is drawn from the training pack itself and the
        # loop can keep each document's own chunks out of what it retrieves. lib and nolib are the same model; nolib
        # trains with no datastore (its library slot outputs zeros). Scored on held-out text with the library
        # attached and removed (stage4_lib_*.jsonl), against the gated Llama on the same source.
        cfg = config_file("library", LIBRARY)
        evals4 = []
        for src in ("en", "code"):
            data = f"--data data/tokpack/bpe32k/train_{src}"
            for seed in (0, 1):
                for name, more in (("lib", f"{data} --datastore_chunks {LIB_CHUNKS}"), ("nolib", data)):
                    jobs.append(train(f"{name}_{src}", cfg, more, seed))
                    run = f"runs/finalize/{name}_{src}_s{seed}"
                    for cond, store in (("attached", f"--datastore_data data/tokpack/bpe32k/train_{src} "
                                                     f"--datastore_chunks {LIB_CHUNKS} "), ("removed", "")):
                        out = f"results/finalize/stage4_lib_{name}_{src}_s{seed}_{cond}.jsonl"
                        evals4.append((f"fin4_{name}_{src}_s{seed}_{cond}", out,
                                       f"rm -f {out} && python -m exealm.eval.swarm_vs_solo --ckpt {run}/model.pt "
                                       f"--baseline runs/finalize/gated_{src}_s0/model.pt --policies random "
                                       f"--data data/tokpack/bpe32k/val_{src} {store}--seq 512 --batches 64 "
                                       f"--seed {seed} --device cuda --out {out}"))
            jobs.append(train(f"gated_{src}", GATED, f"--model llama {data}", 0))
        with open(os.path.join(HERE, "jobs", "stage4_train.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in jobs)
        with open(os.path.join(HERE, "jobs", "stage4_eval.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.writelines("\t".join(j) + "\n" for j in evals4)
        print(f"stage 4: {len(jobs)} training runs, {len(evals4)} evaluation jobs in {REL}/jobs/")
        return
    else:
        over, extra = base_config(a.base)
        for change in filter(None, a.change.split(",")):
            c = STAGE2[change][0]
            over = dict(over, **c) if isinstance(c, dict) else dict(over, pinned={**over.get("pinned", {}), "AG": 2})
        pick = config_file("final", over or None)
        glr = a.gated_lr or a.lr
        for seed in (0, 1):
            for name, cfg, more, lr in (("final", pick, extra, a.lr), ("gated", GATED, "--model llama", glr),
                                        ("core", CORE, "", a.lr)):
                tag = f"long_{name}"
                jobs.append(train(tag, cfg, more, seed, steps=12000, lr=lr, opt=a.opt))
                runs.append(f"runs/finalize/{tag}_s{seed}")
                if name == "final":
                    board.append(runs[-1])
        baseline = "runs/finalize/long_gated_s{seed}"
    with open(os.path.join(HERE, "jobs", f"stage{a.stage}_train.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines("\t".join(j) + "\n" for j in jobs)
    with open(os.path.join(HERE, "jobs", f"stage{a.stage}_eval.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.writelines("\t".join(j) + "\n" for j in evals(a.stage, runs, board, baseline))
    print(f"stage {a.stage}: {len(jobs)} training runs, {1 + len(board)} evaluation jobs in {REL}/jobs/")


if __name__ == "__main__":
    main()
