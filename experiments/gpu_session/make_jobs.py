"""Write the job files of the GPU session into experiments/gpu_session/jobs/ (see README.md in this folder).

    python experiments/gpu_session/make_jobs.py --seconds 500

Run it from the repo root. Each line it writes is TAG <tab> OUTPUT FILE <tab> COMMAND, for runner.py.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BIG = '{"d": 640, "n_layers": 14, "slots": 20, "h": 32}'             # about 0.1B parameters
GATE = {"attn_gate": True, "qk_norm": True}
HYBRID_13M = {"attn_layers": [1, 2, 4, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25, "RP": 0}, **GATE}
HYBRID_100M = {"attn_layers": [1, 2, 4, 5, 7, 8, 10, 11, 13], "init_weights": {"C": 4, "G": 0.25, "D": 0.25, "RP": 0}, **GATE}
QWEN_100M = {"delta_layers": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12], **GATE}
MODELS_100M = [("hybrid", "tonn_core", "--cfg", HYBRID_100M), ("llama_gate", "llama", "--llama_cfg", GATE),
               ("qwen35", "llama", "--llama_cfg", QWEN_100M), ("llama", "llama", "--llama_cfg", {})]


def lm(tag, out_dir, kind, flag, cfg, seed, extra):
    cmd = (f"python experiments/tokenizer_lm.py --toks bpe32k --out {out_dir} --size '{BIG}' --lr 0.001 --models {kind} "
           f"{flag} '{json.dumps(cfg)}' --seeds {seed} --tag {tag} {extra}")
    return tag, f"{out_dir}/lm_compare_{tag}.json", cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=500, help="training time of every equal-time run. 500 s is about 3,000 "
                    "steps of the 0.1B hybrid in bf16 on an MI300X; on a slower GPU raise it so the hybrid still gets ~3,000")
    ap.add_argument("--long_steps", type=int, default=15000, help="steps of the longer runs (5 times the 3,000-step runs)")
    a = ap.parse_args()
    jobs = {}

    # 1. equal training time at 0.1B, bf16. Run these ONE AT A TIME: each run times its own first 50 steps to decide
    #    how many steps fit in --seconds, so nothing else may share the GPU.
    out = "results/gpu_session/eqtime"
    jobs["eqtime"] = [lm(f"eqt100_{name}_s{seed}", out, kind, flag, cfg, seed, f"--amp bf16 --seconds {a.seconds}")
                      for seed in range(3) for name, kind, flag, cfg in MODELS_100M]

    # 2. the combined 0.1B recipe (board + library + weight-alter on text), which trained in 2 of 5 runs. The same
    #    settings again with a log line every 100 steps, and two candidate fixes on the seed that failed.
    out = "results/gpu_session/combined"
    base = ('python experiments/full_small.py --model exealm --wa 1 --preset 100m --p_kinds 0.7,0.15,0.075,0.075 --dense 1 '
            '--relay_chunk 8 --lib_on wiki --board_chunk 8 --eval_every 1000 --stop_at 3000 --log_every 100 --save_every 0 '
            f'--tokenizer results/tokenizer/bpe32k/tokenizer.json --out {out}')
    sched = '--schedule "400:0,0,0.5,0.5;2000:0.3,0.2,0.25,0.25"'
    combined = [("s0_lr2e3", f"--seed 0 --lr 2e-3 {sched}"), ("s1_lr2e3", f"--seed 1 --lr 2e-3 {sched}"),
                ("s0_lr1e3", f"--seed 0 --lr 1e-3 {sched}"), ("s1_lr1e3", f"--seed 1 --lr 1e-3 {sched}"),
                ("s1_lr2e3_growwhen", f"--seed 1 --lr 2e-3 {sched} --grow_when 1.0"),
                ("s1_lr2e3_longstart", '--seed 1 --lr 2e-3 --schedule "800:0,0,0.5,0.5;2000:0.3,0.2,0.25,0.25"')]
    jobs["combined"] = [(f"c100_{n}", f"{out}/full_small_exealm_{n}_seed{'1' if n.startswith('s1') else '0'}.json",
                         f"{base} --tag {n} {extra}") for n, extra in combined]

    # 3. longer runs: 5 times the tokens, on the big training packs so that no text repeats, bf16, equal steps
    out = "results/gpu_session/long"
    long = f"--amp bf16 --steps {a.long_steps} --train train_en_big,train_ja_big,train_code_big"
    jobs["long"] = [lm(f"long100_{name}_s{seed}", out, kind, flag, cfg, seed, long)
                    for seed in range(2) for name, kind, flag, cfg in MODELS_100M[:3]]

    # 4. small runs for two open issues
    out = "results/gpu_session/relay_capacity"                       # issue #58: how many facts can weight edits carry?
    cap = []
    for seed in range(3):
        for n in (4, 8, 16, 32, 64):
            for mem, flags, name in (("AG", "", ""), ("WA", "--wa_key hidden", "_wa_hkey")):
                tag = "" if n == 4 else f"_a2_f{n}"
                cap.append((f"cap_{mem}_f{n}_s{seed}", f"{out}/relay{tag}{name}_seed{seed}.json",
                            f"python experiments/swarm_relay.py --arms swarm --facts {n} --memory {mem} {flags} --seed {seed} --out {out}"))
    out = "results/gpu_session/leave_one_out"                        # issue #59: which circuit types help the start?
    loo = []
    for seed in range(3):
        for t in ("none", "D", "I", "GI", "M", "MI", "MIM", "EP", "G", "SM", "OS"):
            cfg = json.loads(json.dumps(HYBRID_13M))
            if t != "none":
                cfg["init_weights"][t] = 0
            tag = f"f13_leave_{t}_s{seed}"
            loo.append((tag, f"{out}/lm_compare_{tag}.json",
                        f"python experiments/tokenizer_lm.py --toks bpe32k --out {out} --models tonn_core --cfg '{json.dumps(cfg)}' --seeds {seed} --tag {tag}"))
    jobs["small"] = cap + loo

    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    for name, rows in jobs.items():
        with open(os.path.join(HERE, "jobs", f"{name}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join("\t".join(r) + "\n" for r in rows))
        print(f"jobs/{name}.txt: {len(rows)} jobs")


if __name__ == "__main__":
    main()
