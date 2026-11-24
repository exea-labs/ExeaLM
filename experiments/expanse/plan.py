"""The runs of the scaling study, in one place: what each trains on, for how many steps, what it costs, and the lines
that submit it. docs/budget.md says why the plan looks like this.

The study is the evidence for the next grant: ExeaLM against a Qwen3.5-style model (Gated DeltaNet and gated
attention, 3 to 1) of the same size, on the same tokens, trained the same way (docs/decisions.md). Two parts:
    3B        the headline: the pair at 3B on about 12B tokens each
    curve     the same recipe (the same tokens per parameter) at 0.1B, 0.25B and 0.5B, three seeds where cheap: with
              the 3B pair, one curve from 0.1B to 3B per model
ExeaLM is the memory-free hybrid with value residual and XSA, read as a stream; the board, the library and
weight-alter come after pretraining.

    python experiments/expanse/plan.py                                   # the table, at the rates measured on an MI300X
    python experiments/expanse/plan.py --rates results/expanse/throughput_h100.json --speedup 3.6
                                                                         # the same at the H100's measured rates, with
                                                                         # four GPUs 3.6 times as fast as one
    python experiments/expanse/plan.py --lr_check 500m                   # the short runs that pick a learning rate
    python experiments/expanse/plan.py --run exealm_500m_20x_s0 --lr 6e-4    # the lines that submit one run

Every run takes steps of 262,144 tokens, on one GPU or four: --accum makes up the difference. A run on four GPUs that
is S times as fast as on one is charged 4 / S times the GPU-hours of the same run on one GPU.

This script's --batch is the windows one GPU holds in a forward pass, which is what has to fit in its memory. Every row
is STREAM windows read in order with the state carried (--stream), so a GPU reads batch / STREAM rows. The --batch in
the lines it prints is pretrain's: rows over all the run's GPUs, so 32 windows on each of four GPUs is --batch 32.

--variant memory trains ExeaLM with its board and library slots in swarms instead. On the MI300X (results/finalize)
swarms that read one text in four windows cost the 0.1B model about 0.1 bits per byte by step 1,000, so it is not the
default.
"""
import argparse
import glob
import json
import os

SEQ = 512
STREAM = 4                                     # windows per row: pretraining reads 2,048-token stretches as 4 windows
                                               # with the state carried (docs/brain.md), the baselines too
STEP_TOKENS = 262_144
ALLOCATION = 2000                              # GPU-hours
TARGET = (9e9, 1e12)                           # what the study is evidence for: 9B parameters on 1T tokens
FIRST_JOB = 6                                  # first_contact.sb: one node for about an hour and a half
NINE_B = 36                                    # the 9B preset's speed and memory, sharded: 3 hours on one node, then an
                                               # hour each on two and four nodes (needs FSDP, docs/budget.md)
EVALUATION = 100                               # every run on held-out text, long documents and downstream benchmarks
LR_STEPS = {"100m": 1000, "250m": 1000, "500m": 1000, "3b": 500}
LR_RATES = {"100m": 3, "250m": 3, "500m": 3, "3b": 2}  # how many learning rates the budget counts at each size

# Parameters in millions with the 32k tokenizer (tests/test_plan.py counts them from the code), rows per forward pass
# on one 80 GB GPU, and the GPUs a run uses. The 0.1B and 0.25B runs take one GPU each on the shared partition: four
# of them run side by side and none pays for the others' gradients. The rows are sized from two A40 runs
# (results/a40_check: 11.8 GB at 0.1B with 32 rows, 23.1 GB at 0.5B with 16), scaled by width x depth: about 40 GB at
# 0.5B and 1B, and about 59 GB at 3B, where 8 rows would come to about 71 GB. Estimates: throughput.py measures them.
SIZES = {
    "100m": dict(exealm=94.28, core=93.43, gated=95.47, qwen=95.69, batch=64, gpus=1),
    "250m": dict(exealm=246.24, core=245.3, gated=251.56, qwen=252.1, batch=32, gpus=1),
    "500m": dict(exealm=498.08, core=497.07, gated=510.14, qwen=511.27, batch=32, gpus=4),
    "1b": dict(exealm=966.06, core=964.95, gated=989.81, qwen=991.94, batch=16, gpus=4),
    "3b": dict(exealm=2902.88, core=2901.54, gated=2980.06, qwen=2986.19, batch=4, gpus=4),
}
# The study: part, size, tokens per ExeaLM parameter, seeds, models. Every model of a line trains for the same number
# of steps. 4.1 tokens per parameter is what the 3B pair can afford (about 12B tokens each); the curve uses the same
# ratio so that all four sizes lie on one curve.
TPP = 4.1
PAIR = ("exealm", "qwen")
STUDY = [
    ("3B", "3b", TPP, (0,), PAIR),
    ("curve", "100m", TPP, (0, 1, 2), PAIR),
    ("curve", "250m", TPP, (0, 1, 2), PAIR),
    ("curve", "500m", TPP, (0, 1), PAIR),
]
# Effective TFLOP/s (6 x parameters x tokens per second) of one MI300X in bf16: results/funding/README.md, section 6.
# Measured for the memory-free hybrid and the plain Llama at 0.1B and 0.5B. Assumptions until the first job on
# Expanse replaces them: ExeaLM with its memory slots runs at the hybrid's rate, the gated Llama at the plain Llama's,
# 0.25B at the 0.1B rates (smaller shapes run slower, so this overstates its cost), 1B and 3B at the 0.5B rates, and
# the Qwen3.5-style model at ExeaLM's rates (its Gated DeltaNet is plain PyTorch, not a fused kernel).
MI300X = {"100m": dict(exealm=56.2, gated=85.1), "250m": dict(exealm=56.2, gated=85.1),
          "500m": dict(exealm=81.0, gated=124.8), "1b": dict(exealm=81.0, gated=124.8),
          "3b": dict(exealm=81.0, gated=124.8)}
SIZE_NAME = {"0.1B": "100m", "0.25B": "250m", "0.5B": "500m", "1B": "1b", "3B": "3b"}   # throughput.py's names


def steps_of(preset, tpp, variant="core"):
    return round(tpp * SIZES[preset]["core" if variant == "core" else "exealm"] * 1e6 / STEP_TOKENS)


def runs(variant="core"):
    """Every training run of the plan: name, part, preset, model ("exealm", "gated" or "qwen"), seed and steps."""
    return [(f"{model}_{preset}_{tpp:g}x_s{seed}", part, preset, model, seed, steps_of(preset, tpp, variant))
            for part, preset, tpp, seeds, models in STUDY for seed in seeds for model in models]


def measured(rows):
    """Rates from the rows of a throughput_<gpu>.json: the last bf16 row without torch.compile or TF32 for each size
    and model: "hybrid" rows (the memory-free hybrid the plan trains) for ExeaLM, else "exealm"; "gated" rows for the
    Llama, else "llama"; "qwen" rows for the Qwen3.5-style model, else ExeaLM's rate."""
    rates = {p: dict(r, qwen=r["exealm"]) for p, r in MI300X.items()}
    source = {p: dict(exealm="MI300X", gated="MI300X", qwen="MI300X") for p in MI300X}
    for want, kinds in (("exealm", ("exealm", "hybrid")), ("gated", ("llama", "gated")), ("qwen", ("qwen",))):
        for kind in kinds:                                                # the later kind is the closer match
            for r in rows:
                preset = SIZE_NAME.get(r.get("size"))
                if want == "qwen" and preset and not any(x.get("model") == "qwen" for x in rows):
                    rates[preset]["qwen"] = rates[preset]["exealm"]
                if (preset and r.get("model") == kind and r.get("precision") == "bf16"
                        and r.get("effective_TFLOPs", 0) > 0 and not r.get("compile") and not r.get("tf32")):
                    rates[preset][want], source[preset][want] = r["effective_TFLOPs"], kind
    return rates, source


def gpu_hours(params_m, tokens, tflops, gpus=1, speedup=4.0):
    one = 6 * params_m * 1e6 * tokens / (tflops * 1e12 * 3600)
    return one * (gpus / speedup if gpus > 1 else 1)


def params_of(preset, model, a):
    return SIZES[preset]["core" if (model == "exealm" and a.variant == "core") else model]


def rate_of(rates, preset, model):
    return rates[preset].get(model, rates[preset]["exealm"])


def cost(preset, model, rates, a, tokens):
    return gpu_hours(params_of(preset, model, a), tokens, rate_of(rates, preset, model), a.gpus or SIZES[preset]["gpus"],
                     a.speedup)


def lr_check_hours(rates, a):
    return sum(cost(p, m, rates, a, LR_STEPS[p] * STEP_TOKENS) * LR_RATES[p] for p in LR_STEPS for m in PAIR)


def table(rates, source, a):
    print(f"Target: {TARGET[0] / 1e9:.0f}B parameters on {TARGET[1] / 1e12:.0f}T tokens, "
          f"{TARGET[1] / TARGET[0]:.0f} tokens per parameter.\n")
    print("| Part | Run | Parameters | Tokens | Tokens per parameter | Steps | GPUs | GPU-hours |")
    print("|---|---|---|---|---|---|---|---|")
    total, count = 0.0, 0
    for name, part, preset, model, seed, steps in runs(a.variant):
        params, tokens = params_of(preset, model, a), steps * STEP_TOKENS
        h = cost(preset, model, rates, a, tokens)
        total, count = total + h, count + 1
        print(f"| {part} | `{name}` | {params:,.1f}M | {tokens / 1e9:.1f}B | {tokens / params / 1e6:.1f} | "
              f"{steps:,} | {a.gpus or SIZES[preset]['gpus']} | {h:,.0f} |")
    lr = lr_check_hours(rates, a)
    print(f"| | **The {count} runs** | | | | | | **{total:,.0f}** |")
    print(f"| | First job on Expanse | | | | | | {FIRST_JOB} |")
    print(f"| | The 9B model: speed and memory on 1, 2 and 4 nodes | | | | | | {NINE_B} |")
    print(f"| | Learning-rate checks | | | | | | {lr:,.0f} |")
    print(f"| | Evaluation: held-out text, long documents, benchmarks | | | | | | {EVALUATION} |")
    left = ALLOCATION - total - FIRST_JOB - NINE_B - lr - EVALUATION
    print(f"| | **Reserve of the {ALLOCATION:,}** | | | | | | **{left:,.0f}** |")
    if left < 0:
        print(f"\nOver the allocation by {-left:,.0f} GPU-hours. Shorten the 3B pair. Do not drop a baseline.")
    used = sorted({f"{p} {m}: {source[p][m]}" for p in SIZES for m in ("exealm", "gated", "qwen")
                   if source[p].get(m, "MI300X") != "MI300X"})
    print("\nRates: " + ("; ".join(used) + "; the rest as on the MI300X" if used else "as measured on one MI300X")
          + (f". Four GPUs taken as {a.speedup:g} times one." if a.speedup != 4 else
             ". Four GPUs taken as 4 times one: pass --speedup with the measured figure."))


def pack_tokens(data):
    """Tokens in the packs a mix .json names, or None if they cannot be read from here."""
    try:
        total = 0
        for src in json.load(open(data))["sources"].values():
            paths = src["paths"] if isinstance(src["paths"], list) else [src["paths"]]
            for p in paths:
                for d in (glob.glob(p) if any(c in p for c in "*?") else [p]):
                    total += json.load(open(os.path.join(d, "meta.json")))["n_tokens"]
        return total
    except (OSError, KeyError, ValueError):
        return None


def command(run, preset, model, seed, a, steps, lr=None, check=False):
    s = SIZES[preset]
    gpus, windows = a.gpus or s["gpus"], a.batch or s["batch"]
    batch = windows * gpus // STREAM                # pretrain's --batch: rows of STREAM windows, over all the GPUs
    accum, rest = divmod(STEP_TOKENS, batch * SEQ * STREAM)
    if rest or windows % STREAM:
        raise SystemExit(f"--batch {windows} on {gpus} GPU(s) does not divide a step of {STEP_TOKENS:,} tokens into "
                         f"whole rows of {STREAM} windows")
    args = [f"--preset {preset}"]
    if model == "gated":
        args.append("--model llama --config experiments/expanse/gated_llama.json")
    elif model == "qwen":
        args.append(f"--model llama --config experiments/expanse/qwen_{preset}.json")
    else:
        args.append("--search 0" + (" --config experiments/expanse/exealm_core.json" if a.variant == "core" else ""))
    args.append(f"--data {a.data} --val {a.val} --seq {SEQ} --stream {STREAM} --batch {batch} --accum {accum}")
    if check:                                       # a learning-rate check: a tenth of it is warm-up
        args.append(f"--steps {steps} --warmup {steps // 10} --lr {lr:g} --seed {seed} --eval_every 100")
    else:
        args.append(f"--steps {steps} --lr {a.lr:g} --seed {seed} --eval_every {a.eval_every}")
    args.append(f"--eval_batches {max(1, 512 // batch)}")   # about 512 windows per evaluation, each read alone
    lines = [f"RUN={run} \\", 'ARGS="' + " ".join(args) + '" \\']
    flags = ""
    if gpus == 1:                                   # one GPU: the shared partition, and one task's worth of cores
        flags = "--ntasks-per-node=1 "
        lines.append(f'SBATCH_PARTITION=nairr-gpu-shared SBATCH_GPUS=h100:1 SBATCH_EXTRA="{flags.strip()}" \\')
    lines.append(f"sbatch {flags}experiments/expanse/train.sb")
    return "\n".join(lines)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default=None, help="a throughput_<gpu>.json from experiments/throughput.py")
    ap.add_argument("--speedup", type=float, default=4.0, help="how many times as fast four GPUs are as one "
                    "(speedup_on_4_gpus in results/expanse/scaling_500m.json)")
    ap.add_argument("--variant", default="core", choices=["core", "memory"],
                    help="ExeaLM without memory slots, every row alone (core), or with board and library slots in swarms")
    ap.add_argument("--run", default=None, help="print the lines that submit this run (needs --lr)")
    ap.add_argument("--lr_check", default=None, help="print the learning-rate check of this size: " + ", ".join(LR_STEPS))
    ap.add_argument("--lr", type=float, default=None, help="the learning rate the check picked for this model and size")
    ap.add_argument("--lrs", default="3e-4,6e-4,1e-3", help="the learning rates --lr_check tries")
    ap.add_argument("--gpus", type=int, default=0, choices=[0, 1, 4],
                    help="GPUs per run (default: 1 at 0.1B and 0.25B, else 4)")
    ap.add_argument("--batch", type=int, default=0, help="rows per forward pass on one GPU, if the default does not fit")
    ap.add_argument("--data", default="experiments/expanse/mix_tokpack.json")
    ap.add_argument("--val", default="data/tokpack/bpe32k/val_en")
    ap.add_argument("--eval_every", type=int, default=500)
    return ap


def defaults():
    return parser().parse_args([])


def main(argv=None):
    a = parser().parse_args(argv)
    rates, source = measured(json.load(open(a.rates)) if a.rates else [])
    by_name = {r[0]: r for r in runs(a.variant)}
    if a.lr_check:
        if a.lr_check not in LR_STEPS:
            raise SystemExit(f"--lr_check takes one of {', '.join(LR_STEPS)}")
        steps, lrs = LR_STEPS[a.lr_check], [float(x) for x in a.lrs.split(",")]
        each = sum(cost(a.lr_check, m, rates, a, steps * STEP_TOKENS) for m in PAIR)
        print(f"# {len(lrs)} learning rates x 2 models x {steps:,} steps at {a.lr_check}: "
              f"about {each * len(lrs):,.0f} GPU-hours. Each model keeps its own best rate.\n")
        for lr in lrs:
            for model in PAIR:
                print(command(f"lr_{model}_{a.lr_check}_{lr:g}", a.lr_check, model, 0, a, steps, lr, check=True) + "\n")
    elif a.run:
        if a.run not in by_name:
            raise SystemExit(f"no run called {a.run}. The runs: {', '.join(by_name)}")
        _, part, preset, model, seed, steps = by_name[a.run]
        if a.lr is None:
            raise SystemExit("--run needs --lr: the rate the learning-rate check picked for this model at this size "
                             f"(python experiments/expanse/plan.py --lr_check {preset})")
        tokens, have = steps * STEP_TOKENS, pack_tokens(a.data)
        if have is None:
            print(f"# Could not read the size of the packs in {a.data} from here. {a.run} reads {tokens / 1e9:.1f}B "
                  f"tokens:\n# they should hold at least a quarter of that.\n")
        elif tokens > 4 * have:
            print(f"# NOT READY: {a.run} reads {tokens / 1e9:.1f}B tokens and {a.data} holds {have / 1e9:.2f}B, so it "
                  f"would read them {tokens / have:.0f} times.\n# Pack the full data first and pass it with --data.\n")
        print(f"# about {cost(preset, model, rates, a, tokens):,.0f} GPU-hours")
        print(command(a.run, preset, model, seed, a, steps))
    else:
        table(rates, source, a)


if __name__ == "__main__":
    main()
