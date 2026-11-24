# ExeaLM on Expanse: H100 nodes, four GPUs at a time

How the scaling study runs on the SDSC Expanse AI Resource: ExeaLM and a gated Llama side by side at 0.1B, 0.5B, 1B
and 3B, from our NAIRR Pilot allocation of 2,000 GPU-hours. The plan and its reasons are in
[docs/budget.md](../../docs/budget.md).

**Nothing here has run on Expanse yet.** What has been checked:

- On two rented NVIDIA A40s ([results/a40_check](../../results/a40_check/README.md)): `gpu_check.py` passes in full,
  two GPUs stay in sync and train 1.86 times as fast as one at the 0.1B shape, and a resumed run is identical to the
  last bit. That was this branch's first multi-GPU trainer. Training on several GPUs now goes through DDP (the one on
  `main`, with this branch's gradient accumulation and time limits added), which has not run on a GPU yet:
  `gpu_check.py` on any two NVIDIA GPUs checks it before the allocation is spent.
- On a laptop CPU: training with two processes (`tests/test_ddp.py`), the plan's arithmetic and commands
  (`tests/test_plan.py`), the first job's lead check with tiny models, and `train.sb` end to end with Slurm's commands
  faked (a run stopped by its time limit, copied off the "node", resubmitted and finished by the next job).

## The machine

From the [Expanse user guide](https://www.sdsc.edu/systems/expanse/user_guide.html), read on 18 November 2026:

| | |
|---|---|
| Node | 4 NVIDIA H100 (80 GB each), two 36-core Intel CPUs, 1 TB of memory, 6.4 TB of local NVMe |
| Partitions | `nairr-gpu`: whole nodes, up to 4 per job. `nairr-gpu-shared`: one node, fewer than 4 GPUs |
| GPUs | requested with their type: `--gpus=h100:4` |
| Time limit | 48 hours per job |
| Charge | GPUs × hours. A whole node costs 4 GPU-hours per hour |
| Local disk | `/scratch/$USER/job_$SLURM_JOBID`, emptied when the job ends |
| Storage that stays | home (100 GB) and Ceph through S3. The Lustre filesystem is not mounted on these nodes |
| Login | `login.expanse.sdsc.edu`, the same as the rest of Expanse |

Three things follow. A run has to use all four GPUs or it wastes three quarters of what it is charged, so training
runs as four processes (DDP under `torchrun`). A run longer than 48 hours has to stop and continue, and since the
local disk is emptied, its checkpoint has to be copied off the node first. And the data has to be copied onto the
node at the start of every job.

## 1. Setup, on a login node

```bash
git clone https://github.com/exea-labs/ExeaLM && cd ExeaLM
conda create -n exealm python=3.11 && conda activate exealm
huggingface-cli login                       # a token that can read exea-labs/exealm-studio-data (private)
bash experiments/expanse/setup.sh           # installs, downloads 1.2 GB of token packs into ./data, runs the tests
export SBATCH_ACCOUNT=<your project>        # every sbatch below needs it
```

## 2. The first job: is it right, how fast is it, and which ExeaLM do the long runs train?

```bash
sbatch experiments/expanse/first_contact.sb          # about an hour and a half on one node: 6 GPU-hours
```

It writes to `results/expanse/`:

| File | What it answers |
|---|---|
| `gpu_check.txt` | Does every circuit type, and each whole model, compute on the H100 what it computes on the CPU, in 32-bit and in bf16? Does a short run train, do two processes on two GPUs stay in sync, and is a resumed run identical to the last bit? |
| `throughput_h100.json` | Tokens per second and peak memory on one H100. The memory-free hybrid and the plain Llama at 0.1B and 0.5B (the same measurement as `results/funding/throughput_mi300x.json`). The plan's own two models, ExeaLM with its memory slots in swarms and the gated Llama, at all four sizes, and at 0.5B with `torch.compile` and with TF32 |
| `plan_at_h100_rates.txt` | The compute plan at those rates, as if four GPUs were four times one |
| `profile_circuits_*.json` | Which circuit types gain least from bf16 (the open to-do from the MI300X session) |
| `scaling_500m.json` | The real training loop at 0.5B on one GPU and on four: tokens per second, and the speed-up |
| `lead_100m.json` | At 0.1B, 3,000 steps, three seeds: bits per byte on the four held-out sets for the gated Llama, for ExeaLM as the plan trains it, and for the memory-free hybrid the 0.1B lead was measured on |

`gpu_check.py` and `throughput.py` run on any NVIDIA card, so the correctness part can be done before the allocation
starts: `python experiments/gpu_check.py`.

**Then print the plan at what was measured.** The compute plan counts GPU-hours as if every run had one GPU. On four
GPUs a run costs 4 GPU-hours per hour, so if four GPUs are 3.6 times as fast as one, every token costs 4 / 3.6 = 1.11
times that. With the `speedup_on_4_gpus` of `scaling_500m.json`:

```bash
python experiments/expanse/plan.py --rates results/expanse/throughput_h100.json --speedup 3.6
```

If the reserve comes out negative, shorten the largest runs (the order is in docs/budget.md). If the speed-up is well
under 4, add `--gpus 1` to see the plan with one GPU per run on the shared partition, which pays nothing for
gradient averaging.

**And decide which ExeaLM the long runs train**, from the last lines of the job's output (or `lead_100m.json`). If
`lead_exealm` is ahead of `lead_gated` on every seed and no more than 0.01 bits per byte behind `lead_core`, train
ExeaLM as it is. Otherwise add `--variant core` to every `plan.py` line below: the memory-free hybrid, every row
read alone.

Check the 3B rows of `throughput_h100.json` too. The plan trains 3B with 4 rows per GPU in a forward pass, sized
from an estimate. If the row with 8 fit with room to spare, pass `--batch 8` to `plan.py` for the 3B runs.

## 3. The runs of the plan

```bash
python experiments/expanse/plan.py                                  # every run: tokens, steps, GPUs, GPU-hours
python experiments/expanse/plan.py --lr_check 500m                  # six short runs: three learning rates, both models
python experiments/expanse/plan.py --run exealm_500m_s0 --lr 6e-4   # the lines that submit one run
```

The last command prints lines to paste, here with a learning rate made up for the example:

```bash
RUN=exealm_500m_s0 \
ARGS="--preset 500m --search 0 --data experiments/expanse/mix_tokpack.json --val data/tokpack/bpe32k/val_en --seq 512 --batch 128 --accum 4 --steps 114441 --lr 0.0006 --seed 0 --eval_every 500 --eval_batches 4" \
sbatch experiments/expanse/train.sb
```

In order:

1. **Pack the data.** The packs `setup.sh` downloads hold 413M tokens (English, Japanese and code at 0.5, 0.3 and
   0.2). The 0.5B pair reads 30B, which would be these packs 73 times. The study needs at least 7.5B packed tokens.
   Pass the new mix with `--data` to every `plan.py` line. Without it `--run` marks the run NOT READY.
2. **The learning-rate check of a size, then its pair.** `--lr_check` prints short runs of both models at 3e-4, 6e-4
   and 1e-3 (`--lrs` for others). Compare the last `val_bits_per_token` in each `runs/lr_*/log.jsonl`
   (`val_bits_per_token_board_cut` for ExeaLM). Each model keeps its own best rate. If the best is the lowest or
   the highest rate tried, try the next one out before trusting it.
3. **The pairs, smallest first**: `exealm_100m_s0`, `gated_100m_s0`, `exealm_100m_s1`, `gated_100m_s1`, then the
   0.5B, 1B and 3B pairs. The four 0.1B runs take one GPU each on the shared partition and can run at once.
4. **Score each finished pair**: `python experiments/eval_text.py runs/gated_500m_s0 runs/exealm_500m_s0 --out
   results/expanse/text_500m.json`, on a GPU. Bits per byte on the four held-out sets, the same windows for both
   models, every row read alone. docs/budget.md says how to read the logs at equal GPU-hours and across sizes.

A run that is not in the plan is the same three lines with your own `RUN` and `ARGS`. A pilot on the packs at hand:

```bash
export RUN=pilot_500m_s0
export ARGS="--preset 500m --search 0 --data experiments/expanse/mix_tokpack.json --val data/tokpack/bpe32k/val_en --seq 512 --batch 64 --steps 3000 --lr 1e-3"
sbatch experiments/expanse/train.sb
```

That is 3,000 steps of 32,768 tokens, 98M tokens in all.

`train.sb` copies `./data` to the node's NVMe, continues the run from its last checkpoint if it has one, trains on
the job's four GPUs, copies every new checkpoint back to `./runs/$RUN`, stops cleanly 45 minutes before the time
limit and submits the next job, until the run has written `model.pt`. Its settings are listed at the top of the file.
A job that crashes is not resubmitted.

**What one step is.** `--batch` is the rows of one forward pass over all the job's GPUs (each GPU reads batch / GPUs
of them, whole swarms only) and `--accum` how many passes go into one optimizer step. One step trains on
batch × seq × accum tokens (128 × 512 × 4 = 262,144 in the plan's 0.5B runs, 32 rows on each of four GPUs;
64 × 512 = 32,768 in the pilot, 16 rows on each), and `--steps`, the warm-up and the learning-rate schedule count
those steps. `plan.py --batch` is different: the rows one GPU reads, which is what has to fit in its memory. Every evaluation line in `log.jsonl` has the
tokens seen so far (`tokens`), the rate since the last one (`tokens_per_sec`), the peak memory of GPU 0
(`gpu_mem_gb`) and whether the four processes still hold the same model (`processes_in_sync`, which should always be
true).

**The baseline** is the same line with `--model llama --config experiments/expanse/gated_llama.json` instead of
`--search 0`: a Llama with gated attention and QK-norm. Without the `--config` it is a plain Llama.
`--config experiments/expanse/exealm_core.json` is ExeaLM without its board and library slots.

**One GPU instead of four:**

```bash
export SBATCH_PARTITION=nairr-gpu-shared SBATCH_GPUS=h100:1 SBATCH_EXTRA="--ntasks-per-node=1"
sbatch $SBATCH_EXTRA experiments/expanse/train.sb
```

`SBATCH_EXTRA` carries the options to the follow-on jobs. `plan.py` prints these lines for the 0.1B runs, and for
any run with `--gpus 1`.

**Where runs are kept.** A checkpoint holds the weights and two optimizer statistics per weight, 12 bytes per
parameter: about 6 GB at 0.5B, 12 GB at 1B and 35 GB at 3B, and a finished run holds two files of that size. The
default, `./runs` in your 100 GB home directory, is enough for the 0.1B runs and the 0.5B pair. For the larger runs set
`KEEP=s3://<bucket>/runs` (and `DATA=s3://<bucket>/data` once the packed data outgrows home), which uses `s3cmd`.

**Memory at 3B.** Weights, gradients and the two optimizer statistics take 16 bytes per parameter in 32-bit: 46 GB of
an H100's 80 GB at 2.9B parameters, before any activations (DDP keeps the gradients in its communication buffers,
so it adds no second copy). That is arithmetic, not a measurement.
`throughput.py --sizes 3B` reports the real peak and halves the batch until it fits. `plan.py --batch` changes the
rows per GPU in a forward pass and keeps the step at 262,144 tokens.

## To check on the first login

We could not check these from outside:

- The `#SBATCH` lines for tasks and cores in both job files are our guess. SDSC's own sample is in
  `/cm/shared/examples/sdsc/ExpanseAIR/pytorch`: copy what differs (a `--mem` line, for example).
- Whether compute nodes reach the internet (Weights & Biases, Hugging Face). The job files assume they do not: the
  data is downloaded on the login node by `setup.sh`.
- The S3 endpoint and bucket for the Ceph storage, and whether `s3cmd` is configured for it.
- Whether the default `pip install torch` matches the nodes' NVIDIA driver. `first_contact.sb` prints both.

## Not done yet

- A model that does not fit one GPU. Every process holds the whole model, so this is not a way around the 80 GB.
- More than one node per run. The code does not care, but `train.sb` starts its processes on one node.
- The retrieval library searched across GPUs: each process holds and searches the whole datastore.
- Fused kernels for the scans ([#16](https://github.com/exea-labs/ExeaLM/issues/16)). `profile_circuits_*.json` from
  the first job says which circuit to start with.
