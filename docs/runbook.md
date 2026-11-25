# Runbook: the 2,000 GPU-hours on SDSC Expanse

Every command, in order, and what to expect. The plan and its reasons are in [budget.md](budget.md) and
[decisions.md](decisions.md); this page is only how to run it. All of it goes through one script,
`experiments/expanse/go.sh`, from the repo root on a login node (`login.expanse.sdsc.edu`), branch `nairr`.

## Once

```bash
git clone -b nairr https://github.com/exea-labs/ExeaLM.git && cd ExeaLM
conda create -n exealm python=3.11 -y && conda activate exealm
huggingface-cli login                       # a token of an account in exea-labs: reads the data, writes the runs
export SBATCH_ACCOUNT=<your NAIRR project>
export KEEP=<where runs are kept>           # see "Where everything is saved"; default ./runs
bash experiments/expanse/go.sh setup        # the Python environment, checks that torch sees CUDA
bash experiments/expanse/go.sh data         # the 30B-token slice (slice-30b-r2): ~60 GB into data/, ~10-30 min
```

`go.sh data` writes `data/mix_pretrain.json`, `data/mix_pretrain_val.json` and `data/mix_pretrain_anneal.json` with
local paths; every job reads them. `slice-30b-r2` is on Hugging Face (28.5B tokens, published 25 November). Every run of
the study must read the same build: each build has its own tokenizer, so never mix it with the first build,
`slice-30b`.

## The phases

Run them one at a time and look at the results between them, or `bash experiments/expanse/go.sh all` inside `tmux`
(it submits a phase, waits until the queue is empty, then the next). Every command takes `DRY=1` to print the
`sbatch` lines without submitting.

| Phase | Command | Jobs | Wall time | GPU-hours (MI300X rates) | What comes out |
|---|---|---|---|---|---|
| 1. First job | `go.sh first` | 1 node | ~1.5 h | ~6 | `results/expanse/`: gpu_check.txt, throughput, `plan_at_h100_rates.txt`, scaling_500m.json (4-GPU speed-up), lead_100m.json |
| 2. Learning-rate checks | `go.sh lr` | 31 | a few hours | ~74 | `$KEEP/lr_*` |
| 3. Pick rates | `go.sh pick` | none | seconds | 0 | `experiments/expanse/lr_choice.json` |
| 4. The curve and the long pair | `go.sh curve` | 26 | ~1-2 days | ~238 | `$KEEP/{exealm,llama,qwen}_{100m,250m,500m}_4.1x_s*`, `$KEEP/{exealm,qwen}_100m_111x_s0` |
| 5. 3B pair | `go.sh 3b` | 2 (one node each) | ~7 days | ~1,456 | `$KEEP/{exealm,qwen}_3b_4.1x_s0` |
| 6. Score and report | `go.sh eval` | 1 GPU | a few hours | ~10 | `results/expanse/text.json`, `stream8.json`, `bench_*.json`, `report.md`, `scaling.png` |
| 7. The 9B model | `go.sh nine_b` | 3 (1, 2 and 4 nodes) | ~3 h, ~1 h, ~1 h | ~36 | `results/expanse/nine_b.json`; then `go.sh report` prices the 9B run with it |

`go.sh status` shows the queue and every run's latest step, bits per byte and tokens per second.

**Check after phase 1, before spending more:**
- `gpu_check.txt` passes.
- `plan_at_h100_rates.txt` is the plan's table at the H100's measured speed: if the total plus the reserve exceeds
  2,000, shorten the 3B pair (budget.md, "If fewer hours fit"). `python experiments/expanse/plan.py --rates
  results/expanse/throughput_h100.json --speedup <speedup_on_4_gpus from scaling_500m.json>` reprints it.
- `lead_100m.json`: ExeaLM against the base Qwen3.5-style model and the plain Llama at 0.1B, 3 seeds, on this data.
  If ExeaLM is behind either, stop and decide before the curve.
- If the Qwen3.5-style model runs much slower than ExeaLM (`throughput_h100.json`, model `qwen`), its Gated DeltaNet
  should move to the flash-linear-attention kernels before the long runs, so the baseline is not slowed by our code.

## What each training job does

Every run is `experiments/expanse/train.sb` with `RUN` and `ARGS` set by `plan.py`:
- copies `data/` to the node's NVMe and trains from there;
- writes a checkpoint every `SAVE_MIN` minutes (default 30; for the 3B runs use `SAVE_MIN=120`: a 3B checkpoint with
  its optimizer state is ~35 GB) and copies it to `KEEP` at once;
- stops cleanly `MARGIN_MIN` (45) minutes before the 48-hour limit, copies the run back, and submits the next job, which
  continues from the checkpoint (`--resume auto`), up to `JOBS_LEFT` (20) more jobs;
- when the run is finished, uploads `model.pt` and `log.jsonl` to the private Hugging Face repo `HF_REPO` (default
  `exea-labs/exealm-nairr-runs`).
A job that crashes is not resubmitted: read `logs/expanse/<job>.out`, then submit the same run again
(`python experiments/expanse/plan.py --run <name> --lr auto | bash`); it continues from the last checkpoint in `KEEP`.

ExeaLM trains with one 512-token window per row for the first 80% of its steps and rows of four streamed windows for
the last 20% (`--stream 4 --stream_from 0.8`); the plain Llama and the Qwen3.5-style model read one window per row
throughout. Every run reads the anneal mix for the last 10% (`--anneal_data`).

## Where everything is saved

| What | Where |
|---|---|
| Checkpoints and finished models | `KEEP`: a folder (default `./runs`, in your home directory, 100 GB quota) or `s3://bucket/prefix` on SDSC's Ceph (needs `s3cmd` configured). The 3B checkpoints need the S3 option: two runs at ~35 GB each do not fit beside the 60 GB of data. |
| Finished models, off SDSC | Hugging Face `exea-labs/exealm-nairr-runs` (private), `runs/<name>/` |
| Results | `results/expanse/` and Hugging Face `exea-labs/exealm-nairr-runs`, `results/expanse/` |
| Job output | `logs/expanse/` |

## What to expect

- **0.1B on the MI300X** (different data, 3,000 steps, 2 seeds, `results/finalize/final_text.json`): ExeaLM 1.387
  bits per byte, a gated Llama 1.446, the base Qwen3.5-style model 1.460. On older data, 3 seeds, the hybrid without
  value residual or XSA was 0.095 ahead of the plain Llama (1.413 against 1.509). The cluster's numbers are on the new
  data and tokenizer, so they will differ.
- **Watch for:** loss spikes that do not recover (lower that model's learning rate and continue from the last
  checkpoint before the spike), `processes_in_sync: false` in a log (the run resynchronizes itself), runs much slower
  than `plan_at_h100_rates.txt` says.
- **The 9B measurement** (`nine_b.py`, sharded with PyTorch's FSDP2) has not run sharded anywhere: FSDP2 does not
  train on a CPU in torch 2.6. On a CPU, one process trains the model, and the changes FSDP needs (value residual's
  shared values, TONN's scalar parameters, checkpointing TONN's running sizes) give the same losses and gradients as
  the unchanged model (`tests/test_nine_b.py`). Run `go.sh nine_b` after phase 1, when the first job has shown the GPUs
  and NCCL work. It submits the one-node job and makes the two- and four-node jobs wait for it to write a row
  (`afterok`). Compare `nine_b.sb`'s `#SBATCH` lines (`--gpus-per-node`, one torchrun per node) with SDSC's sample
  first.

## The three models

The configs in `experiments/expanse/` are what every phase trains ([decisions.md](decisions.md), 25 November):
`exealm_core.json` is ExeaLM (no memory slots, a TONN layer every fourth layer, value residual, XSA; streamed in the last
20%); `qwen_<size>.json` is the base Qwen3.5-style model, the baseline at every size (Gated DeltaNet in 3 of every 4
layers, gated attention, QK-norm); the plain Llama, on the curve only, is `--model llama` with no config.
`gated_llama.json` is kept for the MI300X record only.
