# Training

## Pretraining

| Setting | Starting value | Why |
|---|---|---|
| Model | `100m` preset, 104.8M parameters (94.3M with the 32k tokenizer) | |
| Tokens | about 100B | about 1,000 tokens per parameter, far past "compute-optimal" on purpose. Past roughly 200B, extra data buys little ([DATA.md](../DATA.md)) |
| Context | 2,048 tokens, extended to 8,192 near the end | |
| Batch | about 1M tokens | |
| Learning rate | warmup, stable, then a linear decay at the end (WSD), where DATA.md's annealing data goes in | during the stable phase we can branch off and try a different final data mix without restarting |
| Optimizer | AdamW (β 0.9/0.95, weight decay 0.1), gradient clip 1.0 | `--optimizer muon` gives the weight matrices Muon steps at the same learning rate and weight decay, and keeps AdamW for the embedding, norms and scalars (`exealm/train/muon.py`). At 13M both were best at lr 4e-3, where Muon was 0.017 and 0.030 bits per byte lower on two seeds ([results/muon](../results/muon/README.md), #21); it has not been tried at 0.1B |
| Precision | bf16 compute; scans and FFTs in float32 | |
| Architecture search | during a pilot run (about the first 5B tokens), then frozen | a search round runs one forward pass per (layer, circuit type) on each held-out batch |
| Retrieval index | re-encoded every few thousand steps | the model's own encoder keeps changing |
| Swarms | 4 agents per board, many swarms per batch, half split / half random | see [swarms.md](swarms.md) |
| Final decay phase | DATA.md's annealing mix (the last 5%) | as in SmolLM2 and OLMo 2 |

**Compute.** 6 × 0.1B parameters × 100B tokens ≈ 6 × 10¹⁹ FLOPs. At about 300 TFLOPs sustained per MI300X that is
about 56 GPU-hours. The default ExeaLM (two attention layers per TONN layer) trains at the same speed as a same-size
transformer on our hardware, and retrieval adds about half again, so plan for about 85 GPU-hours with retrieval on
(results/README.md, section 5).

What exists: `exealm/train/pretrain.py` runs this loop with the search, retrieval, swarms, WSD, evaluation and
checkpoints. A run can stop and pick up again exactly where it left off (`--save_every`, `--stop_at`,
`--resume auto`), which is what lets a long run span several Colab sessions or preemptible GPU jobs.

The full mix: `--data mix.json` names the sources, where each one lives (one packed directory or many) and what
share of the training tokens it gets. The form is in `exealm/data/loader.py` (`open_data`). Token files are
memory-mapped when first used, so nothing is loaded up front. With a mix, name the retrieval library with
`--datastore_data`.

Several GPUs: start the same command with `torchrun --nproc_per_node N -m exealm.train.pretrain ...` (DDP, one process
per GPU; on a CPU, gloo). `--batch` stays the total batch: every process draws the same batch and trains on its own
rows, whole swarms only, so the data is exactly what one process would see. Process 0 runs the search, evaluates, logs
and saves; after each search round the other processes copy its circuits, weights and optimizer state, and after each
datastore refresh its keys. Two CPU processes match one process on the same total batch to within 0.0003 bits per
token after 300 `tiny` steps (two seeds differ by 0.033), and a two-process run resumes exactly. `--accum K` runs K
forward passes of `--batch` rows before each optimizer step (gradients are averaged across processes on the last one
only), so a step is batch × seq × accum tokens however many GPUs there are.

Job time limits: `--save_minutes` writes checkpoints on the clock and `--max_minutes` saves and stops before a limit,
keeping the schedule. [experiments/expanse](../experiments/expanse/README.md) has the Slurm job that chains such
sessions on SDSC Expanse, where a node has four H100s.

What's missing:
- models that do not fit one GPU (FSDP), and more than one node
- the datastore search sharded across GPUs (with DDP every process holds the whole datastore)
- context extension
- `torch.compile` is behind `--compile 1` (after the search has frozen the architecture) and unmeasured on a GPU

## Post-training

1. **SFT** (`exealm/train/sft.py`): conversations in the chat format, loss on the assistant's tokens only. The loss,
   the loader and the loop are written (`python -m exealm.train.sft`). Every conversation gets its own row, padded on
   the right, and conversations of similar length share a batch. The data sources are not downloaded or formatted yet.
2. **DPO** (`exealm/train/dpo.py`): preferred versus rejected answers. The loop is written (`python -m exealm.train.dpo`, tested on `tiny`); the preference data is not chosen yet.
3. **GRPO** (`exealm/train/grpo.py`): RL with verifiable rewards (math answers, code tests, format). The loop is written
   (`python -m exealm.train.grpo`, tested on `tiny`) with math, exact-match and code rewards; the code reward runs
   model-written code and is refused without `--allow_code_exec`.
4. **Swarm SFT** (`python -m exealm.train.sft --swarm N --answer_agent I`): train on the designated agent's answers
   through the shared board. The loop is tested on `tiny`. See the [swarm SFT data format](swarms.md#swarm-sft).
5. **Swarm RL:** whole-swarm attempts with a shared reward are planned ([swarms.md](swarms.md)).

The bet: a 0.1B model alone won't be a strong general assistant, but a swarm of them, each looking things up through
its own retrieval and sharing what it finds on the board, should be able to take on problems far larger than any one
copy could. Post-training is where that has to be shown: the headline eval puts a swarm of 0.1B agents against a
single larger model on the same tasks at the same inference compute.

## Evaluation during training

Validation bits per token and per byte (`--val`). See `exealm/eval/__init__.py` for the benchmark plan.
Benchmarks on every checkpoint, beside the run: `python -m exealm.eval.auto --run <out folder> --watch 300`
([evaluation.md](evaluation.md)).
