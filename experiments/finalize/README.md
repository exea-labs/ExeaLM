# Picking the ExeaLM of the scaling study (MI300X, outside the allocation)

Runs at 0.1B that decide the architecture and training settings before the long runs on Expanse
([docs/budget.md](../../docs/budget.md), "Which ExeaLM"). Chosen on accuracy, not speed. Every run uses the recipe of
results/funding section 3: 3,000 steps of 32 x 512 tokens (49M), learning rate 1e-3, the search off, bf16.

    python experiments/finalize/make_jobs.py                                  # stage 1
    python experiments/gpu_session/runner.py experiments/finalize/jobs/stage1_train.txt 7
    python experiments/gpu_session/runner.py experiments/finalize/jobs/stage1_eval.txt 4
    python experiments/finalize/summarize.py 1

Stage 1, the board on text (7 models x 3 seeds): `gated` (the baseline), `core` (no board or library, every row
alone), `plan` (the plan's ExeaLM), and four with the board only: `board_random` (unrelated texts in every swarm),
`time_mixed`, `doc_mixed` and `doc_split` (`board_order="doc"`: an agent hears every post of the agents reading the
text before its window, from its first token, and nothing from the text after it).

Stage 2 (`--stage 2 --base <the stage-1 pick>`): post every 16 or 32 tokens, learned post pooling, two board slots,
swarms of 2 or 8, and Muon against AdamW at three learning rates for the pick and the Llama. Stage 3
(`--stage 3 --base ... --change ... --opt ... --lr ...`): the final pick, the Llama and `core` at 12,000 steps, 2 seeds.

Reading stage 1: table 1 (every row alone) is the scaling study's number. A board model is worth training only if it is
ahead of `gated` on every seed and no more than 0.01 bits per byte behind `core`. Table 2 is what the board gives when
it is used: four windows of one English text as a swarm, against the same windows alone.
