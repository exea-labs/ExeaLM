# Muon against AdamW on the 13M hybrid (#21, 20 and 21 November 2026)

One RTX 4060 Laptop GPU (8 GB), PyTorch 2.14.1 with CUDA, Windows 11. Runs took 22 to 37 minutes (11 to 12 for the batch-16 check). Other jobs
shared the CPU during some runs, so the Seconds column is not a speed comparison.

How to read the numbers. Bits per byte on held-out English, Japanese Wikipedia, Japanese web text and code; "Mean" is
the mean of those four; lower is better. Both optimizers train the same model, the 13M hybrid from results/funding
(`--models tonn_core` with `HYBRID_13M` from `experiments/gpu_session/make_jobs.py`), with the same data, batch
(32 x 512 tokens), schedule and steps (3,000, about 49M tokens), in 32-bit precision, weight decay 0.1. A seed fixes
the starting weights and the starting circuits, so the same seed gives both optimizers the same model to start from.

The data is not the data behind results/funding, which is private. These runs use public stand-ins built with
`experiments/leave_one_out_data.py --fetch` (FineWeb-Edu, Japanese Wikipedia, FineWeb-2 Japanese, codeparrot-clean
Python) and the same 32k tokenizer. Code here is Python only, so its bits per byte are not comparable. Compare the
two optimizers with each other, not with the numbers in results/funding.

The batch of 32 does not fit an 8 GB card at this size (it needs about 9 GB), so every run here uses `--accum 2`:
each batch runs as two halves of 16. Each step sees the same tokens. The gradient is the same except through TONN's
running slot scale, which updates once per half: on the first step the second half is divided by the first half's
measured size instead of 1, and after that the running average moves twice per step. Both optimizers ran the same way. How much the split moves a score on its own is measured in
"Does --accum change the result?" below.

## Results

Seed 0 at learning rates 1e-3, 2e-3 and 4e-3 for each optimizer. Both were best at 4e-3, the top of that range, so
each got one more run at 8e-3, which was worse for both. Then seed 1 at each optimizer's best rate.

| Optimizer | Learning rate | Seed 0 | Seed 1 | Mean | Seconds |
|---|---|---|---|---|---|
| adamw | 0.001 | 1.3721 |  | 1.3721 | 1360 |
| adamw | 0.002 | 1.3390 |  | 1.3390 | 1361 |
| adamw | 0.004 | 1.3209 | 1.3307 | 1.3258 | 1331 |
| adamw | 0.008 | 1.3377 |  | 1.3377 | 2220 |
| muon | 0.001 | 1.3448 |  | 1.3448 | 1498 |
| muon | 0.002 | 1.3102 |  | 1.3102 | 1497 |
| muon | 0.004 | 1.3041 | 1.3006 | 1.3024 | 1595 |
| muon | 0.008 | 1.3075 |  | 1.3075 | 1378 |

Best learning rate on seed 0: AdamW 0.004, Muon 0.004.
- Muon against AdamW at those rates: ahead by +0.0234 bits per byte (std 0.0094), better on 2 of 2 seeds; per seed 0: +0.0167, 1: +0.0301

Seed 0 also picked the rates, so seed 1 is the cleaner check. For scale, the same model's 10 seeds in
results/funding (section 1, "no rotary attention in the starting menu") have a standard deviation of 0.007 bits per
byte.

At each one's best rate, Muon had lower bits per byte than AdamW on both seeds, by 0.017 on seed 0 and 0.030 on
seed 1, and it was lower on all four kinds of text in both. Both gaps are bigger than the seed-to-seed spread
above (2.3 and 4.2 times it). That spread comes from other data at lr 2e-3, and a difference between two runs
varies more than one run does, so it is only a rough yardstick. It is still two seeds at one size (13M), on
stand-in data and with `--accum 2`, so it does not show what happens at 0.1B or on the real packs.

AdamW was best at 4e-3 here. results/funding trains this model with AdamW at 2e-3, which was 0.018 worse than 4e-3
on seed 0 with this data.

## Does --accum change the result?

A batch of 16 fits the card whole, so AdamW at 4e-3 also ran with batch 16 three ways: one batch of 16 on seeds 0
and 1, and two halves of 8 (`--accum 2`) on seed 0. To see whether Muon's lead depends on the split, Muon at 4e-3
ran with one batch of 16 on seed 0 too. Everything else is as above. These runs see half the tokens (about 25M), so
their scores are higher than the table's. The numbers are the `bits_per_byte` field of each file in `accum_check/`.

| Run | English | Japanese Wikipedia | Japanese web | Code | Mean | Seconds |
|---|---|---|---|---|---|---|
| AdamW, one batch, seed 0 | 1.5522 | 1.3221 | 1.5537 | 1.3440 | 1.4430 | 652 |
| AdamW, two halves, seed 0 | 1.5666 | 1.3324 | 1.5757 | 1.3707 | 1.4613 | 712 |
| AdamW, one batch, seed 1 | 1.5563 | 1.3285 | 1.5722 | 1.3678 | 1.4562 | 661 |
| Muon, one batch, seed 0 | 1.5158 | 1.2945 | 1.5250 | 1.2962 | 1.4079 | 725 |

On seed 0, two halves scored 0.018 worse than one batch, and worse on all four kinds of text. The two runs start
from the same weights and see the same batches, and the split run's training loss was higher at all 12 points of
its curve. A different seed moved the score by 0.013, so the split moved it a little more than a new seed did. With
one run each I can't say how much of the 0.018 comes from the split itself. The part of the code it could come from
is the running slot scale (`_unit_scale` in `exealm/model/tonn.py`): it updates on every forward pass in training,
so with `--accum 2` its running average moves twice per step. I haven't tested whether updating it once per step
closes the gap.

Without the split, Muon was still ahead: 0.035 lower than AdamW on seed 0 at batch 16, and lower on all four kinds of
text. So Muon's lead in the table above does not depend on `--accum 2`. That is one seed at a smaller batch, so it
does not say whether the lead is bigger or smaller with one batch of 32.

## Commands

```
python experiments/leave_one_out_data.py --fetch --raw data/raw --out data/tokpack/bpe32k
python experiments/tokenizer_lm.py --root data/tokpack --toks bpe32k --models tonn_core \
  --cfg '{"attn_layers": [1, 2, 4, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25, "RP": 0}, "attn_gate": true, "qk_norm": true}' \
  --optimizer adamw --lr 4e-3 --seeds 0 --accum 2 --tag m13_adamw_lr4e-3_s0 --out results/muon
python experiments/muon_table.py results/muon
```

The second command once per row of the table, with that row's `--optimizer`, `--lr` and `--seeds`. Each result
file's `args` field has the exact arguments (these runs read the same packs from another folder, so
`root` there is different).

The batch-16 check:

```
python experiments/tokenizer_lm.py --root data/tokpack --toks bpe32k --models tonn_core \
  --cfg '{"attn_layers": [1, 2, 4, 5], "init_weights": {"C": 4, "G": 0.25, "D": 0.25, "RP": 0}, "attn_gate": true, "qk_norm": true}' \
  --optimizer adamw --lr 4e-3 --batch 16 --accum 1 --seeds 0 --tag acc_b16_a1_s0 --out results/muon/accum_check
```

The other three runs change one thing each: `--accum 2` (tag `acc_b16_a2_s0`), `--seeds 1` (`acc_b16_a1_s1`) or
`--optimizer muon` (`acc_b16_muon_a1_s0`).
