# Every run of the Expanse study, by name

Runs are kept on Expanse in `runs/` (a link to `/expanse/lustre/projects/fli100/abhatia7/runs`). The finished,
valid ones are public, weights only, at [exea-labs/exealm-nairr](https://huggingface.co/exea-labs/exealm-nairr), in
folders of the same names. A name is `<design>_<size>_<tokens per parameter>_s<seed>`.

## Pretraining

| Name | Model | Notes |
|---|---|---|
| `exealm_<size>_4.1x_s<seed>` | ExeaLM: one TONN layer at the middle layer, attention with a gate, QK-norm, value residual and XSA elsewhere, streams of 4 windows in the last 20% | The study's ExeaLM. At 0.1B these were trained as `exealm_midstream_100m_4.1x_s*` and renamed on 26 November |
| `exealm_nostream_100m_4.1x_s<seed>` | The same ExeaLM trained without the stream phase | Ablation; trained as `exealm_mid_100m_4.1x_s*` |
| `strongllama_<size>_4.1x_s<seed>` | The same model with attention in place of its TONN layer (`experiments/expanse/diag/llama_two.json`), same recipe, streams included | What the TONN layer adds |
| `qwen_<size>_4.1x_s<seed>` | The base Qwen3.5-style model: Gated DeltaNet in 3 of every 4 layers, gated attention with QK-norm in the 4th (`qwen_<size>.json`) | The 3B run uses the flash-linear-attention kernel (`--fla 1`), the curve runs our PyTorch kernel |
| `llama_<size>_4.1x_s<seed>` | A plain Llama | Up to 0.5B |
| `<design>_100m_111x_s<seed>` | The long pair: 0.1B trained to 111 tokens per parameter, as the 9B would be | ExeaLM, Qwen3.5-style and the strong-attention Llama |
| `lr_<design>_<size>_<rate>` | Learning-rate checks (logs only) | Picks in `experiments/expanse/lr_choice.json` |
| `distill_exealm_100m_s<seed>` | ExeaLM 0.1B pretrained with 30% teacher text (Qwen3.5-9B-Base continuations of training openings) | Distillation into the weights (`experiments/distill`) |

Sizes: `100m` (95M parameters for ExeaLM), `250m`, `500m`, `3b`. `4.1x` is the curve's 4.1 tokens per parameter.

## Post-training tests (0.1B and 0.25B)

| Name | What |
|---|---|
| `libpost_rg_100m`, `libpost_exealm_100m`, `libpost_qwen_100m` | The library test (section 7): ExeaLM with a grafted retrieval slot, without it, and the Qwen3.5-style model |
| `ragpost_<model>_100m_s<seed>` | Retrieval-trained post-training: library items with the BM25-picked entry before them (section 11) |
| `boardpost_ag_100m`, `boardpost_solo_100m`, `boardpost_exealm_big`, `boardpost_qwen_big` | Agents sharing a board against single models (section 14) |

## Not used for results

| Folder | Why |
|---|---|
| `runs/_broken_ddp/` | Trained with the multi-GPU bug of section 17 (the first 0.5B curve, the first 0.5B and 3B learning-rate checks, the 3B pair's first steps) |
| `runs/_core_layout/` | The first plan's ExeaLM (a TONN layer every fourth layer) at 0.1B, superseded by the layout of section 5; its scores are in sections 3 and 9 as a comparison |
| `runs/_aborted/`, `runs/_graft/` | A cancelled run; grafted checkpoints that post-training starts from |
