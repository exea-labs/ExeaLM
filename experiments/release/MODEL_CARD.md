---
license: apache-2.0
language:
- en
- ja
tags:
- exealm
- tonn
- research
- code
- nairr
---

# ExeaLM: models from the NAIRR Pilot scaling study

Exea Labs trains ExeaLM, a language model for research and code, on H100 hours from the National AI Research Resource
(NAIRR) Pilot at the San Diego Supercomputer Center (Expanse). This repository holds the study's finished models:
ExeaLM and the baselines it is compared with, trained on the same data, the same tokens and the same recipe, so the
comparison is between designs. They are small research models (0.1B to 0.5B parameters so far, 0.4B to 10B training
tokens; the 3B pair is training), base language models that continue text. They are not chat assistants and not tuned for safety.

Code, data recipe, every result and its caveats: [github.com/exea-labs/ExeaLM](https://github.com/exea-labs/ExeaLM/tree/nairr)
(`results/expanse/README.md`). Training data: [exea-labs/exealm-pretrain](https://huggingface.co/datasets/exea-labs/exealm-pretrain).
Research notes: [blog.exealabs.org](https://blog.exealabs.org).

## The designs

| Folder prefix | Model |
|---|---|
| `exealm_` | ExeaLM: attention layers with a gate, QK-norm, value residual and XSA, and one TONN layer (searched signal-processing circuits) at the middle layer; reads streams of four windows in the last 20% of training |
| `exealm_nostream_` | the same, trained without the stream phase |
| `exealm_mix20_`, `exealm_mix40_` | the stream phase mixing streams with windows alone, over the last 20% or 40% (the recipe from 26 November on) |
| `distill_` | 30% of the pretraining tokens written by Qwen3.5-9B-Base (a test: no gain at 0.1B) |
| `strongllama_` | the same attention everywhere and no TONN layer: what the TONN layer adds |
| `qwen_` | our build of the Qwen3.5 design: Gated DeltaNet in 3 of every 4 layers, gated attention with QK-norm in the 4th |
| `llama_` | a plain Llama |
| `libpost_`, `ragpost_`, `boardpost_` | post-training tests: a distilled library with retrieval, and agents sharing a board |

Names end in the size (`100m`, `250m`), the tokens per parameter (`4.1x`, `111x`) and the seed (`s0` to `s2`).

## Results so far (bits per byte on 18 held-out sets, lower is better, 3 seeds)

| | ExeaLM | Qwen3.5-style | Plain Llama |
|---|---|---|---|
| 0.1B, windows read alone | **1.157** | 1.190 | 1.200 |
| 0.25B, windows read alone | **1.014** | 1.026 | 1.041 |
| 0.1B, 4,096-token documents read as a stream | **1.096** | 1.157 | 1.584 |
| 0.25B, 4,096-token documents read as a stream | **0.955** | 0.997 | 1.185 |

ExeaLM is ahead of both baselines on all 54 set-and-seed pairs at each size. With a library written by a larger model
and a keyword search, the 0.1B ExeaLM's bits per byte on held-out research questions fall from 1.645 to 0.590. The
GitHub results page has the controls, the failures and what these numbers do not show (among them: the strong-attention
baseline, which matches ExeaLM closely on windows read alone).

## Try the 0.1B model with its library

`ragpost_exealm_100m_s0` is the 0.1B ExeaLM trained to read a library entry before a question. A keyword search (BM25)
picks the entry; the model reads it and answers. With the GitHub code:

```bash
git clone -b nairr https://github.com/exea-labs/ExeaLM && cd ExeaLM
pip install torch numpy safetensors tokenizers huggingface_hub
python experiments/library/ask.py                 # random held-out questions, answered with and without the library
python experiments/library/ask.py --question "What does gradient descent move the parameters along?"
python experiments/library/ask.py --interactive
python experiments/library/ask.py --score 0       # all 6,253 held-out questions, in bits per byte
```

It downloads this model (about 400 MB) and the library,
[exea-labs/exealm-library](https://huggingface.co/datasets/exea-labs/exealm-library) (18,545 entries on research, math
and code topics written by Qwen3-30B-A3B, about 100 MB). The held-out questions are on topics the model never trained
on; their entries are in the library. On all of them it scores 0.588 bits per byte with the library and 1.635 without.
The search picks the question's own entry for 86% of them. It is a 95M-parameter base model: answers are short, often
close rather than exact, and it can only use what the library holds. Every answer shows which entry it read.

## Loading

```python
import json
from safetensors.torch import load_file
from exealm.config import ModelConfig
from exealm.model import ExeaLM, Llama

c = json.load(open("exealm_250m_4.1x_s0/config.json"))
cfg = ModelConfig.from_dict(c["cfg"])
model = ExeaLM(cfg, c["specs"]) if c["kind"] == "tonn" else Llama(cfg)
model.load_state_dict(load_file("exealm_250m_4.1x_s0/model.safetensors"))
```

The tokenizer is in `tokenizer/`. `python -m exealm.chat --ckpt <model.pt> --raw` in the GitHub repo continues text.

## License

Apache 2.0, as the code.
