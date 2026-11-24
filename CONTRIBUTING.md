# Contributing to ExeaLM

You do not need to have trained a model before.

## Getting started

1. Sign up so we can list you as a contributor: https://docs.google.com/forms/d/e/1FAIpQLSdPHVZhuXxBDUf-cJM2zDbj1JZPsduu95Y_kTJ_g4gKAXdDVw/viewform
2. Read the [README](README.md) (the plan) and the [Code of Conduct](CODE_OF_CONDUCT.md).
3. Set up and run the tests (any laptop works, no GPU needed):
   ```bash
   git clone https://github.com/exea-labs/ExeaLM && cd ExeaLM
   python -m venv .venv && source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -e ".[dev,data]"
   pytest
   ```
4. Pick an item from the README's [to-do list](README.md#to-do-list) or an idea from [docs/ideas.md](docs/ideas.md).
   Most run on a laptop or a free Colab GPU. Say which one you're taking in #exealm, or open a draft pull request.
5. Fork the repo, make a branch, make your change, run `pytest`, and open a pull request.

## Where things are

| To work on | Read | Code |
|---|---|---|
| the architecture | [docs/architecture.md](docs/architecture.md) | `exealm/model/` |
| retrieval or agents | [docs/memory.md](docs/memory.md) | `exealm/model/retrieval.py`, `exealm/model/board.py`, `exealm/memory/` |
| data | [docs/data.md](docs/data.md) | `exealm/data/` |
| training and post-training | [docs/training.md](docs/training.md) | `exealm/train/` |
| evaluation | `exealm/eval/__init__.py` | `exealm/eval/` |

Places that need code are marked `TODO(area)` in the source: `grep -rn "TODO(" exealm/`.

## Experiments: the fair-comparison rule

A change to the model or the training (a new circuit, an optimizer, a data filter) has to beat the current default in
a fair comparison before it goes in:
- same data and number of steps
- about the same parameter count
- the same evaluation windows
- at least two seeds if the difference is small: TONN's seed-to-seed spread on small runs is about 0.03 bits

`python -m exealm.train.pretrain ... --model llama` gives the transformer baseline with the same settings. Report
results with the "Experiment result" issue template: config, seeds, both numbers, command lines. Post negative results
too.

## Pull requests

- Keep each pull request to one change.
- Say what changed and why. For experiments, include the config, the baseline, and the result.
- Add or update a test when you change behavior (`tests/`). CI runs `pytest` on every pull request.
- Do not commit large data files, checkpoints, or anything under a restricted license. Link to the source.
- Never commit secrets, API keys, or personal information.
- A maintainer reviews each pull request. Keep review comments kind.

## Data

Every dataset or filter needs a source link, a license, and a note on what we did to it. Add new sources to
`exealm/data/sources.py`. If you cannot state the license, do not add the data.

## Credit

After your first merged contribution, add yourself to [CONTRIBUTORS.md](CONTRIBUTORS.md) in a pull request. List only what you are comfortable making public.

## License

Contributions are licensed under Apache 2.0.
