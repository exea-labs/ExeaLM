#!/bin/bash
# One-time setup on an Expanse login node. Run it from the repo root, inside the Python environment the jobs will use
# (a conda environment in your home directory is what SDSC suggests):
#
#     conda create -n exealm python=3.11 && conda activate exealm
#     huggingface-cli login            # or export HF_TOKEN: a token that can read exea-labs/exealm-studio-data
#     bash experiments/expanse/setup.sh
#
# Login nodes have no GPU, so nothing here touches one: the GPU checks run in first_contact.sb.
set -euo pipefail

python - <<'PY'
import sys
assert sys.version_info >= (3, 10), "Python 3.10 or newer"
PY
python -c "import torch" 2>/dev/null || pip install -q torch          # the default Linux wheel carries CUDA
pip install -q -e ".[dev,data]" huggingface_hub
python - <<'PY'
import torch
assert torch.version.cuda, f"torch {torch.__version__} is a CPU-only build: install a CUDA build before submitting jobs"
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
PY

# The data comes next: bash experiments/expanse/go.sh data (docs/runbook.md)
echo "setup done; next: bash experiments/expanse/go.sh data"
