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

echo "== 1/4 $(date '+%T') Python: $(python --version 2>&1) at $(command -v python)"
python - <<'PY'
import sys
assert sys.version_info >= (3, 10), "Python 3.10 or newer"
PY
echo "== 2/4 $(date '+%T') PyTorch with CUDA: 3-4 GB to download, often 5-15 minutes on a busy login node"
# The default Linux wheel carries CUDA: 3-4 GB with NVIDIA's libraries, so this shows its progress. --no-cache-dir: home
# is 100 GB and pip would otherwise keep a second copy of every wheel in ~/.cache/pip.
if python -c "import torch" 2>/dev/null; then echo "   already installed: torch $(python -c 'import torch; print(torch.__version__)')"
else pip install --no-cache-dir --progress-bar on torch; fi
echo "== 3/4 $(date '+%T') the ExeaLM package and its tools (tokenizers, datasets, pytest, huggingface_hub, tqdm)"
pip install --no-cache-dir --progress-bar on -e ".[dev,data]" huggingface_hub tqdm
echo "== 4/4 $(date '+%T') checking"
python - <<'PY'
import torch
assert torch.version.cuda, f"torch {torch.__version__} is a CPU-only build: install a CUDA build before submitting jobs"
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
PY

# The data comes next: bash experiments/expanse/go.sh data (docs/runbook.md)
echo "setup done; next: bash experiments/expanse/go.sh data"
