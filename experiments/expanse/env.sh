# Sourced by every job script: the exealm conda environment, whatever shell submitted the job. A job takes the
# Python of the shell that ran sbatch; from a shell without the environment it got the system's Python 3.6 and every
# run failed at its first import (Expanse job 56081004).
if ! python -c "import torch, exealm" > /dev/null 2>&1; then
  set +u
  module load anaconda3/2021.05/q4munrg > /dev/null 2>&1
  eval "$(conda shell.bash hook)" && conda activate "${EXEALM_ENV:-exealm}"
  set -u
fi
python -c "import torch, exealm" > /dev/null 2>&1 || { echo "no Python with torch and exealm: activate the exealm environment"; exit 1; }
