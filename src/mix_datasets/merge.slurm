#!/bin/bash
#SBATCH --job-name=mix_reduce
#SBATCH --partition=normal
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=infra01
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/mix_datasets/reduce_mix_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/mix_datasets/reduce_mix_%A.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(cd "$SCRIPT_DIR/../.." && pwd)/config.env}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing config: $CONFIG_FILE"
  exit 1
fi
set -a
source "$CONFIG_FILE"
set +a

mkdir -p "$LOG_ROOT/mix_datasets"

source "$CONDA_PROFILE"
conda activate base

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MIX_MODE="${MIX_MODE:-no_fim}"

echo "Running mix reducer (MIX_MODE=$MIX_MODE)"

srun python "$REPO_ROOT/src/mix_datasets/mix_datasets_reduce.py"
