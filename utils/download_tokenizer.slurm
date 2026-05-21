#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=8
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/download_tokenizer/%j.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/download_tokenizer/%j.out
#SBATCH --gpus-per-node=0
#SBATCH --job-name=download_tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(cd "$SCRIPT_DIR/.." && pwd)/config.env}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing config: $CONFIG_FILE"
  exit 1
fi
set -a
source "$CONFIG_FILE"
set +a

mkdir -p "$LOG_ROOT/download_tokenizer"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Container image not found: $IMAGE_PATH"
  exit 1
fi

CONTAINER_OPTS=(
  --container-image="$IMAGE_PATH"
  --no-container-mount-home
  --container-mounts="$CONTAINER_MOUNTS"
  --container-workdir="$REPO_ROOT"
)

TOKENIZER_ID="${TOKENIZER_ID:-meta-llama/Llama-3.2-3B}"
TOKENIZER_OUTPUT="${TOKENIZER_OUTPUT:-$HF_TOKENIZER_DIR}"

echo "Starting tokenizer download..."
echo "Tokenizer ID: $TOKENIZER_ID"
echo "Output: $TOKENIZER_OUTPUT"
srun "${CONTAINER_OPTS[@]}" python3 utils/download_tokenizer.py \
  --tokenizer-id "$TOKENIZER_ID" \
  --output "$TOKENIZER_OUTPUT"
