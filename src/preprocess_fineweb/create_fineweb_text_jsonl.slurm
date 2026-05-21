#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/fineweb/text_jsonl_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/fineweb/text_jsonl_%A.out
#SBATCH --job-name=fineweb_text_jsonl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=normal

# Step 1: Convert FineWeb parquet shards to text-only JSONL.
# Step 2: Run tokenize_fineweb_llama.slurm after this completes.
#
# Usage:
#   sbatch src/preprocess_fineweb/create_fineweb_text_jsonl.slurm

set -euo pipefail

echo "START TIME: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(cd "$SCRIPT_DIR/../.." && pwd)/config.env}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing config: $CONFIG_FILE"
  exit 1
fi
set -a
source "$CONFIG_FILE"
set +a

mkdir -p "$LOG_ROOT/fineweb"

source "$CONDA_PROFILE"
conda activate base

export PYTHONPATH="$REPO_ROOT/src:$DATATROVE_SRC_DIR/src:${PYTHONPATH:-}"

INPUT_DIR="${FINEWEB_DOWNLOAD_DIR}/sample/100BT"
OUTPUT_DIR="${DATASET_ROOT}/fineweb/fineweb_100BT_text_jsonl"
MIN_CHARS="${MIN_CHARS:-200}"
NUM_WORKERS="${NUM_WORKERS:-64}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "FineWeb parquet dir not found: $INPUT_DIR"
  echo "Download first: huggingface-cli download HuggingFaceFW/fineweb --repo-type dataset --local-dir $FINEWEB_DOWNLOAD_DIR --include 'sample/100BT/*'"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "FineWeb Parquet -> Text JSONL"
echo "=========================================="
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Min chars:  $MIN_CHARS"
echo "Workers:    $NUM_WORKERS"
echo "=========================================="

python "$REPO_ROOT/src/preprocess_fineweb/create_fineweb_text_jsonl.py" \
  --dataset "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --min-chars "$MIN_CHARS" \
  --num-workers "$NUM_WORKERS"

echo ""
echo "Output files:"
ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l
echo "jsonl files created"

echo "END TIME: $(date)"
