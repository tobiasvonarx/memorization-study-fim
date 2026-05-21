#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/fineweb/tokenize_llama_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/fineweb/tokenize_llama_%A.out
#SBATCH --job-name=fineweb_tokenize_llama
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --partition=normal

# Tokenize FineWeb text JSONL into Megatron MMAP format using the LLaMA 3.2 tokenizer.
#
# Two-step pipeline:
#   Step 1 (run separately): create_fineweb_text_jsonl.py  (parquet -> {"text": ...} JSONL)
#   Step 2 (this script):    Megatron preprocess_data.py   (JSONL -> .bin/.idx MMAP)
#
# Usage:
#   sbatch tokenize_fineweb_llama.slurm
#
# Override defaults via environment:
#   MEGATRON_DIR=/path/to/Megatron-LM \
#   LLAMA_TOKENIZER_PATH=/path/to/tokenizer \
#   sbatch tokenize_fineweb_llama.slurm

set -euo pipefail
trap 'echo "ERR: script failed at line $LINENO with exit code $?" | tee -a /dev/stderr' ERR

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

# --- Configurable paths ---
MEGATRON_DIR="${MEGATRON_DIR:-${SCRATCH_ROOT}/megatron-lm}"
LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"

JSONL_DIR="${DATASET_ROOT}/fineweb/fineweb_100BT_text_jsonl"
OUTPUT_PREFIX="${DATASET_ROOT}/fineweb/fineweb_100BT_llama_raw/combined_text_document"

WORKERS="${WORKERS:-256}"
PARTITIONS="${PARTITIONS:-16}"

# --- Validation ---
if [ ! -d "$MEGATRON_DIR" ]; then
  echo "Megatron-LM not found: $MEGATRON_DIR"
  echo "Clone it: git clone https://github.com/NVIDIA/Megatron-LM.git $MEGATRON_DIR"
  exit 1
fi

if [ ! -d "$LLAMA_TOKENIZER_PATH" ]; then
  echo "LLaMA tokenizer not found: $LLAMA_TOKENIZER_PATH"
  echo "Download it: python utils/download_tokenizer.py --model meta-llama/Llama-3.2-3B --output $LLAMA_TOKENIZER_PATH"
  exit 1
fi

JSONL_COUNT=$(find "$JSONL_DIR" -name "*.jsonl" -type f | wc -l)
if [ "$JSONL_COUNT" -eq 0 ]; then
  echo "No JSONL files found in: $JSONL_DIR"
  echo "Run create_fineweb_text_jsonl.py first."
  exit 1
fi
echo "Found $JSONL_COUNT JSONL files in $JSONL_DIR"

# --- Create output directory ---
OUTPUT_DIR="$(dirname "$OUTPUT_PREFIX")"
mkdir -p "$OUTPUT_DIR"

# --- Container image ---
TOKENIZE_IMAGE_PATH="${TOKENIZE_IMAGE_PATH:-${PRETRAIN_IMAGE_PATH:-$IMAGE_PATH}}"
if [ ! -f "$TOKENIZE_IMAGE_PATH" ]; then
  echo "Container image not found: $TOKENIZE_IMAGE_PATH"
  exit 1
fi

echo "=========================================="
echo "FineWeb LLaMA Tokenization (JSONL -> MMAP)"
echo "=========================================="
echo "Megatron-LM:     $MEGATRON_DIR"
echo "Tokenizer:       $LLAMA_TOKENIZER_PATH"
echo "Input JSONL dir: $JSONL_DIR"
echo "Output prefix:   $OUTPUT_PREFIX"
echo "Workers:         $WORKERS"
echo "Partitions:      $PARTITIONS"
echo "Container image: $TOKENIZE_IMAGE_PATH"
echo "=========================================="

# Run inside the container so that transformer_engine (pulled in by megatron.core)
# can find libcudnn and other CUDA libraries at import time.
# Megatron's preprocess_data.py handles:
#   - Tokenization with HuggingFace tokenizer
#   - EOD token appending (--append-eod)
#   - Partitioned processing (--partitions > 1)
#   - Automatic merging of partitions into single .bin/.idx
srun \
  --container-image="$TOKENIZE_IMAGE_PATH" \
  --no-container-mount-home \
  --container-mounts="$CONTAINER_MOUNTS" \
  --ntasks=1 \
  --cpu-bind=none \
  bash -c "
    export PYTHONPATH=${MEGATRON_DIR}:\${PYTHONPATH:-}
    python ${MEGATRON_DIR}/tools/preprocess_data.py \
      --input \"${JSONL_DIR}/*.jsonl\" \
      --output-prefix ${OUTPUT_PREFIX} \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model ${LLAMA_TOKENIZER_PATH} \
      --json-keys text \
      --append-eod \
      --workers ${WORKERS} \
      --partitions ${PARTITIONS}
  "

echo ""
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}"* 2>/dev/null || echo "(no output files found — check for errors above)"

echo "END TIME: $(date)"
