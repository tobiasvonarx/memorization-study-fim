#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/fineweb/build_fim_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/fineweb/build_fim_%A.out
#SBATCH --job-name=fineweb_build_fim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=debug

# Build a 50% FIM / 50% LTR variant from the already-tokenized LLaMA FineWeb
# MMAP dataset. Reads the per-document sequences from
#   ${DATASET_ROOT}/fineweb/fineweb_100BT_llama_raw/combined_text_document.{bin,idx}
# strips BOS/EOD, applies SPM-format FIM with random splits on 50% of docs,
# and writes
#   ${DATASET_ROOT}/fineweb/fineweb_100BT_llama_fim/combined_text_document.{bin,idx}
#
# FIM sentinel mapping (LLaMA 3.2 reserved tokens):
#   FIM_PREFIX = 128002  (<|reserved_special_token_0|>)
#   FIM_MIDDLE = 128003  (<|reserved_special_token_1|>)
#   FIM_SUFFIX = 128005  (<|reserved_special_token_2|>)
#   EOD        = 128001  (<|end_of_text|>)
#
# Usage:
#   sbatch src/preprocess_fineweb/build_fim_from_llama_bin.slurm

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

INPUT_PREFIX="${DATASET_ROOT}/fineweb/fineweb_100BT_llama_raw/combined_text_document"
OUTPUT_PREFIX="${DATASET_ROOT}/fineweb/fineweb_100BT_llama_fim/combined_text_document"
LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"

MODE="${MODE:-fim_hybrid}"
SWAP_PERCENT="${SWAP_PERCENT:-50}"
SPLIT_SEED="${SPLIT_SEED:-42}"
MIN_DOC_LENGTH="${MIN_DOC_LENGTH:-50}"
CONTENT_LENGTH="${CONTENT_LENGTH:-16380}"

if [ ! -f "${INPUT_PREFIX}.bin" ] || [ ! -f "${INPUT_PREFIX}.idx" ]; then
  echo "Input MMAP not found: ${INPUT_PREFIX}.{bin,idx}"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

FIM_IMAGE_PATH="${FIM_IMAGE_PATH:-${PRETRAIN_IMAGE_PATH:-$IMAGE_PATH}}"
if [ ! -f "$FIM_IMAGE_PATH" ]; then
  echo "Container image not found: $FIM_IMAGE_PATH"
  exit 1
fi

echo "=========================================="
echo "FineWeb LLaMA FIM construction (bin -> bin)"
echo "=========================================="
echo "Input prefix:    $INPUT_PREFIX"
echo "Output prefix:   $OUTPUT_PREFIX"
echo "Mode:            $MODE"
echo "Swap percent:    $SWAP_PERCENT"
echo "Split seed:      $SPLIT_SEED"
echo "Min doc length:  $MIN_DOC_LENGTH"
echo "Content length:  $CONTENT_LENGTH (final length <= 16384 with FIM sentinels + EOD)"
echo "Tokenizer:       $LLAMA_TOKENIZER_PATH (only used for EOS sanity check)"
echo "Container image: $FIM_IMAGE_PATH"
echo "=========================================="

srun \
  --container-image="$FIM_IMAGE_PATH" \
  --no-container-mount-home \
  --container-mounts="$CONTAINER_MOUNTS" \
  --ntasks=1 \
  --cpu-bind=none \
  bash -c "
    cd ${REPO_ROOT}/src
    python prepare_fim.py \
      --bin-prefix ${INPUT_PREFIX} \
      --output ${OUTPUT_PREFIX} \
      --tokenizer ${LLAMA_TOKENIZER_PATH} \
      --mode ${MODE} \
      --swap-percent ${SWAP_PERCENT} \
      --min-doc-length ${MIN_DOC_LENGTH} \
      --seq-length ${CONTENT_LENGTH} \
      --deterministic-splits \
      --split-seed ${SPLIT_SEED}
  "

echo ""
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}"* 2>/dev/null || echo "(no output files found — check for errors above)"

echo "END TIME: $(date)"
