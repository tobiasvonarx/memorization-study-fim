#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/fineweb/build_no_fim_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/fineweb/build_no_fim_%A.out
#SBATCH --job-name=fineweb_build_no_fim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=debug

# Build a filtered pure-LTR variant from the already-tokenized LLaMA FineWeb
# MMAP dataset. Reads from fineweb_100BT_llama_raw (unfiltered), applies the
# same min_doc_length=50 filter used by the FIM job, and writes clean LTR docs
# (content + EOD, no FIM sentinels) to fineweb_100BT_llama_no_fim.
#
# This ensures fineweb_100BT_llama_no_fim and fineweb_100BT_llama_fim contain
# exactly the same set of documents (same filter, same 13,836 dropped), making
# the two datasets directly comparable for FIM-vs-LTR pretraining experiments.
#
# Usage:
#   sbatch src/preprocess_fineweb/build_no_fim_from_llama_bin.slurm

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
OUTPUT_PREFIX="${DATASET_ROOT}/fineweb/fineweb_100BT_llama_no_fim/combined_text_document"
LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"

MIN_DOC_LENGTH="${MIN_DOC_LENGTH:-50}"
CONTENT_LENGTH="${CONTENT_LENGTH:-16383}"

if [ ! -f "${INPUT_PREFIX}.bin" ] || [ ! -f "${INPUT_PREFIX}.idx" ]; then
  echo "Input MMAP not found: ${INPUT_PREFIX}.{bin,idx}"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

NO_FIM_IMAGE_PATH="${NO_FIM_IMAGE_PATH:-${PRETRAIN_IMAGE_PATH:-$IMAGE_PATH}}"
if [ ! -f "$NO_FIM_IMAGE_PATH" ]; then
  echo "Container image not found: $NO_FIM_IMAGE_PATH"
  exit 1
fi

echo "=========================================="
echo "FineWeb LLaMA no-FIM (filtered LTR) build"
echo "=========================================="
echo "Input prefix:    $INPUT_PREFIX"
echo "Output prefix:   $OUTPUT_PREFIX"
echo "Mode:            ltr_clean"
echo "Min doc length:  $MIN_DOC_LENGTH"
echo "Content length:  $CONTENT_LENGTH (final length <= 16384 with EOD)"
echo "Container image: $NO_FIM_IMAGE_PATH"
echo "=========================================="

srun \
  --container-image="$NO_FIM_IMAGE_PATH" \
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
      --mode ltr_clean \
      --min-doc-length ${MIN_DOC_LENGTH} \
      --seq-length ${CONTENT_LENGTH}
  "

echo ""
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}"* 2>/dev/null || echo "(no output files found — check for errors above)"

echo "END TIME: $(date)"
