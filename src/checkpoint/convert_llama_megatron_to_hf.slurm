#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=16
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/convert_llama_to_hf/convert_%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/convert_llama_to_hf/convert_%A.out
#SBATCH --gres=gpu:1
#SBATCH --job-name=convert_llama_to_hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --partition=normal

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

mkdir -p "$LOG_ROOT/convert_llama_to_hf"

source "$CONDA_PROFILE"
conda activate base

LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"
CHECKPOINT_RUN_NAME_3B="${CHECKPOINT_RUN_NAME_3B:-${CHECKPOINT_RUN_NAME:-llama3_2_3B-lr-3e-4-minlr-3e-5-gbs-2048-seqlen-16384-tp-1-pp-1}}"
CHECKPOINT_RUN_NAME_1B="${CHECKPOINT_RUN_NAME_1B:-${CHECKPOINT_RUN_NAME:-llama3_2_1B-lr-3e-4-minlr-3e-5-gbs-2048-seqlen-16384-tp-1-pp-1}}"
CONVERTER="${REPO_ROOT}/src/checkpoint/convert_llama_megatron_to_hf.py"
MODEL_VARIANT="${MODEL_VARIANT:-all}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
OVERWRITE_HF="${OVERWRITE_HF:-0}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

if [ ! -f "$CONVERTER" ]; then
  echo "Converter not found: $CONVERTER"
  exit 1
fi
if [ ! -d "$LLAMA_TOKENIZER_PATH" ]; then
  echo "Tokenizer directory not found: $LLAMA_TOKENIZER_PATH"
  exit 1
fi

case "$MODEL_VARIANT" in
  all)
    VARIANTS=(no_fim fim)
    ;;
  all_with_fim_v2)
    VARIANTS=(no_fim fim fim_v2)
    ;;
  all_1B|all_1b)
    VARIANTS=(no_fim_1B fim_v2_1B)
    ;;
  all_with_1B|all_with_1b)
    VARIANTS=(no_fim fim fim_v2 no_fim_1B fim_v2_1B)
    ;;
  no_fim|fim|fim_v2|fineweb_only|no_fim_1B|no_fim_1b|fim_v2_1B|fim_v2_1b)
    VARIANTS=("$MODEL_VARIANT")
    ;;
  *)
    echo "Unsupported MODEL_VARIANT='$MODEL_VARIANT' (use all, all_with_fim_v2, all_1B, all_with_1B, no_fim, fim, fim_v2, fineweb_only, no_fim_1B, or fim_v2_1B)"
    exit 1
    ;;
esac

extra_args=(--max-shard-size "$MAX_SHARD_SIZE")
if [ "$OVERWRITE_HF" = "1" ]; then
  extra_args+=(--overwrite)
fi
if [ "$DRY_RUN_ONLY" = "1" ]; then
  extra_args+=(--dry-run)
fi

canonicalize_variant() {
  case "$1" in
    no_fim_1b) echo "no_fim_1B" ;;
    fim_v2_1b) echo "fim_v2_1B" ;;
    *) echo "$1" ;;
  esac
}

for raw_variant in "${VARIANTS[@]}"; do
  variant="$(canonicalize_variant "$raw_variant")"
  case "$variant" in
    no_fim)
      architecture="3B"
      default_load="${RUNS_ROOT}/llama3_2_3B_pretrain_mix_no_fim/checkpoint/${CHECKPOINT_RUN_NAME_3B}"
      default_save="${RUNS_ROOT}/llama3_2_3B_pretrain_mix_no_fim/hf"
      ;;
    fim)
      architecture="3B"
      default_load="${RUNS_ROOT}/llama3_2_3B_pretrain_mix_fim/checkpoint/${CHECKPOINT_RUN_NAME_3B}"
      default_save="${RUNS_ROOT}/llama3_2_3B_pretrain_mix_fim/hf"
      ;;
    fim_v2)
      architecture="3B"
      default_load="${RUNS_ROOT}/llama3_2_3B_pretrain_mix_fim_v2/checkpoint/${CHECKPOINT_RUN_NAME_3B}"
      default_save="${RUNS_ROOT}/llama3_2_3B_pretrain_mix_fim_v2/hf"
      ;;
    fineweb_only)
      architecture="3B"
      default_load="${RUNS_ROOT}/llama3_2_3B_pretrain_fineweb_only/checkpoint/${CHECKPOINT_RUN_NAME_3B}"
      default_save="${RUNS_ROOT}/llama3_2_3B_pretrain_fineweb_only/hf"
      ;;
    no_fim_1B)
      architecture="1B"
      default_load="${RUNS_ROOT}/llama3_2_1B_pretrain_mix_no_fim/checkpoint/${CHECKPOINT_RUN_NAME_1B}"
      default_save="${RUNS_ROOT}/llama3_2_1B_pretrain_mix_no_fim/hf"
      ;;
    fim_v2_1B)
      architecture="1B"
      default_load="${RUNS_ROOT}/llama3_2_1B_pretrain_mix_fim_v2/checkpoint/${CHECKPOINT_RUN_NAME_1B}"
      default_save="${RUNS_ROOT}/llama3_2_1B_pretrain_mix_fim_v2/hf"
      ;;
  esac

  load_var="LOAD_DIR_${variant^^}"
  save_var="SAVE_DIR_${variant^^}"
  load_dir="${!load_var:-$default_load}"
  save_dir="${!save_var:-$default_save}"

  echo "=========================================="
  echo "LLaMA Megatron -> HF conversion"
  echo "Variant: $variant"
  echo "Architecture: $architecture"
  echo "Load dir: $load_dir"
  echo "Save dir: $save_dir"
  echo "Tokenizer: $LLAMA_TOKENIZER_PATH"
  echo "Max shard size: $MAX_SHARD_SIZE"
  echo "Overwrite: $OVERWRITE_HF"
  echo "Dry run: $DRY_RUN_ONLY"
  echo "=========================================="

  python "$CONVERTER" \
    --load-dir "$load_dir" \
    --save-dir "$save_dir" \
    --tokenizer-dir "$LLAMA_TOKENIZER_PATH" \
    --architecture "$architecture" \
    "${extra_args[@]}"
done
