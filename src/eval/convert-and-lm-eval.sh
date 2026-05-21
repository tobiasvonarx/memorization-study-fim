#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=24
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/lm_eval/%A/lm_eval.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/lm_eval/%A/lm_eval.out
#SBATCH --gpus-per-node=4
#SBATCH --job-name=lm_eval_setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
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

source "$CONDA_PROFILE"
conda activate base

LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"
MODEL_SIZE="${MODEL_SIZE:-3B}"
FIM_VARIANT="${FIM_VARIANT:-fim}"
case "$MODEL_SIZE" in
  1B|1b) MODEL_SIZE="1B" ;;
  3B|3b) MODEL_SIZE="3B" ;;
  *)
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE (use 1B or 3B)"
    exit 1
    ;;
esac
case "$FIM_VARIANT" in
  fim|fim_v2) ;;
  *)
    echo "Unsupported FIM_VARIANT=$FIM_VARIANT (use fim or fim_v2)"
    exit 1
    ;;
esac
RUN_PREFIX="llama3_2_${MODEL_SIZE}"
CHECKPOINT_RUN_NAME="${CHECKPOINT_RUN_NAME:-${RUN_PREFIX}-lr-3e-4-minlr-3e-5-gbs-2048-seqlen-16384-tp-1-pp-1}"
HF_CKPT_PATH_NO_FIM="${HF_CKPT_PATH_NO_FIM:-${RUNS_ROOT}/${RUN_PREFIX}_pretrain_mix_no_fim/hf}"
HF_CKPT_PATH_FIM="${HF_CKPT_PATH_FIM:-${RUNS_ROOT}/${RUN_PREFIX}_pretrain_mix_${FIM_VARIANT}/hf}"
LOAD_DIR_NO_FIM="${LOAD_DIR_NO_FIM:-${RUNS_ROOT}/${RUN_PREFIX}_pretrain_mix_no_fim/checkpoint/${CHECKPOINT_RUN_NAME}}"
LOAD_DIR_FIM="${LOAD_DIR_FIM:-${RUNS_ROOT}/${RUN_PREFIX}_pretrain_mix_${FIM_VARIANT}/checkpoint/${CHECKPOINT_RUN_NAME}}"
CONVERTER="${REPO_ROOT}/src/checkpoint/convert_llama_megatron_to_hf.py"
RUN_CHECKPOINT_CONVERSION="${RUN_CHECKPOINT_CONVERSION:-auto}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
OVERWRITE_HF="${OVERWRITE_HF:-0}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

hf_ready() {
  local path="$1"
  [ -f "$path/config.json" ] && {
    [ -f "$path/model.safetensors" ] || [ -f "$path/model.safetensors.index.json" ]
  }
}

run_conversion() {
  local variant="$1"
  local load_dir="$2"
  local save_dir="$3"
  local args=(--load-dir "$load_dir" --save-dir "$save_dir" --tokenizer-dir "$LLAMA_TOKENIZER_PATH" --architecture "$MODEL_SIZE" --max-shard-size "$MAX_SHARD_SIZE")
  if [ "$OVERWRITE_HF" = "1" ]; then
    args+=(--overwrite)
  fi
  if [ "$DRY_RUN_ONLY" = "1" ]; then
    args+=(--dry-run)
  fi

  echo "=========================================="
  echo "Checkpoint conversion"
  echo "Variant: $variant"
  echo "Architecture: $MODEL_SIZE"
  echo "Load dir: $load_dir"
  echo "Save dir: $save_dir"
  echo "Tokenizer: $LLAMA_TOKENIZER_PATH"
  echo "Overwrite: $OVERWRITE_HF"
  echo "Dry run: $DRY_RUN_ONLY"
  echo "=========================================="
  python "$CONVERTER" "${args[@]}"
}

if [ ! -f "$CONVERTER" ]; then
  echo "Missing converter: $CONVERTER"
  exit 1
fi
if [ ! -d "$LLAMA_TOKENIZER_PATH" ]; then
  echo "Missing tokenizer path: $LLAMA_TOKENIZER_PATH"
  exit 1
fi

case "$RUN_CHECKPOINT_CONVERSION" in
  1|true|yes)
    run_conversion no_fim "$LOAD_DIR_NO_FIM" "$HF_CKPT_PATH_NO_FIM"
    run_conversion "$FIM_VARIANT" "$LOAD_DIR_FIM" "$HF_CKPT_PATH_FIM"
    ;;
  0|false|no)
    echo "Skipping checkpoint conversion because RUN_CHECKPOINT_CONVERSION=$RUN_CHECKPOINT_CONVERSION"
    ;;
  auto)
    if ! hf_ready "$HF_CKPT_PATH_NO_FIM"; then
      run_conversion no_fim "$LOAD_DIR_NO_FIM" "$HF_CKPT_PATH_NO_FIM"
    else
      echo "Found ready no-FIM HF checkpoint: $HF_CKPT_PATH_NO_FIM"
    fi
    if ! hf_ready "$HF_CKPT_PATH_FIM"; then
      run_conversion "$FIM_VARIANT" "$LOAD_DIR_FIM" "$HF_CKPT_PATH_FIM"
    else
      echo "Found ready ${FIM_VARIANT} HF checkpoint: $HF_CKPT_PATH_FIM"
    fi
    ;;
  *)
    echo "Unsupported RUN_CHECKPOINT_CONVERSION=$RUN_CHECKPOINT_CONVERSION (use auto, 1, or 0)"
    exit 1
    ;;
esac

export HF_CKPT_PATH_NO_FIM
export HF_CKPT_PATH_FIM
export TOKENIZER_PATH="${TOKENIZER_PATH:-$LLAMA_TOKENIZER_PATH}"
if [ -z "${RESULTS_TAG:-}" ]; then
  if [ "$MODEL_SIZE" = "3B" ] && [ "$FIM_VARIANT" = "fim" ]; then
    RESULTS_TAG="llama3_2_3B_mix_${SLURM_JOB_ID:-manual}"
  else
    RESULTS_TAG="llama3_2_${MODEL_SIZE}_mix_${FIM_VARIANT}_${SLURM_JOB_ID:-manual}"
  fi
fi
export RESULTS_TAG
export SETUP_LMEVAL=1
if [ "$DRY_RUN_ONLY" = "1" ]; then
  export SKIP_INPUT_CHECKS=1
fi

exec bash "$REPO_ROOT/src/eval/downstream-eval.sh"
