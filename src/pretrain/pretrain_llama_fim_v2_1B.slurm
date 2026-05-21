#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/pretrain_llama_fim_v2_1B/%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/pretrain_llama_fim_v2_1B/%A.out
#SBATCH --gpus-per-node=4
#SBATCH --job-name=llama3_2_1b_mix_fim_v2
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --open-mode=append
#SBATCH --no-requeue

# LLaMA 3.2 1B pretraining on FineWeb + Gutenberg FIM-v2.
# Uses the same LLaMA 3.2 tokenizer as the 3B runs and keeps TRAIN_TOKENS_OR_ITERS=0
# by default, so this ablation trains one epoch over combined_fim_v2 like the 3B FIM-v2 run.
# FIM-v2 keeps the same repeated Gutenberg excerpts but makes each repeated copy
# use a deterministic replica-aware FIM split keyed by excerpt_id and replica_index.
# Expected mixed dataset output prefix: ${DATASET_ROOT}/combined_fim_v2.{bin,idx}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(cd "$SCRIPT_DIR/../.." && pwd)/config.env}"
set -a
source "$CONFIG_FILE"
set +a

PRETRAIN_TITLE="LLaMA 3.2 1B FineWeb + Gutenberg FIM-v2 Pretraining"
DATASET_DESCRIPTION="Mixed FineWeb + Gutenberg dataset with replica-aware FIM splits"
LOG_SUBDIR="pretrain_llama_fim_v2_1B"
LLAMA_MODEL_ID="llama3_2_1B"

MODEL_NUM_LAYERS=16
MODEL_HIDDEN_SIZE=2048
MODEL_NUM_ATTENTION_HEADS=32
MODEL_FFN_HIDDEN_SIZE=8192
MODEL_NUM_QUERY_GROUPS=8
MODEL_KV_CHANNELS=64

DEFAULT_DATASET_RELATIVE_PATH="combined_fim_v2"
DEFAULT_OUTPUT_SUBDIR="llama3_2_1B_pretrain_mix_fim_v2"
DEFAULT_DATA_CACHE_TAG="combined_fim_v2_1B"
DEFAULT_WANDB_EXP_NAME="llama3_2_1B_mix_fim_v2"

source "$REPO_ROOT/src/pretrain/pretrain_llama_common.sh"
