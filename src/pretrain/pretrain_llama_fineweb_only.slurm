#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/pretrain_llama_fineweb_only/%A.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/pretrain_llama_fineweb_only/%A.out
#SBATCH --gpus-per-node=4
#SBATCH --job-name=llama3_2_3b_fineweb_only
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --open-mode=append
#SBATCH --no-requeue

# LLaMA 3.2 3B pretraining on FineWeb only (no FIM).
# Uses Megatron-LM directly with THD-packed sequences via pretrain_llama_packed.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(cd "$SCRIPT_DIR/../.." && pwd)/config.env}"
set -a
source "$CONFIG_FILE"
set +a

PRETRAIN_TITLE="LLaMA 3.2 3B FineWeb-Only Pretraining"
DATASET_DESCRIPTION="FineWeb only, no FIM"
LOG_SUBDIR="pretrain_llama_fineweb_only"
LLAMA_MODEL_ID="llama3_2_3B"

MODEL_NUM_LAYERS=28
MODEL_HIDDEN_SIZE=3072
MODEL_NUM_ATTENTION_HEADS=24
MODEL_FFN_HIDDEN_SIZE=8192
MODEL_NUM_QUERY_GROUPS=8
MODEL_KV_CHANNELS=128

DEFAULT_DATASET_RELATIVE_PATH="fineweb/fineweb_100BT_llama_no_fim/combined_text_document"
DEFAULT_OUTPUT_SUBDIR="llama3_2_3B_pretrain_fineweb_only"
DEFAULT_DATA_CACHE_TAG="fineweb_100BT_llama_no_fim"
DEFAULT_WANDB_EXP_NAME="llama3_2_3B_fineweb_only"

source "$REPO_ROOT/src/pretrain/pretrain_llama_common.sh"
