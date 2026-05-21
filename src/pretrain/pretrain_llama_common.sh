#!/bin/bash

set -euo pipefail

echo "START TIME: $(date)"

required_vars=(
  PRETRAIN_TITLE
  DATASET_DESCRIPTION
  LOG_SUBDIR
  LLAMA_MODEL_ID
  MODEL_NUM_LAYERS
  MODEL_HIDDEN_SIZE
  MODEL_NUM_ATTENTION_HEADS
  MODEL_FFN_HIDDEN_SIZE
  MODEL_NUM_QUERY_GROUPS
  MODEL_KV_CHANNELS
  DEFAULT_DATASET_RELATIVE_PATH
  DEFAULT_OUTPUT_SUBDIR
  DEFAULT_DATA_CACHE_TAG
  DEFAULT_WANDB_EXP_NAME
)

for required_var in "${required_vars[@]}"; do
  if [ -z "${!required_var:-}" ]; then
    echo "Missing required launcher variable: ${required_var}"
    exit 1
  fi
done

# ========== Configuration ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(cd "$SCRIPT_DIR/../.." && pwd)/config.env}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing config: $CONFIG_FILE"
  exit 1
fi
set -a
source "$CONFIG_FILE"
set +a
ENV_FILE="$REPO_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$LOG_ROOT/$LOG_SUBDIR"

# ========== Paths ==========
MEGATRON_DIR="${MEGATRON_DIR:-${SCRATCH_ROOT}/megatron-lm}"
LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/src/pretrain/pretrain_llama_packed.py}"
TRAIN_IMAGE_PATH="${PRETRAIN_IMAGE_PATH:-$IMAGE_PATH}"

if [ ! -f "$TRAIN_IMAGE_PATH" ]; then
  echo "Container image not found: $TRAIN_IMAGE_PATH"
  exit 1
fi
if [ ! -d "$MEGATRON_DIR" ]; then
  echo "Megatron-LM not found: $MEGATRON_DIR"
  exit 1
fi
if [ ! -d "$LLAMA_TOKENIZER_PATH" ]; then
  echo "LLaMA tokenizer not found: $LLAMA_TOKENIZER_PATH"
  exit 1
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Train script not found: $TRAIN_SCRIPT"
  exit 1
fi

# ========== Environment ==========
export NCCL_NVLS_ENABLE=0
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=72

export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# ========== GPU topology ==========
GPUS_PER_NODE_RAW="${SLURM_GPUS_PER_NODE:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE_RAW%%(*}"
GPUS_PER_NODE="${GPUS_PER_NODE%%,*}"
if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ ]]; then
  GPUS_PER_NODE=4
fi

NTASKS="${SLURM_NTASKS:-${SLURM_JOB_NUM_NODES:-16}}"
if ! [[ "$NTASKS" =~ ^[0-9]+$ ]]; then
  NTASKS=16
fi

TOTAL_GPUS=$(( NTASKS * GPUS_PER_NODE ))
echo "Total GPUs (tasks x gpus_per_node): ${TOTAL_GPUS}"

# ========== Training hyperparameters ==========
TP="${TP:-${DEFAULT_TP:-1}}"
PP="${PP:-${DEFAULT_PP:-1}}"
MODEL_PARALLEL_SIZE=$(( TP * PP ))
if (( TOTAL_GPUS % MODEL_PARALLEL_SIZE != 0 )); then
  echo "TOTAL_GPUS (${TOTAL_GPUS}) must be divisible by TP*PP (${MODEL_PARALLEL_SIZE})."
  exit 1
fi
DATA_PARALLEL_SIZE=$(( TOTAL_GPUS / MODEL_PARALLEL_SIZE ))

BATCH_SIZE="${BATCH_SIZE:-1}"            # micro_batch_size=1 required for THD packing
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${DEFAULT_GLOBAL_BATCH_SIZE:-2048}}"
SEQ_LEN="${SEQ_LEN:-${DEFAULT_SEQ_LEN:-16384}}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"

SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL:-3000}}"
LOG_INTERVAL="${LOG_INTERVAL:-${DEFAULT_LOG_INTERVAL:-10}}"
LOG_THROUGHPUT="${LOG_THROUGHPUT:-${DEFAULT_LOG_THROUGHPUT:-1}}"
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS:-8}}"
ENABLE_TP_COMM_OVERLAP="${ENABLE_TP_COMM_OVERLAP:-${DEFAULT_ENABLE_TP_COMM_OVERLAP:-0}}"
GRAD_REDUCE_IN_BF16="${GRAD_REDUCE_IN_BF16:-${DEFAULT_GRAD_REDUCE_IN_BF16:-1}}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-${DEFAULT_DISTRIBUTED_TIMEOUT_MINUTES:-60}}"
TRAIN_TOKENS_OR_ITERS="${TRAIN_TOKENS_OR_ITERS:-0}"  # 0 = auto-detect 1 epoch from dataset
WARMUP_TOKENS_OR_ITERS="${WARMUP_TOKENS_OR_ITERS:-1000000000}"  # ~1B tokens
AUTO_JOB_REQUEUE="${AUTO_JOB_REQUEUE:-1}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"

# Dataset
DATASET_PATH="${DATASET_PATH:-${DATASET_ROOT}/${DEFAULT_DATASET_RELATIVE_PATH}}"
OUTPUT_BASEPATH="${OUTPUT_BASEPATH:-${RUNS_ROOT}/${DEFAULT_OUTPUT_SUBDIR}}"
DATA_SPLIT="${DATA_SPLIT:-100,0,0}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${RUNS_ROOT}/megatron_data_cache/${DEFAULT_DATA_CACHE_TAG}}"
ENABLE_FAST_DATALOADER_CACHE="${ENABLE_FAST_DATALOADER_CACHE:-0}"
ENABLE_DEFER_NPY_INDEX_MMAP="${ENABLE_DEFER_NPY_INDEX_MMAP:-0}"
NEXT_JOB_ID=""

# Gracefully checkpoint and exit before the Slurm wall clock limit.
EXIT_BUFFER_MINUTES="${EXIT_BUFFER_MINUTES:-${DEFAULT_EXIT_BUFFER_MINUTES:-15}}"

slurm_time_to_minutes() {
  local time_str="$1"
  local days=0
  local rest="$time_str"
  local hours=0
  local minutes=0
  local seconds=0

  if [[ "$rest" == *-* ]]; then
    days="${rest%%-*}"
    rest="${rest#*-}"
  fi

  IFS=: read -r hours minutes seconds <<< "$rest"

  if [ -z "${minutes:-}" ]; then
    minutes="${hours:-0}"
    hours=0
    seconds=0
  elif [ -z "${seconds:-}" ]; then
    seconds="${minutes:-0}"
    minutes="${hours:-0}"
    hours=0
  fi

  echo $(( 10#$days * 1440 + 10#$hours * 60 + 10#$minutes + (10#$seconds > 0 ? 1 : 0) ))
}

JOB_TIME_LIMIT_RAW="${SLURM_TIMELIMIT:-11:59:59}"
JOB_TIME_LIMIT_MINUTES="$(slurm_time_to_minutes "$JOB_TIME_LIMIT_RAW")"
EXIT_DURATION_MINS="${EXIT_DURATION_MINS:-$(( JOB_TIME_LIMIT_MINUTES - EXIT_BUFFER_MINUTES ))}"
if (( EXIT_DURATION_MINS < 1 )); then
  EXIT_DURATION_MINS=1
fi

# W&B
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-${WANDB_PROJECT:-llama3-pretrain}}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-$DEFAULT_WANDB_EXP_NAME}"

# ========== Dataset validation ==========
if [ ! -e "${DATASET_PATH}.bin" ] || [ ! -e "${DATASET_PATH}.idx" ]; then
  echo "Missing dataset files for ${DATASET_PATH}"
  exit 1
fi

# ========== Compute training iterations ==========
TOKENS_PER_ITER=$(( GLOBAL_BATCH_SIZE * SEQ_LEN ))
if [ "$TOKENS_PER_ITER" -le 0 ]; then
  echo "Invalid tokens/iter: ${TOKENS_PER_ITER}"
  exit 1
fi

find_latest_completed_iter() {
  local checkpoint_path="$1"
  local latest_completed_iter=0

  if [ -d "$checkpoint_path" ]; then
    while IFS= read -r d; do
      local bn
      local iter
      bn="$(basename "$d")"
      iter="${bn#iter_}"
      iter="${iter##0}"
      if [ -z "$iter" ]; then
        iter=0
      fi
      if [[ "$iter" =~ ^[0-9]+$ ]] && [ "$iter" -gt "$latest_completed_iter" ]; then
        latest_completed_iter="$iter"
      fi
    done < <(find "$checkpoint_path" -maxdepth 1 -type d -name 'iter_*' | sort)
  fi

  echo "$latest_completed_iter"
}

read -r SEQUENCE_COUNT TOTAL_DATASET_TOKENS < <(python3 - "${DATASET_PATH}.idx" "${DATASET_PATH}.bin" <<'PY'
import os
import struct
import sys

idx_path = sys.argv[1]
bin_path = sys.argv[2]

with open(idx_path, "rb") as stream:
    if stream.read(9) != b"MMIDIDX\x00\x00":
        raise SystemExit(f"Invalid idx header: {idx_path}")
    version = struct.unpack("<Q", stream.read(8))[0]
    if version != 1:
        raise SystemExit(f"Unsupported idx version {version}: {idx_path}")
    dtype_code = struct.unpack("<B", stream.read(1))[0]
    sequence_count = struct.unpack("<Q", stream.read(8))[0]

dtype_size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}.get(dtype_code)
if dtype_size is None:
    raise SystemExit(f"Unsupported idx dtype code {dtype_code}: {idx_path}")

bin_size = os.path.getsize(bin_path)
if bin_size % dtype_size != 0:
    raise SystemExit(
        f"Bin size {bin_size} is not divisible by dtype size {dtype_size}: {bin_path}"
    )

print(sequence_count, bin_size // dtype_size)
PY
)

if [ -z "$SEQUENCE_COUNT" ] || [ -z "$TOTAL_DATASET_TOKENS" ]; then
  echo "Failed to parse dataset stats from ${DATASET_PATH}.bin/.idx"
  exit 1
fi

if [ "${TRAIN_TOKENS_OR_ITERS}" -eq 0 ]; then
  TRAIN_TOKENS_OR_ITERS="${TOTAL_DATASET_TOKENS}"
  echo "TRAIN_TOKENS_OR_ITERS=0: defaulting to 1 epoch = ${TOTAL_DATASET_TOKENS} tokens"
fi

if (( TRAIN_TOKENS_OR_ITERS > 0 && TRAIN_TOKENS_OR_ITERS % TOKENS_PER_ITER != 0 )); then
  TRAIN_TOKENS_OR_ITERS=$(( ((TRAIN_TOKENS_OR_ITERS + TOKENS_PER_ITER - 1) / TOKENS_PER_ITER) * TOKENS_PER_ITER ))
fi
if (( WARMUP_TOKENS_OR_ITERS > 0 && WARMUP_TOKENS_OR_ITERS % TOKENS_PER_ITER != 0 )); then
  WARMUP_TOKENS_OR_ITERS=$(( ((WARMUP_TOKENS_OR_ITERS + TOKENS_PER_ITER - 1) / TOKENS_PER_ITER) * TOKENS_PER_ITER ))
fi
if (( WARMUP_TOKENS_OR_ITERS > 0 && WARMUP_TOKENS_OR_ITERS < TOKENS_PER_ITER )); then
  WARMUP_TOKENS_OR_ITERS=$TOKENS_PER_ITER
fi

TRAIN_ITERS=$(( TRAIN_TOKENS_OR_ITERS / TOKENS_PER_ITER ))
WARMUP_ITERS=$(( WARMUP_TOKENS_OR_ITERS / TOKENS_PER_ITER ))

MICRO_BATCHES_PER_ITER_DENOM=$(( DATA_PARALLEL_SIZE * BATCH_SIZE ))
if (( GLOBAL_BATCH_SIZE % MICRO_BATCHES_PER_ITER_DENOM != 0 )); then
  echo "GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by DP*micro_batch (${MICRO_BATCHES_PER_ITER_DENOM})."
  exit 1
fi
GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / MICRO_BATCHES_PER_ITER_DENOM ))

if (( TRAIN_ITERS <= 1 )); then
  WARMUP_ITERS=0
  echo "Disabling warmup for tiny run (${TRAIN_ITERS} train iter)"
elif (( WARMUP_ITERS >= TRAIN_ITERS )); then
  WARMUP_ITERS=$(( TRAIN_ITERS / 10 ))
  if (( WARMUP_ITERS < 1 )); then
    WARMUP_ITERS=1
  fi
  echo "Capping warmup to ${WARMUP_ITERS} iters (warmup >= train_iters)"
fi

# ========== Checkpoint resume ==========
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-$LLAMA_MODEL_ID}"
RUN_NAME="${RUN_NAME_PREFIX}-lr-${LR}-minlr-${MIN_LR}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}-tp-${TP}-pp-${PP}"
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${RUN_NAME}"

LATEST_COMPLETED_ITER="$(find_latest_completed_iter "$CHECKPOINT_PATH")"

if [ "$LATEST_COMPLETED_ITER" -ge "$TRAIN_ITERS" ]; then
  echo "Target already reached (${LATEST_COMPLETED_ITER}/${TRAIN_ITERS}). Nothing to do."
  exit 0
fi

LOAD_ARG=""
if [ "$LATEST_COMPLETED_ITER" -gt 0 ]; then
  LOAD_ARG="--load $CHECKPOINT_PATH"
  RESUME_MODE="resume"
else
  RESUME_MODE="fresh"
fi

# ========== W&B ==========
WANDB_ARGS=""
if [ "$WANDB_ENABLE" = "true" ]; then
  WANDB_ARGS="--wandb-project $WANDB_PROJECT_NAME"
  if [ -n "$WANDB_EXP_NAME" ]; then
    WANDB_ARGS="$WANDB_ARGS --wandb-exp-name $WANDB_EXP_NAME"
  fi
  mkdir -p "$OUTPUT_BASEPATH/wandb"
  export WANDB_DIR="$OUTPUT_BASEPATH/wandb"
fi

# ========== Summary ==========
mkdir -p "$OUTPUT_BASEPATH"

echo "=========================================="
echo "$PRETRAIN_TITLE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Container: $TRAIN_IMAGE_PATH"
echo "Megatron-LM: $MEGATRON_DIR"
echo "Train script: $TRAIN_SCRIPT"
echo "Tokenizer: $LLAMA_TOKENIZER_PATH"
echo "Dataset description: $DATASET_DESCRIPTION"
echo "Dataset: $DATASET_PATH"
echo "Data split: $DATA_SPLIT"
echo "Parallelism TP/PP: ${TP}/${PP}"
echo "Data parallel size: $DATA_PARALLEL_SIZE"
echo "Micro batch: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUM_STEPS"
echo "Global batch: $GLOBAL_BATCH_SIZE"
echo "Sequence length: $SEQ_LEN"
echo "Model: $LLAMA_MODEL_ID"
echo "Model layers/hidden/heads: ${MODEL_NUM_LAYERS}/${MODEL_HIDDEN_SIZE}/${MODEL_NUM_ATTENTION_HEADS}"
echo "FFN hidden / query groups / kv channels: ${MODEL_FFN_HIDDEN_SIZE}/${MODEL_NUM_QUERY_GROUPS}/${MODEL_KV_CHANNELS}"
echo "Total dataset sequences: $SEQUENCE_COUNT"
echo "Total dataset tokens: $TOTAL_DATASET_TOKENS"
echo "Train tokens target: $TRAIN_TOKENS_OR_ITERS"
echo "Train iterations: $TRAIN_ITERS"
echo "Warmup iterations: $WARMUP_ITERS"
echo "LR: $LR / Min LR: $MIN_LR"
echo "Dropout attention/hidden: ${ATTENTION_DROPOUT}/${HIDDEN_DROPOUT}"
echo "Resume mode: $RESUME_MODE"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Latest completed iteration: $LATEST_COMPLETED_ITER"
echo "Output basepath: $OUTPUT_BASEPATH"
echo "Data cache path: $DATA_CACHE_PATH"
echo "Distributed timeout (min): $DISTRIBUTED_TIMEOUT_MINUTES"
echo "Exit after (min): $EXIT_DURATION_MINS (job limit: $JOB_TIME_LIMIT_RAW, buffer: $EXIT_BUFFER_MINUTES)"
echo "Grad reduce in bf16: $GRAD_REDUCE_IN_BF16"
echo "W&B enabled: $WANDB_ENABLE"
echo "Auto job requeue: $AUTO_JOB_REQUEUE"
echo "Dry run only: $DRY_RUN_ONLY"
echo "=========================================="

# ========== Auto-requeue (singleton chain) ==========
if [ "$AUTO_JOB_REQUEUE" = "1" ] && [ "$DRY_RUN_ONLY" != "1" ]; then
  SCRIPT_PATH="$(readlink -f "$0")"
  REQUEUE_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-16}}"
  REQUEUE_NTASKS="${SLURM_NTASKS:-$REQUEUE_NODES}"
  REQUEUE_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
  REQUEUE_NTASKS_PER_NODE="${REQUEUE_NTASKS_PER_NODE%%(*}"
  REQUEUE_NTASKS_PER_NODE="${REQUEUE_NTASKS_PER_NODE%%,*}"
  REQUEUE_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"
  REQUEUE_GPUS_PER_NODE="${REQUEUE_GPUS_PER_NODE%%(*}"
  REQUEUE_GPUS_PER_NODE="${REQUEUE_GPUS_PER_NODE%%,*}"
  REQUEUE_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-288}"
  REQUEUE_TIME="${SLURM_TIMELIMIT:-11:59:59}"
  REQUEUE_JOB_NAME="${SLURM_JOB_NAME:-}"
  REQUEUE_RESERVATION="${SLURM_JOB_RESERVATION:-}"
  if [ -z "$REQUEUE_RESERVATION" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    JOB_INFO="$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null || true)"
    if [[ "$JOB_INFO" =~ (^|[[:space:]])Reservation(Name)?=([^[:space:]]+) ]]; then
      REQUEUE_RESERVATION="${BASH_REMATCH[3]}"
    fi
  fi
  if [ "$REQUEUE_RESERVATION" = "(null)" ] || [ "$REQUEUE_RESERVATION" = "N/A" ] || [ "$REQUEUE_RESERVATION" = "None" ]; then
    REQUEUE_RESERVATION=""
  fi

  REQUEUE_SBATCH_ARGS=(
    --parsable
    --dependency=singleton
    --export=ALL
    --nodes="$REQUEUE_NODES"
    --ntasks="$REQUEUE_NTASKS"
    --ntasks-per-node="$REQUEUE_NTASKS_PER_NODE"
    --gpus-per-node="$REQUEUE_GPUS_PER_NODE"
    --cpus-per-task="$REQUEUE_CPUS_PER_TASK"
    --time="$REQUEUE_TIME"
  )
  if [ -n "$REQUEUE_RESERVATION" ]; then
    REQUEUE_SBATCH_ARGS+=(--reservation="$REQUEUE_RESERVATION")
  fi
  if [ -n "$REQUEUE_JOB_NAME" ]; then
    REQUEUE_SBATCH_ARGS+=(--job-name="$REQUEUE_JOB_NAME")
  fi

  NEXT_JOB_ID="$(sbatch "${REQUEUE_SBATCH_ARGS[@]}" "$SCRIPT_PATH")"
  echo "Auto-queued follow-up job: ${NEXT_JOB_ID}"
fi

if [ "$DRY_RUN_ONLY" = "1" ]; then
  echo "DRY_RUN_ONLY=1, exiting before srun."
  exit 0
fi

# ========== Runtime flags ==========
SEQUENCE_PARALLEL_ARG=""
if [ "$TP" -gt 1 ]; then
  SEQUENCE_PARALLEL_ARG="--sequence-parallel"
fi

TP_COMM_OVERLAP_ARG=""
if [ "$TP" -gt 1 ] && [ "$ENABLE_TP_COMM_OVERLAP" = "1" ]; then
  TP_COMM_OVERLAP_ARG="--tp-comm-overlap"
fi

LOG_THROUGHPUT_ARG=""
if [ "$LOG_THROUGHPUT" = "1" ]; then
  LOG_THROUGHPUT_ARG="--log-throughput"
fi

GRAD_REDUCE_IN_BF16_ARG=""
if [ "$GRAD_REDUCE_IN_BF16" = "1" ]; then
  GRAD_REDUCE_IN_BF16_ARG="--grad-reduce-in-bf16"
fi

mkdir -p "$DATA_CACHE_PATH"
DATA_CACHE_ARG="--data-cache-path ${DATA_CACHE_PATH}"

DATA_CACHE_HAS_CONTENT=0
if [ -n "$(find "$DATA_CACHE_PATH" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  DATA_CACHE_HAS_CONTENT=1
fi

DATALOADER_CACHE_ARGS=""
if [ "$DATA_CACHE_HAS_CONTENT" = "1" ] && [ "$ENABLE_FAST_DATALOADER_CACHE" = "1" ]; then
  DATALOADER_CACHE_ARGS="${DATALOADER_CACHE_ARGS} --dataloader-fast-cache-load"
fi
if [ "$DATA_CACHE_HAS_CONTENT" = "1" ] && [ "$ENABLE_DEFER_NPY_INDEX_MMAP" = "1" ]; then
  DATALOADER_CACHE_ARGS="${DATALOADER_CACHE_ARGS} --dataloader-defer-npy-index-mmap"
fi

# ========== Container & launch ==========
CONTAINER_OPTS=(
  --container-image="$TRAIN_IMAGE_PATH"
  --no-container-mount-home
  --container-mounts="$CONTAINER_MOUNTS"
)

set +e
srun "${CONTAINER_OPTS[@]}" --cpu-bind=none --wait 60 --unbuffered \
  bash -c "
    export PYTHONPATH=${MEGATRON_DIR}:\${PYTHONPATH:-}
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FLASH_ATTN=1
    export NVTE_FUSED_ATTN=0
    export MEGATRON_DIR=${MEGATRON_DIR}

    torchrun \
      --nproc_per_node=${GPUS_PER_NODE} \
      --nnodes=\${SLURM_NTASKS} \
      --node_rank=\${SLURM_PROCID} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      ${TRAIN_SCRIPT} \
      \
      --num-layers ${MODEL_NUM_LAYERS} \
      --hidden-size ${MODEL_HIDDEN_SIZE} \
      --num-attention-heads ${MODEL_NUM_ATTENTION_HEADS} \
      --ffn-hidden-size ${MODEL_FFN_HIDDEN_SIZE} \
      --num-query-groups ${MODEL_NUM_QUERY_GROUPS} \
      --group-query-attention \
      --kv-channels ${MODEL_KV_CHANNELS} \
      --seq-length ${SEQ_LEN} \
      --max-position-embeddings ${MODEL_MAX_POSITION_EMBEDDINGS:-131072} \
      --padded-vocab-size ${MODEL_PADDED_VOCAB_SIZE:-128256} \
      --normalization ${MODEL_NORMALIZATION:-RMSNorm} \
      --norm-epsilon ${MODEL_NORM_EPSILON:-1e-5} \
      --swiglu \
      --disable-bias-linear \
      --position-embedding-type rope \
      --use-rotary-position-embeddings \
      --rotary-base ${MODEL_ROTARY_BASE:-500000} \
      \
      --bf16 \
      ${GRAD_REDUCE_IN_BF16_ARG} \
      --attention-backend flash \
      --transformer-impl transformer_engine \
      --attention-dropout ${ATTENTION_DROPOUT} \
      --hidden-dropout ${HIDDEN_DROPOUT} \
      \
      --micro-batch-size ${BATCH_SIZE} \
      --global-batch-size ${GLOBAL_BATCH_SIZE} \
      --no-create-attention-mask-in-dataloader \
      --reset-position-ids \
      --reset-attention-mask \
      --eod-mask-loss \
      \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model ${LLAMA_TOKENIZER_PATH} \
      \
      --tensor-model-parallel-size ${TP} \
      --pipeline-model-parallel-size ${PP} \
      ${SEQUENCE_PARALLEL_ARG} \
      --use-distributed-optimizer \
      --overlap-grad-reduce \
      --overlap-param-gather \
      ${TP_COMM_OVERLAP_ARG} \
      \
      --lr ${LR} \
      --min-lr ${MIN_LR} \
      --lr-decay-style cosine \
      --lr-warmup-iters ${WARMUP_ITERS} \
      --weight-decay 0.1 \
      --adam-beta1 0.9 \
      --adam-beta2 0.95 \
      --clip-grad 1.0 \
      --init-method-std 0.02 \
      \
      --train-iters ${TRAIN_ITERS} \
      --split ${DATA_SPLIT} \
      --data-path ${DATASET_PATH} \
      ${DATA_CACHE_ARG} \
      ${DATALOADER_CACHE_ARGS} \
      --save ${CHECKPOINT_PATH} \
      ${LOAD_ARG} \
      --save-interval ${SAVE_INTERVAL} \
      --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES} \
      --exit-duration-in-mins ${EXIT_DURATION_MINS} \
      \
      --log-interval ${LOG_INTERVAL} \
      ${LOG_THROUGHPUT_ARG} \
      --eval-interval 10000 \
      --eval-iters 0 \
      --num-workers ${NUM_WORKERS} \
      \
      ${WANDB_ARGS}
  "
SRUN_EXIT_CODE=$?
set -e

FINAL_COMPLETED_ITER="$(find_latest_completed_iter "$CHECKPOINT_PATH")"
if [ -n "$NEXT_JOB_ID" ] && [ "$FINAL_COMPLETED_ITER" -ge "$TRAIN_ITERS" ]; then
  if scancel "$NEXT_JOB_ID"; then
    echo "Canceled queued follow-up job ${NEXT_JOB_ID} after reaching target (${FINAL_COMPLETED_ITER}/${TRAIN_ITERS})."
  else
    echo "Warning: failed to cancel queued follow-up job ${NEXT_JOB_ID} after reaching target (${FINAL_COMPLETED_ITER}/${TRAIN_ITERS})."
  fi
fi

if [ "$SRUN_EXIT_CODE" -ne 0 ]; then
  echo "Training command exited with code ${SRUN_EXIT_CODE}."
  exit "$SRUN_EXIT_CODE"
fi

echo "END TIME: $(date)"
