#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/gutenberg/pipeline_%A_%a.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/gutenberg/pipeline_%A_%a.out
#SBATCH --gres=gpu:4
#SBATCH --job-name=gutenberg_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

mkdir -p "$LOG_ROOT/gutenberg"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Container image not found: $IMAGE_PATH"
  exit 1
fi

source "$CONDA_PROFILE"
conda activate base

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PIPELINE_STAGE="${PIPELINE_STAGE:-build_filter}"
LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"
# Megatron expects --load to point at the checkpoint root directory that contains
# latest_checkpointed_iteration.txt, not directly at an iter_XXXXXXX subdirectory.
GUTENBERG_CHECKPOINT_LOAD_DIR="${GUTENBERG_CHECKPOINT_LOAD_DIR:-$FINEWEB_PRELIM_CHECKPOINT}"
GUTENBERG_CHECKPOINT_LOAD_DIR="${GUTENBERG_CHECKPOINT_LOAD_DIR%/iter_*}"
readonly GUTENBERG_CHECKPOINT_LOAD_DIR
readonly GUTENBERG_MODEL_NUM_LAYERS="${GUTENBERG_MODEL_NUM_LAYERS:-28}"
readonly GUTENBERG_MODEL_HIDDEN_SIZE="${GUTENBERG_MODEL_HIDDEN_SIZE:-3072}"
readonly GUTENBERG_MODEL_NUM_ATTENTION_HEADS="${GUTENBERG_MODEL_NUM_ATTENTION_HEADS:-24}"
readonly GUTENBERG_MODEL_FFN_HIDDEN_SIZE="${GUTENBERG_MODEL_FFN_HIDDEN_SIZE:-8192}"
readonly GUTENBERG_MODEL_NUM_QUERY_GROUPS="${GUTENBERG_MODEL_NUM_QUERY_GROUPS:-8}"
readonly GUTENBERG_MODEL_KV_CHANNELS="${GUTENBERG_MODEL_KV_CHANNELS:-128}"
readonly GUTENBERG_MODEL_MAX_POSITION_EMBEDDINGS="${GUTENBERG_MODEL_MAX_POSITION_EMBEDDINGS:-131072}"
readonly GUTENBERG_MODEL_PADDED_VOCAB_SIZE="${GUTENBERG_MODEL_PADDED_VOCAB_SIZE:-128256}"
readonly GUTENBERG_MODEL_NORM_EPSILON="${GUTENBERG_MODEL_NORM_EPSILON:-1e-5}"
readonly GUTENBERG_MODEL_ROTARY_BASE="${GUTENBERG_MODEL_ROTARY_BASE:-500000}"
GUTENBERG_GPU_PROCS="${GUTENBERG_GPU_PROCS:-4}"
if [ -z "${GUTENBERG_CPU_WORKERS_PER_RANK:-}" ]; then
  total_cpus="${SLURM_CPUS_PER_TASK:-1}"
  workers_per_rank=$(( total_cpus / GUTENBERG_GPU_PROCS ))
  if [ "$workers_per_rank" -lt 1 ]; then
    workers_per_rank=1
  fi
  GUTENBERG_CPU_WORKERS_PER_RANK="$workers_per_rank"
fi

CONTAINER_OPTS=(
  --container-image="$IMAGE_PATH"
  --no-container-mount-home
  --container-mounts="$CONTAINER_MOUNTS"
  --container-workdir="$REPO_ROOT/src"
)

case "$PIPELINE_STAGE" in
  build_filter)
    mkdir -p "$GUTENBERG_FILTERED_DIR"
    srun "${CONTAINER_OPTS[@]}" \
      torchrun --nproc_per_node="$GUTENBERG_GPU_PROCS" "$REPO_ROOT/src/preprocess_gutenberg/build_filter_with_fineweb_checkpoint.py" \
        --output-dir "$GUTENBERG_FILTERED_DIR" \
        --tokenizer-path "$LLAMA_TOKENIZER_PATH" \
        --num-layers "$GUTENBERG_MODEL_NUM_LAYERS" \
        --hidden-size "$GUTENBERG_MODEL_HIDDEN_SIZE" \
        --num-attention-heads "$GUTENBERG_MODEL_NUM_ATTENTION_HEADS" \
        --ffn-hidden-size "$GUTENBERG_MODEL_FFN_HIDDEN_SIZE" \
        --num-query-groups "$GUTENBERG_MODEL_NUM_QUERY_GROUPS" \
        --group-query-attention \
        --kv-channels "$GUTENBERG_MODEL_KV_CHANNELS" \
        --max-position-embeddings "$GUTENBERG_MODEL_MAX_POSITION_EMBEDDINGS" \
        --padded-vocab-size "$GUTENBERG_MODEL_PADDED_VOCAB_SIZE" \
        --normalization RMSNorm \
        --norm-epsilon "$GUTENBERG_MODEL_NORM_EPSILON" \
        --swiglu \
        --disable-bias-linear \
        --position-embedding-type rope \
        --use-rotary-position-embeddings \
        --rotary-base "$GUTENBERG_MODEL_ROTARY_BASE" \
        --bf16 \
        --attention-backend flash \
        --transformer-impl transformer_engine \
        --num-books "${NUM_BOOKS:-34000}" \
        --num-tokens "${NUM_TOKENS:-4096}" \
        --seq-length "${NUM_TOKENS:-4096}" \
        --char-pos-start "${CHAR_POS_START:-10000}" \
        --char-pos-end "${CHAR_POS_END:-80000}" \
        --window-stride-tokens "${WINDOW_STRIDE_TOKENS:-4096}" \
        ${MAX_WINDOWS_PER_BOOK:+--max-windows-per-book "$MAX_WINDOWS_PER_BOOK"} \
        --preprocess-workers "$GUTENBERG_CPU_WORKERS_PER_RANK" \
        --score-batch-size "${GUTENBERG_SCORE_BATCH_SIZE:-16}" \
        --log-every-books "${LOG_EVERY_BOOKS:-25}" \
        ${MEMORIZED_MAX_PPL:+--memorized-max-ppl "$MEMORIZED_MAX_PPL"} \
        --max-allowed-ppl "${MAX_ALLOWED_PPL:-500}" \
        ${DROP_LOW_PPL_FRACTION:+--drop-low-ppl-fraction "$DROP_LOW_PPL_FRACTION"} \
        --use-checkpoint-args \
        --use-mcore-models \
        --load "$GUTENBERG_CHECKPOINT_LOAD_DIR" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "$LLAMA_TOKENIZER_PATH"
    ;;
  semantic_dedup)
    mkdir -p "$GUTENBERG_FILTERED_DIR"
    srun "${CONTAINER_OPTS[@]}" python "$REPO_ROOT/src/preprocess_gutenberg/semantic_dedup.py" \
      --input-dir "$GUTENBERG_FILTERED_DIR" \
      --output-dir "$GUTENBERG_FILTERED_DIR" \
      --semantic-dedup-model "${SEMANTIC_DEDUP_MODEL:-nomic-ai/nomic-embed-text-v1.5}" \
      ${SEMANTIC_DEDUP_CACHE_DIR:+--semantic-dedup-cache-dir "$SEMANTIC_DEDUP_CACHE_DIR"} \
      --semantic-dedup-batch-size "${SEMANTIC_DEDUP_BATCH_SIZE:-8}" \
      --semantic-dedup-max-length "${SEMANTIC_DEDUP_MAX_LENGTH:-4096}" \
      --semantic-dedup-embedding-dim "${SEMANTIC_DEDUP_EMBEDDING_DIM:-256}" \
      --semantic-dedup-similarity-threshold "${SEMANTIC_DEDUP_SIMILARITY_THRESHOLD:-0.96}" \
      --semantic-dedup-jaccard-threshold "${SEMANTIC_DEDUP_JACCARD_THRESHOLD:-0.20}" \
      --semantic-dedup-similarity-block-size "${SEMANTIC_DEDUP_SIMILARITY_BLOCK_SIZE:-512}" \
      --final-count-divisor "${FINAL_COUNT_DIVISOR:-12}" \
      ${KEEP_TOP_K_BY_PPL:+--keep-top-k-by-ppl "$KEEP_TOP_K_BY_PPL"}
    ;;
  create_replicas)
    python "$REPO_ROOT/src/preprocess_gutenberg/create_replicas.py" \
      --input-dir "$GUTENBERG_FILTERED_DIR" \
      --output-dir "$GUTENBERG_REPLICAS_DIR" \
      --bucket-size "${BUCKET_SIZE:-auto}" \
      --score-field ppl \
      --balance-mode "${BUCKET_BALANCE_MODE:-equalize_bucket_mean}" \
      --max-score-mean-spread-ratio "${MAX_SCORE_MEAN_SPREAD_RATIO:-0.01}"
    ;;
  create_replicas_v2)
    python "$REPO_ROOT/src/preprocess_gutenberg/create_replicas.py" \
      --input-dir "$GUTENBERG_FILTERED_DIR" \
      --output-dir "${GUTENBERG_REPLICAS_V2_DIR:-${DATASET_ROOT}/gutenberg/4096_tokens_filtered_with_replicas_v2}" \
      --bucket-size "${BUCKET_SIZE:-auto}" \
      --score-field ppl \
      --balance-mode "${BUCKET_BALANCE_MODE:-equalize_bucket_mean}" \
      --max-score-mean-spread-ratio "${MAX_SCORE_MEAN_SPREAD_RATIO:-0.01}"
    ;;
  prepare_fim)
    mkdir -p "$GUTENBERG_FIM_DIR"
    srun "${CONTAINER_OPTS[@]}" python "$REPO_ROOT/src/prepare_fim.py" \
      --input-glob "$GUTENBERG_REPLICAS_DIR/rep_*_token.jsonl" \
      --output "$GUTENBERG_FIM_DIR/tokens" \
      --mode fim_standard \
      --swap-percent "${SWAP_PERCENT:-100}" \
      --min-doc-length 50 \
      --tokenizer "$LLAMA_TOKENIZER_PATH" \
      --seq-length "${SEQ_LENGTH:-4096}" \
      --deterministic-splits \
      --split-seed "${SPLIT_SEED:-42}"
    ;;
  prepare_fim_v2)
    mkdir -p "${GUTENBERG_FIM_V2_DIR:-${DATASET_ROOT}/gutenberg/4096_tokens_with_fim_v2}"
    srun "${CONTAINER_OPTS[@]}" python "$REPO_ROOT/src/prepare_fim.py" \
      --input-glob "${GUTENBERG_REPLICAS_V2_DIR:-${DATASET_ROOT}/gutenberg/4096_tokens_filtered_with_replicas_v2}/rep_*_token.jsonl" \
      --output "${GUTENBERG_FIM_V2_DIR:-${DATASET_ROOT}/gutenberg/4096_tokens_with_fim_v2}/tokens" \
      --mode fim_standard \
      --swap-percent "${SWAP_PERCENT:-100}" \
      --min-doc-length 50 \
      --tokenizer "$LLAMA_TOKENIZER_PATH" \
      --seq-length "${SEQ_LENGTH:-4096}" \
      --deterministic-splits \
      --replica-aware-splits \
      --split-seed "${SPLIT_SEED:-42}"
    ;;
  prepare_ltr)
    mkdir -p "$GUTENBERG_LTR_DIR"
    srun "${CONTAINER_OPTS[@]}" python "$REPO_ROOT/src/prepare_fim.py" \
      --input-glob "$GUTENBERG_REPLICAS_DIR/rep_*_token.jsonl" \
      --output "$GUTENBERG_LTR_DIR/tokens" \
      --mode ltr_clean \
      --min-doc-length 50 \
      --tokenizer "$LLAMA_TOKENIZER_PATH" \
      --seq-length "${SEQ_LENGTH:-4096}"
    ;;
  *)
    echo "Unknown PIPELINE_STAGE='$PIPELINE_STAGE'"
    echo "Supported: build_filter | semantic_dedup | create_replicas | create_replicas_v2 | prepare_fim | prepare_fim_v2 | prepare_ltr"
    exit 1
    ;;
esac
