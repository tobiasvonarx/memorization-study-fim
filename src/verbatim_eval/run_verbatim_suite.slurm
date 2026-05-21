#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=verbatim_debug
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/verbatim_eval/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/verbatim_eval/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$(cd "$SCRIPT_DIR/../.." && pwd)/config.env}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing config: $CONFIG_FILE"
  exit 1
fi

if [ -z "${SUITE_MANIFEST:-}" ]; then
  echo "Missing SUITE_MANIFEST. Submit with src/verbatim_eval/submit_verbatim_suite.py."
  exit 1
fi

set -a
source "$CONFIG_FILE"
set +a

source "$CONDA_PROFILE"
conda activate base

mkdir -p "$LOG_ROOT/verbatim_eval"

RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
EVAL_REPLICAS_DIR="${EVAL_REPLICAS_DIR:-${GUTENBERG_REPLICAS_V2_DIR:-${DATASET_ROOT}/gutenberg/4096_tokens_filtered_with_replicas_v2}}"
UNIQUE_EVAL_TOKEN_FILE="${UNIQUE_EVAL_TOKEN_FILE:-${GUTENBERG_UNIQUE_TOKEN_FILE:-${DATASET_ROOT}/gutenberg/4096_tokens_filtered/token.jsonl}}"
REPLICA_BUCKETS_DIR="${REPLICA_BUCKETS_DIR:-${GUTENBERG_REPLICAS_V2_DIR:-${DATASET_ROOT}/gutenberg/4096_tokens_filtered_with_replicas_v2}}"
UNIQUE_EVAL_CACHE_DIR="${UNIQUE_EVAL_CACHE_DIR:-$RESULTS_ROOT/verbatim_eval/cache/gutenberg_unique_repetition_buckets}"
USE_REPLICA_JSONL_EVAL="${USE_REPLICA_JSONL_EVAL:-0}"
REBUILD_UNIQUE_EVAL_CACHE="${REBUILD_UNIQUE_EVAL_CACHE:-0}"

NO_FIM_MODEL_PATH="${NO_FIM_MODEL_PATH:-${RUNS_ROOT}/no-fim-no-dropout/hf}"
FIM_V2_MODEL_PATH="${FIM_V2_MODEL_PATH:-${RUNS_ROOT}/fim-v2-no-dropout/hf}"
FINEWEB_ONLY_MODEL_PATH="${FINEWEB_ONLY_MODEL_PATH:-${RUNS_ROOT}/llama3_2_3B_pretrain_fineweb_only/hf}"

MAX_EXCERPTS="${MAX_EXCERPTS:-0}"
MAX_WINDOWS_PER_EXCERPT="${MAX_WINDOWS_PER_EXCERPT:-5}"
WINDOW_SELECTION="${WINDOW_SELECTION:-uniform}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
PROB_EXTRACTION_TOP_K="${PROB_EXTRACTION_TOP_K:-40}"
PROB_EXTRACTION_TEMPERATURE="${PROB_EXTRACTION_TEMPERATURE:-1.0}"
PROB_EXTRACTION_THRESHOLD="${PROB_EXTRACTION_THRESHOLD:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ATTENTION_BATCH_SIZE="${ATTENTION_BATCH_SIZE:-16}"
ATTENTION_MAX_WINDOWS_PER_EXCERPT="${ATTENTION_MAX_WINDOWS_PER_EXCERPT:-4}"
NUM_GPU_WORKERS="${NUM_GPU_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
EVAL_KINDS="${EVAL_KINDS:-all}"
EXPERIMENTS="${EXPERIMENTS:-direct}"
GENERATION_MODE="${GENERATION_MODE:-auto}"
WALLTIME_MINUTES="${WALLTIME_MINUTES:-90}"
STOP_MARGIN_MINUTES="${STOP_MARGIN_MINUTES:-5}"
MIN_TASK_START_SECONDS="${MIN_TASK_START_SECONDS:-300}"
INCLUDE_TOKEN_IDS="${INCLUDE_TOKEN_IDS:-0}"
INCLUDE_TEXT="${INCLUDE_TEXT:-0}"
INCLUDE_FIM_ANNOTATIONS="${INCLUDE_FIM_ANNOTATIONS:-0}"
INCLUDE_LAYER_METRICS="${INCLUDE_LAYER_METRICS:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo
echo "===== Verbatim debug suite ====="
echo "Manifest: $SUITE_MANIFEST"
echo "Results root: $RESULTS_ROOT"
echo "Unique eval token file: $UNIQUE_EVAL_TOKEN_FILE"
echo "Replica bucket source: $REPLICA_BUCKETS_DIR"
echo "Unique eval cache: $UNIQUE_EVAL_CACHE_DIR"
echo "Legacy replica eval mode: $USE_REPLICA_JSONL_EVAL"
echo "Eval kinds: $EVAL_KINDS"
echo "Experiments: $EXPERIMENTS"
echo "Max excerpts/windows per excerpt: $MAX_EXCERPTS/$MAX_WINDOWS_PER_EXCERPT"
echo "Batch size/GPU workers: direct=$BATCH_SIZE attention=$ATTENTION_BATCH_SIZE workers=$NUM_GPU_WORKERS"
echo "Generation mode: $GENERATION_MODE"

RUNNER_FLAGS=()
if [ "$USE_REPLICA_JSONL_EVAL" = "1" ]; then
  RUNNER_FLAGS+=(--use-replica-jsonl-eval)
fi
if [ "$REBUILD_UNIQUE_EVAL_CACHE" = "1" ]; then
  RUNNER_FLAGS+=(--rebuild-unique-eval-cache)
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" src/verbatim_eval/verbatim_debug_runner.py \
  --manifest "$SUITE_MANIFEST" \
  --results-root "$RESULTS_ROOT" \
  --eval-replicas-dir "$EVAL_REPLICAS_DIR" \
  --unique-eval-token-file "$UNIQUE_EVAL_TOKEN_FILE" \
  --replica-buckets-dir "$REPLICA_BUCKETS_DIR" \
  --unique-eval-cache-dir "$UNIQUE_EVAL_CACHE_DIR" \
  --eval-kinds "$EVAL_KINDS" \
  --experiments "$EXPERIMENTS" \
  --max-excerpts "$MAX_EXCERPTS" \
  --max-windows-per-excerpt "$MAX_WINDOWS_PER_EXCERPT" \
  --attention-max-windows-per-excerpt "$ATTENTION_MAX_WINDOWS_PER_EXCERPT" \
  --window-selection "$WINDOW_SELECTION" \
  --sample-seed "$SAMPLE_SEED" \
  --batch-size "$BATCH_SIZE" \
  --attention-batch-size "$ATTENTION_BATCH_SIZE" \
  --num-gpu-workers "$NUM_GPU_WORKERS" \
  --gpu-ids "$GPU_IDS" \
  --generation-mode "$GENERATION_MODE" \
  --walltime-minutes "$WALLTIME_MINUTES" \
  --stop-margin-minutes "$STOP_MARGIN_MINUTES" \
  --min-task-start-seconds "$MIN_TASK_START_SECONDS" \
  --prob-extraction-top-k "$PROB_EXTRACTION_TOP_K" \
  --prob-extraction-temperature "$PROB_EXTRACTION_TEMPERATURE" \
  --prob-extraction-threshold "$PROB_EXTRACTION_THRESHOLD" \
  --no-fim-model-path "$NO_FIM_MODEL_PATH" \
  --fim-v2-model-path "$FIM_V2_MODEL_PATH" \
  --fineweb-only-model-path "$FINEWEB_ONLY_MODEL_PATH" \
  "${RUNNER_FLAGS[@]}"
