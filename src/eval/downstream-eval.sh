#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=24
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/lm_eval/%A/lm_eval.err
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/lm_eval/%A/lm_eval.out
#SBATCH --gpus-per-node=4
#SBATCH --job-name=lm_eval
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

JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p "$LOG_ROOT/lm_eval/${JOB_ID}"

LLAMA_TOKENIZER_PATH="${LLAMA_TOKENIZER_PATH:-${TOKENIZER_ROOT}/llama3_2_3B_tokenizer}"
HF_CKPT_PATH_NO_FIM="${HF_CKPT_PATH_NO_FIM:-${RUNS_ROOT}/llama3_2_3B_pretrain_mix_no_fim/hf}"
HF_CKPT_PATH_FIM="${HF_CKPT_PATH_FIM:-${RUNS_ROOT}/llama3_2_3B_pretrain_mix_fim/hf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$LLAMA_TOKENIZER_PATH}"

RESULTS_TAG="${RESULTS_TAG:-llama3_2_3B_mix_${JOB_ID}}"
RES_ROOT="${RES_ROOT:-${RESULTS_ROOT}/lm_eval/${RESULTS_TAG}}"
RES_PATH_NO_FIM="${RES_PATH_NO_FIM:-${RES_ROOT}/no_fim}"
RES_PATH_FIM="${RES_PATH_FIM:-${RES_ROOT}/fim}"
NO_FIM_RESULTS_JSON="${NO_FIM_RESULTS_JSON:-${RES_PATH_NO_FIM%/}/results_no_fim.json}"
FIM_RESULTS_JSON="${FIM_RESULTS_JSON:-${RES_PATH_FIM%/}/results_fim.json}"
COMPARE_CSV="${COMPARE_CSV:-${RES_ROOT}/compare_fim_vs_no_fim.csv}"
SPIDER_PNG="${SPIDER_PNG:-${RES_ROOT}/spider_fim_vs_no_fim.png}"

TASKS="${TASKS:-hellaswag,mmlu,winogrande,wikitext,arc_easy,arc_challenge,piqa,commonsense_qa}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MODEL_ARGS_EXTRA="${MODEL_ARGS_EXTRA:-dtype=bfloat16}"
SETUP_LMEVAL="${SETUP_LMEVAL:-0}"
GENERATE_COMPARISON="${GENERATE_COMPARISON:-1}"
GENERATE_SPIDER="${GENERATE_SPIDER:-1}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
SKIP_INPUT_CHECKS="${SKIP_INPUT_CHECKS:-0}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${SCRATCH_ROOT}/data/hf_cache}"
WARMUP_LMEVAL_DATASETS="${WARMUP_LMEVAL_DATASETS:-1}"
LMEVAL_OFFLINE_AFTER_WARMUP="${LMEVAL_OFFLINE_AFTER_WARMUP:-1}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-4}"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "Container image not found: $IMAGE_PATH"
  exit 1
fi
if [ "$SKIP_INPUT_CHECKS" != "1" ] && [ "$SETUP_LMEVAL" != "1" ] && [ ! -d "$LMEVAL_DIR" ]; then
  echo "lm-evaluation-harness not found at ${LMEVAL_DIR}"
  echo "Run src/eval/convert-and-lm-eval.sh first, or set SETUP_LMEVAL=1."
  exit 1
fi
if [ "$SKIP_INPUT_CHECKS" != "1" ] && [ ! -d "$HF_CKPT_PATH_NO_FIM" ]; then
  echo "Missing no-FIM HF checkpoint: $HF_CKPT_PATH_NO_FIM"
  echo "Convert first with: sbatch --export=ALL,MODEL_VARIANT=all,OVERWRITE_HF=1 src/checkpoint/convert_llama_megatron_to_hf.slurm"
  exit 1
fi
if [ "$SKIP_INPUT_CHECKS" != "1" ] && [ ! -d "$HF_CKPT_PATH_FIM" ]; then
  echo "Missing FIM HF checkpoint: $HF_CKPT_PATH_FIM"
  echo "Convert first with: sbatch --export=ALL,MODEL_VARIANT=all,OVERWRITE_HF=1 src/checkpoint/convert_llama_megatron_to_hf.slurm"
  exit 1
fi
if [ "$SKIP_INPUT_CHECKS" != "1" ] && [ ! -d "$TOKENIZER_PATH" ]; then
  echo "Missing tokenizer path: $TOKENIZER_PATH"
  exit 1
fi

mkdir -p "$RES_PATH_NO_FIM" "$RES_PATH_FIM" "$(dirname "$COMPARE_CSV")" "$(dirname "$SPIDER_PNG")"

echo "=========================================="
echo "LLaMA downstream lm-eval"
echo "no-FIM checkpoint: $HF_CKPT_PATH_NO_FIM"
echo "FIM checkpoint:    $HF_CKPT_PATH_FIM"
echo "Tokenizer:         $TOKENIZER_PATH"
echo "Harness:           $LMEVAL_DIR"
echo "Tasks:             $TASKS"
echo "Batch size:        $BATCH_SIZE"
echo "Results root:      $RES_ROOT"
echo "Setup harness:     $SETUP_LMEVAL"
echo "HF cache root:     $HF_CACHE_ROOT"
echo "Dataset warmup:    $WARMUP_LMEVAL_DATASETS"
echo "Offline eval:      $LMEVAL_OFFLINE_AFTER_WARMUP"
echo "Accelerate procs:  $ACCELERATE_NUM_PROCESSES"
echo "Dry run:           $DRY_RUN_ONLY"
echo "=========================================="

if [ "$DRY_RUN_ONLY" = "1" ]; then
  exit 0
fi

if [ "$SETUP_LMEVAL" = "1" ]; then
  if [ ! -d "$LMEVAL_DIR" ]; then
    echo "Cloning lm-evaluation-harness into $LMEVAL_DIR"
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$LMEVAL_DIR"
  else
    echo "Updating lm-evaluation-harness in $LMEVAL_DIR"
    git -C "$LMEVAL_DIR" pull --ff-only
  fi
fi

CONTAINER_OPTS=(
  --container-image="$IMAGE_PATH"
  --no-container-mount-home
  --container-mounts="$CONTAINER_MOUNTS"
  --container-workdir="$LMEVAL_DIR"
)

mkdir -p "$HF_CACHE_ROOT"/{hub,datasets,transformers}
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers"
export HF_ALLOW_CODE_EVAL="1"
export CONFIRM_RUN_UNSAFE_CODE="1"
export HF_HUB_DISABLE_TELEMETRY="1"
export TASKS

if [ -n "$MODEL_ARGS_EXTRA" ]; then
  MODEL_ARGS_NO_FIM="pretrained=${HF_CKPT_PATH_NO_FIM},tokenizer=${TOKENIZER_PATH},${MODEL_ARGS_EXTRA}"
  MODEL_ARGS_FIM="pretrained=${HF_CKPT_PATH_FIM},tokenizer=${TOKENIZER_PATH},${MODEL_ARGS_EXTRA}"
else
  MODEL_ARGS_NO_FIM="pretrained=${HF_CKPT_PATH_NO_FIM},tokenizer=${TOKENIZER_PATH}"
  MODEL_ARGS_FIM="pretrained=${HF_CKPT_PATH_FIM},tokenizer=${TOKENIZER_PATH}"
fi

srun "${CONTAINER_OPTS[@]}" bash -lc '
  set -euo pipefail

  if [ "'"$SETUP_LMEVAL"'" = "1" ]; then
    python -m pip install "lm_eval[hf]"
    python -m pip install -e .
  fi

  if [ "'"$WARMUP_LMEVAL_DATASETS"'" = "1" ]; then
    echo "Warming lm-eval task datasets on one process"
    python - <<'"'"'PY'"'"'
import os

from lm_eval.tasks import TaskManager

tasks = [task.strip() for task in os.environ["TASKS"].split(",") if task.strip()]
print("Loading tasks:", ",".join(tasks), flush=True)
loaded = TaskManager().load(tasks)
print("Warmed {} leaf tasks".format(len(loaded["tasks"])), flush=True)
PY
  fi

  if [ "'"$LMEVAL_OFFLINE_AFTER_WARMUP"'" = "1" ]; then
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
  fi

  rm -f "'"$NO_FIM_RESULTS_JSON"'" "'"$FIM_RESULTS_JSON"'"

  resolve_lmeval_output() {
    local expected="$1"
    local dir
    local stem
    local resolved
    dir="$(dirname "$expected")"
    stem="$(basename "$expected" .json)"

    if [ -f "$expected" ]; then
      printf "%s\n" "$expected"
      return 0
    fi

    resolved="$(
      find "$dir" -maxdepth 1 -type f -name "${stem}*.json" -printf "%T@ %p\n" \
        | sort -nr \
        | awk "NR == 1 { print substr(\$0, index(\$0, \$2)) }"
    )"
    if [ -z "$resolved" ]; then
      echo "Missing lm-eval output. Expected $expected or a timestamped ${stem}*.json in $dir" >&2
      return 1
    fi

    cp "$resolved" "$expected"
    echo "Copied timestamped lm-eval output to stable path: $expected" >&2
    printf "%s\n" "$expected"
  }

  accelerate launch --num_processes "'"$ACCELERATE_NUM_PROCESSES"'" -m lm_eval --model hf \
    --model_args "'"$MODEL_ARGS_NO_FIM"'" \
    --tasks "'"$TASKS"'" \
    --batch_size "'"$BATCH_SIZE"'" \
    --output_path "'"$NO_FIM_RESULTS_JSON"'"
  NO_FIM_RESULTS_FOR_COMPARE="$(resolve_lmeval_output "'"$NO_FIM_RESULTS_JSON"'")"

  accelerate launch --num_processes "'"$ACCELERATE_NUM_PROCESSES"'" -m lm_eval --model hf \
    --model_args "'"$MODEL_ARGS_FIM"'" \
    --tasks "'"$TASKS"'" \
    --batch_size "'"$BATCH_SIZE"'" \
    --output_path "'"$FIM_RESULTS_JSON"'"
  FIM_RESULTS_FOR_COMPARE="$(resolve_lmeval_output "'"$FIM_RESULTS_JSON"'")"

  if [ "'"$GENERATE_COMPARISON"'" = "1" ]; then
    python "'"$REPO_ROOT"'/src/eval/compare_eval.py" \
      --fim "$FIM_RESULTS_FOR_COMPARE" \
      --no-fim "$NO_FIM_RESULTS_FOR_COMPARE" \
      > "'"$COMPARE_CSV"'"
  fi

  if [ "'"$GENERATE_SPIDER"'" = "1" ]; then
    if python -c "import matplotlib" >/dev/null 2>&1; then
      python "'"$REPO_ROOT"'/src/eval/plot_benchmark_spider.py" \
        --fim-json "$FIM_RESULTS_FOR_COMPARE" \
        --no-fim-json "$NO_FIM_RESULTS_FOR_COMPARE" \
        --output "'"$SPIDER_PNG"'"
    else
      echo "Skipping spider plot because matplotlib is not available in the eval environment." >&2
    fi
  fi
'

echo "Saved no-FIM results: $NO_FIM_RESULTS_JSON"
echo "Saved FIM results:    $FIM_RESULTS_JSON"
if [ "$GENERATE_COMPARISON" = "1" ]; then
  echo "Saved comparison CSV: $COMPARE_CSV"
fi
if [ "$GENERATE_SPIDER" = "1" ]; then
  echo "Saved spider plot:    $SPIDER_PNG"
fi
