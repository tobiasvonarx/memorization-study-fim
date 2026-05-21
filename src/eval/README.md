# eval

Reproduce downstream quality-control evaluations from Hugging Face checkpoints.

## Entry Points

- `downstream-eval.sh` - run the lm-eval suite.
- `convert-and-lm-eval.sh` - convert missing checkpoints first, then evaluate.
- `compare_eval.py` - compare two lm-eval JSON result files.
- `plot_benchmark_spider.py` - generate the benchmark spider plot.

## Run From Existing Checkpoints

```bash
cd "$REPO_ROOT"

export HF_CKPT_PATH_NO_FIM=/path/to/ltr_hf_checkpoint
export HF_CKPT_PATH_FIM=/path/to/fim_hf_checkpoint
export LLAMA_TOKENIZER_PATH=/path/to/llama32_tokenizer
export RESULTS_TAG=memfim_downstream_repro

sbatch --export=ALL,HF_CKPT_PATH_NO_FIM,HF_CKPT_PATH_FIM,LLAMA_TOKENIZER_PATH,RESULTS_TAG \
  src/eval/downstream-eval.sh
```

Outputs land in `$RESULTS_ROOT/lm_eval/$RESULTS_TAG/`. Set `HF_TOKEN` if task
dataset downloads need authenticated Hugging Face access.
