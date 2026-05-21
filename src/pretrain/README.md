# pretrain

Reproduce Megatron-LM LLaMA 3.2 training runs with THD-packed sequences.

## Entry Points

- `pretrain_llama_packed.py` - Megatron GPT training wrapper with
  EOD-derived `PackedSeqParams`.
- `pretrain_llama_common.sh` - shared Slurm launcher core.
- `pretrain_llama_fineweb_only.slurm` - FineWeb-only baseline.
- `pretrain_llama_no_fim.slurm` - LTR mixture run.
- `pretrain_llama_fim_v2.slurm` - FIM mixture run.
- `pretrain_llama_no_fim_1B.slurm` / `pretrain_llama_fim_v2_1B.slurm` - 1B ablations.

## Stack and Defaults

- Megatron-LM with Transformer Engine / FlashAttention support.
- THD packing, `micro_batch_size=1`, no padding.
- Sequence length `16384`, global batch size `2048`.
- `TP=1`, `PP=1`, data parallel by default.
- Cosine LR schedule `3e-4` to `3e-5`.
- `TRAIN_TOKENS_OR_ITERS=0` infers one epoch from dataset size.

## Run

```bash
cd "$REPO_ROOT"

# smoke test
sbatch --nodes=1 --ntasks=1 --gpus-per-node=4 --time=00:30:00 --partition=debug \
  --export=ALL,TRAIN_TOKENS_OR_ITERS=65536,SAVE_INTERVAL=1,OUTPUT_BASEPATH="$SCRATCH_ROOT/runs/llama3_smoke" \
  src/pretrain/pretrain_llama_fineweb_only.slurm

# maintained comparison
sbatch --export=ALL,AUTO_JOB_REQUEUE=1 src/pretrain/pretrain_llama_no_fim.slurm
sbatch --export=ALL,AUTO_JOB_REQUEUE=1 src/pretrain/pretrain_llama_fim_v2.slurm
```

Use `DRY_RUN_ONLY=1` to print resolved paths. Enable W&B with
`WANDB_ENABLE=true` and `WANDB_PROJECT_NAME=...`.
