# checkpoint

Reproducible conversion from Megatron distributed checkpoints to Hugging Face
model directories.

## Inputs

- Megatron checkpoint root under `$RUNS_ROOT`.
- LLaMA tokenizer path in `$LLAMA_TOKENIZER_PATH`.
- Project sentinel ids from `src/prepare_fim.py`.

## Convert

```bash
cd "$REPO_ROOT"
sbatch --export=ALL,MODEL_VARIANT=all_with_fim_v2,OVERWRITE_HF=1 \
  src/checkpoint/convert_llama_megatron_to_hf.slurm
```

Useful variants: `no_fim`, `fim_v2`, `fineweb_only`, `no_fim_1B`,
`fim_v2_1B`, `all_with_fim_v2`, `all_1B`.

Use `DRY_RUN_ONLY=1` to print resolved paths without writing weights.

## Outputs

Converted checkpoints are written under `$RUNS_ROOT/.../hf` unless overridden by
the launcher environment. Each output includes `fim_tokens.json` so evaluation
code can recover the FIM sentinel ids.
