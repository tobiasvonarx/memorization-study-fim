# mix_datasets

Reproduce the final Megatron indexed training prefixes by concatenating
processed Gutenberg and FineWeb sources.

## Entry Points

- `submit_mix.slurm` - array launcher for source-copy workers.
- `mix_datasets_worker.py` - copies one source indexed dataset into a shard.
- `merge.slurm` - reducer launcher.
- `mix_datasets_reduce.py` - merges shards into one `.bin/.idx` prefix.

## Modes

- `MIX_MODE=no_fim` - Gutenberg LTR plus FineWeb LTR.
- `MIX_MODE=fim_v2` - Gutenberg replica-aware FIM plus FineWeb FIM.

`MIX_MODE=fim` and `MIX_MODE=hybrid` remain for compatibility with existing
artifacts, but the maintained comparison is LTR vs FIM.

## Run

```bash
cd "$REPO_ROOT"

LTR_MIX=$(sbatch --parsable --export=ALL,MIX_MODE=no_fim src/mix_datasets/submit_mix.slurm)
sbatch --export=ALL,MIX_MODE=no_fim --dependency=afterok:${LTR_MIX} src/mix_datasets/merge.slurm

FIM_MIX=$(sbatch --parsable --export=ALL,MIX_MODE=fim_v2 src/mix_datasets/submit_mix.slurm)
sbatch --export=ALL,MIX_MODE=fim_v2 --dependency=afterok:${FIM_MIX} src/mix_datasets/merge.slurm
```

Override inputs with `MIX_FINEWEB_GLOB`, `MIX_GUTENBERG_PREFIX`,
`MIX_SHARD_DIR`, or `MIX_OUTPUT_PREFIX`.

The mixer copies indexed datasets byte-for-byte. It does not retokenize or edit
EOS/EOD boundaries.
