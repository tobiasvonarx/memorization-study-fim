# preprocess_gutenberg

Reproduce the Gutenberg memorization corpus and repetition buckets.

## Stages

`gutenberg.slurm` dispatches stages with `PIPELINE_STAGE`:

- `build_filter` - score candidate windows with the FineWeb checkpoint.
- `semantic_dedup` - remove near-duplicates using embedding similarity plus
  token 5-gram Jaccard.
- `create_replicas` - assign excerpts to repetition buckets.
- `create_replicas_v2` - add replica metadata for FIM split randomization.
- `prepare_ltr` - write LTR indexed data.
- `prepare_fim_v2` - write replica-aware FIM indexed data.

## Data Input

The filtering stage reads the English split of the Hugging Face dataset
`manu/project_gutenberg`. The script first looks for cached Arrow shards under
the Gutenberg cache directory, then falls back to `datasets.load_dataset`:

```python
load_dataset("manu/project_gutenberg", split="en", cache_dir=...)
```

To prefetch the dataset into the cache used by the Slurm job:

```bash
python - <<'PY'
import os
from datasets import load_dataset
load_dataset(
    "manu/project_gutenberg",
    split="en",
    cache_dir=os.environ.get(
        "GUTENBERG_CACHE_DIR",
        os.path.join(os.environ["DATASET_ROOT"], "gutenberg", "cache"),
    ),
)
PY
```

`build_book_metadata_manifest.py` separately downloads or reuses the official
Project Gutenberg catalog CSV for book metadata.

## Run

```bash
cd "$REPO_ROOT"

sbatch --export=ALL,PIPELINE_STAGE=build_filter src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=semantic_dedup src/preprocess_gutenberg/gutenberg.slurm
python src/preprocess_gutenberg/build_book_metadata_manifest.py \
  --filtered-dir "$GUTENBERG_FILTERED_DIR"
sbatch --export=ALL,PIPELINE_STAGE=create_replicas src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=create_replicas_v2 src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=prepare_ltr src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=prepare_fim_v2 src/preprocess_gutenberg/gutenberg.slurm
```

To rescore with another FineWeb checkpoint:

```bash
sbatch --export=ALL,PIPELINE_STAGE=build_filter,GUTENBERG_CHECKPOINT_LOAD_DIR=/path/to/checkpoint/root \
  src/preprocess_gutenberg/gutenberg.slurm
```

## Outputs

- filtered manifests: `$GUTENBERG_FILTERED_DIR`
- replicated JSONL: `$GUTENBERG_REPLICAS_DIR`
- FIM replicated JSONL: `$GUTENBERG_REPLICAS_V2_DIR`
- LTR indexed data: `$GUTENBERG_LTR_DIR/tokens.{bin,idx}`
- FIM indexed data: `$GUTENBERG_FIM_V2_DIR/tokens.{bin,idx}`

Current repetition buckets are `1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128`.
