# preprocess_fineweb

Reproduce the FineWeb side of the training mixture.

## Entry Points

- `create_fineweb_text_jsonl.py` / `.slurm` - FineWeb parquet to text JSONL.
- `tokenize_fineweb_llama.slurm` - Megatron `preprocess_data.py` with the LLaMA
  tokenizer.
- `build_no_fim_from_llama_bin.slurm` - clean LTR documents.
- `build_fim_from_llama_bin.slurm` - matched FineWeb FIM/LTR mixture via
  `src/prepare_fim.py`.

## Data Input

Download the FineWeb 100BT sample parquet shards before running the conversion
stage:

```bash
huggingface-cli download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --local-dir "$FINEWEB_DOWNLOAD_DIR" \
  --include 'sample/100BT/*'
```

`create_fineweb_text_jsonl.slurm` expects those shards under
`$FINEWEB_DOWNLOAD_DIR/sample/100BT`.

## Run

```bash
cd "$REPO_ROOT"

STEP1=$(sbatch --parsable src/preprocess_fineweb/create_fineweb_text_jsonl.slurm)
RAW=$(sbatch --parsable --dependency=afterok:${STEP1} src/preprocess_fineweb/tokenize_fineweb_llama.slurm)
sbatch --dependency=afterok:${RAW} src/preprocess_fineweb/build_no_fim_from_llama_bin.slurm
sbatch --dependency=afterok:${RAW} src/preprocess_fineweb/build_fim_from_llama_bin.slurm
```

## Outputs

- text JSONL: `$FINEWEB_TEXT_JSONL_DIR`
- raw LLaMA MMAP: `$FINEWEB_LLAMA_RAW_DIR/combined_text_document`
- LTR MMAP: `$FINEWEB_NO_FIM_DIR/combined_text_document`
- FIM MMAP: `$FINEWEB_FIM_DIR/combined_text_document`

Both filtered variants use the same source document set. LTR writes
`content + EOD`; FIM writes
`FIM_PREFIX + prefix + FIM_SUFFIX + suffix + FIM_MIDDLE + middle + EOD`.
