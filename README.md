# Memorization Dynamics of Fill-in-the-Middle Pretraining

[*ICML 2026 Workshop on the Impact of Memorization on Trustworthy Foundation Models*]

[**Paper**](TODO) | [**Checkpoints and Training Logs**](#released-checkpoints)

This repository contains the code accompanying the paper *Memorization Dynamics of Fill-in-the-Middle Pretraining*. It includes data
preparation, Megatron-LM training code, downstream
evaluation, and verbatim memorization probes.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Released Checkpoints](#released-checkpoints)
- [Evaluation](#evaluation)
- [Full Training Reproduction](#full-training-reproduction)
- [Repo Tree](#repo-tree)
- [Citation](#citation)

## Overview

The study asks how fill-in-the-middle pretraining changes memorization dynamics of repeated training samples when the training data, model scale, and repetition schedule are controlled. The pipeline builds:

- FineWeb LTR and FIM background corpora.
- A filtered Gutenberg memorization corpus with repetition buckets
  `1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128`.
- Matched LTR and FIM training mixtures.
- Converted HuggingFace checkpoints for downstream and verbatim evaluation.

Expected external stack:

- NVIDIA Megatron-LM for preprocessing and training.
- Transformer Engine / FlashAttention through the project container.
- Hugging Face `transformers`, `datasets`, `huggingface_hub`, and
  `lm-evaluation-harness`.
- Slurm for the provided launchers.

## Setup

Create a Conda environment for the Python scripts:

```bash
conda env create -f environment.yml
conda activate memfim
```

## Released Checkpoints

Model checkpoints are available on HuggingFace:
- [memfim-fim-3b](https://huggingface.co/tvonarx/memfim-fim-3b)
- [memfim-ltr-3b](https://huggingface.co/tvonarx/memfim-ltr-3b)
- [memfim-fim-1b](https://huggingface.co/tvonarx/memfim-fim-1b)
- [memfim-ltr-1b](https://huggingface.co/tvonarx/memfim-ltr-1b)

Training logs are available on WandB:
- [memorization-study-fim-team/memorization-study-fim](https://wandb.ai/memorization-study-fim-team/memorization-study-fim)

## Evaluation

Given converted or downloaded checkpoints, downstream evaluation can be rerun
without repeating pretraining:

```bash
export HF_CKPT_PATH_NO_FIM=/path/to/ltr_hf_checkpoint
export HF_CKPT_PATH_FIM=/path/to/fim_hf_checkpoint
export LLAMA_TOKENIZER_PATH=/path/to/llama32_tokenizer
export RESULTS_TAG=memfim_downstream_repro

sbatch --export=ALL,HF_CKPT_PATH_NO_FIM,HF_CKPT_PATH_FIM,LLAMA_TOKENIZER_PATH,RESULTS_TAG \
  src/eval/downstream-eval.sh
```

Verbatim memorization evaluation requires the Gutenberg repetition-bucket JSONL
inputs. With those data available and checkpoint paths set:

```bash
export NO_FIM_MODEL_PATH=/path/to/ltr_hf_checkpoint
export FIM_V2_MODEL_PATH=/path/to/fim_hf_checkpoint
export FINEWEB_ONLY_MODEL_PATH=/path/to/fineweb_only_hf_checkpoint

python src/verbatim_eval/submit_verbatim_suite.py \
  --suite core --max-excerpts 256
python src/verbatim_eval/compare_direct_overlap_results.py \
  --suite core --suite-report ltr
python src/verbatim_eval/compare_direct_overlap_results.py \
  --suite core --suite-report native_geometry
python src/verbatim_eval/compare_prefix_rescue_results.py \
  --suite core
python src/verbatim_eval/compare_attention_results.py \
  --suite core --suite-report attention_ltr
python src/verbatim_eval/compare_attention_results.py \
  --suite core --suite-report attention_native_geometry
```

The core suite covers LTR extraction, native geometry, prefix rescue, and
attention reports. Generated summaries and figures are stored under `results/`.

## Full Training Reproduction

Training uses LLaMA 3.2 style models with THD-packed sequences in Megatron-LM:
`micro_batch_size=1`, no padding, EOD-derived `cu_seqlens`, and standard
data-parallel training by default. The main results are the 3B variant; 1B is an ablation.

Some command identifiers use internal slugs: `no_fim` means
LTR, and `fim_v2` means FIM.

The commands below are the Slurm-based workflow used for our runs. Treat them
as a reference recipe: update Slurm headers, container/module setup, filesystem
paths in `config.env`, and resource settings for your own cluster. The
pretraining launchers are calibrated for our hardware, so batch sizes,
walltimes, and partitions may need adjustment. For WandB logging, `.env` should
define `WANDB_API_KEY`, `WANDB_PROJECT`, and `WANDB_ENTITY`.

Reference stage order:

```bash
# FineWeb text -> raw LLaMA MMAP -> matched LTR/FIM variants
STEP1=$(sbatch --parsable src/preprocess_fineweb/create_fineweb_text_jsonl.slurm)
RAW=$(sbatch --parsable --dependency=afterok:${STEP1} src/preprocess_fineweb/tokenize_fineweb_llama.slurm)
sbatch --dependency=afterok:${RAW} src/preprocess_fineweb/build_no_fim_from_llama_bin.slurm
sbatch --dependency=afterok:${RAW} src/preprocess_fineweb/build_fim_from_llama_bin.slurm

# Gutenberg filtering, deduplication, repetition buckets, and FIM preparation
sbatch --export=ALL,PIPELINE_STAGE=build_filter src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=semantic_dedup src/preprocess_gutenberg/gutenberg.slurm
python src/preprocess_gutenberg/build_book_metadata_manifest.py
sbatch --export=ALL,PIPELINE_STAGE=create_replicas src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=create_replicas_v2 src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=prepare_ltr src/preprocess_gutenberg/gutenberg.slurm
sbatch --export=ALL,PIPELINE_STAGE=prepare_fim_v2 src/preprocess_gutenberg/gutenberg.slurm

# Final mixed datasets
LTR_MIX=$(sbatch --parsable --export=ALL,MIX_MODE=no_fim src/mix_datasets/submit_mix.slurm)
sbatch --export=ALL,MIX_MODE=no_fim --dependency=afterok:${LTR_MIX} src/mix_datasets/merge.slurm
FIM_MIX=$(sbatch --parsable --export=ALL,MIX_MODE=fim_v2 src/mix_datasets/submit_mix.slurm)
sbatch --export=ALL,MIX_MODE=fim_v2 --dependency=afterok:${FIM_MIX} src/mix_datasets/merge.slurm

# Training
sbatch --export=ALL,AUTO_JOB_REQUEUE=1 src/pretrain/pretrain_llama_no_fim.slurm
sbatch --export=ALL,AUTO_JOB_REQUEUE=1 src/pretrain/pretrain_llama_fim_v2.slurm
```

Convert Megatron checkpoints to HuggingFace format:

```bash
sbatch --export=ALL,MODEL_VARIANT=all_with_fim_v2,OVERWRITE_HF=1 \
  src/checkpoint/convert_llama_megatron_to_hf.slurm
```

## Repo Tree

```text
config.env                         shared reproducibility config
utils/                             setup helpers
src/prepare_fim.py                 LTR/FIM indexed-dataset writer
src/preprocess_fineweb/            FineWeb preprocessing
src/preprocess_gutenberg/          Gutenberg filtering and repetition buckets
src/mix_datasets/                  final indexed-dataset mixing
src/pretrain/                      Megatron-LM training launchers
src/checkpoint/                    Megatron -> HuggingFace conversion
src/eval/                          downstream lm-eval utilities
src/verbatim_eval/                 memorization and attention probes
results/                           generated summaries, figures, and caches
```

## Citation

If you find our work helpful, please use the following citation:
```bibtex
TODO
```
