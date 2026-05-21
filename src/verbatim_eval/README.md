# verbatim_eval

Suite-based memorization and attention probes for converted LTR, FIM, and
FineWeb-only checkpoints on Gutenberg repetition-bucket inputs.

## Entry Points

- `verbatim_suite.py` - definitions and report arm selection.
- `submit_verbatim_suite.py` - write `manifest.json` / `tasks.jsonl` and submit
  one Slurm runner job.
- `run_verbatim_suite.slurm` - execute selected `(arm, repetition)` tasks.
- `direct_overlap_eval.py` - batched direct extraction evaluator.
- `attention_probe.py` - attention collection for target-token prediction.
- `compare_direct_overlap_results.py` - direct LTR and native-geometry collation.
- `compare_prefix_rescue_results.py` - distractor intervention collation.
- `compare_attention_results.py` - attention collation and plotting.
- `make_ltr_audit_figures.py` - optional LTR appendix audit figures.

## Metrics

- `cooper_p_z` - exact target probability under top-k sampling (`k=40`, `T=1`)
- `cooper_extractable` - `1` when `cooper_p_z >= 0.001`
- `cooper_token_geomean_p_z` - length-normalized target probability
- `cooper_supported_token_rate` - fraction of target tokens in top-k
- `greedy_exact_match` - greedy exact-match baseline

ROUGE-L/LCS, Ref NLL/PPL, token accuracy, and attention partition fields are
further metrics.

## Core Suite

```bash
set -a; source config.env; set +a
cd "$REPO_ROOT"

export NO_FIM_MODEL_PATH=/path/to/ltr_hf_checkpoint
export FIM_V2_MODEL_PATH=/path/to/fim_hf_checkpoint
export FINEWEB_ONLY_MODEL_PATH=/path/to/fineweb_only_hf_checkpoint

"$PYTHON_BIN" src/verbatim_eval/submit_verbatim_suite.py \
  --suite core --max-excerpts 256
```

The default suite includes:

- direct LTR extraction: LTR, FIM, FineWeb-only with `P100 -> M32`
- native FIM geometry: FIM prefix/suffix splits from full-context
  prefix-rescue arms
- prefix rescue: FIM-v2 native prompts under full and distractor interventions
- attention LTR: LTR and FIM with `P100 -> M32`
- attention native geometry: FIM split sweep with `C=100`, `M32`

## Collate

```bash
"$PYTHON_BIN" src/verbatim_eval/compare_direct_overlap_results.py \
  --suite core --suite-report ltr

"$PYTHON_BIN" src/verbatim_eval/compare_direct_overlap_results.py \
  --suite core --suite-report native_geometry

"$PYTHON_BIN" src/verbatim_eval/compare_prefix_rescue_results.py \
  --suite core

"$PYTHON_BIN" src/verbatim_eval/compare_attention_results.py \
  --suite core --suite-report attention_ltr

"$PYTHON_BIN" src/verbatim_eval/compare_attention_results.py \
  --suite core --suite-report attention_native_geometry
```

Collators write `.per_arm.csv`, `.summary.json`, insights, and figures under:

```text
results/verbatim_eval/suites/<suite>/
```
