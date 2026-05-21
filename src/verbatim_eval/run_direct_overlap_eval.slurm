#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=verbatim_legacy_retired
#SBATCH --partition=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/log/verbatim_eval/%x_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/log/verbatim_eval/%x_%A_%a.err

set -euo pipefail

echo "run_direct_overlap_eval.slurm is retired."
echo "Use: /users/\$USER/miniconda3/bin/python src/verbatim_eval/submit_verbatim_suite.py --suite core"
exit 2
