#!/bin/bash -l
#SBATCH -o .log/diagnose_end.out
#SBATCH -e .log/diagnose_end.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J diag-end
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:15:00

set -euo pipefail
module purge
module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -u scripts/inference/diagnose_end_firing.py
