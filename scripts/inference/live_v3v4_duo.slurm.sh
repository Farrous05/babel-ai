#!/bin/bash -l

#SBATCH -o .log/live_v3v4_duo.out
#SBATCH -e .log/live_v3v4_duo.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J v3v4-duo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00

# ONE cross-model conversation: v3 (agent A) and v4 (agent B) talk to each other.
# Both models load in-process; watch which one fires <end> under a system prompt.
# RUNS defaults to 1 (a single dialogue); bump it to sample a few seeds.

set -euo pipefail

RUNS="${RUNS:-1}"
CONFIG="configs/livetest/v3_vs_v4_duo.yaml"

module purge
module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
mkdir -p .log
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "v3 vs v4 duo: $RUNS conversation(s), config=$CONFIG"
args=""; for _ in $(seq 1 "$RUNS"); do args="$args $CONFIG"; done
SEED_OFFSET="${SEED_OFFSET:-0}" python -u src/main.py $args

echo "done -> /dais/fs/scratch/fash/results/livetest_v3v4_duo"
