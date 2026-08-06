#!/bin/bash -l
#SBATCH -o .log/live_multiagent_clean.out
#SBATCH -e .log/live_multiagent_clean.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J multiagent2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:30:00
# Multi-agent reruns on the FILTERED (substantive) seeds. Same seeds each config.
#   selfloop_v4 · duo_v3v3 · duo_v4v4 · duo_v3v4 (v3 first) · duo_v4v3 (v4 first)
set -euo pipefail
RUNS="${RUNS:-10}"
CONFS="${CONFS:-selfloop_v4 duo_v3v3 duo_v4v4 duo_v3v4 duo_v4v3}"
module purge; module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"; mkdir -p .log
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
for cfg in $CONFS; do
    echo "================ $cfg  ($RUNS runs, clean seeds) ================"
    args=""; for _ in $(seq 1 "$RUNS"); do args="$args configs/livetest/$cfg.yaml"; done
    SEED_OFFSET=0 python -u src/main.py $args
done
echo "done -> /dais/fs/scratch/fash/results/livetest_{v4_selfloop,v3v3,v4v4,v3v4,v4v3}_clean"
