#!/bin/bash -l

#SBATCH -o .log/live_multiagent.out
#SBATCH -e .log/live_multiagent.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J multiagent
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00

# TWO MODELS TALKING TO EACH OTHER -- the most realistic (non-self-loop) test, and
# the most compelling result so far. Three pairings, SAME seeds each (SEED_OFFSET
# reset per pairing), so differences are the models, not the openers:
#   v3v3  -- two v3s        (baseline: do they escape at all, how do they pivot)
#   v4v4  -- two v4s        (both improved: escape rate + pivot quality)
#   v3v4  -- one of each    (side by side in one conversation)
# babel records agent_id per turn, so the transcript says who said what.

set -euo pipefail
RUNS="${RUNS:-6}"
PAIRINGS="${PAIRINGS:-duo_v3v3 duo_v4v4 v3_vs_v4_duo}"

module purge; module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"; mkdir -p .log
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

for cfg in $PAIRINGS; do
    echo "================ PAIRING: $cfg  ($RUNS conversations) ================"
    args=""; for _ in $(seq 1 "$RUNS"); do args="$args configs/livetest/$cfg.yaml"; done
    SEED_OFFSET=0 python -u src/main.py $args        # same seeds 0..RUNS-1 each pairing
done
echo "done -> /dais/fs/scratch/fash/results/livetest_duo_{v3v3,v4v4} + livetest_v3v4_duo"
