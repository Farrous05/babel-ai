#!/bin/bash -l

#SBATCH -o .log/live_v6_longrun.out
#SBATCH -e .log/live_v6_longrun.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J live_v6_long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00

# ONE seed, MANY iterations -- replicate a specific short run at length, to watch
# the full arc (collapse -> escape -> re-collapse -> ...) instead of a 28-turn
# snapshot that ends mid-story.
#
# PINNING THE SEED: prompt_fetcher.py picks index = (SEED_OFFSET + local counter)
# % len(corpus). A fresh process starts the counter at 0, so with RUNS=1 the run
# gets EXACTLY index SEED_OFFSET. Verified against seeds_unseen_hgh_500.json:
#     SEED_OFFSET=4 -> eQa1Sp4_0   (was run 14ca4688, never collapsed in 28 turns)
#     SEED_OFFSET=6 -> HIdxm2l_0   (was run 56e6a81a, collapse onset round 18)
#
# Everything else is byte-identical to selfloop_v6_L3.yaml -- same model, same
# temperature/top_p/max_tokens, same history_window 5 -- so the only difference
# from the original run is how long we let it go.
#
#   SEED_OFFSET=4 CONFIG=selfloop_v6_L3_long_eQa1Sp4 sbatch scripts/inference/live_v6_longrun.slurm.sh

set -euo pipefail
RUNS="${RUNS:-1}"
CONFIG="${CONFIG:?set CONFIG}"
SEED_OFFSET="${SEED_OFFSET:?set SEED_OFFSET to pin the seed}"

module purge; module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"; mkdir -p .log
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "=== LONG RUN: $CONFIG | SEED_OFFSET=$SEED_OFFSET | RUNS=$RUNS ==="
args=""; for _ in $(seq 1 "$RUNS"); do args="$args configs/livetest/$CONFIG.yaml"; done
SEED_OFFSET="$SEED_OFFSET" python -u src/main.py $args

echo "done. CHECK the meta.json seed_id matches the intended seed before analysing."
