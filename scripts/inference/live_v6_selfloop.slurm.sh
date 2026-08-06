#!/bin/bash -l

#SBATCH -o .log/live_v6_selfloop.out
#SBATCH -e .log/live_v6_selfloop.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J live_v6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00

# v6 LIVE TEST -- the actual gate. Runs ONE v6 variant in the self-loop on the
# same unseen [H,G,H] seeds v4/v5/v5.1 were run on, so transcripts are directly
# comparable and a blind read can be done on paired seeds.
#
# WHY THIS AND NOT THE OFFLINE EVAL: doc/Project/v51_blind_read_for_v6.md is
# explicit that v5.1's offline eval was PERFECT (AUC 1.0) and missed the
# regression entirely, because offline only tests the fire DECISION, never live
# generation. The "." / empty collapse only appears here.
#
# SEED_OFFSET=0 for every variant so all of them see seeds 0..RUNS-1 -- the
# differences are the models, not the openers.
#
# One variant per job so the six run in PARALLEL (8 H200s/node, MaxJobs=100):
#   CONFIG=selfloop_v6_L9-bh sbatch scripts/inference/live_v6_selfloop.slurm.sh

set -euo pipefail
RUNS="${RUNS:-10}"
CONFIG="${CONFIG:-selfloop_v6_L9-bh}"

module purge; module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"; mkdir -p .log
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "=========== v6 SELF-LOOP: $CONFIG  ($RUNS conversations) ==========="
args=""; for _ in $(seq 1 "$RUNS"); do args="$args configs/livetest/$CONFIG.yaml"; done
SEED_OFFSET=0 python -u src/main.py $args

echo "done -> see output_dir in configs/livetest/$CONFIG.yaml"
echo ""
echo "NEXT: scripts/measure_recovery_length_v6.py --run-root <output_dir>=<label>"
echo "      then a BLIND READ vs v4 on the same seeds. The fire/empty/loop table"
echo "      is necessary but NOT sufficient -- v5.1 passed it and read worse."
