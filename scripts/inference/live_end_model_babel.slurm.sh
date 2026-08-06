#!/bin/bash -l

#SBATCH -o .log/babel_live_history.out
#SBATCH -e .log/babel_live_history.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J babel-live

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00

# ===========================================================================
#  THE TRAINED <end> MODEL, SELF-CONVERSING INSIDE BABEL-AI
#
#  JUST RUN:   sbatch scripts/inference/live_end_model_babel.slurm.sh
#  WATCH:      tail -f .log/babel_live.err
#
#  Options:
#    sbatch --export=ALL,RUNS=32 ...        # how many conversations (default 8)
#    sbatch --export=ALL,SEED_OFFSET=100 .. # start at a different seed
#
#  WHY BABEL-AI AND NOT MY OWN HARNESS
#  My standalone harness (internship repo, live_self_conversation.py) showed the
#  escape once, but it does not reliably reproduce collapse -- base collapsed 98%
#  of the time in the harvest and 0% in that harness. babel-ai IS the harness that
#  produced the 9,349 collapses, and it also computes the similarity analyzer,
#  the collapse detector, the recovery metric and a per-run PDF. Running here is
#  what turns "it escaped once" into "it escapes N% of its own collapses".
#
#  Model is loaded IN-PROCESS (provider local_hf) -- no vLLM, no HTTP, no second
#  job to babysit. See src/api/local_hf.py.
#
#  SEEDS: data/seeds_unseen_500.json, openers held out of seeds_10k -- so nothing
#  the model sees here was in its training data. SEED_OFFSET makes each run take a
#  DIFFERENT opener (without it babel picks at random and repeats).
# ===========================================================================

set -euo pipefail

RUNS="${RUNS:-32}"
SEED_OFFSET="${SEED_OFFSET:-0}"
CONFIG="${CONFIG:-configs/livetest/end_model_selfloop.yaml}"
PLAN=".log/livetest_plan.txt"

module purge
module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
mkdir -p .log

# src/main.py takes one config path per run; N copies = N conversations.
: > "$PLAN"
for _ in $(seq 1 "$RUNS"); do echo "$CONFIG" >> "$PLAN"; done

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "runs=$RUNS  seed_offset=$SEED_OFFSET  config=$CONFIG"
START=$(date +%s)

# Run in CHUNKS, not all at once. babel writes a run's folder only when that
# run COMPLETES, so one big parallel batch means nothing is on disk until the
# very end -- a wall-clock timeout then loses everything. Chunks also run at
# full speed: 8 concurrent conversations cost ~2.1 s/turn, 32 cost ~6.6 s/turn,
# because they all share one GPU. Each chunk flushes its results before the
# next begins.
CHUNK="${CHUNK:-8}"
done_n=0
while [ $done_n -lt $RUNS ]; do
    n=$CHUNK; [ $(( done_n + n )) -gt $RUNS ] && n=$(( RUNS - done_n ))
    echo "--- chunk: runs $done_n..$(( done_n + n - 1 )) of $RUNS"
    args=$(head -n "$n" "$PLAN" | tr '\n' ' ')
    SEED_OFFSET=$(( SEED_OFFSET + done_n )) python -u src/main.py $args
    done_n=$(( done_n + n ))
done

echo "elapsed $(( ($(date +%s) - START) / 60 )) min"
echo "results: $(grep -m1 output_dir "$CONFIG" | awk '{print $2}')"
