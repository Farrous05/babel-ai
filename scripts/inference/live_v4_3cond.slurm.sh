#!/bin/bash -l

#SBATCH -o .log/live_v4_3cond.out
#SBATCH -e .log/live_v4_3cond.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J live-v4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00

# v4 LIVE TEST -- 3 system-prompt conditions x 48 runs, temperature DISTRIBUTION.
#
# Configs + per-condition plans come from make_v4_livetest_configs.py (run it
# first). Each condition runs the SAME seeds (SEED_OFFSET 0..47, reset per
# condition), so the only thing that differs between conditions is the system
# prompt. Runs in chunks of 8 (full-speed, results flush per chunk).
#
#   nosys   -> vs v3's 47%
#   seen    -> did training-with-system-prompts work (v3 scored 0 here)
#   unseen  -> generalization to a prompt NOT in the training bank

set -euo pipefail

CONDS="${CONDS:-nosys seen unseen}"
CHUNK="${CHUNK:-8}"

module purge
module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
mkdir -p .log
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

START=$(date +%s)
for cond in $CONDS; do
    PLAN=".log/livetest_v4_${cond}_plan.txt"
    [ -s "$PLAN" ] || { echo "MISSING plan $PLAN -- run make_v4_livetest_configs.py"; exit 1; }
    N=$(wc -l < "$PLAN")
    echo "================ CONDITION: $cond  ($N runs) ================"
    done_n=0
    while [ $done_n -lt $N ]; do
        n=$CHUNK; [ $(( done_n + n )) -gt $N ] && n=$(( N - done_n ))
        echo "--- [$cond] runs $done_n..$(( done_n + n - 1 )) of $N (seeds reset per condition)"
        args=$(sed -n "$(( done_n + 1 )),$(( done_n + n ))p" "$PLAN" | tr '\n' ' ')
        # SEED_OFFSET starts at 0 for EACH condition -> same seeds across conditions
        SEED_OFFSET=$done_n python -u src/main.py $args
        done_n=$(( done_n + n ))
    done
done

echo "elapsed $(( ($(date +%s) - START) / 60 )) min"
echo "results in /dais/fs/scratch/fash/results/livetest_v4_{nosys,seen,unseen}"
