#!/bin/bash -l

#SBATCH -o .log/dagger_rollout.out
#SBATCH -e .log/dagger_rollout.err
#SBATCH -D /dais/u/fash/babel-ai
#SBATCH -J dagger-roll
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00

# ===========================================================================
#  DAgger ROLLOUTS -- the current policy self-converses; we record what it does.
#
#  Plan of record: internship-hpc-repo-template/doc/Project/dagger_plan.md
#
#  ⛔ SEQUENCING GATE: do NOT submit this until v5 has landed and v6's length
#     curve is readable, and ASK THE USER FIRST. One H200, shared with v5 and
#     v6; this is the heaviest of the three tracks. Check `squeue -u fash`.
#
#  FIRST:  python scripts/make_dagger_configs.py --iter N --phase train|eval \
#              --model <policy path>
#          (it writes .log/dagger_i{N}_{phase}_{cond}_plan.txt and the configs,
#           and refuses if the model is not in LocalHFModels)
#
#  THEN:   ITER=1 PHASE=train SEED_BASE=0 sbatch scripts/inference/dagger_rollout.slurm.sh
#
#  Options:
#    ITER       DAgger iteration (0 = the v5 starting policy)          [required]
#    PHASE      train | eval                                           [required]
#    SEED_BASE  30 for train, 0 for eval -- MUST match the generator   [required]
#    TAG        result-dir suffix, e.g. closed / open at iteration 1   [optional]
#    CONDS      subset of "nosys seen unseen"                          [default all]
#    CHUNK      concurrent conversations                               [default 8]
#
#  WHY THE SEED SPLIT MATTERS: eval rollouts (seeds 0-29) are the BEFORE/AFTER
#  measurement -- v6 already ran them, so BEFORE costs no GPU. Train rollouts
#  (seeds 30-279) become training data. A pilot on 2026-08-04 nearly trained on
#  the eval seeds because those runs already existed and were convenient; that
#  is exactly what this guard is for. The generator enforces the split; this
#  script just has to pass the matching SEED_BASE through.
#
#  WHY CHUNKS: babel writes a run's folder only when that run COMPLETES, so one
#  big parallel batch means nothing is on disk until the very end -- a wall-clock
#  timeout then loses everything. Chunks also run at full speed: 8 concurrent
#  conversations cost ~2.1 s/turn, 32 cost ~6.6 s/turn (one shared GPU).
#  Measured throughput at CHUNK=8, 28 iterations: ~80 runs/hour.
#
#  SEED_OFFSET is reset per condition (SEED_BASE + position in the plan), so all
#  three conditions see the SAME openers and the only difference between them is
#  the system prompt -- same design as the v4 live test.
# ===========================================================================

set -euo pipefail

ITER="${ITER:?set ITER (DAgger iteration, 0 = v5 starting policy)}"
PHASE="${PHASE:?set PHASE=train or PHASE=eval}"
SEED_BASE="${SEED_BASE:?set SEED_BASE (0 for train, 250 for eval)}"
TAG="${TAG:-}"
CONDS="${CONDS:-nosys seen unseen}"
CHUNK="${CHUNK:-8}"

case "$PHASE" in
    train) [ "$SEED_BASE" -eq 30 ] || { echo "PHASE=train needs SEED_BASE=30"; exit 1; } ;;
    eval)  [ "$SEED_BASE" -eq 0 ]  || { echo "PHASE=eval needs SEED_BASE=0";   exit 1; } ;;
    *)     echo "PHASE must be train or eval"; exit 1 ;;
esac

STEM_TAG=""; [ -n "$TAG" ] && STEM_TAG="_${TAG}"

module purge
module load apptainer/1.5.2
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
mkdir -p .log
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "DAgger rollouts | iter=$ITER phase=$PHASE seed_base=$SEED_BASE tag=${TAG:-none}"
START=$(date +%s)

for cond in $CONDS; do
    PLAN=".log/dagger_i${ITER}${STEM_TAG}_${PHASE}_${cond}_plan.txt"
    [ -s "$PLAN" ] || { echo "MISSING plan $PLAN -- run make_dagger_configs.py first"; exit 1; }
    N=$(wc -l < "$PLAN")
    echo "================ CONDITION: $cond  ($N runs, seeds ${SEED_BASE}..$(( SEED_BASE + N - 1 ))) ================"
    done_n=0
    while [ $done_n -lt $N ]; do
        n=$CHUNK; [ $(( done_n + n )) -gt $N ] && n=$(( N - done_n ))
        echo "--- [$cond] runs $done_n..$(( done_n + n - 1 )) of $N"
        args=$(sed -n "$(( done_n + 1 )),$(( done_n + n ))p" "$PLAN" | tr '\n' ' ')
        # seeds reset per condition -> identical openers across conditions
        SEED_OFFSET=$(( SEED_BASE + done_n )) python -u src/main.py $args
        done_n=$(( done_n + n ))
    done
done

echo "elapsed $(( ($(date +%s) - START) / 60 )) min"
echo "results: /dais/fs/scratch/fash/results/dagger_i${ITER}${STEM_TAG}_${PHASE}_{${CONDS// /,}}"
