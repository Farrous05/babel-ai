#!/bin/bash -l

#SBATCH -o .log/harvest_10k.out
#SBATCH -e .log/harvest_10k.err
#SBATCH -D /u/fash/babel-ai
#SBATCH -J qwen10k

#SBATCH --nodes=1
#SBATCH --cpus-per-task 12   # cluster cap: 1 GPU => max 12 cores, 250GB
#SBATCH --mem 200GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

# 10k-run Qwen2.5-3B collapse harvest.
# Temperature ~ normal(1.0, 0.2) truncated to [0.2, 1.2]; 15 iterations.
#
# WHY IT IS SHAPED LIKE THIS (do not "simplify" without reading):
#
# 1. BATCHED, NOT ONE BIG CALL. `main.py cfg cfg cfg...` does
#    asyncio.gather() over ALL configs at once with no cap. Passing 10k configs
#    would build 10k coroutines/configs up front. We feed it CHUNK at a time.
#
# 2. MULTIPLE PROCESSES, NOT MORE THREADS. Each run is asyncio.to_thread(), and
#    Python's default executor caps at min(32, cpu_count+4) -- so threads are
#    capped anyway. More importantly SimilarityAnalyzer._embed_lock is a CLASS
#    attribute: every thread in a process serializes on embedding. Only separate
#    PROCESSES give the analyzer real parallelism (own lock, own models).
#
# 3. TEMPERATURE DISTRIBUTION VIA A PLAN FILE. Babel has no per-run temperature
#    override, so a distribution has to become one config per grid temperature
#    plus a shuffled plan naming the config for each of the TOTAL runs. Build it
#    with scripts/make_temp_configs.py; the plan is shuffled, so any contiguous
#    slice is an unbiased sample and workers can just take slices.
#
# 4. 15 ITERATIONS. We harvest COLLAPSE samples only -- healthy negatives are
#    GENERATED (91% yield), not mined from late-collapse runs, so the old
#    24-iteration budget bought nothing. Measured on 100 pilot runs: onset+3
#    <= 15 for 80% of runs at 62% of the compute -> ~29% more usable positives
#    per GPU-hour. Cost: loses the ~10% that onset at 16+.
#
# 5. SEED_OFFSET => 1 RUN PER SEED. Each Experiment builds its own fetcher and
#    draws ONE conversation, so plain random.choice samples with replacement:
#    10k draws over a 10k corpus touch only ~63% of it. Each chunk exports a
#    distinct SEED_OFFSET; the fetcher walks offset+counter, so the 10,000 runs
#    consume the 10,000 seeds exactly once. Requires the SEED_OFFSET branch in
#    babel_ai/prompt_fetcher.py.
#
# Prereq: analyzer embedding cache must be present (9.3x); without it the
# analyzer alone is ~111h for 10k x 20 and this job cannot finish in 24h.

set -euo pipefail

# ---- knobs -----------------------------------------------------------------
PLAN="${PLAN:-.log/harvest_plan.txt}"   # one config path per run (see above)
PROCS="${PROCS:-6}"    # parallel main.py processes (analyzer parallelism)
CHUNK="${CHUNK:-50}"   # conversations per main.py invocation (bounds memory)
MODEL=qwen2.5:3b-instruct-fp16
OUT="${OUT:-/dais/fs/scratch/$USER/results/qwen10k}"
# TOTAL comes from the plan, so the plan is the single source of truth.
TOTAL=$(grep -c . "$PLAN")

# --- PARALLEL-JOB SLICING ---------------------------------------------------
# To split the 10k harvest across N GPUs, each job processes the plan slice
# [JOB_START, JOB_END). SEED_OFFSET stays the GLOBAL plan index, so seeds never
# collide across jobs. Defaults = the whole plan (single-job behaviour).
JOB_START="${JOB_START:-0}"
JOB_END="${JOB_END:-$TOTAL}"
JOB_TAG="${JOB_TAG:-0}"       # distinguishes co-located jobs' logs + ollama port
# Per-job port so two jobs on the SAME node don't collide on Ollama's default.
PORT="${PORT:-$(( 11434 + JOB_TAG ))}"
# ---------------------------------------------------------------------------

module purge
module load apptainer/1.5.2

export OLLAMA_MODELS="/dais/fs/scratch/$USER/ollama"
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_NUM_PARALLEL=32     # was 8; a 3B barely dents a 141GB H200
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_HOST="0.0.0.0:$PORT"   # serve on this job's port
mkdir -p .log "$OUT"

# 1. serve Qwen locally (per-job log so parallel jobs don't clobber each other).
#    OLLAMA_HOST/MODELS are passed EXPLICITLY via --env: a plain `export` does
#    NOT reliably reach the process inside apptainer, so without this two
#    co-located jobs both fall back to the default port 11434 and collide
#    ("bind: address already in use"). Per-job PORT is what lets jobs share a node.
srun apptainer run --nv \
    --env OLLAMA_HOST="0.0.0.0:$PORT" \
    --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    --env OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" \
    --env OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL" \
    --env OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS" \
    -B .:"$HOME",$OLLAMA_MODELS container/ollama.sif \
    > ".log/ollama_10k_j${JOB_TAG}.log" 2>&1 &
SERVER_PID=$!
trap 'echo "stopping Ollama"; kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
    curl -sf "http://localhost:$PORT/api/tags" >/dev/null 2>&1 && { echo "Ollama up (port $PORT)"; break; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "Ollama died - see .log/ollama_10k_j${JOB_TAG}.log"; exit 1; }
    sleep 5
done
apptainer exec --env OLLAMA_HOST="0.0.0.0:$PORT" --env OLLAMA_MODELS="$OLLAMA_MODELS" \
    -B $OLLAMA_MODELS container/ollama.sif ollama pull "$MODEL" || \
    echo "pull skipped - assuming cached"

source .venv/bin/activate
export QWEN_2_5_3B_BASE_URL="http://localhost:$PORT/v1/"
export QWEN_2_5_3B_API_KEY="ollama"

JOB_TOTAL=$(( JOB_END - JOB_START ))
PER_PROC=$(( (JOB_TOTAL + PROCS - 1) / PROCS ))   # ceil, so nothing is dropped
echo "job $JOB_TAG: runs [$JOB_START,$JOB_END) = $JOB_TOTAL of $TOTAL, port $PORT"
echo "  $PROCS procs x ~$PER_PROC each, chunks of $CHUNK"
START=$(date +%s)

# 2. PROCS workers. Worker p owns the contiguous plan slice
#    [JOB_START + p*PER_PROC, ...), clamped to JOB_END. Within it we step CHUNK
#    at a time; the chunk's GLOBAL start index is BOTH the plan offset and the
#    SEED_OFFSET, so run i gets seed index start+i -- globally unique across ALL
#    procs AND all parallel jobs (each job owns a disjoint [JOB_START,JOB_END)).
WPIDS=()
for p in $(seq 0 $(( PROCS - 1 ))); do
  (
    start=$(( JOB_START + p * PER_PROC ))
    end=$(( start + PER_PROC )); [ $end -gt $JOB_END ] && end=$JOB_END
    while [ $start -lt $end ]; do
      n=$CHUNK; [ $(( start + n )) -gt $end ] && n=$(( end - start ))
      # plan is 0-indexed here, sed is 1-indexed
      args=$(sed -n "$(( start + 1 )),$(( start + n ))p" "$PLAN" | tr '\n' ' ')
      SEED_OFFSET=$start python src/main.py $args \
        >> ".log/worker_j${JOB_TAG}_${p}.log" 2>&1 || \
        echo "worker j${JOB_TAG}_$p chunk @${start} failed (continuing)" \
          >> ".log/worker_j${JOB_TAG}_${p}.log"
      start=$(( start + n ))
      echo "job $JOB_TAG worker $p: $(( start - JOB_START - p * PER_PROC ))/$(( end - JOB_START - p * PER_PROC ))"
    done
  ) &
  WPIDS+=($!)
done
# Wait ONLY for the workers, NOT the backgrounded Ollama server (which never
# exits on its own). A bare `wait` here hangs the job forever after the runs are
# done -- it blocks on Ollama until wall-time/scancel (the smoke-test symptom).
wait "${WPIDS[@]}"

END=$(date +%s)
N=$(find "$OUT" -name "*_meta.json" 2>/dev/null | wc -l)
echo "harvest done: $N runs in $(( (END-START)/60 )) min -> $OUT"
echo "rate: $(python -c "print(f'{$N/max(($END-$START)/3600,1e-9):.0f} runs/hour')")"
