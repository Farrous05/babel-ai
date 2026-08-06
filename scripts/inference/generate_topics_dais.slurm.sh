#!/bin/bash -l

#SBATCH -o .log/generate_topics.out
#SBATCH -e .log/generate_topics.err
#SBATCH -D /u/fash/babel-ai
#SBATCH -J gen-topics

#SBATCH --nodes=1
#SBATCH --cpus-per-task 12   # cluster cap: 1 GPU => max 12 cores, 250GB
#SBATCH --mem 125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00

# Build a 10,000-paragraph TOPIC BANK for the 10k harvest.
#
# WHY 10k: topics are grafted after <end> to teach the pivot. The 10k harvest will
# yield thousands of positives; the current bank is 60 HAND-WRITTEN paragraphs, so
# each would be reused ~100x and the model would memorize 60 recitations instead of
# learning "switch to something unrelated". 10k topics ~= 1 use each -> nothing to
# memorize, and no single topic-change to fixate on.
#
# WHY SHAREGPT SUBJECTS: a hand-picked domain list makes the author's biases the
# topic distribution. ShareGPT's 94k questions are a real, human-written subject
# list. We take the SUBJECT of a real question and have a model write a standalone
# paragraph about it (NOT an answer -- answers reference an absent question).
#
# DISJOINT BY ID from data/seeds_10k.json, so a run seeded from subject X can never
# be handed a "fresh" topic that is also about X.
#
# Only Mixtral is resident (28GB of 141GB), so we push parallelism hard. bge-large
# (dedupe) auto-uses CUDA on this node.

set -euo pipefail

GEN_MODEL="${GEN_MODEL:-mixtral:8x7b-instruct-v0.1-q4_K_M}"
N="${N:-10000}"
OVERGEN="${OVERGEN:-2.0}"     # dedupe + answer-rejection remove a lot
WORKERS="${WORKERS:-16}"
PORT=11434
REPO=/u/fash/internship-hpc-repo-template
SG=$(ls /dais/fs/scratch/$USER/hf/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_unfiltered_cleaned_split.json | head -1)

module purge
module load apptainer/1.5.2

export OLLAMA_MODELS="/dais/fs/scratch/$USER/ollama"
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_NUM_PARALLEL="$WORKERS"   # one model, tiny footprint -> go wide
export OLLAMA_MAX_LOADED_MODELS=1
mkdir -p .log

srun apptainer run --nv -B .:"$HOME",$OLLAMA_MODELS container/ollama.sif \
    > .log/ollama_topics.log 2>&1 &
SERVER_PID=$!
trap 'echo "stopping Ollama"; kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
    curl -sf "http://localhost:$PORT/api/tags" >/dev/null 2>&1 && { echo "Ollama up"; break; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "Ollama died"; exit 1; }
    sleep 5
done
apptainer exec -B $OLLAMA_MODELS container/ollama.sif ollama pull "$GEN_MODEL" || \
    echo "pull skipped - assuming cached"

cd /u/fash/babel-ai
source .venv/bin/activate
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="ollama"
export HF_HOME=/u/fash/.cache/huggingface     # bge-large for dedupe

echo "=== generating $N topics (overgen ${OVERGEN}x, $WORKERS workers) ==="
echo "    subjects: $SG  (excluding data/seeds_10k.json ids)"
python scripts/generate_topics.py \
    --sharegpt "$SG" \
    --exclude data/seeds_10k.json \
    --n "$N" --overgen "$OVERGEN" \
    --model "$GEN_MODEL" --workers "$WORKERS" \
    --out data/topics_10k.json

echo "done -> data/topics_10k.json"
