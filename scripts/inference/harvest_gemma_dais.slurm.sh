#!/bin/bash -l

#SBATCH -o .log/harvest.out
#SBATCH -e .log/harvest.err
#SBATCH -D /u/fash/babel-ai
#SBATCH -J gemma_harvest

#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --mem 125GB

#SBATCH --partition="gpu1"        # gpu1 for real runs; swap to a debug partition first
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1

# Wall clock limit (max 24h):
#SBATCH --time=04:00:00

# Harvest gemma self-loop conversations, following the company Ollama guide:
# start Ollama in the container, then run the babel-ai harvest against it over
# localhost. One job:
#   1. start the Ollama server (container) with models cached on scratch
#   2. wait for it, ensure gemma4:31b-it-bf16 is present
#   3. run NUM_CONVOS babel-ai self-loops concurrently against :11434/v1
#   4. stop the server
#
# One-time prereqs:
#   * build the image:  sbatch scripts/container/build-ollama-apptainer-dais.sh
#   * install babel-ai deps so ./.venv exists (poetry install  OR  uv sync)
#   * PRE-PULL the model to scratch on a node with internet (login node):
#       export OLLAMA_MODELS=/dais/fs/scratch/$USER/ollama
#       apptainer exec --nv -B $OLLAMA_MODELS container/ollama.sif \
#           bash -c 'ollama serve & sleep 5; ollama pull gemma4:31b-it-bf16'
#     (the runtime pull below is only a fallback and needs internet on the node)

set -euo pipefail

# ---- config ----------------------------------------------------------------
MODEL=gemma4:31b-it-bf16          # bf16 full precision (matches the fine-tune target)
CONFIG=configs/gemma_collapse.yaml
PORT=11434
NUM_CONVOS=4
# ---------------------------------------------------------------------------

module purge
module load apptainer

export OLLAMA_MODELS="/dais/fs/scratch/$USER/ollama"
# Defaults are too small: 4096 truncates long self-loops, NUM_PARALLEL=1
# serializes our concurrent conversations. Raise both (propagate into the
# container since we don't use --cleanenv).
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_NUM_PARALLEL=8
mkdir -p "$OLLAMA_MODELS" .log

# 1. start the Ollama server (OpenAI-compatible at :$PORT/v1)
srun apptainer run --nv -B .:"$HOME",$OLLAMA_MODELS container/ollama.sif \
    > .log/ollama_server.log 2>&1 &
SERVER_PID=$!
trap 'echo "stopping Ollama"; kill $SERVER_PID 2>/dev/null || true' EXIT

# 2. wait for the server, then make sure the model is present
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$PORT/api/tags" >/dev/null 2>&1; then
        echo "Ollama up after ${i} checks"; break
    fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "Ollama died - see .log/ollama_server.log"; exit 1; }
    sleep 5
done
# no-op if already pulled to scratch; needs internet otherwise
apptainer exec -B $OLLAMA_MODELS container/ollama.sif ollama pull "$MODEL" || \
    echo "pull skipped/failed - assuming model already cached on scratch"

# 3. run the harvest (babel-ai client) via its host venv. Repeat the config
#    NUM_CONVOS times so babel-ai runs them concurrently; Ollama serves them
#    together (raise concurrency with OLLAMA_NUM_PARALLEL if needed).
source .venv/bin/activate
export GEMMA_4_31B_IT_BASE_URL="http://localhost:$PORT/v1/"
export GEMMA_4_31B_IT_API_KEY="ollama"
ARGS=$(for _ in $(seq 1 "$NUM_CONVOS"); do printf '%s ' "$CONFIG"; done)

echo "harvesting $NUM_CONVOS conversations with $MODEL"
python src/main.py $ARGS

echo "harvest done -> results/gemma_harvest"
