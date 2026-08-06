#!/bin/bash -l

#SBATCH -o .log/smoke_openai_embed.out
#SBATCH -e .log/smoke_openai_embed.err
#SBATCH -D /u/fash/babel-ai
#SBATCH -J smoke_openai_embed

#SBATCH --nodes=1
#SBATCH --cpus-per-task 2
#SBATCH --mem 8GB
#SBATCH --partition="gpu1"        # same partition as the working harvest job
#SBATCH --gres=gpu:h200:1         # gpu1 allocates a GPU; test doesn't use it
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00

# Proves a COMPUTE node can reach the OpenAI embedding API (text-embedding-3-
# large) via babel's reference_embedder() -- the path the recovery metric and
# topic-distance need. No apptainer/GPU needed; just babel's venv (has openai)
# and OPENAI_API_KEY from .env (loaded by embeddings.py).

set -euo pipefail
mkdir -p .log
source .venv/bin/activate
export PYTHONPATH=src
echo "running OpenAI embedding smoke test..."
python scripts/inference/smoke_openai_embed.py
echo "exit code: $?"
