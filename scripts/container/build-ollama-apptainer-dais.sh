#!/bin/bash -l

#SBATCH -o .log/job.out
#SBATCH -e .log/job.err
#SBATCH -D ./
#SBATCH -J build_ollama_image

#SBATCH --nodes=1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1

#SBATCH --time=03:00:00

# Build the Ollama Apptainer image (company HPC guide). Produces container/ollama.sif.
#   sbatch scripts/container/build-ollama-apptainer-dais.sh

module load apptainer

mkdir -p container
rm -rf container/ollama.sif

TMP=$(mktemp -p /dais/fs/scratch/$USER/ -d)
export APPTAINER_TMPDIR=$TMP

# gemma4 is new -> needs a recent Ollama. 0.12.9 (from the company guide, which
# targeted older gpt-oss models) is too old and 412s on pull. Use latest to
# unblock; after it works, pin to the exact version (`ollama --version`) for
# reproducibility.
apptainer pull container/ollama.sif docker://ollama/ollama:latest

rm -rf $TMP
