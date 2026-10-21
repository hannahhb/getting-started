#!/bin/bash
# Sample one model with vLLM on reduced settings (first 100 problems,
# 4 samples per original, 2 per variant, up to 8k tokens). Submit from the repo root:
#
#   sbatch datagen/sample_vllm.slurm deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
#   PROBLEMS=50 sbatch datagen/sample_vllm.slurm Qwen/Qwen3.5-4B
#
# Resubmit the same command after a timeout: finished prompts are skipped.
# Output: data/labelgen/samples/<model>__default.jsonl; log: sample_<jobid>.out
#SBATCH --job-name=sample
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=sample_%j.out
set -eo pipefail

MODEL="${1:?usage: sbatch datagen/sample_vllm.slurm <model-id>}"
PROBLEMS="${PROBLEMS:-100}"
BASE="${BASE:-4}"              # samples per original problem
VARIANT="${VARIANT:-2}"        # samples per variant
MAX_TOKENS="${MAX_TOKENS:-8192}"
VLLM_PYTHON="${VLLM_PYTHON:-/usr/bin/python}"   # the Python that has vLLM on Spartan

cd "$SLURM_SUBMIT_DIR"
"$VLLM_PYTHON" datagen/sample.py --model "$MODEL" --limit-problems "$PROBLEMS" \
  --base-samples "$BASE" --variant-samples "$VARIANT" --max-tokens "$MAX_TOKENS"
