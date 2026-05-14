#!/bin/bash
#SBATCH --job-name=tail-reward-stageC
#SBATCH --output=logs/stageC_%j.out
#SBATCH --error=logs/stageC_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p logs outputs/stageC data

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python experiment/run_reward_gap_experiment.py prepare-prompts \
  --dataset alpaca \
  --num-prompts 100 \
  --seed 7 \
  --prompt-file data/prompts_stageC.jsonl

python experiment/run_reward_gap_experiment.py generate \
  --run-mode live \
  --prompt-file data/prompts_stageC.jsonl \
  --generator-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-prompts 100 \
  --num-candidates 20 \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-new-tokens 128 \
  --output-dir outputs/stageC

python experiment/run_reward_gap_experiment.py score \
  --run-mode live \
  --output-dir outputs/stageC \
  --proxy-scorer-kind hf_reward_model \
  --proxy-model-name OpenAssistant/reward-model-deberta-v3-large-v2 \
  --true-scorer-kind hf_reward_model \
  --true-model-name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
  --min-prompt-count 100 \
  --min-candidates-per-prompt 20

python experiment/run_reward_gap_experiment.py analyze \
  --run-mode live \
  --output-dir outputs/stageC \
  --n-values 1 2 5 10 20 \
  --analysis-seeds 7 13 29 \
  --alpha 0.95 \
  --top-proxy-quantile 0.95 \
  --num-percentile-bins 20 \
  --min-prompt-count 100 \
  --min-candidates-per-prompt 20
