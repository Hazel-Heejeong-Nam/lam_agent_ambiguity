#!/bin/bash
#SBATCH --job-name=eval_quantitative
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=logs/otf_vqvae/%x_%j.out
#SBATCH --error=logs/otf_vqvae/%x_%j.err
#SBATCH --exclude=gpu16[00-99],gpu17[00-99],gpu18[00-99]

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--smoke-test" ]]; then
  "$PYTHON" otf_vqvae/eval_quantitative.py --help >/dev/null
  echo "smoke-test ok: $0"
  exit 0
fi

mkdir -p logs/otf_vqvae

RUNS_ROOT="${RUNS_ROOT:-/users/hnam16/scratch/otf_vqvae_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/users/hnam16/scratch/otf_vqvae_runs/quantitative_eval}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-500}"
SAMPLES_PER_TRAJECTORY="${SAMPLES_PER_TRAJECTORY:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
DEVICE="${DEVICE:-auto}"
RUN_ID="${RUN_ID:-}"

ARGS=(
  --runs-root "$RUNS_ROOT"
  --output-root "$OUTPUT_ROOT"
  --max-trajectories "$MAX_TRAJECTORIES"
  --samples-per-trajectory "$SAMPLES_PER_TRAJECTORY"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$RUN_ID" ]]; then
  ARGS+=(--run-id "$RUN_ID")
fi

if [[ -n "${EXPERIMENTS:-}" ]]; then
  IFS=',' read -ra EXPERIMENT_NAMES <<< "$EXPERIMENTS"
  for EXPERIMENT_NAME in "${EXPERIMENT_NAMES[@]}"; do
    ARGS+=(--experiment "$EXPERIMENT_NAME")
  done
fi

"$PYTHON" -u otf_vqvae/eval_quantitative.py "${ARGS[@]}"
