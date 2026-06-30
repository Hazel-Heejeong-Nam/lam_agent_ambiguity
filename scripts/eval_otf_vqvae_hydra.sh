#!/bin/bash
#SBATCH --job-name=otf-vq-eval
#SBATCH --partition=gpus
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--smoke-test" ]]; then
  "$PYTHON" otf_vqvae/eval.py --help >/dev/null
  echo "smoke-test ok: $0"
  exit 0
fi

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an OTF-VQ-VAE checkpoint.}"
"$PYTHON" otf_vqvae/eval.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --num-samples "${NUM_SAMPLES:-10}"
