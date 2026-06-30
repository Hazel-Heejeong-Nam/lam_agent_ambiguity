#!/bin/bash
#SBATCH --job-name=otf-vq-eval
#SBATCH --partition=cs-all-gcondo
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err

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
