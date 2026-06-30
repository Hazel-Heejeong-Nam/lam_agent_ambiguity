#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--smoke-test" ]]; then
  "$PYTHON" -m tests.eval_moving_mnist_transfer --help >/dev/null
  echo "smoke-test ok: $0"
  exit 0
fi

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an OTF-VQ-VAE checkpoint.}"

ARGS=(
  --checkpoint_path "$CHECKPOINT_PATH"
  --data_dir "${DATA_DIR:-data/controlled_moving_mnist}"
  --output_dir "${OUTPUT_DIR:-eval/moving_mnist_transfer}"
  --batch_size "${BATCH_SIZE:-64}"
  --num_workers "${NUM_WORKERS:-4}"
  --device "${DEVICE:-auto}"
  --num_vis_samples "${NUM_VIS_SAMPLES:-8}"
  --seed "${SEED:-0}"
)

if [[ -n "${MAX_SEQUENCES:-}" ]]; then
  ARGS+=(--max_sequences "$MAX_SEQUENCES")
fi
if [[ "${USE_WANDB:-false}" == "true" ]]; then
  ARGS+=(--use_wandb)
fi

"$PYTHON" -u -m tests.eval_moving_mnist_transfer "${ARGS[@]}"
