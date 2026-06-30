#!/bin/bash
#SBATCH --job-name=mnist_transfer_eval
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=1
#SBATCH --gres=gpu:geforce_gtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--smoke-test" ]]; then
    "$PYTHON" -m tests.eval_moving_mnist_transfer --help >/dev/null
    echo "smoke-test ok: $0"
    exit 0
fi

if [[ -z "${MANIFEST:-}" || -z "${EVAL_ROOT:-}" || -z "${DATA_DIR:-}" ]]; then
    echo "MANIFEST, EVAL_ROOT, and DATA_DIR must be exported for this Slurm job." >&2
    exit 2
fi

line_number=$((SLURM_ARRAY_TASK_ID + 2))
line=$(sed -n "${line_number}p" "$MANIFEST")
if [[ -z "$line" ]]; then
    echo "No manifest line for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 3
fi

experiment=$(printf '%s\n' "$line" | cut -f1)
checkpoint_path=$(printf '%s\n' "$line" | cut -f2)
output_dir="${EVAL_ROOT}/${experiment}"

mkdir -p "$output_dir"

echo "experiment=${experiment}"
echo "checkpoint_path=${checkpoint_path}"
echo "output_dir=${output_dir}"
echo "data_dir=${DATA_DIR}"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"

"$PYTHON" -u -m tests.eval_moving_mnist_transfer \
    --checkpoint_path "$checkpoint_path" \
    --data_dir "$DATA_DIR" \
    --output_dir "$output_dir" \
    --batch_size "${BATCH_SIZE:-64}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --device cuda \
    --num_vis_samples "${NUM_VIS_SAMPLES:-8}" \
    --seed "${SEED:-0}"
