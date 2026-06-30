#!/bin/bash
#SBATCH --job-name=otf_vqvae
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --qos=slow_nice
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--smoke-test" ]]; then
     "$PYTHON" otf_vqvae/main_walker.py --help >/dev/null
     echo "smoke-test ok: $0"
     exit 0
fi

# TASK="cheetah-run"
# AGENT="cheetah"
CODEBOOK_SIZE=64
TASK="walker-run"
AGENT="walker"

# Train OTF-VQ-VAE
OTF_VQVAE_RUN_NAME="otf_vqvae_K_${CODEBOOK_SIZE}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
DATA_PATH="/grid/klindt/home/nam/data/dcs/${TASK}.hdf5"
EXPECTED_NUM_TRAJS=2000
TRAIN_NUM_TRAJS=1000

NUM_TRAJS=$(h5ls "$DATA_PATH" | wc -l)
if [ "$NUM_TRAJS" -ne "$EXPECTED_NUM_TRAJS" ]; then
     echo "Expected ${EXPECTED_NUM_TRAJS} trajectories in ${DATA_PATH}, got ${NUM_TRAJS}." >&2
     exit 1
fi
echo "Using first ${TRAIN_NUM_TRAJS} of ${NUM_TRAJS} trajectories from ${DATA_PATH}."

"$PYTHON" -u otf_vqvae/main_"${AGENT}".py \
     run_name="$OTF_VQVAE_RUN_NAME" \
     output_dir="/grid/klindt/home/nam/ckpt/otf_vqvae_runs/${OTF_VQVAE_RUN_NAME}" \
     data.type="${TASK}" \
     data.path="$DATA_PATH" \
     data.max_sequences=${TRAIN_NUM_TRAJS} \
     model.codebook_size=${CODEBOOK_SIZE} \
     model.motion_transform="gradient" \
     train.batch_size=512

