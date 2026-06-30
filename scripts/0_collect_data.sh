#!/bin/bash
#SBATCH --job-name=data-collection
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --qos=slow_nice
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--smoke-test" ]]; then
    "$PYTHON" envs/dcs/experts/collect_data.py --help >/dev/null
    echo "smoke-test ok: $0"
    exit 0
fi

# module load gcc
# module load mesa
# module load glew

export MUJOCO_GL=egl

for TASK in ${TASKS:-cheetah-run}; do

    "$PYTHON" envs/dcs/experts/collect_data.py \
        --checkpoint_path="envs/dcs/experts/checkpoints/${TASK}-expert" \
        --checkpoint_name="checkpoint.pt" \
        --dcs_backgrounds_path="/grid/klindt/home/nam/data/DAVIS/JPEGImages/480p" \
        --save_path="/grid/klindt/home/nam/data/dcs/${TASK}.hdf5" \
        --num_trajectories=2000 \
        --dcs_difficulty="scale_easy_video_hard" \
        --dcs_backgrounds_split="train" \
        --dcs_img_hw=64 \
        --seed=0 \
        --cuda=False

done
