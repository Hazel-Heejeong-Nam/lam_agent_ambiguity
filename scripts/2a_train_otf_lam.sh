#!/bin/bash
#SBATCH --job-name=lam-cheetah
#SBATCH --partition=gpus
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --time=7-00:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--smoke-test" ]]; then
     "$PYTHON" otf_lam/train_otf_lam.py --help >/dev/null
     echo "smoke-test ok: $0"
     exit 0
fi

# TASK="walker-run"
# AGENT="walker"
TASK="cheetah-run"
AGENT="cheetah"
LAM_SEED=0



# Train OTF-LAM once with a fixed seed.
DT="$(date +%Y%m%d_%H%M%S)"

# cheetah-run
# PT="otf_vqvae_K_16_28005_20260603_142448/checkpoints/otf_vqvae_K_16_28005_20260603_142448_final.pt" 
# PT="otf_vqvae_K_32_28006_20260603_142538/checkpoints/otf_vqvae_K_32_28006_20260603_142538_final.pt" 
# PT="otf_vqvae_K_64_28007_20260603_142543/checkpoints/otf_vqvae_K_64_28007_20260603_142543_final.pt" 
PT="otf_vqvae_K_128_28008_20260603_142550/checkpoints/otf_vqvae_K_128_28008_20260603_142550_final.pt" 

# walker-run
# PT="otf_vqvae_K_16_28009_20260603_142616/checkpoints/otf_vqvae_K_16_28009_20260603_142616_final.pt" 
# PT="otf_vqvae_K_32_28010_20260603_142621/checkpoints/otf_vqvae_K_32_28010_20260603_142621_final.pt" 
# PT="otf_vqvae_K_64_28011_20260603_142628/checkpoints/otf_vqvae_K_64_28011_20260603_142628_final.pt"
# PT="otf_vqvae_K_128_28012_20260603_142635/checkpoints/otf_vqvae_K_128_28012_20260603_142635_final.pt" 


OTF_VQVAE_CHECKPOINT_PATH="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/${PT}"

LAM_RUN_NAME="lam_${SLURM_JOB_ID}_otf_vqvae_${AGENT}"
LAM_OUTPUT_ROOT="/cs/data/people/hnam16/lam_camready/lam_runs/${LAM_RUN_NAME}"

"$PYTHON" -u otf_lam/train_otf_lam.py \
     --otf_vqvae_checkpoint_path="$OTF_VQVAE_CHECKPOINT_PATH" \
     --data_dir="/cs/data/people/hnam16/dcs/"${TASK}".hdf5" \
     --output_dir="$LAM_OUTPUT_ROOT" \
     --prediction_mode="residual" \
     --max_steps=20000 \
     --occupancy_encoder_type="mlp" \
     --aggregator_type="gate" \
     --seed="$LAM_SEED" \
     --wandb_project="lam" \
     --wandb_run_name="${LAM_RUN_NAME}" \
     --num_workers=4 \
     --batch_size=512
