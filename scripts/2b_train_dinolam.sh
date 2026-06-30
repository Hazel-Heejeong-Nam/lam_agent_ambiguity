#!/bin/bash
#SBATCH --job-name=dinolam
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --qos=bio_ai
#SBATCH --cpus-per-task=5
#SBATCH --exclude=bamgpu02,bamgpu07,bamgpu17
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
  "$PYTHON" dinolam/train.py --help >/dev/null
  echo "smoke-test ok: $0"
  exit 0
fi


export PYTHONUNBUFFERED=1

# cheetah-run
# CKPT="otf_vqvae_K_16_28005_20260603_142448/checkpoints/otf_vqvae_K_16_28005_20260603_142448_final.pt" 
# CKPT="otf_vqvae_K_32_28006_20260603_142538/checkpoints/otf_vqvae_K_32_28006_20260603_142538_final.pt" 
CKPT="otf_vqvae_K_64_28007_20260603_142543/checkpoints/otf_vqvae_K_64_28007_20260603_142543_final.pt" 
# CKPT="otf_vqvae_K_128_28008_20260603_142550/checkpoints/otf_vqvae_K_128_28008_20260603_142550_final.pt" 

DATA_DIR="${DATA_DIR:-/grid/klindt/home/nam/data/dcs/cheetah-run.hdf5}"


# walker-run
# CKPT="otf_vqvae_K_16_28009_20260603_142616/checkpoints/otf_vqvae_K_16_28009_20260603_142616_final.pt" 
# CKPT="otf_vqvae_K_32_28010_20260603_142621/checkpoints/otf_vqvae_K_32_28010_20260603_142621_final.pt" 
# CKPT="otf_vqvae_K_64_2514166_20260612_204551/checkpoints/otf_vqvae_K_64_2514166_20260612_204551_final.pt"
# CKPT="otf_vqvae_K_128_28012_20260603_142635/checkpoints/otf_vqvae_K_128_28012_20260603_142635_final.pt" 

# DATA_DIR="${DATA_DIR:-/grid/klindt/home/nam/data/dcs/walker-run.hdf5}"


OUTPUT_DIR="${OUTPUT_DIR:-/grid/klindt/home/nam/ckpt/dinolam_runs}"


LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-20000}"
DINO_MODEL_NAME="${DINO_MODEL_NAME:-facebook/dinov2-small}"
DINO_IMAGE_SIZE="${DINO_IMAGE_SIZE:-196}"


"$PYTHON" -u dinolam/train.py \
  --otf_vqvae_checkpoint_path "/grid/klindt/home/nam/ckpt/otf_vqvae_runs/${CKPT}" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_steps "$MAX_STEPS" \
  --lr "$LR" \
  --num_epochs 20 \
  --dino_model_name "$DINO_MODEL_NAME" \
  --dino_image_size "$DINO_IMAGE_SIZE" \
  --grid_alignment exact \
  --jepa_loss_type mse \
  --target_mode future \
  --vq_finetune_mode frozen \
  --use_ema_codebook_update false \
  --use_dead_code_reinit false \
  --use_codebook_orthogonality_loss false \
  --codebook_orthogonality_weight 0 \
  --action_aggregator_type perceiver \
  --aggregator_dim 256 \
  --aggregator_depth 2 \
  --aggregator_heads 4 \
  --aggregator_mlp_dim 1024 \
  --num_action_queries 4 \
  --z_action_dim 256 \
  --predictor_dim 384 \
  --predictor_depth 2 \
  --predictor_heads 6 \
  --predictor_mlp_dim 1536 \
  --use_motion_codes true \
  --use_global_action_token true \
  --use_patch_motion_codes_in_predictor false \
  --log_every_steps 50 \
  --eval_every_steps 1000 \
  --checkpoint_every_steps 5000 \
  --qual_every_steps 5000 \
  --wandb_project dinolam \
  --wandb_run_name cheetah_dinolam_global_only
