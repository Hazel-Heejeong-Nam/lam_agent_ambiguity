#!/bin/bash
#SBATCH --job-name=policy-eval
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
     "$PYTHON" otf_lam/train_downstream.py --help >/dev/null
     echo "smoke-test ok: $0"
     exit 0
fi


# cheetah-run
# OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_16_28005_20260603_142448/checkpoints/otf_vqvae_K_16_28005_20260603_142448_final.pt" 
# OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_32_28006_20260603_142538/checkpoints/otf_vqvae_K_32_28006_20260603_142538_final.pt" 
# OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_64_28007_20260603_142543/checkpoints/otf_vqvae_K_64_28007_20260603_142543_final.pt" 
# OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_128_28008_20260603_142550/checkpoints/otf_vqvae_K_128_28008_20260603_142550_final.pt" 

# walker-run
# OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_16_28009_20260603_142616/checkpoints/otf_vqvae_K_16_28009_20260603_142616_final.pt" 
# OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_32_28010_20260603_142621/checkpoints/otf_vqvae_K_32_28010_20260603_142621_final.pt" 
# OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_64_28011_20260603_142628/checkpoints/otf_vqvae_K_64_28011_20260603_142628_final.pt"
OTF_VQPT="/cs/data/people/hnam16/lam_camready/otf_vqvae_runs/otf_vqvae_K_128_28012_20260603_142635/checkpoints/otf_vqvae_K_128_28012_20260603_142635_final.pt" 



# PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28090_otf_vqvae_cheetah/lam_28090_otf_vqvae_cheetah_job28090/checkpoints/otf_lam_final.pt 
# PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28091_otf_vqvae_cheetah/lam_28091_otf_vqvae_cheetah_job28091/checkpoints/otf_lam_final.pt 
# PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28092_otf_vqvae_cheetah/lam_28092_otf_vqvae_cheetah_job28092/checkpoints/otf_lam_final.pt 
# PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28093_otf_vqvae_cheetah/lam_28093_otf_vqvae_cheetah_job28093/checkpoints/otf_lam_final.pt

# PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28085_otf_vqvae_walker/lam_28085_otf_vqvae_walker_job28085/checkpoints/otf_lam_final.pt 
# PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28086_otf_vqvae_walker/lam_28086_otf_vqvae_walker_job28086/checkpoints/otf_lam_final.pt 
# PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28087_otf_vqvae_walker/lam_28087_otf_vqvae_walker_job28087/checkpoints/otf_lam_final.pt 
PT=/cs/data/people/hnam16/lam_camready/lam_runs/lam_28089_otf_vqvae_walker/lam_28089_otf_vqvae_walker_job28089/checkpoints/otf_lam_final.pt 



TASK="walker-run"
export MUJOCO_GL=egl
DCS_BACKGROUNDS_PATH="${DCS_BACKGROUNDS_PATH:-/cs/data/people/hnam16/DAVIS/JPEGImages/480p}"


# Train OTF-LAM once with a fixed seed.
DT="$(date +%Y%m%d_%H%M%S)"
LAM_RUN_NAME="lam_${SLURM_JOB_ID}_${DT}"
BC_EVAL_SEED=0
ACT_DECODER_EVAL_SEED=0

# To reuse completed BC checkpoints, set this to the run prefix before
# "_downstream_seed<N>", for example:
OTF_VQDIR="${OTF_VQPT#*/otf_vqvae_runs/}"
OTF_VQDIR="${OTF_VQDIR%%/*}"
# BC_RESUME_RUN_PREFIX="/cs/data/people/hnam16/lam_camready/downstream/lam_28167_20260607_152242"
BC_RESUME_RUN_PREFIX="/cs/data/people/hnam16/lam_camready/downstream/${OTF_VQDIR}"

# Train downstream task with varied training seeds and fixed eval seeds.
for TRAIN_SEED in 0 1 2; do
     DOWNSTREAM_RUN_NAME="${BC_RESUME_RUN_PREFIX}_seed${TRAIN_SEED}"
     DOWNSTREAM_OUTPUT_DIR="/cs/data/people/hnam16/lam_camready/downstream/${DOWNSTREAM_RUN_NAME}"
     BC_ARGS=(--train_bc=true)
     if [[ -n "${BC_RESUME_RUN_PREFIX}" ]]; then
          BC_RESUME_PATH="${BC_RESUME_RUN_PREFIX}_seed${TRAIN_SEED}/bc-epoch_10.pt"
          BC_ARGS=(--train_bc=false --bc_resume_path="${BC_RESUME_PATH}")
          echo "Resuming BC training from ${BC_RESUME_PATH}"
     fi

     "$PYTHON" -u otf_lam/train_downstream.py \
          --otf_lam_checkpoint_path="$PT" \
          --otf_vqvae_checkpoint_path="$OTF_VQPT" \
          --data_path="/cs/data/people/hnam16/dcs/"${TASK}".hdf5" \
          --output_dir="${DOWNSTREAM_OUTPUT_DIR}" \
          --seed="${TRAIN_SEED}" \
          --bc_eval_seed="${BC_EVAL_SEED}" \
          --act_decoder_eval_seed="${ACT_DECODER_EVAL_SEED}" \
          --bc_dcs_backgrounds_path="${DCS_BACKGROUNDS_PATH}" \
          --act_decoder_dcs_backgrounds_path="${DCS_BACKGROUNDS_PATH}" \
          "${BC_ARGS[@]}" \
          --wandb_project="lam" \
          --wandb_run_name="${DOWNSTREAM_RUN_NAME}" \
          --wandb_tags "training_seed_${TRAIN_SEED}" "bc_eval_seed_${BC_EVAL_SEED}" "act_decoder_eval_seed_${ACT_DECODER_EVAL_SEED}"
done
