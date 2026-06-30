# OTF Latent Action Models

This repository trains an OTF-VQ-VAE motion vocabulary, an OTF-LAM or DINO-LAM latent action model, and downstream policies on DCS datasets.

## Layout

- `scripts/`: runnable data collection, training, downstream, and evaluation entrypoints.
- `configs/otf_vqvae/`: Hydra configurations for OTF-VQ-VAE.
- `otf_vqvae/`: OTF-VQ-VAE model, training, and evaluation code.
- `otf_lam/`: OTF-LAM model, training, and downstream policy code.
- `dinolam/`: DINO-LAM Stage 1 implementation.
- `envs/`: DCS environment and expert data collection code.
- `tests/`: transfer evaluation programs and dataset checks.

## Pipeline

Run commands from the repository root:

```bash
sbatch scripts/0_collect_data.sh
sbatch scripts/1_train_otf_vqvae.sh
sbatch scripts/2a_train_otf_lam.sh
sbatch scripts/2b_train_dinolam.sh
sbatch scripts/3a_train_otf_lam_downstream.sh
sbatch scripts/3b_train_dinolam_downstream.sh
```

The OTF-VQ-VAE entrypoint selects `configs/otf_vqvae/cheetah.yaml` or `configs/otf_vqvae/walker.yaml` and applies the overrides in the submission script.

## Transfer evaluation

Set `CHECKPOINT_PATH` and optionally override the data/output settings:

```bash
CHECKPOINT_PATH=/path/to/checkpoint.pt bash scripts/eval_moving_mnist_transfer.sh
CHECKPOINT_PATH=/path/to/checkpoint.pt bash scripts/eval_walker_to_cheetah_transfer.sh
```

Every shell entrypoint supports `--smoke-test`. Run all smoke tests in the project environment with:

```bash
conda run -n dino310 bash scripts/smoke_test.sh
```
