from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from otf_vqvae.model import tensor_to_uint8_image

from otf_lam.utils import (
    choose_device,
    load_otf_vqvae_from_checkpoint,
    move_batch_to_device,
    resolve_otf_vqvae_checkpoint_path,
    torch_load,
    unique_path,
)

from .model import DINOJEPALatentActionModel, FrozenDINOv2Encoder, JEPAVQMotionExtractor


def make_run_name(prefix: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    run_id = f"job{job_id}" if job_id else f"pid{os.getpid()}"
    base_name = prefix if prefix is not None and str(prefix).strip() else "dinolam"
    safe_base_name = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(base_name).strip()
    )
    return f"{safe_base_name}_{run_id}_{timestamp}"


def code_usage_statistics(factors: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    if factors is None:
        return {}
    active_mask = factors["active_mask"].detach().bool()
    weights = factors["weights"].detach().float()
    eps = torch.finfo(weights.dtype).eps
    entropy = -(weights * (weights + eps).log()).sum(dim=1)
    return {
        "code_usage_active_ratio": float(active_mask.any(dim=0).float().mean().item()),
        "code_usage_entropy": float(entropy.mean().item()),
        "dead_codes": float((~active_mask.any(dim=0)).sum().item()),
        "avg_active_codes_per_sample": float(active_mask.float().sum(dim=1).mean().item()),
    }


def attention_statistics(attn_weights: Optional[torch.Tensor]) -> Dict[str, float]:
    if attn_weights is None:
        return {}
    weights = attn_weights.detach().float()
    eps = torch.finfo(weights.dtype).eps
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)
    return {
        "action_query_attn_entropy": float(entropy.mean().item()),
        "action_query_attn_max": float(weights.max(dim=-1).values.mean().item()),
    }


def parameter_grad_norm(parameters) -> Optional[float]:
    grads = [parameter.grad.detach().flatten() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return None
    return float(torch.cat(grads).norm().item())


@torch.no_grad()
def evaluate_jepa(
    model: DINOJEPALatentActionModel,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    cosine_sum = 0.0
    squared_error_sum = 0.0
    num_tokens = 0
    num_values = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        pred = output["pred_tokens"]
        target = output["target_tokens"]
        batch_tokens = int(pred.shape[0] * pred.shape[1])
        loss_sum += float(output["jepa_loss"].item()) * batch_tokens
        cosine_sum += float(F.cosine_similarity(pred, target, dim=-1).sum().item())
        squared_error_sum += float((pred - target).pow(2).sum().item())
        num_tokens += batch_tokens
        num_values += int(pred.numel())

    if was_training:
        model.train()
    if num_tokens == 0 or num_values == 0:
        raise RuntimeError("No evaluation batches were processed")

    return {
        "jepa_loss": loss_sum / num_tokens,
        "cosine_similarity": cosine_sum / num_tokens,
        "feature_mse": squared_error_sum / num_values,
    }


def save_dinolam_checkpoint(
    *,
    model: DINOJEPALatentActionModel,
    optimizer: torch.optim.Optimizer,
    cfg_snapshot: Dict[str, Any],
    otf_vqvae_checkpoint_path: Optional[str | Path],
    otf_vqvae_cfg_snapshot: Dict[str, Any],
    epoch: int,
    step: int,
    output_dir: str | Path,
    filename_prefix: str = "dinolam",
) -> Path:
    checkpoint_dir = Path(output_dir).expanduser() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = unique_path(checkpoint_dir / f"{filename_prefix}_step{int(step):06d}.pt")
    checkpoint = {
        "model_state_dict": model.second_stage_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg_snapshot": cfg_snapshot,
        "otf_vqvae_checkpoint_path": None if otf_vqvae_checkpoint_path is None else str(otf_vqvae_checkpoint_path),
        "otf_vqvae_cfg_snapshot": otf_vqvae_cfg_snapshot,
        "dino_model_name": str(model.dino_encoder.model_name),
        "epoch": int(epoch),
        "step": int(step),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _plot_frame(axis, frame: torch.Tensor, title: str) -> None:
    axis.imshow(tensor_to_uint8_image(frame))
    axis.set_title(title)
    axis.axis("off")


@torch.no_grad()
def save_jepa_debug_examples(
    model: DINOJEPALatentActionModel,
    dataset: Dataset,
    output_dir: str | Path,
    device: torch.device,
    *,
    num_examples: int = 4,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(dataset) == 0:
        return {"examples": []}

    count = min(int(num_examples), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=int).tolist()
    summary: Dict[str, Any] = {"examples": []}
    was_training = model.training
    model.eval()

    for rank, dataset_index in enumerate(indices):
        sample = dataset[int(dataset_index)]
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in sample.items()
            if torch.is_tensor(value)
        }
        output = model(batch)
        current = batch["current"][0].detach().cpu()
        future_key = "target" if "target" in batch else ("future" if "future" in batch else "next")
        future = batch[future_key][0].detach().cpu()
        patch_error = output["patch_error"][0].detach().cpu().reshape(*output["dino_grid"])
        motion_indices = output["motion_indices"]

        num_panels = 4 if motion_indices is not None else 3
        fig, axes = plt.subplots(1, num_panels, figsize=(3.0 * num_panels, 3))
        if num_panels == 1:
            axes = [axes]
        _plot_frame(axes[0], current, "current")
        _plot_frame(axes[1], future, "future")
        error_axis = axes[2]
        if motion_indices is not None:
            axes[2].imshow(motion_indices[0].detach().cpu().numpy(), cmap="tab20")
            axes[2].set_title("motion codes")
            axes[2].axis("off")
            error_axis = axes[3]
        im = error_axis.imshow(patch_error.numpy(), cmap="magma")
        error_axis.set_title("cosine error")
        error_axis.axis("off")
        fig.colorbar(im, ax=error_axis, fraction=0.046, pad=0.04)
        fig.tight_layout()

        prefix = f"sample_{rank:03d}_dataset_{int(dataset_index):06d}"
        image_path = output_dir / f"{prefix}_jepa_diagnostics.png"
        metadata_path = output_dir / f"{prefix}_metadata.json"
        fig.savefig(image_path, dpi=160)
        plt.close(fig)

        metadata = {
            "dataset_index": int(dataset_index),
            "jepa_loss": float(output["jepa_loss"].item()),
            "cosine_similarity": float(
                F.cosine_similarity(output["pred_tokens"], output["target_tokens"], dim=-1).mean().item()
            ),
            "feature_mse": float(F.mse_loss(output["pred_tokens"], output["target_tokens"]).item()),
            "patch_error_mean": float(patch_error.mean().item()),
        }
        z_act = output.get("z_act")
        if z_act is not None:
            metadata["z_act_norm"] = float(z_act[0].detach().norm().item())
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary["examples"].append(
            {
                "diagnostics": str(image_path),
                "metadata": str(metadata_path),
            }
        )

    if was_training:
        model.train()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_dinolam_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
    *,
    otf_vqvae_checkpoint_path: Optional[str | Path] = None,
) -> Tuple[DINOJEPALatentActionModel, DictConfig]:
    if "cfg_snapshot" not in checkpoint:
        raise KeyError("DINO-LAM checkpoint does not contain cfg_snapshot")
    cfg = OmegaConf.create(checkpoint["cfg_snapshot"])

    use_motion_codes = bool(cfg.get("use_motion_codes", True))
    otf_vqvae_extractor = None
    otf_vqvae_cfg = OmegaConf.create(checkpoint.get("otf_vqvae_cfg_snapshot", {}))
    if use_motion_codes:
        resolved_otf_vqvae_path = otf_vqvae_checkpoint_path or checkpoint.get("otf_vqvae_checkpoint_path")
        if resolved_otf_vqvae_path is None:
            raise KeyError("DINO-LAM checkpoint does not include otf_vqvae_checkpoint_path; pass one explicitly")
        otf_vqvae, otf_vqvae_cfg, _ = load_otf_vqvae_from_checkpoint(resolved_otf_vqvae_path, device)
        otf_vqvae_extractor = JEPAVQMotionExtractor(
            otf_vqvae,
            finetune_mode=str(cfg.get("vq_finetune_mode", "frozen")),
            use_ema_codebook_update=bool(cfg.get("use_ema_codebook_update", True)),
            use_dead_code_reinit=bool(cfg.get("use_dead_code_reinit", True)),
            dead_code_threshold_steps=int(cfg.get("dead_code_threshold_steps", 1000)),
        )

    dino_encoder = FrozenDINOv2Encoder(
        model_name=str(checkpoint.get("dino_model_name", cfg.get("dino_model_name", "facebook/dinov2-small"))),
        image_size=int(cfg.get("dino_image_size", 224)),
        mean=tuple(float(value) for value in cfg.get("dino_mean", (0.485, 0.456, 0.406))),
        std=tuple(float(value) for value in cfg.get("dino_std", (0.229, 0.224, 0.225))),
    ).to(device)
    model = DINOJEPALatentActionModel(otf_vqvae_extractor, dino_encoder, cfg).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("dinolam_state_dict"))
    if state_dict is None:
        raise KeyError("DINO-LAM checkpoint does not contain model_state_dict")
    model.load_second_stage_state_dict(state_dict)
    model.eval()
    return model, otf_vqvae_cfg


def load_dinolam_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    otf_vqvae_checkpoint_path: Optional[str | Path] = None,
) -> Tuple[DINOJEPALatentActionModel, Dict[str, Any], DictConfig]:
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch_load(checkpoint_path, map_location=device)
    model, otf_vqvae_cfg = build_dinolam_from_checkpoint(
        checkpoint,
        device,
        otf_vqvae_checkpoint_path=otf_vqvae_checkpoint_path,
    )
    return model, checkpoint, otf_vqvae_cfg

