from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from otf_vqvae.model import (
    OTFVQVAE,
    apply_checkpoint_model_settings,
    apply_model_config_defaults,
    tensor_to_uint8_image,
)

from .model import OTFLAM, FrozenOTFVQVAEFactorExtractor


def torch_load(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def make_run_name(prefix: Optional[str] = None) -> str:
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    run_id = f"job{job_id}" if job_id else f"pid{os.getpid()}"
    base_name = prefix if prefix is not None and str(prefix).strip() else "otf_lam"
    safe_base_name = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(base_name).strip()
    )
    return f"{safe_base_name}_{run_id}"


def resolve_otf_vqvae_checkpoint_path(
    checkpoint_path: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if path.is_dir():
            checkpoint_dir = path
        elif path.exists():
            return path
        else:
            raise FileNotFoundError(path)

    if checkpoint_dir is None:
        raise ValueError("Provide either --otf_vqvae_checkpoint_path or --otf_vqvae_checkpoint_dir")

    root = Path(checkpoint_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = sorted(root.rglob("otf_vqvae_*.pt"))
    if not candidates:
        candidates = sorted(root.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def load_otf_vqvae_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[OTFVQVAE, DictConfig, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch_load(checkpoint_path, map_location=device)
    if "config" not in checkpoint:
        raise KeyError(f"OTF-VQ-VAE checkpoint does not contain a saved config: {checkpoint_path}")

    cfg = OmegaConf.create(checkpoint["config"])
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from checkpoint config, got {type(cfg)}")
    if "model" in cfg:
        apply_checkpoint_model_settings(cfg, checkpoint)
        model_cfg = cfg.model
    else:
        model_cfg = cfg
    apply_model_config_defaults(model_cfg)

    model = OTFVQVAE(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, cfg, checkpoint


def build_otf_lam(
    otf_vqvae: OTFVQVAE,
    cfg: DictConfig,
    device: torch.device,
) -> OTFLAM:
    extractor = FrozenOTFVQVAEFactorExtractor(otf_vqvae)
    model = OTFLAM(extractor, cfg).to(device)
    model.otf_vqvae_extractor.otf_vqvae.eval()
    return model


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(1, 10000):
        candidate = path.with_name(f"{stem}_v{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique path for {path}")


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_rgb_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    mse = F.mse_loss(pred, target).detach()
    mae = F.l1_loss(pred, target).detach()
    rmse = torch.sqrt(mse.clamp_min(0.0))
    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse.clamp_min(1.0e-12)))
    return {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "psnr": float(psnr.item()),
    }


def gate_statistics(alpha: torch.Tensor) -> Dict[str, float]:
    if alpha.ndim == 3:
        alpha = alpha.squeeze(-1)
    alpha = alpha.detach().float()
    eps = torch.finfo(alpha.dtype).eps
    entropy = -(alpha * (alpha + eps).log() + (1.0 - alpha) * (1.0 - alpha + eps).log())
    stats = {
        "gate_mean": float(alpha.mean().item()),
        "gates_gt_0.1": float((alpha > 0.1).float().sum(dim=1).mean().item()),
        "gates_gt_0.3": float((alpha > 0.3).float().sum(dim=1).mean().item()),
        "gates_gt_0.5": float((alpha > 0.5).float().sum(dim=1).mean().item()),
        "gate_entropy": float(entropy.mean().item()),
    }
    top_count = min(5, alpha.shape[1])
    top_values = torch.topk(alpha, k=top_count, dim=1).values.mean(dim=0)
    for idx, value in enumerate(top_values, start=1):
        stats[f"top_gate_{idx}"] = float(value.item())
    return stats


@torch.no_grad()
def evaluate_lam(
    model: OTFLAM,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    num_values = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        output = model(batch)
        pred = output["pred"]
        target = output["target"]
        squared_error_sum += float((pred - target).pow(2).sum().item())
        absolute_error_sum += float((pred - target).abs().sum().item())
        num_values += int(target.numel())

    if was_training:
        model.train()
    if num_values == 0:
        raise RuntimeError("No evaluation batches were processed")

    mse = squared_error_sum / num_values
    mae = absolute_error_sum / num_values
    rmse = math.sqrt(max(mse, 0.0))
    psnr = 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1.0e-12)))
    return {"mse": mse, "rmse": rmse, "mae": mae, "psnr": psnr}


def _error_to_uint8(error: torch.Tensor) -> np.ndarray:
    if error.ndim == 3:
        error = error.mean(dim=0, keepdim=True)
    max_value = float(error.max().item())
    if max_value > 0.0:
        error = error / max_value
    error = error.clamp(0.0, 1.0).expand(3, -1, -1)
    return (error * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _save_prediction_grid(
    path: Path,
    current: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
) -> None:
    error = (pred - target).abs()
    images = [
        tensor_to_uint8_image(current),
        tensor_to_uint8_image(target),
        tensor_to_uint8_image(pred),
        _error_to_uint8(error),
    ]
    titles = ["current", "target", "prediction", "absolute error"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_top_occupancy_maps(
    path: Path,
    occupancy: torch.Tensor,
    top_indices: Iterable[int],
) -> None:
    top_indices = list(top_indices)
    if not top_indices:
        return
    fig, axes = plt.subplots(1, len(top_indices), figsize=(2.4 * len(top_indices), 2.4))
    if len(top_indices) == 1:
        axes = [axes]
    for axis, code_id in zip(axes, top_indices):
        axis.imshow(occupancy[int(code_id)].detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"code {int(code_id)}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_gate_bar_plot(path: Path, alpha: torch.Tensor, top_count: int = 20) -> None:
    alpha = alpha.detach().cpu().float()
    top_count = min(int(top_count), alpha.numel())
    values, indices = torch.topk(alpha, k=top_count)
    fig, axis = plt.subplots(1, 1, figsize=(max(6, top_count * 0.35), 3))
    axis.bar(np.arange(top_count), values.numpy())
    axis.set_xticks(np.arange(top_count))
    axis.set_xticklabels([str(int(idx)) for idx in indices], rotation=90)
    axis.set_xlabel("code id")
    axis.set_ylabel("gate")
    axis.set_ylim(0.0, max(1.0, float(values.max().item()) * 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_qualitative_examples(
    model: OTFLAM,
    dataset: Dataset,
    output_dir: str | Path,
    device: torch.device,
    *,
    num_examples: int = 8,
    top_factors: int = 6,
) -> Dict[str, Any]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(dataset) == 0:
        return {"examples": []}

    count = min(int(num_examples), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=int).tolist()
    summary = {"examples": []}

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
        target = output["target"][0].detach().cpu()
        pred = output["pred"][0].detach().cpu()
        alpha = output["alpha"][0, :, 0].detach().cpu()
        weights = output["weights"][0].detach().cpu()
        occupancy = output["occupancy"][0].detach().cpu()

        top_count = min(int(top_factors), alpha.numel())
        top_values, top_indices = torch.topk(alpha, k=top_count)
        prefix = f"sample_{rank:03d}_dataset_{int(dataset_index):06d}"
        grid_path = output_dir / f"{prefix}_prediction_grid.png"
        gates_path = output_dir / f"{prefix}_gates.png"
        occupancy_path = output_dir / f"{prefix}_top_occupancy.png"
        metadata_path = output_dir / f"{prefix}_metadata.json"

        _save_prediction_grid(grid_path, current, target, pred)
        _save_gate_bar_plot(gates_path, alpha)
        _save_top_occupancy_maps(occupancy_path, occupancy, top_indices.tolist())

        top_records = [
            {
                "code_id": int(code_id),
                "gate": float(gate_value),
                "weight": float(weights[int(code_id)]),
            }
            for code_id, gate_value in zip(top_indices.tolist(), top_values.tolist())
        ]
        metadata = {
            "dataset_index": int(dataset_index),
            "mse": float(F.mse_loss(pred, target).item()),
            "mae": float(F.l1_loss(pred, target).item()),
            "top_factors": top_records,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary["examples"].append(
            {
                "grid": str(grid_path),
                "gates": str(gates_path),
                "occupancy": str(occupancy_path),
                "metadata": str(metadata_path),
            }
        )

    if was_training:
        model.train()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_checkpoint(
    *,
    model: OTFLAM,
    optimizer: torch.optim.Optimizer,
    cfg_snapshot: Dict[str, Any],
    otf_vqvae_checkpoint_path: str | Path,
    otf_vqvae_cfg_snapshot: Dict[str, Any],
    epoch: int,
    step: int,
    output_dir: str | Path,
    filename_prefix: str = "otf_lam",
    final: bool = False,
) -> Path:
    checkpoint_dir = Path(output_dir).expanduser() / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "final" if final else f"step{int(step):06d}"
    checkpoint_path = unique_path(checkpoint_dir / f"{filename_prefix}_{suffix}.pt")
    checkpoint = {
        "otf_lam_state_dict": model.second_stage_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg_snapshot": cfg_snapshot,
        "otf_vqvae_checkpoint_path": str(otf_vqvae_checkpoint_path),
        "otf_vqvae_cfg_snapshot": otf_vqvae_cfg_snapshot,
        "epoch": int(epoch),
        "step": int(step),
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_otf_lam_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    otf_vqvae_checkpoint_path: Optional[str | Path] = None,
) -> Tuple[OTFLAM, Dict[str, Any], DictConfig]:
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch_load(checkpoint_path, map_location=device)
    resolved_otf_vqvae_path = otf_vqvae_checkpoint_path or checkpoint.get("otf_vqvae_checkpoint_path")
    if resolved_otf_vqvae_path is None:
        raise KeyError("Checkpoint does not include otf_vqvae_checkpoint_path; pass one explicitly")
    otf_vqvae, otf_vqvae_cfg, _ = load_otf_vqvae_from_checkpoint(resolved_otf_vqvae_path, device)
    cfg = OmegaConf.create(checkpoint["cfg_snapshot"])
    model = build_otf_lam(otf_vqvae, cfg, device)
    model.load_second_stage_state_dict(checkpoint["otf_lam_state_dict"])
    model.eval()
    return model, checkpoint, otf_vqvae_cfg
