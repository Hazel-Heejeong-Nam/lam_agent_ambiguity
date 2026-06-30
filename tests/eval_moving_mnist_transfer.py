from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from otf_vqvae import model as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Moving MNIST digit-transfer reconstruction from a frozen OTF-VQ-VAE checkpoint."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/controlled_moving_mnist")
    parser.add_argument("--split", type=str, choices=("test", "train"), default="test")
    parser.add_argument("--output_dir", type=str, default="eval/moving_mnist_transfer_eval_baselines")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_vis_samples", type=int, default=8)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_mixed_digits", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="motion-reusability")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_torch_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict at {path}, got {type(checkpoint)}")
    return checkpoint


def load_checkpoint_and_config(checkpoint_path: Path, device: torch.device) -> tuple[Dict[str, Any], DictConfig]:
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a saved config: {checkpoint_path}")
    cfg = OmegaConf.create(checkpoint["config"])
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected checkpoint config to produce DictConfig, got {type(cfg)}")
    train_main.apply_checkpoint_model_settings(cfg, checkpoint)
    return checkpoint, cfg


def build_frozen_model(cfg: DictConfig, checkpoint: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    train_main.get_model_type(cfg)
    train_main.apply_model_config_defaults(cfg.model)
    model = train_main.OTFVQVAE(cfg.model).to(device)
    model.model_type = "otf_vqvae"
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def get_cfg_value(cfg: DictConfig, section: str, key: str, default: Any) -> Any:
    if section not in cfg:
        return default
    section_cfg = cfg[section]
    return section_cfg[key] if key in section_cfg else default


def get_data_tau(cfg: DictConfig) -> int:
    return int(get_cfg_value(cfg, "data", "tau", 1))


def ensure_mnist_channels(frames: np.ndarray, channels: int, scale_uint8: bool) -> torch.Tensor:
    tensor = torch.as_tensor(frames)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(2)
    elif tensor.ndim == 5 and tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(0, 1, 4, 2, 3)
    elif tensor.ndim != 5:
        raise ValueError(f"Expected frames with shape [N,T,H,W] or channelized equivalent, got {tuple(tensor.shape)}")

    if tensor.shape[2] == channels:
        pass
    elif tensor.shape[2] == 1 and channels == 3:
        tensor = tensor.repeat(1, 1, 3, 1, 1)
    else:
        raise ValueError(f"Cannot adapt frame channels from {tensor.shape[2]} to model channels={channels}")

    tensor = tensor.contiguous()
    if scale_uint8 and tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    return tensor.float()


def resize_sequences_if_needed(
    frames: torch.Tensor,
    image_height: int,
    image_width: int,
    resize_to_input: bool,
) -> torch.Tensor:
    if tuple(frames.shape[-2:]) == (image_height, image_width):
        return frames
    if not resize_to_input:
        raise ValueError(
            f"Test frames have size {tuple(frames.shape[-2:])}, but the checkpoint expects "
            f"{(image_height, image_width)}. The checkpoint config has data.resize_to_input=false."
        )
    n, t, c, _, _ = frames.shape
    flat = frames.reshape(n * t, c, frames.shape[-2], frames.shape[-1])
    flat = F.interpolate(flat, size=(image_height, image_width), mode="bilinear", align_corners=False)
    return flat.reshape(n, t, c, image_height, image_width)


class ControlledMovingMNISTTransferDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        cfg: DictConfig,
        motion_input_type: str,
        motion_transform: str,
        use_reference_conditioning: bool,
        allow_mixed_digits: bool,
        split: str = "test",
        max_sequences: Optional[int] = None,
    ) -> None:
        path = data_dir / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as archive:
            required = {"frames", "digit_labels", "motion_ids"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise KeyError(f"{path} is missing required keys: {missing}")
            frames_np = archive["frames"]
            digit_labels_np = archive["digit_labels"]
            motion_ids_np = archive["motion_ids"]

        if max_sequences is not None:
            frames_np = frames_np[: int(max_sequences)]
            digit_labels_np = digit_labels_np[: int(max_sequences)]
            motion_ids_np = motion_ids_np[: int(max_sequences)]

        if digit_labels_np.size == 0:
            raise ValueError(f"{path} contains empty digit_labels")
        min_digit = int(digit_labels_np.min())
        max_digit = int(digit_labels_np.max())
        if split == "test" and not allow_mixed_digits and (min_digit < 5 or max_digit > 9):
            raise ValueError(
                f"Expected held-out digits 5-9 in {path}, got min={min_digit}, max={max_digit}. "
                "Pass --allow_mixed_digits to evaluate anyway."
            )

        self.split = split
        self.motion_input_type = train_main.validate_motion_input_type(motion_input_type)
        self.motion_transform = train_main.validate_motion_transform(motion_transform)
        self.use_reference_conditioning = bool(use_reference_conditioning)
        self.tau = get_data_tau(cfg)
        self.digit_labels = torch.as_tensor(digit_labels_np, dtype=torch.long)
        self.motion_ids = torch.as_tensor(motion_ids_np, dtype=torch.long)
        self.frames_shape = tuple(frames_np.shape)

        channels = int(cfg.model.channels)
        self.channels = channels
        self.image_height = int(cfg.model.image_height)
        self.image_width = int(cfg.model.image_width)
        scale_uint8 = bool(get_cfg_value(cfg, "data", "scale_uint8", True))
        resize_to_input = bool(get_cfg_value(cfg, "data", "resize_to_input", False))
        frames = ensure_mnist_channels(frames_np, channels=channels, scale_uint8=scale_uint8)
        frames = resize_sequences_if_needed(
            frames,
            image_height=self.image_height,
            image_width=self.image_width,
            resize_to_input=resize_to_input,
        )

        sequence_length = int(frames.shape[1])
        if self.tau < 1:
            raise ValueError(f"tau must be positive, got {self.tau}")
        required_context = 2 * self.tau if self.motion_input_type == "acceleration" else self.tau
        if sequence_length <= required_context:
            raise ValueError(
                f"{path.name} sequence_length={sequence_length} is too short for tau={self.tau} "
                f"and motion_input_type={self.motion_input_type}"
            )

        if self.motion_input_type == "acceleration":
            time_indices = range(self.tau, sequence_length - self.tau)
        else:
            time_indices = range(0, sequence_length - self.tau)
        self.indices = [
            (video_idx, time_idx)
            for video_idx in range(frames.shape[0])
            for time_idx in time_indices
        ]
        if not self.indices:
            raise ValueError(f"No valid transitions found in {path}")
        self.frames = frames.contiguous()

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_videos(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_idx, time_idx = self.indices[index]
        sample = {
            "current": self.frames[video_idx, time_idx],
            "next": self.frames[video_idx, time_idx + self.tau],
            "video_index": torch.tensor(video_idx, dtype=torch.long),
            "time_index": torch.tensor(time_idx, dtype=torch.long),
            "digit_labels": self.digit_labels[video_idx],
            "motion_ids": self.motion_ids[video_idx],
        }
        if self.motion_input_type == "acceleration":
            sample["previous"] = self.frames[video_idx, time_idx - self.tau]
        sample["motion"] = train_main.make_motion_signal(sample, self.motion_input_type, self.motion_transform)
        return train_main.add_reference_frame_if_needed(
            sample,
            self.motion_input_type,
            self.use_reference_conditioning,
        )


def move_model_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = {"previous", "current", "next", "motion", "reference_frame"}
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if key in keys}


def update_motion_breakdown(
    by_motion: Dict[int, Dict[str, float]],
    motion_ids: torch.Tensor,
    sample_squared_error: torch.Tensor,
    sample_absolute_error: torch.Tensor,
    pixels_per_transition: int,
) -> None:
    motion_ids = motion_ids.detach().cpu()
    squared = sample_squared_error.detach().cpu()
    absolute = sample_absolute_error.detach().cpu()
    for batch_idx in range(motion_ids.shape[0]):
        for motion_id in torch.unique(motion_ids[batch_idx]).tolist():
            motion_id = int(motion_id)
            stats = by_motion.setdefault(
                motion_id,
                {"num_transitions": 0, "sum_squared_error": 0.0, "sum_absolute_error": 0.0, "count": 0.0},
            )
            stats["num_transitions"] += 1
            stats["sum_squared_error"] += float(squared[batch_idx].item())
            stats["sum_absolute_error"] += float(absolute[batch_idx].item())
            stats["count"] += float(pixels_per_transition)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_count = 0
    by_motion: Dict[int, Dict[str, float]] = {}
    for batch in loader:
        model_batch = move_model_batch(batch, device)
        target = model_batch["motion"]
        output = model(
            target,
            reference_frame=model_batch.get("reference_frame"),
            use_quantization=True,
        )
        reconstruction = output["reconstruction"]
        error = reconstruction - target
        squared_error = error.pow(2)
        absolute_error = error.abs()

        total_squared_error += float(squared_error.sum(dtype=torch.float64).item())
        total_absolute_error += float(absolute_error.sum(dtype=torch.float64).item())
        total_count += int(target.numel())

        batch_size = int(target.shape[0])
        pixels_per_transition = int(target[0].numel())
        sample_squared_error = squared_error.reshape(batch_size, -1).sum(dim=1, dtype=torch.float64)
        sample_absolute_error = absolute_error.reshape(batch_size, -1).sum(dim=1, dtype=torch.float64)
        update_motion_breakdown(
            by_motion,
            batch["motion_ids"],
            sample_squared_error,
            sample_absolute_error,
            pixels_per_transition,
        )

    mse = total_squared_error / float(total_count)
    mae = total_absolute_error / float(total_count)
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae}, by_motion


def finalize_motion_breakdown(by_motion: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    finalized = {}
    for motion_id, stats in sorted(by_motion.items()):
        count = float(stats["count"])
        mse = float(stats["sum_squared_error"]) / count
        mae = float(stats["sum_absolute_error"]) / count
        finalized[str(motion_id)] = {
            "num_transitions": int(stats["num_transitions"]),
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": mae,
        }
    return finalized


def tensor_to_frame_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    elif tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")
    return (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def tensor_to_normalized_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    elif tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[0]}")
    tensor = tensor - tensor.min()
    denom = tensor.max().clamp_min(torch.finfo(tensor.dtype).eps)
    tensor = tensor / denom
    return (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def make_panel(images: Iterable[tuple[str, np.ndarray]]) -> Image.Image:
    items = list(images)
    if not items:
        raise ValueError("Cannot build an empty panel")
    label_height = 20
    gap = 4
    widths = [array.shape[1] for _, array in items]
    heights = [array.shape[0] for _, array in items]
    panel_width = sum(widths) + gap * (len(items) - 1)
    panel_height = max(heights) + label_height
    panel = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    x_offset = 0
    for label, array in items:
        draw.text((x_offset + 2, 3), label, fill=(0, 0, 0))
        panel.paste(Image.fromarray(array).convert("RGB"), (x_offset, label_height))
        x_offset += array.shape[1] + gap
    return panel


def choose_visual_indices(dataset_size: int, num_samples: int, seed: int) -> list[int]:
    num_samples = max(0, min(int(num_samples), dataset_size))
    if num_samples == 0:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, size=num_samples, replace=False)
    return sorted(int(index) for index in indices.tolist())


@torch.no_grad()
def save_qualitative_samples(
    model: torch.nn.Module,
    dataset: ControlledMovingMNISTTransferDataset,
    output_dir: Path,
    indices: list[int],
    device: torch.device,
    wandb_run: Optional[Any],
) -> None:
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    wandb_images = {}

    for sample_number, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        model_batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in sample.items()
            if key in {"previous", "current", "next", "motion", "reference_frame"}
        }

        sample_dir = qualitative_dir / f"sample_{sample_number:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_items = []
        if dataset.motion_input_type == "acceleration" and "previous" in sample:
            previous = tensor_to_frame_image(sample["previous"])
            save_image(sample_dir / "previous.png", previous)
            image_items.append(("previous", previous))
        current = tensor_to_frame_image(sample["current"])
        next_frame = tensor_to_frame_image(sample["next"])
        save_image(sample_dir / "current.png", current)
        save_image(sample_dir / "next.png", next_frame)

        target = model_batch["motion"]
        output = model(
            target,
            reference_frame=model_batch.get("reference_frame"),
            use_quantization=True,
        )
        reconstruction = output["reconstruction"][0]
        target_motion = target[0]
        error = (reconstruction - target_motion).abs()
        target_image = tensor_to_normalized_image(target_motion)
        recon_image = tensor_to_normalized_image(reconstruction)
        error_image = tensor_to_normalized_image(error)

        save_image(sample_dir / "target_motion.png", target_image)
        save_image(sample_dir / "recon_motion.png", recon_image)
        save_image(sample_dir / "error.png", error_image)
        image_items.extend(
            [
                ("current", current),
                ("next", next_frame),
                ("target", target_image),
                ("recon", recon_image),
                ("error", error_image),
            ]
        )
        panel = make_panel(image_items)
        panel_path = sample_dir / "panel.png"
        panel.save(panel_path)

        if wandb_run is not None:
            import wandb

            video_idx, time_idx = dataset.indices[dataset_index]
            caption = f"dataset_index={dataset_index}, video={video_idx}, time={time_idx}"
            wandb_images[f"qualitative/sample_{sample_number:03d}"] = wandb.Image(panel, caption=caption)

    if wandb_run is not None and wandb_images:
        wandb_run.log(wandb_images)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    device = train_main.choose_device(args.device)

    checkpoint_path = Path(args.checkpoint_path).expanduser()
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, cfg = load_checkpoint_and_config(checkpoint_path, device)
    model = build_frozen_model(cfg, checkpoint, device)
    model_type = model.model_type
    eval_motion_input_type = model.motion_input_type
    eval_motion_transform = model.motion_transform
    eval_use_reference_conditioning = bool(model.use_reference_conditioning)
    dataset = ControlledMovingMNISTTransferDataset(
        data_dir=data_dir,
        cfg=cfg,
        motion_input_type=eval_motion_input_type,
        motion_transform=eval_motion_transform,
        use_reference_conditioning=eval_use_reference_conditioning,
        allow_mixed_digits=bool(args.allow_mixed_digits),
        split=args.split,
        max_sequences=args.max_sequences,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_dir = output_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            dir=str(wandb_dir),
            config={
                "checkpoint_path": str(checkpoint_path),
                "data_dir": str(data_dir),
                "split": args.split,
                "output_dir": str(output_dir),
                "model_type": model_type,
                "motion_input_type": eval_motion_input_type,
                "motion_transform": eval_motion_transform,
                "use_reference_conditioning": eval_use_reference_conditioning,
                "tau": dataset.tau,
                "image_height": dataset.image_height,
                "image_width": dataset.image_width,
                "channels": dataset.channels,
                "batch_size": int(args.batch_size),
                "max_sequences": args.max_sequences,
                "seed": int(args.seed),
                "config_source": "checkpoint",
            },
        )

    metrics, by_motion = evaluate(model, loader, device)
    metrics_payload = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "config_source": "checkpoint",
        "data_dir": str(data_dir),
        "split": args.split,
        "model_type": model_type,
        "num_videos": dataset.num_videos,
        "num_transitions": len(dataset),
        "max_sequences": args.max_sequences,
        "tau": dataset.tau,
        "image_height": dataset.image_height,
        "image_width": dataset.image_width,
        "channels": dataset.channels,
        "motion_input_type": eval_motion_input_type,
        "motion_transform": eval_motion_transform,
        "use_reference_conditioning": eval_use_reference_conditioning,
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
    }
    write_json(output_dir / "moving_mnist_transfer_metrics.json", metrics_payload)

    by_motion_payload = {
        "checkpoint_path": str(checkpoint_path),
        "config_source": "checkpoint",
        "data_dir": str(data_dir),
        "split": args.split,
        "max_sequences": args.max_sequences,
        "tau": dataset.tau,
        "model_type": model_type,
        "motion_input_type": eval_motion_input_type,
        "motion_transform": eval_motion_transform,
        "use_reference_conditioning": eval_use_reference_conditioning,
        "metrics_by_motion": finalize_motion_breakdown(by_motion),
    }
    write_json(output_dir / "moving_mnist_transfer_metrics_by_motion.json", by_motion_payload)

    visual_indices = choose_visual_indices(len(dataset), int(args.num_vis_samples), int(args.seed))
    save_qualitative_samples(model, dataset, output_dir, visual_indices, device, wandb_run)

    if wandb_run is not None:
        wandb_run.log(
            {
                "transfer/mse": metrics["mse"],
                "transfer/rmse": metrics["rmse"],
                "transfer/mae": metrics["mae"],
                "transfer/num_transitions": len(dataset),
            }
        )
        wandb_run.finish()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print("Config source: checkpoint")
    print(f"Model type: {model_type}")
    print(f"Tau: {dataset.tau}")
    print(f"Motion input type: {eval_motion_input_type}")
    print(f"Motion transform: {eval_motion_transform}")
    print(f"Reference conditioning: {eval_use_reference_conditioning}")
    print(f"Model input shape: channels={dataset.channels}, height={dataset.image_height}, width={dataset.image_width}")
    print(f"Data split: {args.split}")
    print(f"Frames shape: {dataset.frames_shape}")
    print(f"Number of evaluated transitions: {len(dataset)}")
    print(f"MSE/RMSE/MAE: {metrics['mse']:.8f} / {metrics['rmse']:.8f} / {metrics['mae']:.8f}")
    print(f"Saved metrics: {output_dir / 'moving_mnist_transfer_metrics.json'}")
    print(f"Saved qualitative samples: {output_dir / 'qualitative'}")


if __name__ == "__main__":
    main()
