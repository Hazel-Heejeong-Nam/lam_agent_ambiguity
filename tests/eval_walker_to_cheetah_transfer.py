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


DEFAULT_DATA_DIR = "/users/hnam16/scratch/dcs"
DEFAULT_TEST_FILE = "cheetah-run.hdf5"
DEFAULT_TRAIN_FILE = "walker-run.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate walker-run OTF-VQ-VAE checkpoints on cheetah-run motion reconstruction."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional explicit HDF5 path. Overrides --split and --data_dir/--test_file/--train_file.",
    )
    parser.add_argument("--split", type=str, choices=("test", "train"), default="test")
    parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--train_file", type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output_dir", type=str, default="eval/walker_to_cheetah_transfer_eval")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_vis_samples", type=int, default=8)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
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


def sorted_hdf5_keys(keys: Iterable[str]) -> list[str]:
    return sorted(keys, key=lambda value: int(value) if str(value).isdigit() else str(value))


class WalkerToCheetahTransferDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        cfg: DictConfig,
        motion_input_type: str,
        motion_transform: str,
        use_reference_conditioning: bool,
        max_sequences: Optional[int] = None,
    ) -> None:
        import h5py

        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if data_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"Expected one cheetah-run HDF5 file, got {data_path}")

        self.path = data_path
        self.obs_key = str(get_cfg_value(cfg, "data", "obs_key", "obs"))
        self.motion_input_type = train_main.validate_motion_input_type(motion_input_type)
        self.motion_transform = train_main.validate_motion_transform(motion_transform)
        self.use_reference_conditioning = bool(use_reference_conditioning)
        self.tau = get_data_tau(cfg)
        if self.tau < 1:
            raise ValueError(f"tau must be positive, got {self.tau}")

        self.channels = int(cfg.model.channels)
        self.image_height = int(cfg.model.image_height)
        self.image_width = int(cfg.model.image_width)
        self.scale_uint8 = bool(get_cfg_value(cfg, "data", "scale_uint8", True))
        self.resize_to_input = bool(get_cfg_value(cfg, "data", "resize_to_input", False))
        self._handle = None

        with h5py.File(data_path, "r") as handle:
            trajectory_keys = [str(key) for key in handle.keys() if self.obs_key in handle[key]]
            trajectory_keys = sorted_hdf5_keys(trajectory_keys)
            if max_sequences is not None:
                trajectory_keys = trajectory_keys[: int(max_sequences)]
            if not trajectory_keys:
                raise ValueError(f"No trajectories with obs_key='{self.obs_key}' found in {data_path}")

            first_shape = tuple(handle[trajectory_keys[0]][self.obs_key].shape)
            self.frames_shape = (len(trajectory_keys),) + first_shape
            self.trajectory_lengths = {
                key: int(handle[key][self.obs_key].shape[0])
                for key in trajectory_keys
            }

            required_context = 2 * self.tau if self.motion_input_type == "acceleration" else self.tau
            self.indices = []
            for key in trajectory_keys:
                sequence_length = self.trajectory_lengths[key]
                if sequence_length <= required_context:
                    continue
                if self.motion_input_type == "acceleration":
                    time_indices = range(self.tau, sequence_length - self.tau)
                else:
                    time_indices = range(0, sequence_length - self.tau)
                for time_idx in time_indices:
                    self.indices.append((key, time_idx))

        if not self.indices:
            raise ValueError(
                f"No valid transitions found in {data_path} for tau={self.tau} "
                f"and motion_input_type={self.motion_input_type}"
            )

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _get_handle(self):
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def num_videos(self) -> int:
        return len(self.trajectory_lengths)

    def _load_frame(self, trajectory_key: str, time_idx: int) -> torch.Tensor:
        frame = self._get_handle()[trajectory_key][self.obs_key][time_idx]
        frame = train_main.frame_to_chw(frame, self.channels, self.scale_uint8)
        if tuple(frame.shape[-2:]) != (self.image_height, self.image_width):
            if not self.resize_to_input:
                raise ValueError(
                    f"Frame size {tuple(frame.shape[-2:])} does not match expected "
                    f"{(self.image_height, self.image_width)}. The checkpoint config has "
                    "data.resize_to_input=false."
                )
            frame = train_main.resize_frame(frame, self.image_height, self.image_width)
        return frame

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        trajectory_key, time_idx = self.indices[index]
        sample = {
            "current": self._load_frame(trajectory_key, time_idx),
            "next": self._load_frame(trajectory_key, time_idx + self.tau),
            "trajectory_index": torch.tensor(int(trajectory_key) if trajectory_key.isdigit() else index, dtype=torch.long),
            "time_index": torch.tensor(time_idx, dtype=torch.long),
            "motion_ids": torch.tensor(0, dtype=torch.long),
        }
        if self.motion_input_type == "acceleration":
            sample["previous"] = self._load_frame(trajectory_key, time_idx - self.tau)
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
    dataset: WalkerToCheetahTransferDataset,
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

            trajectory_key, time_idx = dataset.indices[dataset_index]
            caption = f"dataset_index={dataset_index}, trajectory={trajectory_key}, time={time_idx}"
            wandb_images[f"qualitative/sample_{sample_number:03d}"] = wandb.Image(panel, caption=caption)

    if wandb_run is not None and wandb_images:
        wandb_run.log(wandb_images)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_data_path(args: argparse.Namespace, cfg: DictConfig) -> Path:
    if args.data_path is not None:
        return Path(args.data_path).expanduser()
    if args.split == "train":
        train_data_path = str(get_cfg_value(cfg, "data", "path", ""))
        if train_data_path:
            return Path(train_data_path).expanduser()
        return Path(args.data_dir).expanduser() / args.train_file
    return Path(args.data_dir).expanduser() / args.test_file


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    device = train_main.choose_device(args.device)

    checkpoint_path = Path(args.checkpoint_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, cfg = load_checkpoint_and_config(checkpoint_path, device)
    data_path = resolve_data_path(args, cfg)
    model = build_frozen_model(cfg, checkpoint, device)
    model_type = model.model_type
    eval_dataset = "walker-run" if args.split == "train" else "cheetah-run"
    eval_motion_input_type = model.motion_input_type
    eval_motion_transform = model.motion_transform
    eval_use_reference_conditioning = bool(model.use_reference_conditioning)

    dataset = WalkerToCheetahTransferDataset(
        data_path=data_path,
        cfg=cfg,
        motion_input_type=eval_motion_input_type,
        motion_transform=eval_motion_transform,
        use_reference_conditioning=eval_use_reference_conditioning,
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
                "data_path": str(data_path),
                "split": args.split,
                "output_dir": str(output_dir),
                "source_dataset": "walker-run",
                "target_dataset": eval_dataset,
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
        "train_data_path": str(get_cfg_value(cfg, "data", "path", "")),
        "test_data_path": str(data_path),
        "source_dataset": "walker-run",
        "target_dataset": eval_dataset,
        "split": args.split,
        "model_type": model_type,
        "num_videos": dataset.num_videos,
        "num_trajectories": dataset.num_videos,
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
    write_json(output_dir / "walker_to_cheetah_transfer_metrics.json", metrics_payload)

    by_motion_payload = {
        "checkpoint_path": str(checkpoint_path),
        "config_source": "checkpoint",
        "train_data_path": str(get_cfg_value(cfg, "data", "path", "")),
        "test_data_path": str(data_path),
        "source_dataset": "walker-run",
        "target_dataset": eval_dataset,
        "split": args.split,
        "max_sequences": args.max_sequences,
        "tau": dataset.tau,
        "model_type": model_type,
        "motion_input_type": eval_motion_input_type,
        "motion_transform": eval_motion_transform,
        "use_reference_conditioning": eval_use_reference_conditioning,
        "metrics_by_motion": finalize_motion_breakdown(by_motion),
    }
    write_json(output_dir / "walker_to_cheetah_transfer_metrics_by_motion.json", by_motion_payload)

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
    print(f"Train data path: {get_cfg_value(cfg, 'data', 'path', '')}")
    print(f"Data split: {args.split}")
    print(f"Eval data path: {data_path}")
    print(f"Model type: {model_type}")
    print(f"Tau: {dataset.tau}")
    print(f"Motion input type: {eval_motion_input_type}")
    print(f"Motion transform: {eval_motion_transform}")
    print(f"Reference conditioning: {eval_use_reference_conditioning}")
    print(f"Model input shape: channels={dataset.channels}, height={dataset.image_height}, width={dataset.image_width}")
    print(f"Frames shape: {dataset.frames_shape}")
    print(f"Number of evaluated transitions: {len(dataset)}")
    print(f"MSE/RMSE/MAE: {metrics['mse']:.8f} / {metrics['rmse']:.8f} / {metrics['mae']:.8f}")
    print(f"Saved metrics: {output_dir / 'walker_to_cheetah_transfer_metrics.json'}")
    print(f"Saved qualitative samples: {output_dir / 'qualitative'}")


if __name__ == "__main__":
    main()
