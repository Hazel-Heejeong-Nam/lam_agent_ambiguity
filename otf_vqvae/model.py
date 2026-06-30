from __future__ import annotations

import math
import io
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

try:
    from .motion_transforms import (
        compute_motion_signal,
        describe_motion_computation,
        motion_transform_output_channels,
        save_debug_motion_inputs,
        validate_motion_transform,
    )
except ImportError:
    from motion_transforms import (  # type: ignore
        compute_motion_signal,
        describe_motion_computation,
        motion_transform_output_channels,
        save_debug_motion_inputs,
        validate_motion_transform,
    )


BASE_OVERLAY_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (80, 70, 229),
    (143, 76, 173),
    (76, 175, 80),
)
MOTION_VISUAL_MAX_ABS_VALUE = 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def validate_motion_input_type(motion_input_type: str) -> str:
    motion_input_type = str(motion_input_type)
    if motion_input_type not in {"acceleration", "velocity"}:
        raise ValueError(
            "motion_input_type must be either 'acceleration' or 'velocity', "
            f"got '{motion_input_type}'"
        )
    return motion_input_type


def canonicalize_model_type(model_type: str) -> str:
    model_type = str(model_type).lower()
    if model_type != "otf_vqvae":
        raise ValueError(f"Unknown model type '{model_type}'. Expected 'otf_vqvae'")
    return model_type


def get_model_type(cfg: DictConfig) -> str:
    model_cfg = cfg.model if "model" in cfg else cfg
    if "model_type" in model_cfg:
        return canonicalize_model_type(model_cfg.model_type)
    if "type" in model_cfg:
        return canonicalize_model_type(model_cfg.type)
    return "otf_vqvae"


def describe_motion_signal(motion_input_type: str) -> str:
    motion_input_type = validate_motion_input_type(motion_input_type)
    if motion_input_type == "acceleration":
        return "next - 2 * current + previous"
    return "next - current"


def get_motion_input_type(cfg: DictConfig) -> str:
    value = cfg["motion_input_type"] if "motion_input_type" in cfg else "acceleration"
    return validate_motion_input_type(value)


def get_motion_transform(cfg: DictConfig) -> str:
    value = cfg["motion_transform"] if "motion_transform" in cfg else "none"
    return validate_motion_transform(value)


def get_use_reference_conditioning(cfg: DictConfig) -> bool:
    value = cfg["use_reference_conditioning"] if "use_reference_conditioning" in cfg else True
    return bool(value)


def get_reference_channels(cfg: DictConfig) -> int:
    value = cfg["reference_channels"] if "reference_channels" in cfg else cfg.channels
    return int(value)


def apply_model_config_defaults(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.motion_input_type = get_motion_input_type(cfg)
        cfg.motion_transform = get_motion_transform(cfg)
        cfg.use_reference_conditioning = get_use_reference_conditioning(cfg)
        cfg.reference_channels = get_reference_channels(cfg)
        cfg.motion_channels = motion_transform_output_channels(cfg.motion_transform, int(cfg.channels))


def make_motion_signal(
    batch: Dict[str, torch.Tensor],
    motion_input_type: str = "acceleration",
    motion_transform: str = "none",
) -> torch.Tensor:
    motion_input_type = validate_motion_input_type(motion_input_type)
    motion_transform = validate_motion_transform(motion_transform)
    if "motion" in batch:
        return batch["motion"]
    return compute_motion_signal(batch, motion_input_type, motion_transform)


def add_reference_frame_if_needed(
    sample: Dict[str, torch.Tensor],
    motion_input_type: str,
    use_reference_conditioning: bool,
) -> Dict[str, torch.Tensor]:
    if not use_reference_conditioning:
        return sample
    if motion_input_type == "acceleration":
        sample["reference_frame"] = sample["previous"]
    else:
        sample["reference_frame"] = sample["current"]
    return sample


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(tensor.shape)}")
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    elif image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 image channels, got {image.shape[0]}")
    image = (image * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return image


def signed_tensor_to_uint8_image(
    tensor: torch.Tensor,
    max_abs_value: float = MOTION_VISUAL_MAX_ABS_VALUE,
) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(tensor.shape)}")
    if max_abs_value <= 0.0:
        raise ValueError(f"Expected positive max_abs_value, got {max_abs_value}")

    image = tensor.detach().cpu().clamp(-max_abs_value, max_abs_value)
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    elif image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 image channels, got {image.shape[0]}")
    image = image / (2.0 * max_abs_value) + 0.5
    image = (image * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return image


def draw_patch_grid(
    image: Image.Image,
    patch_height: int,
    patch_width: int,
    foreground: Tuple[int, int, int, int] = (255, 255, 255, 255),
    background: Tuple[int, int, int, int] = (0, 0, 0, 255),
) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    def _draw_lines(color: Tuple[int, int, int, int], line_width: int) -> None:
        for x_coord in range(0, width + 1, patch_width):
            draw.line([(x_coord, 0), (x_coord, height)], fill=color, width=line_width)
        for y_coord in range(0, height + 1, patch_height):
            draw.line([(0, y_coord), (width, y_coord)], fill=color, width=line_width)

    _draw_lines(background, 3)
    _draw_lines(foreground, 1)
    return canvas


def extend_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    if num_colors <= len(BASE_OVERLAY_COLORS):
        return list(BASE_OVERLAY_COLORS[:num_colors])

    palette = list(BASE_OVERLAY_COLORS)
    cmap = plt.get_cmap("tab20", num_colors)
    for color_idx in range(len(BASE_OVERLAY_COLORS), num_colors):
        rgba = cmap(color_idx)
        palette.append(tuple(int(round(channel * 255.0)) for channel in rgba[:3]))
    return palette


def make_quantized_overlay_image(
    frame: torch.Tensor,
    assignment_grid: np.ndarray,
    patch_height: int,
    patch_width: int,
    alpha: float = 1,
) -> Image.Image:
    base = Image.fromarray(tensor_to_uint8_image(frame)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    unique_codes = sorted(int(code) for code in np.unique(assignment_grid))
    colors = extend_palette(len(unique_codes))
    alpha_value = int(round(255.0 * alpha))

    for color, code_id in zip(colors, unique_codes):
        positions = np.argwhere(assignment_grid == code_id)
        for row_idx, col_idx in positions:
            x0 = int(col_idx * patch_width)
            y0 = int(row_idx * patch_height)
            x1 = int((col_idx + 1) * patch_width) - 1
            y1 = int((row_idx + 1) * patch_height) - 1
            draw.rectangle([(x0, y0), (x1, y1)], fill=(*color, alpha_value))

    composited = Image.alpha_composite(base, overlay)
    return draw_patch_grid(composited, patch_height=patch_height, patch_width=patch_width)


def make_source_frame_triptych_image(
    previous_frame: torch.Tensor,
    current_frame: torch.Tensor,
    next_frame: torch.Tensor,
) -> Image.Image:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for axis, title, frame in zip(
        axes,
        ("t-tau", "t", "t+tau"),
        (previous_frame, current_frame, next_frame),
    ):
        axis.imshow(tensor_to_uint8_image(frame))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()
    return image


def ensure_channel_first(
    array: np.ndarray | torch.Tensor,
    channels: int,
    scale_uint8: bool,
) -> torch.Tensor:
    tensor = torch.as_tensor(array)
    if tensor.ndim == 4:
        if channels == 1 and tensor.shape[1] != channels and tensor.shape[-1] != channels:
            tensor = tensor.unsqueeze(2)
        else:
            tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(
            "Expected frames with shape [N,T,C,H,W], [N,T,H,W,C], [N,T,H,W] "
            "for single-channel data, [T,C,H,W], or [T,H,W,C], "
            f"got {tuple(tensor.shape)}"
        )

    if tensor.shape[2] == channels:
        pass
    elif tensor.shape[-1] == channels:
        tensor = tensor.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(
            f"Could not infer channel axis for shape {tuple(tensor.shape)} "
            f"with channels={channels}"
        )

    tensor = tensor.contiguous()
    if scale_uint8 and tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    return tensor


def frame_to_chw(
    array: np.ndarray | torch.Tensor,
    channels: int,
    scale_uint8: bool,
) -> torch.Tensor:
    tensor = torch.as_tensor(array)
    if tensor.ndim != 3:
        raise ValueError(f"Expected one frame with 3 dimensions, got {tuple(tensor.shape)}")

    if tensor.shape[0] == channels:
        pass
    elif tensor.shape[-1] == channels:
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(
            f"Could not infer channel axis for frame shape {tuple(tensor.shape)} "
            f"with channels={channels}"
        )

    tensor = tensor.contiguous()
    if scale_uint8 and tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    return tensor.float()


def resize_frame(frame: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    if tuple(frame.shape[-2:]) == (image_height, image_width):
        return frame
    return F.interpolate(
        frame.unsqueeze(0),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def resize_sequences(frames: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    n, t, c, h, w = frames.shape
    if (h, w) == (image_height, image_width):
        return frames
    flat = frames.reshape(n * t, c, h, w)
    flat = F.interpolate(flat, size=(image_height, image_width), mode="bilinear", align_corners=False)
    return flat.reshape(n, t, c, image_height, image_width)


class SequenceMotionDataset(Dataset):
    def __init__(
        self,
        frames: torch.Tensor,
        tau: int,
        image_height: int,
        image_width: int,
        resize_to_input: bool,
        motion_input_type: str,
        motion_transform: str,
        use_reference_conditioning: bool,
    ) -> None:
        if frames.ndim != 5:
            raise ValueError(f"Expected [N,T,C,H,W] frames, got {tuple(frames.shape)}")
        if tau < 1:
            raise ValueError(f"tau must be positive, got {tau}")

        self.motion_input_type = validate_motion_input_type(motion_input_type)
        self.motion_transform = validate_motion_transform(motion_transform)
        self.use_reference_conditioning = bool(use_reference_conditioning)

        _, sequence_length, _, height, width = frames.shape
        required_context = 2 * tau if self.motion_input_type == "acceleration" else tau
        if sequence_length <= required_context:
            raise ValueError(
                f"sequence_length={sequence_length} is too short for tau={tau}; "
                f"need at least {required_context + 1} frames for {self.motion_input_type}"
            )
        if (height, width) != (image_height, image_width):
            if not resize_to_input:
                raise ValueError(
                    f"Frame size {(height, width)} does not match expected "
                    f"{(image_height, image_width)}. Set data.resize_to_input=true "
                    "to resize loaded data."
                )
            frames = resize_sequences(frames, image_height, image_width)

        self.frames = frames.contiguous()
        self.tau = tau
        if self.motion_input_type == "acceleration":
            time_indices = range(tau, self.frames.shape[1] - tau)
        else:
            time_indices = range(0, self.frames.shape[1] - tau)
        self.indices = [
            (sequence_idx, time_idx)
            for sequence_idx in range(self.frames.shape[0])
            for time_idx in time_indices
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sequence_idx, time_idx = self.indices[index]
        sample = {
            "current": self.frames[sequence_idx, time_idx],
            "next": self.frames[sequence_idx, time_idx + self.tau],
        }
        if self.motion_input_type == "acceleration":
            sample["previous"] = self.frames[sequence_idx, time_idx - self.tau]
        return add_reference_frame_if_needed(
            sample,
            self.motion_input_type,
            self.use_reference_conditioning,
        )


class DCSHDF5MotionDataset(Dataset):
    def __init__(
        self,
        path: Path,
        obs_key: str,
        tau: int,
        channels: int,
        image_height: int,
        image_width: int,
        resize_to_input: bool,
        scale_uint8: bool,
        max_sequences: Optional[int],
        motion_input_type: str,
        motion_transform: str,
        use_reference_conditioning: bool,
        split: str = "train",                  # "train" or "test"
        test_n_trajectories: Optional[int] = None,  # N trajectories to sample at test time
        seed: int = 0,
    ) -> None:
        import h5py

        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"DCS dataset must be one .hdf5 file, got {path}")
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if split == "test" and test_n_trajectories is not None and test_n_trajectories < 1:
            raise ValueError(f"test_n_trajectories must be >= 1, got {test_n_trajectories}")

        self.path = path
        self.obs_key = obs_key
        self.tau = tau
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        self.resize_to_input = resize_to_input
        self.scale_uint8 = scale_uint8
        self.motion_input_type = validate_motion_input_type(motion_input_type)
        self.motion_transform = validate_motion_transform(motion_transform)
        self.use_reference_conditioning = bool(use_reference_conditioning)
        self.split = split
        self.test_n_trajectories = test_n_trajectories
        self.seed = seed
        self._handle = None

        with h5py.File(path, "r") as handle:
            trajectory_keys = [key for key in handle.keys() if obs_key in handle[key]]
            trajectory_keys = sorted(trajectory_keys, key=lambda value: int(value) if value.isdigit() else value)
            if max_sequences is not None:
                trajectory_keys = trajectory_keys[: int(max_sequences)]

            # --- split trajectories into two halves ---
            mid = len(trajectory_keys) // 2
            if split == "train":
                split_keys = trajectory_keys[:mid]
            else:
                split_keys = trajectory_keys[mid:]
                if test_n_trajectories is not None:
                    rng = random.Random(seed)
                    n = min(test_n_trajectories, len(split_keys))
                    split_keys = rng.sample(split_keys, n)

            self.trajectory_keys = split_keys  # expose for inspection if needed

            self.indices = []
            for key in split_keys:
                sequence_length = int(handle[key][obs_key].shape[0])
                required_context = 2 * tau if self.motion_input_type == "acceleration" else tau
                if sequence_length <= required_context:
                    continue
                if self.motion_input_type == "acceleration":
                    time_indices = range(tau, sequence_length - tau)
                else:
                    time_indices = range(0, sequence_length - tau)
                for time_idx in time_indices:
                    self.indices.append((key, time_idx))

        if not self.indices:
            raise ValueError(
                f"No valid motion samples found in {path} with tau={tau} for split={split!r}"
            )

    def __getstate__(self):
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

    def _load_frame(self, trajectory_key: str, time_idx: int) -> torch.Tensor:
        frame = self._get_handle()[trajectory_key][self.obs_key][time_idx]
        frame = frame_to_chw(frame, self.channels, self.scale_uint8)
        if tuple(frame.shape[-2:]) != (self.image_height, self.image_width):
            if not self.resize_to_input:
                raise ValueError(
                    f"Frame size {tuple(frame.shape[-2:])} does not match expected "
                    f"{(self.image_height, self.image_width)}. Set data.resize_to_input=true."
                )
            frame = resize_frame(frame, self.image_height, self.image_width)
        return frame

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        trajectory_key, time_idx = self.indices[index]
        sample = {
            "current": self._load_frame(trajectory_key, time_idx),
            "next": self._load_frame(trajectory_key, time_idx + self.tau),
        }
        if self.motion_input_type == "acceleration":
            sample["previous"] = self._load_frame(trajectory_key, time_idx - self.tau)
        return add_reference_frame_if_needed(
            sample,
            self.motion_input_type,
            self.use_reference_conditioning,
        )


class ClevrerVideoMotionDataset(Dataset):
    def __init__(
        self,
        path: Path,
        video_glob: str,
        recursive: bool,
        tau: int,
        channels: int,
        image_height: int,
        image_width: int,
        resize_to_input: bool,
        scale_uint8: bool,
        max_sequences: Optional[int],
        max_frames_per_video: Optional[int],
        motion_input_type: str,
        motion_transform: str,
        use_reference_conditioning: bool,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_dir():
            raise ValueError(f"CLEVRER data.path must be a directory of .mp4 files, got {path}")

        pattern = f"**/{video_glob}" if recursive else video_glob
        video_paths = sorted(path.glob(pattern))
        if max_sequences is not None:
            video_paths = video_paths[: int(max_sequences)]
        if not video_paths:
            raise ValueError(f"No videos matching '{pattern}' found under {path}")

        self.video_paths = video_paths
        self.tau = tau
        self.channels = channels
        self.image_height = image_height
        self.image_width = image_width
        self.resize_to_input = resize_to_input
        self.scale_uint8 = scale_uint8
        self.max_frames_per_video = max_frames_per_video
        self.motion_input_type = validate_motion_input_type(motion_input_type)
        self.motion_transform = validate_motion_transform(motion_transform)
        self.use_reference_conditioning = bool(use_reference_conditioning)
        self.indices = []

        for video_idx, video_path in enumerate(self.video_paths):
            num_frames = self._count_video_frames(video_path)
            if self.max_frames_per_video is not None:
                num_frames = min(num_frames, int(self.max_frames_per_video))
            required_context = 2 * tau if self.motion_input_type == "acceleration" else tau
            if num_frames <= required_context:
                continue
            if self.motion_input_type == "acceleration":
                time_indices = range(tau, num_frames - tau)
            else:
                time_indices = range(0, num_frames - tau)
            for time_idx in time_indices:
                self.indices.append((video_idx, time_idx))

        if not self.indices:
            raise ValueError(f"No valid motion samples found in {path} with tau={tau}")

    def _count_video_frames(self, video_path: Path) -> int:
        import imageio.v3 as iio

        count = 0
        for _ in iio.imiter(video_path):
            count += 1
            if self.max_frames_per_video is not None and count >= int(self.max_frames_per_video):
                break
        return count

    def _convert_video_frame(self, frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = frame_to_chw(frame, self.channels, self.scale_uint8)
        if tuple(tensor.shape[-2:]) != (self.image_height, self.image_width):
            if not self.resize_to_input:
                raise ValueError(
                    f"Frame size {tuple(tensor.shape[-2:])} does not match expected "
                    f"{(self.image_height, self.image_width)}. Set data.resize_to_input=true."
                )
            tensor = resize_frame(tensor, self.image_height, self.image_width)
        return tensor

    def _read_frames(self, video_path: Path, time_indices: Tuple[int, ...]) -> Dict[int, torch.Tensor]:
        import imageio.v3 as iio

        targets = set(time_indices)
        frames = {}
        for frame_idx, frame in enumerate(iio.imiter(video_path)):
            if frame_idx in targets:
                frames[frame_idx] = self._convert_video_frame(frame)
                if len(frames) == len(targets):
                    return frames
        missing = sorted(targets.difference(frames))
        raise IndexError(f"Frames {missing} were not found in {video_path}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        video_idx, time_idx = self.indices[index]
        video_path = self.video_paths[video_idx]
        if self.motion_input_type == "acceleration":
            time_indices = (time_idx - self.tau, time_idx, time_idx + self.tau)
        else:
            time_indices = (time_idx, time_idx + self.tau)
        frames = self._read_frames(video_path, time_indices)
        if self.motion_input_type == "acceleration":
            sample = {
                "previous": frames[time_indices[0]],
                "current": frames[time_indices[1]],
                "next": frames[time_indices[2]],
            }
        else:
            sample = {
                "current": frames[time_indices[0]],
                "next": frames[time_indices[1]],
            }
        return add_reference_frame_if_needed(
            sample,
            self.motion_input_type,
            self.use_reference_conditioning,
        )


def make_mock_frames(cfg: DictConfig) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    num_sequences = int(cfg.num_sequences)
    sequence_length = int(cfg.sequence_length)
    channels = int(cfg.channels)
    height = int(cfg.image_height)
    width = int(cfg.image_width)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    frames = torch.empty(num_sequences, sequence_length, channels, height, width)

    for sequence_idx in range(num_sequences):
        background = 0.08 * torch.rand(channels, height, width, generator=generator)
        color = 0.35 + 0.65 * torch.rand(channels, 1, 1, generator=generator)
        phase_x = 2.0 * math.pi * torch.rand((), generator=generator).item()
        phase_y = 2.0 * math.pi * torch.rand((), generator=generator).item()
        sigma = float(torch.randint(5, 11, (), generator=generator).item())
        amp_x = 0.25 * width + 0.08 * width * torch.rand((), generator=generator).item()
        amp_y = 0.25 * height + 0.08 * height * torch.rand((), generator=generator).item()

        for time_idx in range(sequence_length):
            progress = time_idx / max(1, sequence_length - 1)
            center_x = width / 2.0 + amp_x * math.sin(2.0 * math.pi * progress + phase_x)
            center_y = height / 2.0 + amp_y * math.cos(2.0 * math.pi * progress + phase_y)
            dist_sq = (x_grid - center_x).pow(2) + (y_grid - center_y).pow(2)
            blob = torch.exp(-dist_sq / (2.0 * sigma * sigma)).unsqueeze(0)
            frame = background + color * blob
            frames[sequence_idx, time_idx] = frame.clamp(0.0, 1.0)

    return frames


def load_frame_array(cfg: DictConfig) -> torch.Tensor:
    path = Path(str(cfg.path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pt" or suffix == ".pth":
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            loaded = loaded[str(cfg.sequence_key)]
    elif suffix == ".npy":
        loaded = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        loaded = archive[str(cfg.sequence_key)]
    elif suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as handle:
            loaded = handle[str(cfg.sequence_key)][()]
    else:
        raise ValueError(f"Unsupported dataset suffix '{suffix}' for {path}")

    frames = ensure_channel_first(loaded, int(cfg.channels), bool(cfg.scale_uint8))
    if cfg.max_sequences is not None:
        frames = frames[: int(cfg.max_sequences)]
    return frames


def load_mnist_frame_array(cfg: DictConfig) -> torch.Tensor:
    path = Path(str(cfg.path)).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Moving MNIST dataset must be one .npz file, got {path}")

    sequence_key = str(cfg.sequence_key)
    with np.load(path) as archive:
        if sequence_key not in archive:
            available = ", ".join(archive.files)
            raise KeyError(f"Key '{sequence_key}' not found in {path}. Available keys: {available}")
        loaded = archive[sequence_key]

    frames = ensure_channel_first(loaded, int(cfg.channels), bool(cfg.scale_uint8))
    if cfg.max_sequences is not None:
        frames = frames[: int(cfg.max_sequences)]
    return frames


def build_dataset(
    cfg: DictConfig,
    motion_input_type: str = "acceleration",
    motion_transform: str = "none",
    use_reference_conditioning: bool = True,
) -> Dataset:
    motion_input_type = validate_motion_input_type(motion_input_type)
    motion_transform = validate_motion_transform(motion_transform)
    if cfg.type == "mock":
        frames = make_mock_frames(cfg)
    elif cfg.type == "frames":
        frames = load_frame_array(cfg)
    elif cfg.type == "mnist":
        frames = load_mnist_frame_array(cfg)
    elif cfg.type == "clevrer":
        return ClevrerVideoMotionDataset(
            path=Path(str(cfg.path)).expanduser(),
            video_glob=str(cfg.video_glob),
            recursive=bool(cfg.recursive),
            tau=int(cfg.tau),
            channels=int(cfg.channels),
            image_height=int(cfg.image_height),
            image_width=int(cfg.image_width),
            resize_to_input=bool(cfg.resize_to_input),
            scale_uint8=bool(cfg.scale_uint8),
            max_sequences=cfg.max_sequences,
            max_frames_per_video=cfg.max_frames_per_video,
            motion_input_type=motion_input_type,
            motion_transform=motion_transform,
            use_reference_conditioning=use_reference_conditioning,
        )
    elif cfg.type in {"cheetah-run", "walker-run"}:
        return DCSHDF5MotionDataset(
            path=Path(str(cfg.path)).expanduser(),
            obs_key=str(cfg.obs_key),
            tau=int(cfg.tau),
            channels=int(cfg.channels),
            image_height=int(cfg.image_height),
            image_width=int(cfg.image_width),
            resize_to_input=bool(cfg.resize_to_input),
            scale_uint8=bool(cfg.scale_uint8),
            max_sequences=cfg.max_sequences,
            motion_input_type=motion_input_type,
            motion_transform=motion_transform,
            use_reference_conditioning=use_reference_conditioning,
            split=str(cfg.get("split", "train")),
            test_n_trajectories=cfg.get("test_n_trajectories", None),
            seed=int(cfg.get("seed", 0)),
        )
    else:
        raise ValueError(f"Unknown data.type '{cfg.type}'")

    return SequenceMotionDataset(
        frames=frames,
        tau=int(cfg.tau),
        image_height=int(cfg.image_height),
        image_width=int(cfg.image_width),
        resize_to_input=bool(cfg.resize_to_input),
        motion_input_type=motion_input_type,
        motion_transform=motion_transform,
        use_reference_conditioning=use_reference_conditioning,
    )


def make_patch_coordinates(grid_height: int, grid_width: int, normalized: bool) -> torch.Tensor:
    if normalized:
        rows = torch.linspace(-1.0, 1.0, grid_height)
        cols = torch.linspace(-1.0, 1.0, grid_width)
    else:
        rows = torch.arange(grid_height, dtype=torch.float32)
        cols = torch.arange(grid_width, dtype=torch.float32)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([row_grid, col_grid], dim=-1).reshape(grid_height * grid_width, 2)


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'")


class PatchEncoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.channels = int(cfg.motion_channels) if "motion_channels" in cfg else int(cfg.channels)
        self.image_height = int(cfg.image_height)
        self.image_width = int(cfg.image_width)
        self.patch_height = int(cfg.patch_height)
        self.patch_width = int(cfg.patch_width)
        self.grid_height = self.image_height // self.patch_height
        self.grid_width = self.image_width // self.patch_width
        self.use_positional_encoding = bool(cfg.use_positional_encoding)

        if self.image_height % self.patch_height != 0 or self.image_width % self.patch_width != 0:
            raise ValueError(
                "image_height and image_width must be divisible by patch_height and patch_width"
            )

        patch_dim = self.channels * self.patch_height * self.patch_width
        pe_dim = 2 if self.use_positional_encoding else 0
        hidden_dim = int(cfg.encoder_hidden_dim)
        latent_dim = int(cfg.latent_dim)
        activation = make_activation(str(cfg.activation))

        self.unfold = nn.Unfold(
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        self.net = nn.Sequential(
            nn.Linear(patch_dim + pe_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, latent_dim),
        )
        self.register_buffer(
            "positional_encoding",
            make_patch_coordinates(self.grid_height, self.grid_width, normalized=True),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.unfold(x).transpose(1, 2)
        if self.use_positional_encoding:
            pe = self.positional_encoding.unsqueeze(0).expand(patches.shape[0], -1, -1)
            patches = torch.cat([patches, pe.to(patches.dtype)], dim=-1)
        return self.net(patches)


class ReferenceFrameEncoder(nn.Module):
    def __init__(self, cfg: DictConfig, grid_height: int, grid_width: int) -> None:
        super().__init__()
        self.target_size = (grid_height, grid_width)
        channels = get_reference_channels(cfg)
        hidden_dim = int(cfg.reference_hidden_dim)
        self.output_channels = int(cfg.reference_feature_dim)
        activation_name = str(cfg.activation)

        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            make_activation(activation_name),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            make_activation(activation_name),
            nn.Conv2d(hidden_dim, self.output_channels, kernel_size=3, stride=2, padding=1),
            make_activation(activation_name),
        )

    def forward(self, reference_frame: torch.Tensor) -> torch.Tensor:
        features = self.net(reference_frame)
        if tuple(features.shape[-2:]) != self.target_size:
            features = F.interpolate(
                features,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
        return features


class FactorDecoder(nn.Module):
    def __init__(self, cfg: DictConfig, grid_height: int, grid_width: int) -> None:
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.image_height = int(cfg.image_height)
        self.image_width = int(cfg.image_width)
        self.channels = int(cfg.motion_channels) if "motion_channels" in cfg else int(cfg.channels)
        latent_dim = int(cfg.latent_dim)
        descriptor_hidden_dim = int(cfg.descriptor_hidden_dim)
        decoder_feature_dim = int(cfg.decoder_feature_dim)
        decoder_hidden_dim = int(cfg.decoder_hidden_dim)
        self.use_reference_conditioning = get_use_reference_conditioning(cfg)
        reference_feature_dim = int(cfg.reference_feature_dim) if self.use_reference_conditioning else 0
        activation_name = str(cfg.activation)

        self.descriptor_mlp = nn.Sequential(
            nn.Linear(latent_dim + 1, descriptor_hidden_dim),
            make_activation(activation_name),
            nn.Linear(descriptor_hidden_dim, decoder_feature_dim),
        )
        self.continuous_projection = nn.Sequential(
            nn.Linear(latent_dim, decoder_feature_dim),
            make_activation(activation_name),
            nn.Linear(decoder_feature_dim, decoder_feature_dim),
        )
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(
                decoder_feature_dim + reference_feature_dim,
                decoder_hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            make_activation(activation_name),
            nn.Upsample(size=(self.image_height, self.image_width), mode="bilinear", align_corners=False),
            nn.Conv2d(decoder_hidden_dim, decoder_hidden_dim, kernel_size=3, padding=1),
            make_activation(activation_name),
            nn.Conv2d(decoder_hidden_dim, self.channels, kernel_size=3, padding=1),
        )

    def _decode_motion_features(
        self,
        motion_features: torch.Tensor,
        reference_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_reference_conditioning:
            if reference_features is None:
                raise ValueError("reference_features are required when use_reference_conditioning=true")
            if tuple(reference_features.shape[-2:]) != (self.grid_height, self.grid_width):
                reference_features = F.interpolate(
                    reference_features,
                    size=(self.grid_height, self.grid_width),
                    mode="bilinear",
                    align_corners=False,
                )
            features = torch.cat([motion_features, reference_features], dim=1)
        else:
            features = motion_features
        return self.conv_decoder(features)

    def decode_continuous(
        self,
        patch_embeddings: torch.Tensor,
        reference_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = patch_embeddings.shape[0]
        features = self.continuous_projection(patch_embeddings)
        features = features.reshape(batch_size, self.grid_height, self.grid_width, -1)
        features = features.permute(0, 3, 1, 2).contiguous()
        return self._decode_motion_features(features, reference_features)

    def decode_factors(
        self,
        code_embeddings: torch.Tensor,
        weights: torch.Tensor,
        occupancy_maps: torch.Tensor,
        reference_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # h_{t,k}: shared code feature from code content and transition-level usage weight.
        descriptors = torch.cat([code_embeddings, weights.unsqueeze(-1)], dim=-1)
        code_features = self.descriptor_mlp(descriptors)
        occupancy_maps = occupancy_maps.to(code_features.dtype)
        # Broadcast h_{t,k} only to patch locations assigned to code k, then sum over codes.
        feature_map = torch.einsum("bkhw,bkf->bfhw", occupancy_maps, code_features)
        return self._decode_motion_features(feature_map, reference_features)


class VectorQuantizer(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.num_codes = int(cfg.codebook_size)
        self.latent_dim = int(cfg.latent_dim)
        self.update_mode = str(cfg.codebook_update)
        self.ema_decay = float(cfg.ema_decay)
        self.dead_code_steps = int(cfg.dead_code_steps)
        self.random_std = float(cfg.random_std)

        if self.update_mode not in {"ema", "gradient"}:
            raise ValueError("model.codebook_update must be 'ema' or 'gradient'")

        initial = torch.randn(self.num_codes, self.latent_dim) * self.random_std
        self.embedding = nn.Parameter(initial)
        self.register_buffer("last_used_step", torch.zeros(self.num_codes, dtype=torch.long))
        initialized = str(cfg.codebook_init) == "random_normal"
        self.register_buffer("initialized", torch.tensor(initialized, dtype=torch.bool))
        self._initialized = bool(initialized)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._initialized = bool(self.initialized.detach().cpu().item())

    def is_initialized(self) -> bool:
        return self._initialized

    def set_initialized(self, initialized: bool = True) -> None:
        self.initialized.fill_(bool(initialized))
        self._initialized = bool(initialized)

    def set_codebook(self, centers: torch.Tensor, step: int) -> None:
        if centers.shape != self.embedding.shape:
            raise ValueError(f"Expected centers {tuple(self.embedding.shape)}, got {tuple(centers.shape)}")
        with torch.no_grad():
            self.embedding.copy_(centers.to(self.embedding.device, self.embedding.dtype))
            self.last_used_step.fill_(int(step))
            self.set_initialized(True)

    def forward(self, patch_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self._initialized:
            raise RuntimeError("Codebook is not initialized. Run k-means initialization first.")

        flat = patch_embeddings.reshape(-1, self.latent_dim)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1).unsqueeze(0)
        )
        indices = torch.argmin(distances, dim=1).reshape(patch_embeddings.shape[0], patch_embeddings.shape[1])
        quantized = F.embedding(indices, self.embedding)
        quantized_st = patch_embeddings + (quantized - patch_embeddings).detach()

        if self.update_mode == "gradient":
            code_loss = F.mse_loss(quantized, patch_embeddings.detach())
        else:
            code_loss = F.mse_loss(quantized.detach(), patch_embeddings.detach())
        commit_loss = F.mse_loss(patch_embeddings, quantized.detach())

        return {
            "indices": indices,
            "quantized": quantized,
            "quantized_st": quantized_st,
            "code_loss": code_loss,
            "commit_loss": commit_loss,
        }

    @torch.no_grad()
    def update_codebook(self, patch_embeddings: torch.Tensor, indices: torch.Tensor, step: int) -> Dict[str, int]:
        flat_embeddings = patch_embeddings.detach().reshape(-1, self.latent_dim)
        flat_indices = indices.detach().reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.num_codes)
        active = counts > 0
        self.last_used_step[active] = int(step)

        if self.update_mode == "ema":
            sums = torch.zeros_like(self.embedding)
            sums.index_add_(0, flat_indices, flat_embeddings)
            means = sums[active] / counts[active].to(sums.dtype).unsqueeze(-1)
            self.embedding.data[active] = (
                self.ema_decay * self.embedding.data[active] + (1.0 - self.ema_decay) * means
            )

        reinitialized = 0
        if self.dead_code_steps > 0:
            dead = (int(step) - self.last_used_step) >= self.dead_code_steps
            if dead.any():
                dead_indices = torch.nonzero(dead, as_tuple=False).flatten()
                replacement_ids = torch.randint(
                    low=0,
                    high=flat_embeddings.shape[0],
                    size=(dead_indices.numel(),),
                    device=flat_embeddings.device,
                )
                self.embedding.data[dead_indices] = flat_embeddings[replacement_ids]
                self.last_used_step[dead_indices] = int(step)
                reinitialized = int(dead_indices.numel())
        return {"reinitialized_codes": reinitialized}

    def orthogonality_loss(self, indices: Optional[torch.Tensor], active_only: bool) -> torch.Tensor:
        if active_only and indices is not None:
            code_ids = torch.unique(indices.detach())
        else:
            code_ids = torch.arange(self.num_codes, device=self.embedding.device)
        if code_ids.numel() <= 1:
            return self.embedding.sum() * 0.0

        codes = F.normalize(self.embedding[code_ids], p=2, dim=-1)
        gram = codes @ codes.t()
        eye = torch.eye(code_ids.numel(), device=gram.device, dtype=gram.dtype)
        return (gram - eye).pow(2).sum()


class OTFVQVAE(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        apply_model_config_defaults(cfg)
        self.cfg = cfg
        self.motion_input_type = get_motion_input_type(cfg)
        self.motion_transform = get_motion_transform(cfg)
        self.use_reference_conditioning = get_use_reference_conditioning(cfg)
        self.reference_channels = get_reference_channels(cfg)
        self.motion_channels = int(cfg.motion_channels)
        self.encoder = PatchEncoder(cfg)
        if self.use_reference_conditioning:
            self.reference_encoder = ReferenceFrameEncoder(cfg, self.encoder.grid_height, self.encoder.grid_width)
        self.quantizer = VectorQuantizer(cfg)
        self.decoder = FactorDecoder(cfg, self.encoder.grid_height, self.encoder.grid_width)
        self.num_patches = self.encoder.grid_height * self.encoder.grid_width
        self.num_codes = int(cfg.codebook_size)
        self.summary_eps = float(cfg.summary_eps)
        self.orth_active_only = bool(cfg.orth_active_only)
        self.register_buffer(
            "patch_coordinates",
            make_patch_coordinates(self.encoder.grid_height, self.encoder.grid_width, normalized=False),
            persistent=False,
        )

    def summarize_assignments(
        self,
        indices: torch.Tensor,
        quantized_st: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Summarize hard patch assignments without discarding their patch-grid layout."""
        one_hot = F.one_hot(indices, num_classes=self.num_codes).to(quantized_st.dtype)
        counts = one_hot.sum(dim=1)
        weights = counts / float(self.num_patches)
        # [batch, K, H_p, W_p], one binary map per code from integer patch assignments.
        occupancy_maps = one_hot.transpose(1, 2).reshape(
            indices.shape[0],
            self.num_codes,
            self.encoder.grid_height,
            self.encoder.grid_width,
        )
        centroids = torch.einsum("bpk,pd->bkd", one_hot, self.patch_coordinates.to(one_hot.dtype))
        centroids = centroids / (counts.unsqueeze(-1) + self.summary_eps)
        active_mask = counts > 0
        code_embeddings = torch.einsum("bpk,bpd->bkd", one_hot, quantized_st)
        code_embeddings = code_embeddings / (counts.unsqueeze(-1) + self.summary_eps)
        return weights, occupancy_maps, active_mask, code_embeddings, centroids

    def forward(
        self,
        motion: torch.Tensor,
        reference_frame: Optional[torch.Tensor] = None,
        use_quantization: bool = True,
    ) -> Dict[str, torch.Tensor]:
        patch_embeddings = self.encoder(motion)
        reference_features = None
        if self.use_reference_conditioning:
            if reference_frame is None:
                raise ValueError("reference_frame is required when use_reference_conditioning=true")
            reference_features = self.reference_encoder(reference_frame)
        zero = motion.new_tensor(0.0)

        if not use_quantization:
            reconstruction = self.decoder.decode_continuous(patch_embeddings, reference_features)
            return {
                "reconstruction": reconstruction,
                "patch_embeddings": patch_embeddings,
                "indices": None,
                "code_loss": zero,
                "commit_loss": zero,
                "orth_loss": zero,
                "active_codes": zero,
            }

        vq_output = self.quantizer(patch_embeddings)
        weights, occupancy_maps, active_mask, code_embeddings, centroids = self.summarize_assignments(
            vq_output["indices"],
            vq_output["quantized_st"],
        )
        reconstruction = self.decoder.decode_factors(
            code_embeddings,
            weights,
            occupancy_maps,
            reference_features,
        )
        orth_loss = self.quantizer.orthogonality_loss(vq_output["indices"], self.orth_active_only)

        return {
            "reconstruction": reconstruction,
            "patch_embeddings": patch_embeddings,
            "indices": vq_output["indices"],
            "weights": weights,
            "occupancy_maps": occupancy_maps,
            "centroids": centroids,
            "code_loss": vq_output["code_loss"],
            "commit_loss": vq_output["commit_loss"],
            "orth_loss": orth_loss,
            "active_codes": active_mask.any(dim=0).sum().to(motion.dtype),
        }


@torch.no_grad()
def collect_patch_embeddings(
    model: OTFVQVAE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    max_tokens: int,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    chunks = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        motion = make_motion_signal(batch, model.motion_input_type, model.motion_transform)
        embeddings = model.encoder(motion).reshape(-1, model.cfg.latent_dim)
        chunks.append(embeddings.cpu())
        if sum(chunk.shape[0] for chunk in chunks) >= max_tokens:
            break
    if was_training:
        model.train()
    if not chunks:
        raise RuntimeError("No patch embeddings were collected for k-means initialization")
    return torch.cat(chunks, dim=0)[:max_tokens]


def run_kmeans(samples: torch.Tensor, num_centers: int, num_iters: int, seed: int) -> torch.Tensor:
    if samples.shape[0] < num_centers:
        repeat = math.ceil(num_centers / samples.shape[0])
        samples = samples.repeat(repeat, 1)

    generator = torch.Generator(device=samples.device).manual_seed(seed)
    perm = torch.randperm(samples.shape[0], generator=generator, device=samples.device)
    centers = samples[perm[:num_centers]].clone()

    for _ in range(num_iters):
        distances = (
            samples.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * samples @ centers.t()
            + centers.pow(2).sum(dim=1).unsqueeze(0)
        )
        assignments = torch.argmin(distances, dim=1)
        counts = torch.bincount(assignments, minlength=num_centers).to(samples.dtype)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, samples)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)
        if (~nonempty).any():
            replacement = torch.randint(
                low=0,
                high=samples.shape[0],
                size=(int((~nonempty).sum().item()),),
                generator=generator,
                device=samples.device,
            )
            centers[~nonempty] = samples[replacement]

    return centers


def initialize_codebook_with_kmeans(
    model: OTFVQVAE,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
    step: int,
) -> None:
    samples = collect_patch_embeddings(
        model=model,
        loader=loader,
        device=device,
        max_batches=int(cfg.kmeans_batches),
        max_tokens=int(cfg.kmeans_max_tokens),
    ).to(device)
    centers = run_kmeans(
        samples=samples,
        num_centers=int(cfg.codebook_size),
        num_iters=int(cfg.kmeans_iters),
        seed=int(cfg.kmeans_seed),
    )
    model.quantizer.set_codebook(centers, step=step)


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )


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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    step: int,
    final: bool,
) -> Path:
    checkpoint_dir = Path(str(cfg.checkpoint.dir)).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "final" if final else f"step{step:06d}"
    filename = f"{cfg.run_name}_{suffix}.pt"
    path = unique_path(checkpoint_dir / filename)
    checkpoint = {
        "global_step": step,
        "model_type": get_model_type(cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    for attr_name in (
        "motion_input_type",
        "motion_transform",
        "use_reference_conditioning",
        "latent_action_dim",
    ):
        if hasattr(model, attr_name):
            checkpoint[attr_name] = getattr(model, attr_name)
    torch.save(checkpoint, path)
    return path


def get_checkpoint_model_setting(checkpoint: Dict[str, object], key: str, default):
    if key in checkpoint:
        return checkpoint[key]
    config = checkpoint.get("config", {})
    if isinstance(config, DictConfig):
        if "model" in config and key in config.model:
            return config.model[key]
    elif isinstance(config, dict):
        model_config = config.get("model", {})
        if isinstance(model_config, dict) and key in model_config:
            return model_config[key]
    return default


def apply_checkpoint_model_settings(cfg: DictConfig, checkpoint: Dict[str, object]) -> None:
    with open_dict(cfg.model):
        cfg.model.model_type = canonicalize_model_type(
            get_checkpoint_model_setting(checkpoint, "model_type", get_model_type(cfg))
        )
        cfg.model.motion_input_type = validate_motion_input_type(
            get_checkpoint_model_setting(checkpoint, "motion_input_type", "acceleration")
        )
        cfg.model.motion_transform = validate_motion_transform(
            get_checkpoint_model_setting(checkpoint, "motion_transform", "none")
        )
        cfg.model.use_reference_conditioning = bool(
            get_checkpoint_model_setting(checkpoint, "use_reference_conditioning", True)
        )


def load_resume_checkpoint_and_apply_model_config(
    cfg: DictConfig,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    if cfg.train.resume_path is None:
        apply_model_config_defaults(cfg.model)
        return None
    checkpoint = torch.load(Path(str(cfg.train.resume_path)).expanduser(), map_location=device)
    apply_checkpoint_model_settings(cfg, checkpoint)
    return checkpoint


def maybe_load_checkpoint(
    model: OTFVQVAE,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    checkpoint: Optional[Dict[str, object]] = None,
) -> int:
    if cfg.train.resume_path is None:
        return 0
    if checkpoint is None:
        checkpoint = torch.load(Path(str(cfg.train.resume_path)).expanduser(), map_location=device)
        apply_checkpoint_model_settings(cfg, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    if bool(cfg.train.resume_optimizer) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("global_step", 0))


def write_resolved_config(cfg: DictConfig) -> None:
    output_dir = Path(str(cfg.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = unique_path(output_dir / "resolved_config.yaml")
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def get_optional(cfg: DictConfig, key: str, default):
    return cfg[key] if key in cfg else default


def initialize_wandb(cfg: DictConfig):
    if not bool(cfg.wandb.enabled):
        return None

    import wandb

    wandb_dir = Path(str(cfg.wandb.dir)).expanduser()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    init_kwargs = {
        "project": str(cfg.wandb.project),
        "name": str(cfg.wandb.name),
        "mode": str(cfg.wandb.mode),
        "dir": str(wandb_dir),
        "tags": list(cfg.wandb.tags),
    }
    if cfg.wandb.entity is not None:
        init_kwargs["entity"] = str(cfg.wandb.entity)
    if bool(cfg.wandb.log_config):
        init_kwargs["config"] = OmegaConf.to_container(cfg, resolve=True)
    return wandb.init(**init_kwargs)


def compute_total_steps(cfg: DictConfig, steps_per_epoch: int) -> int:
    num_epochs = int(get_optional(cfg.train, "num_epochs", 1))
    return num_epochs * steps_per_epoch


def compute_warmup_step_count(cfg: DictConfig, steps_per_epoch: int) -> int:
    warmup_epoch = float(get_optional(cfg.train, "warmup_epoch", 1))
    if warmup_epoch < 0:
        raise ValueError("train.warmup_epoch must be non-negative.")
    return math.ceil(warmup_epoch * steps_per_epoch)


def select_validation_sample_indices(dataset_size: int, num_samples: int = 3) -> List[int]:
    if dataset_size <= 0:
        return []
    count = min(num_samples, dataset_size)
    indices = np.linspace(0, dataset_size - 1, num=count, dtype=int).tolist()
    return list(dict.fromkeys(int(index) for index in indices))


@torch.no_grad()
def make_validation_image_log(
    model: OTFVQVAE,
    dataset: Dataset,
    sample_indices: List[int],
    device: torch.device,
    epoch: int,
) -> Dict[str, object]:
    model_was_training = model.training
    model.eval()

    patch_height = int(model.encoder.patch_height)
    patch_width = int(model.encoder.patch_width)
    grid_height = int(model.encoder.grid_height)
    grid_width = int(model.encoder.grid_width)
    log_images: Dict[str, object] = {}

    import wandb

    for sample_rank, dataset_index in enumerate(sample_indices):
        sample = dataset[dataset_index]
        batch = {
            key: value.unsqueeze(0).to(device, non_blocking=True)
            for key, value in sample.items()
        }
        motion = make_motion_signal(batch, model.motion_input_type, model.motion_transform)
        output = model(
            motion,
            reference_frame=batch.get("reference_frame"),
            use_quantization=True,
        )
        assignment_grid = (
            output["indices"][0]
            .reshape(grid_height, grid_width)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        reconstruction = output["reconstruction"][0]

        prefix = f"validation/sample_{sample_rank:02d}_dataset_{dataset_index:06d}"
        current_frame = sample["current"]
        motion_image = Image.fromarray(signed_tensor_to_uint8_image(motion[0]))

        log_images[f"{prefix}/grid"] = wandb.Image(
            draw_patch_grid(
                Image.fromarray(tensor_to_uint8_image(current_frame)),
                patch_height=patch_height,
                patch_width=patch_width,
            ),
            caption=f"epoch {epoch}, frame t",
        )
        log_images[f"{prefix}/motion_grid"] = wandb.Image(
            draw_patch_grid(motion_image, patch_height=patch_height, patch_width=patch_width),
            caption=f"epoch {epoch}, motion grid",
        )
        if "previous" in sample:
            log_images[f"{prefix}/source_frames"] = wandb.Image(
                make_source_frame_triptych_image(
                    previous_frame=sample["previous"],
                    current_frame=sample["current"],
                    next_frame=sample["next"],
                ),
                caption=f"epoch {epoch}: t-tau, t, t+tau",
            )
        log_images[f"{prefix}/quantized_overlay"] = wandb.Image(
            make_quantized_overlay_image(
                frame=current_frame,
                assignment_grid=assignment_grid,
                patch_height=patch_height,
                patch_width=patch_width,
            ),
            caption=f"epoch {epoch}, patch assignments",
        )
        log_images[f"{prefix}/motion"] = wandb.Image(
            motion_image,
            caption=f"epoch {epoch}, {describe_motion_signal(model.motion_input_type)}",
        )
        log_images[f"{prefix}/reconstruction"] = wandb.Image(
            Image.fromarray(signed_tensor_to_uint8_image(reconstruction)),
            caption=f"epoch {epoch}, reconstruction",
        )

    if model_was_training:
        model.train()
    return log_images



def train_otf_vqvae(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = choose_device(str(cfg.device))
    resume_checkpoint = load_resume_checkpoint_and_apply_model_config(cfg, device)
    write_resolved_config(cfg)
    wandb_run = initialize_wandb(cfg)

    dataset = build_dataset(
        cfg.data,
        motion_input_type=get_motion_input_type(cfg.model),
        motion_transform=get_motion_transform(cfg.model),
        use_reference_conditioning=get_use_reference_conditioning(cfg.model),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=bool(cfg.train.shuffle),
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=bool(cfg.train.drop_last),
    )
    if len(loader) == 0:
        raise RuntimeError("The DataLoader is empty. Reduce batch_size or set train.drop_last=false.")

    total_steps = compute_total_steps(cfg, len(loader))
    warmup_step_count = compute_warmup_step_count(cfg, len(loader))
    validation_sample_indices = (
        select_validation_sample_indices(len(dataset), num_samples=3)
        if wandb_run is not None
        else []
    )

    model = OTFVQVAE(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.optim)
    global_step = maybe_load_checkpoint(model, optimizer, cfg, device, checkpoint=resume_checkpoint)
    model.train()

    print(f"device={device}")
    print(f"dataset_size={len(dataset)} num_batches={len(loader)}")
    print(f"total_steps={total_steps}")
    print(
        f"warmup_epoch={float(get_optional(cfg.train, 'warmup_epoch', 1)):g} "
        f"warmup_step_count={warmup_step_count}"
    )
    print(f"output_dir={Path(str(cfg.output_dir)).expanduser()}")
    print(f"Motion input type: {model.motion_input_type}")
    print(f"Motion transform: {model.motion_transform}")
    print(f"Motion computation: {describe_motion_computation(model.motion_input_type)}")
    print(f"Motion channels: {model.motion_channels}")

    if str(cfg.model.codebook_init) == "random_normal":
        model.quantizer.set_initialized(True)

    last_checkpoint_path = None
    epoch = 0
    while global_step < total_steps:
        epoch += 1
        epoch_start_step = global_step
        for batch in loader:
            if global_step >= total_steps:
                break

            next_step = global_step + 1
            use_quantization = next_step > warmup_step_count
            if use_quantization and not model.quantizer.is_initialized():
                print(f"step={next_step} initializing codebook with k-means")
                initialize_codebook_with_kmeans(model, loader, device, cfg.model, step=next_step)
                model.train()

            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            motion = make_motion_signal(batch, model.motion_input_type, model.motion_transform)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                motion,
                reference_frame=batch.get("reference_frame"),
                use_quantization=use_quantization,
            )
            rec_loss = F.mse_loss(output["reconstruction"], motion)
            weighted_code_loss = float(cfg.loss.lambda_code) * output["code_loss"]
            weighted_commit_loss = float(cfg.loss.lambda_commit) * output["commit_loss"]
            weighted_orth_loss = float(cfg.loss.lambda_orth) * output["orth_loss"]
            loss = (
                rec_loss
                + weighted_code_loss
                + weighted_commit_loss
                + weighted_orth_loss
            )
            loss.backward()
            if cfg.optim.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.optim.grad_clip_norm))
            optimizer.step()

            update_info = {"reinitialized_codes": 0}
            if use_quantization:
                update_info = model.quantizer.update_codebook(
                    output["patch_embeddings"],
                    output["indices"],
                    step=next_step,
                )

            global_step = next_step
            log_values = {
                "train/loss": loss.item(),
                "train/reconstruction_loss": rec_loss.item(),
                "train/code_loss": output["code_loss"].item(),
                "train/commitment_loss": output["commit_loss"].item(),
                "train/orthogonality_loss": output["orth_loss"].item(),
                "train/weighted_code_loss": weighted_code_loss.item(),
                "train/weighted_commitment_loss": weighted_commit_loss.item(),
                "train/weighted_orthogonality_loss": weighted_orth_loss.item(),
                "train/active_motion_codes": float(output["active_codes"].item()),
                "train/reinitialized_codes": update_info["reinitialized_codes"],
                "train/use_quantization": float(use_quantization),
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if wandb_run is not None:
                wandb_run.log(log_values, step=global_step)

            if global_step % int(cfg.train.log_every) == 0 or global_step == 1:
                phase = "quantized" if use_quantization else "warmup"
                print(
                    " ".join(
                        [
                            f"step={global_step:06d}",
                            f"phase={phase}",
                            f"loss={loss.item():.6f}",
                            f"rec={rec_loss.item():.6f}",
                            f"code={output['code_loss'].item():.6f}",
                            f"commit={output['commit_loss'].item():.6f}",
                            f"orth={output['orth_loss'].item():.6f}",
                            f"active_codes={float(output['active_codes'].item()):.0f}",
                            f"reinit={update_info['reinitialized_codes']}",
                        ]
                    )
                )

            if (
                bool(cfg.checkpoint.save)
                and int(cfg.checkpoint.every_steps) > 0
                and global_step % int(cfg.checkpoint.every_steps) == 0
            ):
                last_checkpoint_path = save_checkpoint(model, optimizer, cfg, global_step, final=False)
                print(f"saved_checkpoint={last_checkpoint_path}")

        if (
            wandb_run is not None
            and validation_sample_indices
            and global_step > epoch_start_step
            and model.quantizer.is_initialized()
        ):
            wandb_run.log(
                make_validation_image_log(
                    model=model,
                    dataset=dataset,
                    sample_indices=validation_sample_indices,
                    device=device,
                    epoch=epoch,
                ),
                step=global_step,
            )

    if bool(cfg.checkpoint.save) and bool(cfg.checkpoint.save_final):
        last_checkpoint_path = save_checkpoint(model, optimizer, cfg, global_step, final=True)
        print(f"saved_checkpoint={last_checkpoint_path}")
    elif last_checkpoint_path is not None:
        print(f"last_checkpoint={last_checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


def train(cfg: DictConfig) -> None:
    get_model_type(cfg)
    with open_dict(cfg.model):
        cfg.model.model_type = "otf_vqvae"
    train_otf_vqvae(cfg)
