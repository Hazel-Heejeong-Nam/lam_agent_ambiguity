from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


VALID_MOTION_TRANSFORMS: Tuple[str, ...] = (
    "none",
    "grayscale_sobel",
    "gradient",
    "log",
    "hpf",
)
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0
LAPLACIAN_KERNEL = (
    (0.0, 1.0, 0.0),
    (1.0, -4.0, 1.0),
    (0.0, 1.0, 0.0),
)
SOBEL_X_KERNEL = (
    (-1.0, 0.0, 1.0),
    (-2.0, 0.0, 2.0),
    (-1.0, 0.0, 1.0),
)
SOBEL_Y_KERNEL = (
    (-1.0, -2.0, -1.0),
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 1.0),
)


def validate_motion_transform(motion_transform: str) -> str:
    motion_transform = str(motion_transform)
    if motion_transform not in VALID_MOTION_TRANSFORMS:
        expected = ", ".join(VALID_MOTION_TRANSFORMS)
        raise ValueError(f"Unknown motion_transform: {motion_transform}. Expected one of: {expected}.")
    return motion_transform


def ensure_batched(frame: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if frame.ndim == 3:
        return frame.unsqueeze(0), True
    if frame.ndim == 4:
        return frame, False
    raise ValueError(f"Expected frame tensor with shape [C,H,W] or [B,C,H,W], got {tuple(frame.shape)}")


def restore_batch_shape(frame: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return frame.squeeze(0) if squeezed else frame


def rgb_to_grayscale(frame: torch.Tensor) -> torch.Tensor:
    frame, squeezed = ensure_batched(frame)
    channels = frame.shape[1]
    if channels == 1:
        gray = frame
    elif channels == 3:
        weights = frame.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        gray = (frame * weights).sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Expected 1 or 3 input channels for grayscale transform, got {channels}")
    return restore_batch_shape(gray, squeezed)


def conv2d_single_channel(frame: torch.Tensor, kernel_values: Tuple[Tuple[float, ...], ...]) -> torch.Tensor:
    frame, squeezed = ensure_batched(frame)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one channel for single-channel convolution, got {frame.shape[1]}")
    kernel = frame.new_tensor(kernel_values).view(1, 1, len(kernel_values), len(kernel_values[0]))
    output = F.conv2d(frame, kernel, padding=1)
    return restore_batch_shape(output, squeezed)


def gaussian_kernel(size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size % 2 != 1 or size < 3:
        raise ValueError(f"Gaussian kernel size must be odd and >= 3, got {size}")
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def gaussian_blur(
    frame: torch.Tensor,
    kernel_size: int = GAUSSIAN_KERNEL_SIZE,
    sigma: float = GAUSSIAN_SIGMA,
) -> torch.Tensor:
    frame, squeezed = ensure_batched(frame)
    channels = frame.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, device=frame.device, dtype=frame.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(channels, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    padded = F.pad(frame, (padding, padding, padding, padding), mode="replicate")
    blurred = F.conv2d(padded, kernel, groups=channels)
    return restore_batch_shape(blurred, squeezed)


def sobel_xy(gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        conv2d_single_channel(gray, SOBEL_X_KERNEL),
        conv2d_single_channel(gray, SOBEL_Y_KERNEL),
    )


def apply_frame_transform(frame: torch.Tensor, motion_transform: str) -> torch.Tensor:
    motion_transform = validate_motion_transform(motion_transform)
    if motion_transform == "none":
        return frame
    if motion_transform == "hpf":
        return frame - gaussian_blur(frame)

    gray = rgb_to_grayscale(frame)
    if motion_transform == "grayscale_sobel":
        sobel_x, sobel_y = sobel_xy(gray)
        return torch.cat([gray, sobel_x, sobel_y], dim=-3)
    if motion_transform == "gradient":
        grad_x, grad_y = sobel_xy(gray)
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + torch.finfo(gray.dtype).eps)
        return torch.cat([grad_x, grad_y, grad_mag], dim=-3)
    if motion_transform == "log":
        blurred = gaussian_blur(gray)
        log_response = conv2d_single_channel(blurred, LAPLACIAN_KERNEL)
        return torch.cat([log_response, F.relu(log_response), F.relu(-log_response)], dim=-3)

    raise AssertionError(f"Unexpected validated motion_transform: {motion_transform}")


def compute_motion_signal(
    frames: Dict[str, torch.Tensor],
    motion_input_type: str,
    motion_transform: str,
) -> torch.Tensor:
    motion_transform = validate_motion_transform(motion_transform)
    transformed_current = apply_frame_transform(frames["current"], motion_transform)
    transformed_next = apply_frame_transform(frames["next"], motion_transform)
    if motion_input_type == "velocity":
        return transformed_next - transformed_current
    if motion_input_type == "acceleration":
        transformed_previous = apply_frame_transform(frames["previous"], motion_transform)
        return transformed_next - 2.0 * transformed_current + transformed_previous
    raise ValueError(
        "motion_input_type must be either 'acceleration' or 'velocity', "
        f"got '{motion_input_type}'"
    )


def describe_motion_computation(motion_input_type: str) -> str:
    if motion_input_type == "velocity":
        return "transform(next) - transform(current)"
    if motion_input_type == "acceleration":
        return "transform(next) - 2*transform(current) + transform(previous)"
    raise ValueError(
        "motion_input_type must be either 'acceleration' or 'velocity', "
        f"got '{motion_input_type}'"
    )


def motion_transform_output_channels(motion_transform: str, input_channels: int) -> int:
    motion_transform = validate_motion_transform(motion_transform)
    if motion_transform in {"grayscale_sobel", "gradient", "log"}:
        return 3
    return int(input_channels)


def tensor_to_image_array(tensor: torch.Tensor, signed: bool) -> np.ndarray:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor for image export, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    elif tensor.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels for image export, got {tensor.shape[0]}")

    if signed:
        max_abs = float(tensor.abs().max().item())
        if math.isclose(max_abs, 0.0):
            tensor = torch.zeros_like(tensor) + 0.5
        else:
            tensor = tensor.clamp(-max_abs, max_abs) / (2.0 * max_abs) + 0.5
    elif float(tensor.min().item()) < 0.0 or float(tensor.max().item()) > 1.0:
        min_value = tensor.min()
        max_value = tensor.max()
        denom = max_value - min_value
        tensor = torch.zeros_like(tensor) if float(denom.item()) == 0.0 else (tensor - min_value) / denom
    else:
        tensor = tensor.clamp(0.0, 1.0)

    return (tensor * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()


def save_debug_motion_inputs(
    batch: Dict[str, torch.Tensor],
    output_root: str | Path = "outputs/debug_motion_inputs",
    *,
    motion_input_type: str,
    motion_transform: str,
    max_samples: int = 4,
) -> Path:
    motion_transform = validate_motion_transform(motion_transform)
    output_dir = Path(output_root) / f"{motion_input_type}_{motion_transform}"
    output_dir.mkdir(parents=True, exist_ok=True)

    first_tensor = next(iter(batch.values()))
    batch_size = first_tensor.shape[0] if first_tensor.ndim == 4 else 1
    num_samples = min(int(max_samples), batch_size)
    frame_keys = ["current", "next"]
    if motion_input_type == "acceleration":
        frame_keys.insert(0, "previous")

    for sample_idx in range(num_samples):
        sample = {
            key: value[sample_idx] if value.ndim == 4 else value
            for key, value in batch.items()
            if key in frame_keys
        }
        motion = compute_motion_signal(sample, motion_input_type, motion_transform)
        for key in frame_keys:
            original = sample[key]
            transformed = apply_frame_transform(original, motion_transform)
            Image.fromarray(tensor_to_image_array(original, signed=False)).save(
                output_dir / f"sample_{sample_idx:03d}_original_{key}.png"
            )
            Image.fromarray(tensor_to_image_array(transformed, signed=True)).save(
                output_dir / f"sample_{sample_idx:03d}_transformed_{key}.png"
            )
        Image.fromarray(tensor_to_image_array(motion, signed=True)).save(
            output_dir / f"sample_{sample_idx:03d}_motion.png"
        )

    return output_dir
