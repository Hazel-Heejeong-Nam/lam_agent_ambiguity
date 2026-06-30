from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from otf_vqvae.model import build_dataset, validate_motion_input_type
from otf_vqvae.motion_transforms import validate_motion_transform


class RawFramePredictionDataset(Dataset):
    """Wrap a OTF-VQ-VAE motion dataset and expose raw prediction frames."""

    def __init__(self, base_dataset: Dataset, motion_input_type: str) -> None:
        self.base_dataset = base_dataset
        self.motion_input_type = validate_motion_input_type(motion_input_type)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[index]
        future = sample["future"] if "future" in sample else sample["next"]
        output = {
            "current": sample["current"],
            "future": future,
            "target": future,
            "next": future,
        }
        if self.motion_input_type == "acceleration":
            output["previous"] = sample["previous"]
        return output


def clone_data_config(
    data_cfg: DictConfig,
    data_path: Optional[str | Path] = None,
    max_sequences: Optional[int] = None,
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
    if data_path is not None:
        cfg.path = str(Path(data_path).expanduser())
    if max_sequences is not None:
        cfg.max_sequences = int(max_sequences)
    return cfg


def build_raw_frame_dataset(
    data_cfg: DictConfig,
    *,
    motion_input_type: str,
    motion_transform: str,
    data_path: Optional[str | Path] = None,
    max_sequences: Optional[int] = None,
) -> RawFramePredictionDataset:
    motion_input_type = validate_motion_input_type(motion_input_type)
    motion_transform = validate_motion_transform(motion_transform)
    cfg = clone_data_config(data_cfg, data_path=data_path, max_sequences=max_sequences)
    base_dataset = build_dataset(
        cfg,
        motion_input_type=motion_input_type,
        motion_transform=motion_transform,
        use_reference_conditioning=False,
    )
    return RawFramePredictionDataset(base_dataset, motion_input_type=motion_input_type)
