from __future__ import annotations

import hydra
from omegaconf import DictConfig

try:
    from .model import *  # noqa: F401,F403
    from .model import train
except ImportError:
    from model import *  # noqa: F401,F403
    from model import train


@hydra.main(version_base="1.3", config_path="../configs/otf_vqvae", config_name="walker")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
