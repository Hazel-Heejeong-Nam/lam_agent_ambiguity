from __future__ import annotations

import argparse
import math
import re
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from otf_lam.utils import choose_device, load_otf_lam_checkpoint, torch_load
from utils.augmentations import Augmenter
from utils.datasets.dcs import DCSInMemoryDataset
from utils.nn import (
    ActionDecoder,
    Actor,
    get_optim_groups,
    linear_annealing_with_warmup,
    normalize_img,
    set_seed,
)


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def resolve_otf_lam_checkpoint(
    checkpoint_path: Optional[str | Path],
    checkpoint_dir: Optional[str | Path],
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
        raise ValueError("Provide --otf_lam_checkpoint_path or --otf_lam_checkpoint_dir")
    root = Path(checkpoint_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    candidates = sorted(root.rglob("otf_lam_final_*.pt"))
    if not candidates:
        candidates = sorted(root.rglob("otf_lam_*.pt"))
    if not candidates:
        candidates = sorted(root.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No OTF-LAM checkpoints found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_dinolam_checkpoint(
    checkpoint_path: Optional[str | Path],
    checkpoint_dir: Optional[str | Path],
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
        raise ValueError("Provide --dinolam_checkpoint_path or --dinolam_checkpoint_dir")
    root = Path(checkpoint_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)

    candidates = []
    for pattern in ("dinolam_final*.pt", "dinolam*.pt", "*.pt"):
        candidates = sorted(root.rglob(pattern))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"No DINO-LAM checkpoints found under {root}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_bc_resume_path(
    output_dir: Path,
    explicit_resume_path: Optional[str | Path],
    *,
    auto_resume: bool,
) -> Optional[str]:
    if explicit_resume_path:
        path = Path(explicit_resume_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)
    if not auto_resume:
        return None

    final_path = output_dir / "bc-final.pt"
    if final_path.exists():
        return str(final_path)

    epoch_checkpoints = []
    for path in output_dir.glob("bc-epoch_*.pt"):
        match = re.fullmatch(r"bc-epoch_(\d+)\.pt", path.name)
        if match is not None:
            epoch_checkpoints.append((int(match.group(1)), path.stat().st_mtime, path))
    if epoch_checkpoints:
        return str(max(epoch_checkpoints, key=lambda item: (item[0], item[1], item[2].name))[2])

    latest_path = output_dir / "bc-latest.pt"
    if latest_path.exists():
        return str(latest_path)
    return None


def _bc_checkpoint_state(
    cfg,
    epoch,
    total_steps,
    total_tokens,
    actor,
    optim,
    scheduler,
    act_decoder,
    act_decoder_optim,
    act_decoder_scheduler,
):
    return {
        "actor_state_dict": actor.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "act_decoder_state_dict": act_decoder.state_dict(),
        "act_decoder_optimizer_state_dict": act_decoder_optim.state_dict(),
        "act_decoder_scheduler_state_dict": act_decoder_scheduler.state_dict(),
        "epoch": epoch,
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "cfg_snapshot": OmegaConf.to_container(cfg, resolve=True),
    }


def _act_decoder_checkpoint_state(cfg, total_steps, total_tokens, action_decoder, optim, scheduler):
    return {
        "action_decoder_state_dict": action_decoder.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "total_steps": total_steps,
        "total_tokens": total_tokens,
        "cfg_snapshot": OmegaConf.to_container(cfg, resolve=True),
    }


def read_dcs_metadata(hdf5_path: str | Path) -> tuple[int, int]:
    with h5py.File(hdf5_path, "r") as df:
        first_traj = next(iter(df.keys()))
        img_hw = int(df.attrs["img_hw"])
        act_dim = int(df[first_traj]["actions"][0].shape[-1])
    return img_hw, act_dim


def build_actor(cfg, img_hw: int, num_actions: int, device: torch.device) -> Actor:
    return Actor(
        shape=(3 * cfg.bc.frame_stack, img_hw, img_hw),
        num_actions=num_actions,
        encoder_scale=cfg.bc.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if cfg.bc.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=cfg.bc.encoder_num_res_blocks,
        dropout=cfg.bc.dropout,
    ).to(device)


def infer_actor_num_actions(actor_state_dict) -> int:
    for key in ("actor_mean.1.weight", "actor_mean.2.weight"):
        weight = actor_state_dict.get(key)
        if weight is not None:
            return int(weight.shape[0])
    raise KeyError("Could not infer actor output dimension from BC checkpoint actor_state_dict")


def load_bc_actor_from_checkpoint(cfg, checkpoint_path: str | Path, device: torch.device) -> tuple[Actor, dict]:
    ckpt = torch_load(checkpoint_path, map_location=device)
    saved_cfg = ckpt.get("cfg_snapshot", {})
    for key in ("otf_lam_checkpoint_path", "dinolam_checkpoint_path", "otf_vqvae_checkpoint_path"):
        current_value = cfg.get(key)
        saved_value = saved_cfg.get(key) if isinstance(saved_cfg, dict) else None
        if current_value and saved_value and str(current_value) != str(saved_value):
            print(
                f"WARNING: BC checkpoint {key} differs from current config: "
                f"checkpoint={saved_value} current={current_value}"
            )
    num_actions = infer_actor_num_actions(ckpt["actor_state_dict"])
    img_hw, _ = read_dcs_metadata(cfg.bc.data_path)
    actor = build_actor(cfg, img_hw, num_actions, device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor, ckpt


class OTFLAMActionLabeler(nn.Module):
    """Expose OTFLAM z_act labels from DCS stacked-frame observations."""

    def __init__(self, model, frame_stack: int) -> None:
        super().__init__()
        self.model = model
        self.frame_stack = int(frame_stack)
        self.channels = int(model.channels)
        self.image_size = (int(model.image_height), int(model.image_width))
        self.motion_input_type = model.otf_vqvae_extractor.otf_vqvae.motion_input_type
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.latent_act_dim = int(model.z_action_dim)

    def _split_stacked_obs(self, stacked_obs: torch.Tensor) -> torch.Tensor:
        if stacked_obs.ndim != 4:
            raise ValueError(f"Expected [B,H,W,C*frame_stack], got {tuple(stacked_obs.shape)}")
        batch_size, height, width, stacked_channels = stacked_obs.shape
        expected_channels = self.frame_stack * self.channels
        if stacked_channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} stacked channels from frame_stack={self.frame_stack} "
                f"and channels={self.channels}, got {stacked_channels}"
            )
        frames = stacked_obs.float()
        if float(frames.detach().max().item()) > 1.5:
            frames = frames / 255.0
        frames = frames.reshape(batch_size, height, width, self.frame_stack, self.channels)
        return frames.permute(0, 3, 4, 1, 2).contiguous()

    def _resize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if tuple(frame.shape[-2:]) == self.image_size:
            return frame
        return F.interpolate(frame, size=self.image_size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def label(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        obs_frames = self._split_stacked_obs(obs)
        next_obs_frames = self._split_stacked_obs(next_obs)
        current = self._resize_frame(obs_frames[:, -1])
        future = self._resize_frame(next_obs_frames[:, -1])
        batch = {
            "current": current,
            "future": future,
            "target": future,
        }
        if self.motion_input_type == "acceleration":
            if self.frame_stack < 2:
                raise ValueError("Acceleration checkpoints require frame_stack >= 2 for previous frame labels")
            batch["previous"] = self._resize_frame(obs_frames[:, -2])
        output = self.model(batch, decode=False)
        return output["z_act"].detach()


class DINOLAMActionLabeler(nn.Module):
    """Expose DINO-LAM global z_act labels from DCS stacked-frame observations."""

    def __init__(self, model, frame_stack: int, channels: int = 3) -> None:
        super().__init__()
        if model.z_action_dim is None:
            raise ValueError("DINO-LAM downstream evaluation requires use_global_action_token=true")
        if model.otf_vqvae_extractor is None:
            raise ValueError("DINO-LAM downstream evaluation requires a VQ-backed Stage 1 checkpoint")
        self.model = model
        self.frame_stack = int(frame_stack)
        self.channels = int(channels)
        otf_vqvae = model.otf_vqvae_extractor.otf_vqvae
        self.image_size = (int(otf_vqvae.encoder.image_height), int(otf_vqvae.encoder.image_width))
        self.motion_input_type = otf_vqvae.motion_input_type
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.latent_act_dim = int(model.z_action_dim)
        self.policy_state_dim = int(model.state_dim)

    def _split_stacked_obs(self, stacked_obs: torch.Tensor) -> torch.Tensor:
        if stacked_obs.ndim != 4:
            raise ValueError(f"Expected [B,H,W,C*frame_stack], got {tuple(stacked_obs.shape)}")
        batch_size, height, width, stacked_channels = stacked_obs.shape
        expected_channels = self.frame_stack * self.channels
        if stacked_channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} stacked channels from frame_stack={self.frame_stack} "
                f"and channels={self.channels}, got {stacked_channels}"
            )
        frames = stacked_obs.float()
        if float(frames.detach().max().item()) > 1.5:
            frames = frames / 255.0
        frames = frames.reshape(batch_size, height, width, self.frame_stack, self.channels)
        return frames.permute(0, 3, 4, 1, 2).contiguous()

    def _resize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if tuple(frame.shape[-2:]) == self.image_size:
            return frame
        return F.interpolate(frame, size=self.image_size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def policy_state_from_actor_input(self, actor_input: torch.Tensor) -> torch.Tensor:
        if actor_input.ndim != 4:
            raise ValueError(f"Expected [B,C*frame_stack,H,W], got {tuple(actor_input.shape)}")
        expected_channels = self.frame_stack * self.channels
        if actor_input.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels from frame_stack={self.frame_stack} "
                f"and channels={self.channels}, got {actor_input.shape[1]}"
            )
        current = actor_input[:, -self.channels :].float()
        if float(current.detach().min().item()) < -0.01:
            current = current / 2.0 + 0.5
        elif float(current.detach().max().item()) > 1.5:
            current = current / 255.0
        current = self._resize_frame(current.clamp(0.0, 1.0))
        return self.model.encode_policy_state(current).detach()

    @torch.no_grad()
    def label(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        obs_frames = self._split_stacked_obs(obs)
        next_obs_frames = self._split_stacked_obs(next_obs)
        current = self._resize_frame(obs_frames[:, -1])
        future = self._resize_frame(next_obs_frames[:, -1])
        batch = {
            "current": current,
            "future": future,
            "target": future,
        }
        if self.motion_input_type == "acceleration":
            if self.frame_stack < 2:
                raise ValueError("Acceleration checkpoints require frame_stack >= 2 for previous frame labels")
            batch["previous"] = self._resize_frame(obs_frames[:, -2])
        return self.model.action_labels(batch)["z_act"].detach()


@torch.no_grad()
def evaluate_bc(env, actor, num_episodes, seed=0, device="cpu", action_decoder=None, policy_state_labeler=None):
    returns = []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        total_reward = 0.0
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_ = torch.tensor(obs.copy(), device=device)[None].permute(0, 3, 1, 2)
            obs_ = normalize_img(obs_)
            action, obs_emb = actor(obs_)
            if action_decoder is not None:
                if isinstance(action_decoder, ActionDecoder):
                    if policy_state_labeler is not None:
                        obs_emb = policy_state_labeler.policy_state_from_actor_input(obs_)
                    action = action_decoder(obs_emb, action)
                else:
                    action = action_decoder(action)

            obs, reward, terminated, truncated, info = env.step(action.squeeze().cpu().numpy())
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def train_bc(cfg, lam_labeler, device, wandb_run=None):
    from envs.dcs import create_env_from_df

    dataset = DCSInMemoryDataset(
        cfg.bc.data_path,
        num_trajs=cfg.bc.num_trajs,
        frame_stack=cfg.bc.frame_stack,
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.bc.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_env = create_env_from_df(
        cfg.bc.data_path,
        cfg.bc.dcs_backgrounds_path,
        cfg.bc.dcs_backgrounds_split,
        frame_stack=cfg.bc.frame_stack,
        seed=cfg.bc.eval_seed,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    num_actions = lam_labeler.latent_act_dim
    actor = build_actor(cfg, dataset.img_hw, num_actions, device)

    optim = torch.optim.AdamW(
        params=get_optim_groups(actor, cfg.bc.weight_decay),
        lr=cfg.bc.learning_rate,
        fused=device.type == "cuda",
    )
    total_updates = len(dataloader) * cfg.bc.num_epochs
    warmup_updates = len(dataloader) * cfg.bc.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    print("Latent action dim:", num_actions)
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, dataset.act_dim),
    ).to(device)

    act_decoder_optim = torch.optim.AdamW(
        params=act_decoder.parameters(),
        lr=cfg.bc.learning_rate,
        fused=device.type == "cuda",
    )
    act_decoder_scheduler = linear_annealing_with_warmup(act_decoder_optim, warmup_updates, total_updates)

    if cfg.bc.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    max_steps = getattr(cfg.bc, "max_steps", None)
    start_epoch = 0
    resume_path = getattr(cfg.bc, "resume_path", None)
    checkpoint_every_epochs = int(getattr(cfg.bc, "checkpoint_every_epochs", 5))
    if resume_path:
        ckpt = torch_load(resume_path, map_location=device)
        actor.load_state_dict(ckpt["actor_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        act_decoder.load_state_dict(ckpt["act_decoder_state_dict"])
        act_decoder_optim.load_state_dict(ckpt["act_decoder_optimizer_state_dict"])
        act_decoder_scheduler.load_state_dict(ckpt["act_decoder_scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        total_steps = int(ckpt.get("total_steps", 0))
        total_tokens = int(ckpt.get("total_tokens", 0))
        print(f"Resumed BC from {resume_path} at epoch {start_epoch}")
        if start_epoch >= int(cfg.bc.num_epochs):
            print(f"BC checkpoint already reached num_epochs={cfg.bc.num_epochs}; skipping BC optimizer steps.")

    epoch = start_epoch - 1
    for epoch in trange(start_epoch, cfg.bc.num_epochs, desc="BC Epochs"):
        actor.train()
        for batch in dataloader:
            if max_steps is not None and total_steps >= int(max_steps):
                break
            total_tokens += cfg.bc.batch_size
            total_steps += 1

            obs, next_obs, true_actions = [b.to(device) for b in batch]
            with torch.no_grad():
                target_actions = lam_labeler.label(obs, next_obs)

            obs_for_actor = normalize_img(obs.permute((0, 3, 1, 2)))
            if cfg.bc.use_aug:
                obs_for_actor = augmenter(obs_for_actor)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_actions, _ = actor(obs_for_actor)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                decoder_loss = F.mse_loss(pred_true_actions, true_actions)

            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            log_values = {
                "bc/mse_loss": loss.item(),
                "bc/throughput": total_tokens / (time.time() - start_time),
                "bc/learning_rate": scheduler.get_last_lr()[0],
                "bc/act_decoder_probe_mse_loss": decoder_loss.item(),
                "bc/epoch": epoch,
                "bc/total_steps": total_steps,
            }
            if wandb_run is not None:
                wandb_run.log(log_values)

        bc_ckpt_state = _bc_checkpoint_state(
            cfg=cfg,
            epoch=epoch,
            total_steps=total_steps,
            total_tokens=total_tokens,
            actor=actor,
            optim=optim,
            scheduler=scheduler,
            act_decoder=act_decoder,
            act_decoder_optim=act_decoder_optim,
            act_decoder_scheduler=act_decoder_scheduler,
        )
        torch.save(bc_ckpt_state, f"{cfg.ckpt_dir}/bc-latest.pt")
        if (epoch + 1) % checkpoint_every_epochs == 0:
            torch.save(bc_ckpt_state, f"{cfg.ckpt_dir}/bc-epoch_{epoch + 1}.pt")
        if max_steps is not None and total_steps >= int(max_steps):
            break

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=cfg.bc.eval_episodes,
        seed=cfg.bc.eval_seed,
        device=device,
        action_decoder=act_decoder,
    )
    log_values = {
        "bc/eval_returns_mean": eval_returns.mean(),
        "bc/eval_returns_std": eval_returns.std(),
        "bc/epoch": epoch,
        "bc/total_steps": total_steps,
    }
    if wandb_run is not None:
        wandb_run.log(log_values)
    print(f"bc_eval_returns_mean={eval_returns.mean():.4f} std={eval_returns.std():.4f}")

    bc_ckpt_state = _bc_checkpoint_state(
        cfg=cfg,
        epoch=epoch,
        total_steps=total_steps,
        total_tokens=total_tokens,
        actor=actor,
        optim=optim,
        scheduler=scheduler,
        act_decoder=act_decoder,
        act_decoder_optim=act_decoder_optim,
        act_decoder_scheduler=act_decoder_scheduler,
    )
    torch.save(bc_ckpt_state, f"{cfg.ckpt_dir}/bc-final.pt")

    del dataset, dataloader
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return actor


def train_act_decoder(cfg, actor, device, lam_labeler=None, wandb_run=None):
    from envs.dcs import create_env_from_df

    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    dataset = DCSInMemoryDataset(
        cfg.act_decoder.data_path,
        num_trajs=cfg.act_decoder.num_trajs,
        frame_stack=cfg.bc.frame_stack,
        device=device,
        precompute_stacked_obs=bool(getattr(cfg.act_decoder, "precompute_stacked_obs", False)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.act_decoder.batch_size,
        shuffle=True,
    )
    use_dino_policy_state = str(cfg.lam_type) == "dino"
    latentaction_type = str(getattr(cfg.act_decoder, "latentaction_type", "pred")).lower()
    if latentaction_type not in {"posterior", "pred"}:
        raise ValueError(f"action_decoder_latentaction_type must be 'posterior' or 'pred', got {latentaction_type!r}")
    if use_dino_policy_state and lam_labeler is None:
        raise ValueError("lam_type=dino action decoder training requires a DINO-LAM labeler for policy state.")
    if latentaction_type == "posterior" and lam_labeler is None:
        raise ValueError("action_decoder_latentaction_type=posterior requires a LAM labeler.")

    decoder_obs_emb_dim = (
        int(lam_labeler.policy_state_dim)
        if use_dino_policy_state
        else int(actor.final_encoder_shape[0])
    )

    action_decoder = ActionDecoder(
        obs_emb_dim=decoder_obs_emb_dim,
        latent_act_dim=actor.num_actions,
        true_act_dim=dataset.act_dim,
        hidden_dim=cfg.act_decoder.hidden_dim,
    ).to(device)

    optim = torch.optim.AdamW(
        params=get_optim_groups(action_decoder, cfg.act_decoder.weight_decay),
        lr=cfg.act_decoder.learning_rate,
        fused=device.type == "cuda",
    )
    total_updates = cfg.act_decoder.total_updates
    warmup_updates = len(dataloader) * cfg.act_decoder.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)
    checkpoint_every_steps = int(getattr(cfg.act_decoder, "checkpoint_every_steps", 0))
    log_every_steps = int(getattr(cfg.act_decoder, "log_every_steps", 1))

    if cfg.act_decoder.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    resume_path = getattr(cfg.act_decoder, "resume_path", None)
    if resume_path:
        ckpt = torch_load(resume_path, map_location=device)
        action_decoder.load_state_dict(ckpt["action_decoder_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        total_steps = int(ckpt.get("total_steps", 0))
        total_tokens = int(ckpt.get("total_tokens", 0))
        print(f"Resumed action decoder from {resume_path} at step {total_steps}")
        if total_steps >= int(total_updates):
            print(
                f"Action decoder checkpoint already reached total_updates={total_updates}; "
                "skipping action decoder optimizer steps."
            )

    def infinite_dataloader(loader):
        while True:
            for batch in loader:
                yield batch

    def maybe_to_device(tensor):
        if tensor.device.type == device.type and (
            device.type != "cuda" or device.index is None or tensor.device.index == device.index
        ):
            return tensor
        return tensor.to(device, non_blocking=True)

    def save_decoder_checkpoint(path):
        state = _act_decoder_checkpoint_state(
            cfg=cfg,
            total_steps=total_steps,
            total_tokens=total_tokens,
            action_decoder=action_decoder,
            optim=optim,
            scheduler=scheduler,
        )
        torch.save(state, path)
        return state

    decoder_ckpt_state = None
    if total_steps < int(total_updates):
        for step, batch in enumerate(
            islice(infinite_dataloader(dataloader), int(total_updates) - total_steps),
            start=total_steps,
        ):
            total_tokens += cfg.act_decoder.batch_size
            total_steps = step + 1

            obs, next_obs, true_actions = batch
            obs = maybe_to_device(obs)
            next_obs = maybe_to_device(next_obs)
            true_actions = maybe_to_device(true_actions)
            obs_for_actor = normalize_img(obs.permute((0, 3, 1, 2)))

            if cfg.act_decoder.use_aug:
                obs_for_actor = augmenter(obs_for_actor)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.no_grad():
                    pred_latent_actions, actor_obs_emb = actor(obs_for_actor)
                    obs_emb = (
                        lam_labeler.policy_state_from_actor_input(obs_for_actor)
                        if use_dino_policy_state
                        else actor_obs_emb
                    )
                    latent_actions = (
                        lam_labeler.label(obs, next_obs)
                        if latentaction_type == "posterior"
                        else pred_latent_actions
                    )
                pred_actions = action_decoder(obs_emb, latent_actions)
                loss = F.mse_loss(pred_actions, true_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            log_values = {
                "decoder/mse_loss": loss.item(),
                "decoder/throughput": total_tokens / (time.time() - start_time),
                "decoder/learning_rate": scheduler.get_last_lr()[0],
                "decoder/total_steps": total_steps,
            }
            should_log = log_every_steps <= 1 or total_steps % log_every_steps == 0 or total_steps == int(total_updates)
            if wandb_run is not None and should_log:
                wandb_run.log(log_values)

            if checkpoint_every_steps > 0 and total_steps % checkpoint_every_steps == 0:
                decoder_ckpt_state = save_decoder_checkpoint(f"{cfg.ckpt_dir}/act-decoder-latest.pt")
                torch.save(decoder_ckpt_state, f"{cfg.ckpt_dir}/act-decoder-step_{total_steps:06d}.pt")

    decoder_ckpt_state = save_decoder_checkpoint(f"{cfg.ckpt_dir}/act-decoder-latest.pt")

    eval_env = create_env_from_df(
        cfg.act_decoder.data_path,
        cfg.act_decoder.dcs_backgrounds_path,
        cfg.act_decoder.dcs_backgrounds_split,
        frame_stack=cfg.bc.frame_stack,
        seed=cfg.act_decoder.eval_seed,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=cfg.act_decoder.eval_episodes,
        seed=cfg.act_decoder.eval_seed,
        device=device,
        action_decoder=action_decoder,
        policy_state_labeler=lam_labeler if use_dino_policy_state else None,
    )
    log_values = {
        "decoder/eval_returns_mean": eval_returns.mean(),
        "decoder/eval_returns_std": eval_returns.std(),
        "decoder/total_steps": total_steps,
    }
    if wandb_run is not None:
        wandb_run.log(log_values)
    print(f"decoder_eval_returns_mean={eval_returns.mean():.4f} std={eval_returns.std():.4f}")

    torch.save(decoder_ckpt_state, f"{cfg.ckpt_dir}/act-decoder-final.pt")
    return action_decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train downstream DCS policy for latent action models.")
    parser.add_argument("--lam_type", choices=("otf", "dino"), default="otf")
    parser.add_argument("--otf_lam_checkpoint_path", type=str, default=None)
    parser.add_argument("--otf_lam_checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--dinolam_checkpoint_path",
        "--stage1_lam_checkpoint_path",
        dest="dinolam_checkpoint_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dinolam_checkpoint_dir",
        "--stage1_lam_checkpoint_dir",
        dest="dinolam_checkpoint_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--otf_vqvae_checkpoint_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1002)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lam")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])

    parser.add_argument("--train_bc", type=str2bool, default=True)
    parser.add_argument("--train_act_decoder", type=str2bool, default=True)

    parser.add_argument("--bc_num_trajs", type=int, default=1000)
    parser.add_argument("--bc_num_epochs", type=int, default=5)
    parser.add_argument("--bc_max_steps", type=int, default=None)
    parser.add_argument("--bc_batch_size", type=int, default=512)
    parser.add_argument("--bc_learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--bc_weight_decay", type=float, default=0.0)
    parser.add_argument("--bc_warmup_epochs", type=int, default=0)
    parser.add_argument("--bc_encoder_scale", type=int, default=32)
    parser.add_argument("--bc_encoder_num_res_blocks", type=int, default=2)
    parser.add_argument("--bc_encoder_deep", type=str2bool, default=False)
    parser.add_argument("--bc_dropout", type=float, default=0.0)
    parser.add_argument("--bc_use_aug", type=str2bool, default=False)
    parser.add_argument("--bc_frame_stack", type=int, default=3)
    parser.add_argument("--bc_dcs_backgrounds_path", type=str, default="data/dcs/DAVIS/JPEGImages/480p")
    parser.add_argument("--bc_dcs_backgrounds_split", type=str, default="train")
    parser.add_argument("--bc_eval_episodes", type=int, default=10)
    parser.add_argument("--bc_eval_seed", type=int, default=0)
    parser.add_argument("--bc_checkpoint_every_epochs", type=int, default=5)
    parser.add_argument("--bc_resume_path", type=str, default=None)
    parser.add_argument("--no_auto_bc_resume", action="store_true")

    parser.add_argument("--act_decoder_num_trajs", type=int, default=32)
    parser.add_argument("--act_decoder_total_updates", type=int, default=5000)
    parser.add_argument("--act_decoder_checkpoint_every_steps", type=int, default=0)
    parser.add_argument("--act_decoder_log_every_steps", type=int, default=1)
    parser.add_argument("--act_decoder_batch_size", type=int, default=512)
    parser.add_argument("--act_decoder_learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--act_decoder_weight_decay", type=float, default=0.0)
    parser.add_argument("--act_decoder_warmup_epochs", type=int, default=0)
    parser.add_argument("--act_decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--action_decoder_latentaction_type", choices=("posterior", "pred"), default="pred")
    parser.add_argument("--act_decoder_use_aug", type=str2bool, default=False)
    parser.add_argument("--act_decoder_precompute_stacked_obs", type=str2bool, default=False)
    parser.add_argument("--act_decoder_dcs_backgrounds_path", type=str, default=None)
    parser.add_argument("--act_decoder_dcs_backgrounds_split", type=str, default=None)
    parser.add_argument("--act_decoder_eval_episodes", type=int, default=10)
    parser.add_argument("--act_decoder_eval_seed", type=int, default=0)
    parser.add_argument("--act_decoder_resume_path", type=str, default=None)
    return parser.parse_args()


def build_cfg(
    args: argparse.Namespace,
    otf_lam_checkpoint_path: Optional[Path],
    dinolam_checkpoint_path: Optional[Path],
    output_dir: Path,
):
    otf_lam_checkpoint_path = (
        str(otf_lam_checkpoint_path)
        if otf_lam_checkpoint_path is not None
        else args.otf_lam_checkpoint_path
    )
    dinolam_checkpoint_path = (
        str(dinolam_checkpoint_path)
        if dinolam_checkpoint_path is not None
        else args.dinolam_checkpoint_path
    )
    return OmegaConf.create(
        {
            "seed": args.seed,
            "ckpt_dir": str(output_dir),
            "lam_type": args.lam_type,
            "otf_lam_checkpoint_path": otf_lam_checkpoint_path,
            "dinolam_checkpoint_path": dinolam_checkpoint_path,
            "otf_vqvae_checkpoint_path": args.otf_vqvae_checkpoint_path,
            "train_bc": bool(args.train_bc),
            "train_act_decoder": bool(args.train_act_decoder),
            "bc": {
                "num_trajs": args.bc_num_trajs,
                "num_epochs": args.bc_num_epochs,
                "max_steps": args.bc_max_steps,
                "batch_size": args.bc_batch_size,
                "learning_rate": args.bc_learning_rate,
                "weight_decay": args.bc_weight_decay,
                "warmup_epochs": args.bc_warmup_epochs,
                "encoder_scale": args.bc_encoder_scale,
                "encoder_num_res_blocks": args.bc_encoder_num_res_blocks,
                "encoder_deep": bool(args.bc_encoder_deep),
                "dropout": args.bc_dropout,
                "use_aug": bool(args.bc_use_aug),
                "frame_stack": args.bc_frame_stack,
                "data_path": args.data_path,
                "dcs_backgrounds_path": args.bc_dcs_backgrounds_path,
                "dcs_backgrounds_split": args.bc_dcs_backgrounds_split,
                "eval_episodes": args.bc_eval_episodes,
                "eval_seed": args.bc_eval_seed,
                "checkpoint_every_epochs": args.bc_checkpoint_every_epochs,
                "resume_path": args.bc_resume_path,
            },
            "act_decoder": {
                "num_trajs": args.act_decoder_num_trajs,
                "total_updates": args.act_decoder_total_updates,
                "checkpoint_every_steps": args.act_decoder_checkpoint_every_steps,
                "log_every_steps": args.act_decoder_log_every_steps,
                "batch_size": args.act_decoder_batch_size,
                "learning_rate": args.act_decoder_learning_rate,
                "weight_decay": args.act_decoder_weight_decay,
                "warmup_epochs": args.act_decoder_warmup_epochs,
                "hidden_dim": args.act_decoder_hidden_dim,
                "latentaction_type": args.action_decoder_latentaction_type,
                "use_aug": bool(args.act_decoder_use_aug),
                "precompute_stacked_obs": bool(args.act_decoder_precompute_stacked_obs),
                "data_path": args.data_path,
                "dcs_backgrounds_path": args.act_decoder_dcs_backgrounds_path
                or args.bc_dcs_backgrounds_path,
                "dcs_backgrounds_split": args.act_decoder_dcs_backgrounds_split
                or args.bc_dcs_backgrounds_split,
                "eval_episodes": args.act_decoder_eval_episodes,
                "eval_seed": args.act_decoder_eval_seed,
                "resume_path": args.act_decoder_resume_path,
            },
        }
    )


def init_wandb(args: argparse.Namespace, cfg, output_dir: Path):
    if args.no_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=args.wandb_tags,
        dir=str(output_dir),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if (
        args.lam_type == "otf"
        and (args.dinolam_checkpoint_path or args.dinolam_checkpoint_dir)
        and not (args.otf_lam_checkpoint_path or args.otf_lam_checkpoint_dir)
    ):
        args.lam_type = "dino"

    otf_lam_checkpoint_path = None
    dinolam_checkpoint_path = None
    needs_labeler = bool(args.train_bc) or (
        bool(args.train_act_decoder)
        and (args.lam_type == "dino" or args.action_decoder_latentaction_type == "posterior")
    )
    needs_lam_checkpoint = needs_labeler or (args.output_dir is None and args.bc_resume_path is None)
    if needs_lam_checkpoint:
        if args.lam_type == "otf":
            otf_lam_checkpoint_path = resolve_otf_lam_checkpoint(
                args.otf_lam_checkpoint_path,
                args.otf_lam_checkpoint_dir,
            )
        elif args.lam_type == "dino":
            dinolam_checkpoint_path = resolve_dinolam_checkpoint(
                args.dinolam_checkpoint_path,
                args.dinolam_checkpoint_dir,
            )
        else:
            raise ValueError(f"Unknown lam_type={args.lam_type!r}")
    checkpoint_path = otf_lam_checkpoint_path if args.lam_type == "otf" else dinolam_checkpoint_path
    output_dir = Path(args.output_dir).expanduser() if args.output_dir is not None else None
    if output_dir is None:
        if checkpoint_path is not None:
            output_dir = checkpoint_path.parent.parent / "downstream"
        else:
            output_dir = Path(args.bc_resume_path).expanduser().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    explicit_bc_resume_path = args.bc_resume_path
    args.bc_resume_path = resolve_bc_resume_path(
        output_dir,
        args.bc_resume_path,
        auto_resume=not args.no_auto_bc_resume,
    )
    cfg = build_cfg(args, otf_lam_checkpoint_path, dinolam_checkpoint_path, output_dir)
    (output_dir / "resolved_downstream_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    labeler = None
    if needs_labeler:
        if args.lam_type == "otf":
            model, _, _ = load_otf_lam_checkpoint(
                otf_lam_checkpoint_path,
                device,
                otf_vqvae_checkpoint_path=args.otf_vqvae_checkpoint_path,
            )
            labeler = OTFLAMActionLabeler(model, frame_stack=cfg.bc.frame_stack).to(device)
        elif args.lam_type == "dino":
            from dinolam.utils import load_dinolam_checkpoint

            model, _, _ = load_dinolam_checkpoint(
                dinolam_checkpoint_path,
                device,
                otf_vqvae_checkpoint_path=args.otf_vqvae_checkpoint_path,
            )
            labeler = DINOLAMActionLabeler(model, frame_stack=cfg.bc.frame_stack).to(device)
        else:
            raise ValueError(f"Unknown lam_type={args.lam_type!r}")
    wandb_run = init_wandb(args, cfg, output_dir)

    print(f"device={device}")
    print(f"lam_type={cfg.lam_type}")
    if otf_lam_checkpoint_path is not None:
        print(f"otf_lam_checkpoint={otf_lam_checkpoint_path}")
    elif dinolam_checkpoint_path is not None:
        print(f"dinolam_checkpoint={dinolam_checkpoint_path}")
    elif not cfg.train_bc:
        print("lam_checkpoint=None (skipped because --train_bc=false and BC checkpoint is used)")
    print(f"output_dir={output_dir}")
    print(f"data_path={cfg.bc.data_path}")
    print(f"action_decoder_latentaction_type={cfg.act_decoder.latentaction_type}")
    if labeler is not None:
        print(f"latent_action_dim={labeler.latent_act_dim}")
        if hasattr(labeler, "policy_state_dim"):
            print(f"policy_state_dim={labeler.policy_state_dim}")
    if cfg.bc.resume_path is not None:
        resume_source = "explicit" if explicit_bc_resume_path else "auto"
        print(f"bc_resume_path={cfg.bc.resume_path} ({resume_source})")

    actor = None
    if cfg.train_bc:
        actor = train_bc(cfg, labeler, device, wandb_run=wandb_run)
    if cfg.train_act_decoder:
        if actor is None:
            bc_checkpoint_path = (
                Path(str(cfg.bc.resume_path)) if cfg.bc.resume_path else Path(str(cfg.ckpt_dir)) / "bc-final.pt"
            )
            if not bc_checkpoint_path.exists():
                raise ValueError(
                    f"Actor is required for action-decoder training. Enable --train_bc or provide --bc_resume_path."
                )
            actor, _ = load_bc_actor_from_checkpoint(cfg, bc_checkpoint_path, device)
            print(f"Loaded BC checkpoint for action-decoder training from {bc_checkpoint_path}")
        train_act_decoder(cfg, actor, device, lam_labeler=labeler, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
