from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from otf_vqvae.model import set_seed

from otf_lam.dataset import build_raw_frame_dataset
from otf_lam.utils import (
    build_otf_lam,
    choose_device,
    compute_rgb_metrics,
    evaluate_lam,
    gate_statistics,
    load_otf_vqvae_from_checkpoint,
    make_run_name,
    move_batch_to_device,
    resolve_otf_vqvae_checkpoint_path,
    save_checkpoint,
    save_qualitative_examples,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OTF-LAM on top of a frozen OTF-VQ-VAE.")
    parser.add_argument("--otf_vqvae_checkpoint_path", type=str, default=None)
    parser.add_argument("--otf_vqvae_checkpoint_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prediction_mode", choices=("residual", "direct"), default="residual")
    parser.add_argument("--occupancy_embed_dim", type=int, default=16)
    parser.add_argument("--occupancy_encoder_type", choices=("small_cnn", "mlp"), default="small_cnn")
    parser.add_argument("--factor_hidden_dim", type=int, default=128)
    parser.add_argument("--factor_embed_dim", type=int, default=128)
    parser.add_argument("--state_feature_dim", type=int, default=128)
    parser.add_argument("--gate_hidden_dim", type=int, default=128)
    parser.add_argument("--z_action_dim", type=int, default=128)
    parser.add_argument("--decoder_hidden_dim", type=int, default=128)
    parser.add_argument("--aggregator_type", type=str, default="gate")
    parser.add_argument("--use_gate_sparsity_loss", type=bool, default=False)
    parser.add_argument("--gate_sparsity_weight", type=float, default=0.0)
    parser.add_argument("--no_mask_inactive_factors", action="store_true")
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--eval_every_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_every_steps", type=int, default=5000)
    parser.add_argument("--qual_every_steps", type=int, default=5000)
    parser.add_argument("--num_qual_examples", type=int, default=4)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="otf-lam")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def make_lam_config(
    args: argparse.Namespace,
    otf_vqvae_cfg: DictConfig,
    output_root: Path,
    output_dir: Path,
    run_name: str,
) -> DictConfig:
    model_cfg = otf_vqvae_cfg.model if "model" in otf_vqvae_cfg else otf_vqvae_cfg
    cfg = {
        "image_height": int(model_cfg.image_height),
        "image_width": int(model_cfg.image_width),
        "channels": int(model_cfg.channels),
        "prediction_mode": args.prediction_mode,
        "occupancy_embed_dim": args.occupancy_embed_dim,
        "occupancy_encoder_type": args.occupancy_encoder_type,
        "factor_hidden_dim": args.factor_hidden_dim,
        "factor_embed_dim": args.factor_embed_dim,
        "state_feature_dim": args.state_feature_dim,
        "gate_hidden_dim": args.gate_hidden_dim,
        "z_action_dim": args.z_action_dim,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "mask_inactive_factors": not args.no_mask_inactive_factors,
        "activation": str(model_cfg.activation) if "activation" in model_cfg else "gelu",
        "aggregator_type": args.aggregator_type,
        "use_gate_sparsity_loss": bool(args.use_gate_sparsity_loss),
        "gate_sparsity_weight": float(args.gate_sparsity_weight),
        "train": {
            "batch_size": int(args.batch_size),
            "num_epochs": int(args.num_epochs),
            "max_steps": args.max_steps,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": args.grad_clip_norm,
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
        },
        "run_name": run_name,
        "output_root": str(output_root),
        "output_dir": str(output_dir),
    }
    return OmegaConf.create(cfg)


def init_wandb(args: argparse.Namespace, cfg: DictConfig, output_dir: Path, run_name: str):
    if not args.use_wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(output_dir),
    )


def log_wandb(run, values: Dict[str, float], step: int) -> None:
    if run is not None:
        run.log(values, step=step)


def maybe_log_qualitative_to_wandb(run, summary: Dict, step: int, prefix: str) -> None:
    if run is None:
        return
    import wandb

    images = {}
    for idx, record in enumerate(summary.get("examples", [])[:2]):
        images[f"{prefix}/sample_{idx}_prediction"] = wandb.Image(record["grid"])
        images[f"{prefix}/sample_{idx}_gates"] = wandb.Image(record["gates"])
        images[f"{prefix}/sample_{idx}_occupancy"] = wandb.Image(record["occupancy"])
    if images:
        run.log(images, step=step)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = choose_device(args.device)
    otf_vqvae_checkpoint_path = resolve_otf_vqvae_checkpoint_path(
        args.otf_vqvae_checkpoint_path,
        args.otf_vqvae_checkpoint_dir,
    )
    otf_vqvae, otf_vqvae_cfg, _ = load_otf_vqvae_from_checkpoint(otf_vqvae_checkpoint_path, device)
    if "data" not in otf_vqvae_cfg:
        raise KeyError("OTF-VQ-VAE checkpoint config does not include data settings")

    run_name = make_run_name(args.wandb_run_name)
    output_root = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else Path("/cs/data/people/hnam16/otf_lam_runs")
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_lam_config(args, otf_vqvae_cfg, output_root, output_dir, run_name)
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    train_dataset = build_raw_frame_dataset(
        otf_vqvae_cfg.data,
        motion_input_type=otf_vqvae.motion_input_type,
        motion_transform=otf_vqvae.motion_transform,
        data_path=args.data_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    if len(train_loader) == 0:
        raise RuntimeError("The training DataLoader is empty")

    val_dataset = None
    val_loader = None
    if args.val_data_dir is not None:
        val_dataset = build_raw_frame_dataset(
            otf_vqvae_cfg.data,
            motion_input_type=otf_vqvae.motion_input_type,
            motion_transform=otf_vqvae.motion_transform,
            data_path=args.val_data_dir,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = build_otf_lam(otf_vqvae, cfg, device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch_load(args.resume_checkpoint, map_location=device)
        model.load_second_stage_state_dict(checkpoint["otf_lam_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))
        print(f"resumed_checkpoint={args.resume_checkpoint} step={global_step}")

    wandb_run = init_wandb(args, cfg, output_dir, run_name)
    total_steps = int(args.max_steps) if args.max_steps is not None else int(args.num_epochs) * len(train_loader)
    cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    otf_vqvae_cfg_snapshot = OmegaConf.to_container(otf_vqvae_cfg, resolve=True)

    print(f"device={device}")
    print(f"otf_vqvae_checkpoint={otf_vqvae_checkpoint_path}")
    print(f"run_name={run_name}")
    print(f"output_root={output_root}")
    print(f"output_dir={output_dir}")
    print(f"train_dataset_size={len(train_dataset)} num_batches={len(train_loader)}")
    print(f"motion_input_type={otf_vqvae.motion_input_type} motion_transform={otf_vqvae.motion_transform}")
    print(f"prediction_mode={cfg.prediction_mode} total_steps={total_steps}")
    if val_loader is None:
        print("validation=disabled (pass --val_data_dir to enable periodic validation)")

    epoch = start_epoch
    while global_step < total_steps:
        model.train()
        epoch += 1
        for batch in train_loader:
            if global_step >= total_steps:
                break
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            loss_rgb = F.mse_loss(output["pred"], output["target"])
            loss = loss_rgb
            if bool(cfg.use_gate_sparsity_loss) and float(cfg.gate_sparsity_weight) > 0.0:
                loss = loss + float(cfg.gate_sparsity_weight) * output["alpha"].mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), float(args.grad_clip_norm))
            optimizer.step()

            global_step += 1
            metrics = compute_rgb_metrics(output["pred"].detach(), output["target"].detach())
            gates = gate_statistics(output["alpha"].detach())
            log_values = {
                "train/loss_rgb": float(loss_rgb.item()),
                "train/loss": float(loss.item()),
                "train/rmse": metrics["rmse"],
                "train/mae": metrics["mae"],
                "train/psnr": metrics["psnr"],
                "train/gate_mean": gates["gate_mean"],
                "train/gates_gt_0.1": gates["gates_gt_0.1"],
                "train/gates_gt_0.3": gates["gates_gt_0.3"],
                "train/gates_gt_0.5": gates["gates_gt_0.5"],
                "train/gate_entropy": gates["gate_entropy"],
                "train/z_act_norm": float(output["z_act"].detach().norm(dim=-1).mean().item()),
                "train/epoch": float(epoch),
            }
            log_wandb(wandb_run, log_values, global_step)

            if global_step == 1 or global_step % int(args.log_every_steps) == 0:
                print(
                    " ".join(
                        [
                            f"step={global_step:06d}",
                            f"epoch={epoch}",
                            f"loss_rgb={loss_rgb.item():.6f}",
                            f"rmse={metrics['rmse']:.6f}",
                            f"mae={metrics['mae']:.6f}",
                            f"gate_mean={gates['gate_mean']:.4f}",
                            f"z_norm={log_values['train/z_act_norm']:.4f}",
                        ]
                    )
                )

            if val_loader is not None and int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
                val_metrics = evaluate_lam(model, val_loader, device)
                prefixed = {f"val/{key if key != 'mse' else 'loss_rgb'}": value for key, value in val_metrics.items()}
                log_wandb(wandb_run, prefixed, global_step)
                print(
                    f"validation step={global_step:06d} "
                    f"mse={val_metrics['mse']:.6f} rmse={val_metrics['rmse']:.6f} "
                    f"mae={val_metrics['mae']:.6f}"
                )

            if (
                val_dataset is not None
                and int(args.qual_every_steps) > 0
                and global_step % int(args.qual_every_steps) == 0
            ):
                summary = save_qualitative_examples(
                    model,
                    val_dataset,
                    output_dir / "qualitative" / f"step_{global_step:06d}",
                    device,
                    num_examples=int(args.num_qual_examples),
                )
                maybe_log_qualitative_to_wandb(wandb_run, summary, global_step, "val/qualitative")

            if int(args.checkpoint_every_steps) > 0 and global_step % int(args.checkpoint_every_steps) == 0:
                checkpoint_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    cfg_snapshot=cfg_snapshot,
                    otf_vqvae_checkpoint_path=otf_vqvae_checkpoint_path,
                    otf_vqvae_cfg_snapshot=otf_vqvae_cfg_snapshot,
                    epoch=epoch,
                    step=global_step,
                    output_dir=output_dir,
                )
                print(f"saved_checkpoint={checkpoint_path}")

    final_checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg_snapshot=cfg_snapshot,
        otf_vqvae_checkpoint_path=otf_vqvae_checkpoint_path,
        otf_vqvae_cfg_snapshot=otf_vqvae_cfg_snapshot,
        epoch=epoch,
        step=global_step,
        output_dir=output_dir,
        final=True,
    )
    print(f"saved_final_checkpoint={final_checkpoint_path}")

    if val_dataset is not None:
        final_summary = save_qualitative_examples(
            model,
            val_dataset,
            output_dir / "qualitative" / "final",
            device,
            num_examples=int(args.num_qual_examples),
        )
        maybe_log_qualitative_to_wandb(wandb_run, final_summary, global_step, "val/final_qualitative")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
