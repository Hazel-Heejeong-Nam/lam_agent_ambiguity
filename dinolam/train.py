from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from otf_vqvae.model import set_seed

from dinolam.dataset import build_raw_frame_dataset
from dinolam.model import (
    DINOJEPALatentActionModel,
    FrozenDINOv2Encoder,
    JEPAVQMotionExtractor,
    jepa_token_metrics,
)
from dinolam.utils import (
    attention_statistics,
    choose_device,
    code_usage_statistics,
    evaluate_jepa,
    load_otf_vqvae_from_checkpoint,
    make_run_name,
    move_batch_to_device,
    parameter_grad_norm,
    resolve_otf_vqvae_checkpoint_path,
    save_dinolam_checkpoint,
    save_jepa_debug_examples,
    torch_load,
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


def _cfg_default(cfg: Optional[DictConfig], key: str, default):
    if cfg is None:
        return default
    if key in cfg:
        return cfg[key]
    if "train" in cfg and key in cfg.train:
        return cfg.train[key]
    return default


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=None)
    known_args, _ = base_parser.parse_known_args()
    config = OmegaConf.load(known_args.config) if known_args.config is not None else None

    parser = argparse.ArgumentParser(
        description="Train DINO-LAM Stage 1 DINO-JEPA latent action labeler.",
        parents=[base_parser],
    )
    parser.add_argument("--otf_vqvae_checkpoint_path", type=str, default=_cfg_default(config, "otf_vqvae_checkpoint_path", None))
    parser.add_argument("--otf_vqvae_checkpoint_dir", type=str, default=_cfg_default(config, "otf_vqvae_checkpoint_dir", None))
    parser.add_argument("--data_config_path", type=str, default=_cfg_default(config, "data_config_path", None))
    parser.add_argument("--data_dir", type=str, default=_cfg_default(config, "data_dir", None))
    parser.add_argument("--output_dir", type=str, default=_cfg_default(config, "output_dir", None))
    parser.add_argument("--resume_checkpoint", type=str, default=_cfg_default(config, "resume_checkpoint", None))
    parser.add_argument("--resume_vq_ckpt", type=str2bool, default=_cfg_default(config, "resume_vq_ckpt", False))
    parser.add_argument("--resume_vq_search_root", type=str, default=_cfg_default(config, "resume_vq_search_root", None))
    parser.add_argument("--batch_size", type=int, default=_cfg_default(config, "batch_size", 64))
    parser.add_argument("--num_epochs", type=int, default=_cfg_default(config, "num_epochs", 10))
    parser.add_argument("--max_steps", type=int, default=_cfg_default(config, "max_steps", None))
    parser.add_argument("--lr", type=float, default=_cfg_default(config, "lr", 1.0e-4))
    parser.add_argument("--weight_decay", type=float, default=_cfg_default(config, "weight_decay", 0.0))
    parser.add_argument("--grad_clip_norm", type=float, default=_cfg_default(config, "grad_clip_norm", 1.0))
    parser.add_argument("--device", type=str, default=_cfg_default(config, "device", "auto"))
    parser.add_argument("--num_workers", type=int, default=_cfg_default(config, "num_workers", 4))
    parser.add_argument("--seed", type=int, default=_cfg_default(config, "seed", 0))

    parser.add_argument("--dino_model_name", type=str, default=_cfg_default(config, "dino_model_name", "facebook/dinov2-small"))
    parser.add_argument("--dino_image_size", type=int, default=_cfg_default(config, "dino_image_size", 224))
    parser.add_argument("--dino_mean", type=float, nargs=3, default=_cfg_default(config, "dino_mean", (0.485, 0.456, 0.406)))
    parser.add_argument("--dino_std", type=float, nargs=3, default=_cfg_default(config, "dino_std", (0.229, 0.224, 0.225)))
    parser.add_argument("--freeze_dino", type=str2bool, default=_cfg_default(config, "freeze_dino", True))
    parser.add_argument("--grid_alignment", choices=("exact", "nearest_resize"), default=_cfg_default(config, "grid_alignment", "exact"))

    parser.add_argument("--action_aggregator_type", choices=("perceiver", "transformer_pool", "mean_pool"), default=_cfg_default(config, "action_aggregator_type", "perceiver"))
    parser.add_argument("--aggregator_dim", type=int, default=_cfg_default(config, "aggregator_dim", 256))
    parser.add_argument("--aggregator_depth", type=int, default=_cfg_default(config, "aggregator_depth", 2))
    parser.add_argument("--aggregator_heads", type=int, default=_cfg_default(config, "aggregator_heads", 4))
    parser.add_argument("--aggregator_mlp_dim", type=int, default=_cfg_default(config, "aggregator_mlp_dim", 1024))
    parser.add_argument("--num_action_queries", type=int, default=_cfg_default(config, "num_action_queries", 4))
    parser.add_argument("--z_action_dim", type=int, default=_cfg_default(config, "z_action_dim", 256))
    parser.add_argument("--aggregator_dropout", type=float, default=_cfg_default(config, "aggregator_dropout", 0.1))

    parser.add_argument("--predictor_dim", type=int, default=_cfg_default(config, "predictor_dim", 384))
    parser.add_argument("--predictor_depth", type=int, default=_cfg_default(config, "predictor_depth", 2))
    parser.add_argument("--predictor_heads", type=int, default=_cfg_default(config, "predictor_heads", 6))
    parser.add_argument("--predictor_mlp_dim", type=int, default=_cfg_default(config, "predictor_mlp_dim", 1536))
    parser.add_argument("--predictor_dropout", type=float, default=_cfg_default(config, "predictor_dropout", 0.1))
    parser.add_argument("--predictor_emb_dropout", type=float, default=_cfg_default(config, "predictor_emb_dropout", 0.1))
    parser.add_argument("--inject_global_each_layer", type=str2bool, default=_cfg_default(config, "inject_global_each_layer", True))
    parser.add_argument("--global_conditioning_type", choices=("additive", "film"), default=_cfg_default(config, "global_conditioning_type", "additive"))
    parser.add_argument("--jepa_loss_type", choices=("cosine", "mse", "smooth_l1"), default=_cfg_default(config, "jepa_loss_type", "mse"))
    parser.add_argument("--target_mode", choices=("future", "delta"), default=_cfg_default(config, "target_mode", "future"))
    parser.add_argument("--use_motion_codes", type=str2bool, default=_cfg_default(config, "use_motion_codes", True))
    parser.add_argument("--use_global_action_token", type=str2bool, default=_cfg_default(config, "use_global_action_token", True))
    parser.add_argument(
        "--use_patch_motion_codes_in_predictor",
        "--use_patch_motion_codes",
        dest="use_patch_motion_codes_in_predictor",
        type=str2bool,
        default=_cfg_default(config, "use_patch_motion_codes_in_predictor", False),
    )
    parser.add_argument("--vq_finetune_mode", choices=("frozen", "encoder", "encoder_codebook"), default=_cfg_default(config, "vq_finetune_mode", "frozen"))
    parser.add_argument("--use_ema_codebook_update", type=str2bool, default=_cfg_default(config, "use_ema_codebook_update", True))
    parser.add_argument("--use_dead_code_reinit", type=str2bool, default=_cfg_default(config, "use_dead_code_reinit", True))
    parser.add_argument("--dead_code_threshold_steps", type=int, default=_cfg_default(config, "dead_code_threshold_steps", 1000))
    parser.add_argument("--use_codebook_orthogonality_loss", type=str2bool, default=_cfg_default(config, "use_codebook_orthogonality_loss", True))
    parser.add_argument("--codebook_orthogonality_weight", type=float, default=_cfg_default(config, "codebook_orthogonality_weight", 1.0e-4))
    parser.add_argument("--motion_input_type", choices=("velocity", "acceleration"), default=_cfg_default(config, "motion_input_type", "velocity"))
    parser.add_argument("--motion_transform", type=str, default=_cfg_default(config, "motion_transform", "none"))

    parser.add_argument("--log_every_steps", type=int, default=_cfg_default(config, "log_every_steps", 50))
    parser.add_argument("--eval_every_steps", type=int, default=_cfg_default(config, "eval_every_steps", 1000))
    parser.add_argument("--checkpoint_every_steps", type=int, default=_cfg_default(config, "checkpoint_every_steps", 10000))
    parser.add_argument("--qual_every_steps", type=int, default=_cfg_default(config, "qual_every_steps", 5000))
    parser.add_argument("--num_qual_examples", type=int, default=_cfg_default(config, "num_qual_examples", 4))
    parser.add_argument("--eval_max_batches", type=int, default=_cfg_default(config, "eval_max_batches", None))
    parser.add_argument("--use_wandb", type=str2bool, default=_cfg_default(config, "use_wandb", True))
    parser.add_argument("--wandb_project", type=str, default=_cfg_default(config, "wandb_project", "dinolam"))
    parser.add_argument("--wandb_run_name", type=str, default=_cfg_default(config, "wandb_run_name", None))
    return parser.parse_args()


def load_config_snapshot(path: str | Path) -> DictConfig:
    path = Path(path).expanduser()
    if path.suffix.lower() in {".yaml", ".yml"}:
        return OmegaConf.load(path)
    checkpoint = torch_load(path, map_location="cpu")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain config: {path}")
    return OmegaConf.create(checkpoint["config"])


def _resolved_path_string(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _checkpoint_otf_vqvae_path(checkpoint: Dict) -> Optional[str]:
    path = checkpoint.get("otf_vqvae_checkpoint_path")
    if path is None:
        cfg_snapshot = checkpoint.get("cfg_snapshot") or {}
        path = cfg_snapshot.get("otf_vqvae_checkpoint_path")
    return None if path is None else str(path)


def find_latest_resume_checkpoint_for_vq(
    *,
    search_root: str | Path,
    otf_vqvae_checkpoint_path: str | Path,
) -> Path:
    root = Path(search_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Stage 1 resume search root does not exist: {root}")

    target_otf_vqvae_path = _resolved_path_string(otf_vqvae_checkpoint_path)
    matches = []
    for checkpoint_dir in root.rglob("checkpoints"):
        checkpoint_files = sorted(
            checkpoint_dir.glob("dinolam*.pt"),
            key=lambda path: (path.stat().st_mtime, str(path)),
        )
        if not checkpoint_files:
            continue
        latest_checkpoint = checkpoint_files[-1]
        try:
            checkpoint = torch_load(latest_checkpoint, map_location="cpu")
        except Exception as exc:
            print(f"[WARN] Skipping unreadable Stage 1 checkpoint: {latest_checkpoint} ({exc})")
            continue
        saved_otf_vqvae_path = _checkpoint_otf_vqvae_path(checkpoint)
        if saved_otf_vqvae_path is None:
            continue
        if _resolved_path_string(saved_otf_vqvae_path) == target_otf_vqvae_path:
            matches.append((latest_checkpoint.stat().st_mtime, str(checkpoint_dir.parent), latest_checkpoint))

    if not matches:
        raise FileNotFoundError(
            "Could not find a Stage 1 checkpoint trained with "
            f"otf_vqvae_checkpoint_path={target_otf_vqvae_path} under {root}"
        )
    matches.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return matches[-1][2]


def should_use_vq(args: argparse.Namespace) -> bool:
    return bool(args.use_motion_codes) or str(args.vq_finetune_mode) != "frozen"


def resolve_data_and_motion_settings(args: argparse.Namespace, otf_vqvae, source_cfg: DictConfig):
    data_cfg = source_cfg.data if "data" in source_cfg else source_cfg
    if otf_vqvae is not None:
        return data_cfg, otf_vqvae.motion_input_type, otf_vqvae.motion_transform
    model_cfg = source_cfg.model if "model" in source_cfg else None
    motion_input_type = (
        str(model_cfg.motion_input_type)
        if model_cfg is not None and "motion_input_type" in model_cfg
        else str(args.motion_input_type)
    )
    motion_transform = (
        str(model_cfg.motion_transform)
        if model_cfg is not None and "motion_transform" in model_cfg
        else str(args.motion_transform)
    )
    return data_cfg, motion_input_type, motion_transform


def make_dinolam_config(
    args: argparse.Namespace,
    source_cfg: DictConfig,
    output_root: Path,
    output_dir: Path,
    run_name: str,
) -> DictConfig:
    model_cfg = source_cfg.model if "model" in source_cfg else None
    data_cfg = source_cfg.data if "data" in source_cfg else source_cfg
    image_height = int(model_cfg.image_height) if model_cfg is not None and "image_height" in model_cfg else int(data_cfg.image_height)
    image_width = int(model_cfg.image_width) if model_cfg is not None and "image_width" in model_cfg else int(data_cfg.image_width)
    channels = int(model_cfg.channels) if model_cfg is not None and "channels" in model_cfg else int(data_cfg.channels)
    cfg = {
        "otf_vqvae_checkpoint_path": None if args.otf_vqvae_checkpoint_path is None else str(Path(args.otf_vqvae_checkpoint_path).expanduser()),
        "otf_vqvae_checkpoint_dir": None if args.otf_vqvae_checkpoint_dir is None else str(Path(args.otf_vqvae_checkpoint_dir).expanduser()),
        "image_height": image_height,
        "image_width": image_width,
        "channels": channels,
        "dino_model_name": args.dino_model_name,
        "dino_image_size": int(args.dino_image_size),
        "dino_mean": [float(value) for value in args.dino_mean],
        "dino_std": [float(value) for value in args.dino_std],
        "freeze_dino": bool(args.freeze_dino),
        "grid_alignment": args.grid_alignment,
        "action_aggregator_type": args.action_aggregator_type,
        "aggregator_dim": int(args.aggregator_dim),
        "aggregator_depth": int(args.aggregator_depth),
        "aggregator_heads": int(args.aggregator_heads),
        "aggregator_mlp_dim": int(args.aggregator_mlp_dim),
        "num_action_queries": int(args.num_action_queries),
        "z_action_dim": int(args.z_action_dim),
        "aggregator_dropout": float(args.aggregator_dropout),
        "predictor_dim": int(args.predictor_dim),
        "predictor_depth": int(args.predictor_depth),
        "predictor_heads": int(args.predictor_heads),
        "predictor_mlp_dim": int(args.predictor_mlp_dim),
        "predictor_dropout": float(args.predictor_dropout),
        "predictor_emb_dropout": float(args.predictor_emb_dropout),
        "inject_global_each_layer": bool(args.inject_global_each_layer),
        "global_conditioning_type": args.global_conditioning_type,
        "jepa_loss_type": args.jepa_loss_type,
        "target_mode": args.target_mode,
        "use_motion_codes": bool(args.use_motion_codes),
        "use_global_action_token": bool(args.use_global_action_token),
        "use_patch_motion_codes_in_predictor": bool(args.use_patch_motion_codes_in_predictor),
        "vq_finetune_mode": args.vq_finetune_mode,
        "use_ema_codebook_update": bool(args.use_ema_codebook_update),
        "use_dead_code_reinit": bool(args.use_dead_code_reinit),
        "dead_code_threshold_steps": int(args.dead_code_threshold_steps),
        "use_codebook_orthogonality_loss": bool(args.use_codebook_orthogonality_loss),
        "codebook_orthogonality_weight": float(args.codebook_orthogonality_weight),
        "activation": str(model_cfg.activation) if model_cfg is not None and "activation" in model_cfg else "gelu",
        "train": {
            "batch_size": int(args.batch_size),
            "num_epochs": int(args.num_epochs),
            "max_steps": args.max_steps,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": args.grad_clip_norm,
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "data_dir": args.data_dir,
            "train_split": "train",
            "val_split": "test",
            "resume_checkpoint": args.resume_checkpoint,
            "resume_vq_ckpt": bool(args.resume_vq_ckpt),
            "resume_vq_search_root": args.resume_vq_search_root,
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


def maybe_log_debug_to_wandb(run, summary: Dict, step: int, prefix: str) -> None:
    if run is None:
        return
    import wandb

    images = {}
    for idx, record in enumerate(summary.get("examples", [])[:2]):
        images[f"{prefix}/sample_{idx}_diagnostics"] = wandb.Image(record["diagnostics"])
    if images:
        run.log(images, step=step)


def main() -> None:
    args = parse_args()
    if args.use_global_action_token and not args.use_motion_codes:
        raise ValueError("Global action token requires motion codes")
    if args.vq_finetune_mode != "frozen" and not args.use_motion_codes:
        raise ValueError("VQ fine-tuning requires use_motion_codes=true")

    set_seed(int(args.seed))
    device = choose_device(args.device)
    use_vq = should_use_vq(args)
    otf_vqvae_checkpoint_path = None
    otf_vqvae = None
    if use_vq:
        otf_vqvae_checkpoint_path = resolve_otf_vqvae_checkpoint_path(
            args.otf_vqvae_checkpoint_path,
            args.otf_vqvae_checkpoint_dir,
        )
        otf_vqvae, source_cfg, _ = load_otf_vqvae_from_checkpoint(otf_vqvae_checkpoint_path, device)
    else:
        if args.data_config_path is not None:
            source_cfg = load_config_snapshot(args.data_config_path)
        elif args.otf_vqvae_checkpoint_path is not None or args.otf_vqvae_checkpoint_dir is not None:
            otf_vqvae_checkpoint_path = resolve_otf_vqvae_checkpoint_path(
                args.otf_vqvae_checkpoint_path,
                args.otf_vqvae_checkpoint_dir,
            )
            source_cfg = load_config_snapshot(otf_vqvae_checkpoint_path)
            print(f"dino_only_mode=enabled; using data config from checkpoint: {otf_vqvae_checkpoint_path}")
        else:
            raise ValueError(
                "DINO-only mode without a OTF-VQ-VAE checkpoint requires --data_config_path "
                "with data settings."
            )
    if "data" not in source_cfg and "type" not in source_cfg:
        raise KeyError("Config does not include data settings")
    data_cfg, motion_input_type, motion_transform = resolve_data_and_motion_settings(args, otf_vqvae, source_cfg)

    run_name = make_run_name(args.wandb_run_name or "dinolam")
    output_root = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else Path("/cs/data/people/hnam16/dinolam_runs")
    )
    if args.resume_vq_ckpt:
        if otf_vqvae_checkpoint_path is None:
            raise ValueError("--resume_vq_ckpt requires a resolved OTF-VQ-VAE checkpoint path")
        if args.resume_checkpoint is not None:
            print(f"resume_vq_ckpt=true explicit_resume_checkpoint={args.resume_checkpoint}")
        else:
            resume_search_root = (
                Path(args.resume_vq_search_root).expanduser()
                if args.resume_vq_search_root is not None
                else output_root
            )
            args.resume_checkpoint = str(
                find_latest_resume_checkpoint_for_vq(
                    search_root=resume_search_root,
                    otf_vqvae_checkpoint_path=otf_vqvae_checkpoint_path,
                )
            )
            print(f"resume_vq_ckpt=true search_root={resume_search_root}")
            print(f"auto_resume_checkpoint={args.resume_checkpoint}")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_dinolam_config(args, source_cfg, output_root, output_dir, run_name)
    if not bool(cfg.use_motion_codes):
        cfg.mode = "dino_only"
    elif not bool(cfg.use_global_action_token):
        cfg.mode = f"local_direct_{cfg.vq_finetune_mode}"
    elif bool(cfg.use_patch_motion_codes_in_predictor):
        cfg.mode = f"local_global_{cfg.vq_finetune_mode}"
    else:
        cfg.mode = f"global_only_{cfg.vq_finetune_mode}"

    train_dataset = build_raw_frame_dataset(
        data_cfg,
        motion_input_type=motion_input_type,
        motion_transform=motion_transform,
        data_path=args.data_dir,
        split="train",
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

    val_dataset = build_raw_frame_dataset(
        data_cfg,
        motion_input_type=motion_input_type,
        motion_transform=motion_transform,
        data_path=args.data_dir,
        split="test",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    dino_encoder = FrozenDINOv2Encoder(
        model_name=str(cfg.dino_model_name),
        image_size=int(cfg.dino_image_size),
        mean=tuple(float(value) for value in cfg.dino_mean),
        std=tuple(float(value) for value in cfg.dino_std),
    ).to(device)
    otf_vqvae_extractor = None
    if use_vq:
        otf_vqvae_extractor = JEPAVQMotionExtractor(
            otf_vqvae,
            finetune_mode=str(cfg.vq_finetune_mode),
            use_ema_codebook_update=bool(cfg.use_ema_codebook_update),
            use_dead_code_reinit=bool(cfg.use_dead_code_reinit),
            dead_code_threshold_steps=int(cfg.dead_code_threshold_steps),
        )
    model = DINOJEPALatentActionModel(otf_vqvae_extractor, dino_encoder, cfg).to(device)
    cfg.patch_grid_match = None if model.patch_grid_match is None else bool(model.patch_grid_match)
    cfg.dino_patch_grid = [int(model.dino_grid_size[0]), int(model.dino_grid_size[1])]
    cfg.vq_patch_grid = None if model.vq_grid_size is None else [int(model.vq_grid_size[0]), int(model.vq_grid_size[1])]
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint is not None:
        checkpoint = torch_load(args.resume_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("dinolam_state_dict"))
        if state_dict is None:
            raise KeyError("Resume checkpoint does not contain model_state_dict")
        model.load_second_stage_state_dict(state_dict)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))
        print(f"resumed_checkpoint={args.resume_checkpoint} step={global_step}")

    wandb_run = init_wandb(args, cfg, output_dir, run_name)
    total_steps = int(args.max_steps) if args.max_steps is not None else int(args.num_epochs) * len(train_loader)
    cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    otf_vqvae_cfg_snapshot = OmegaConf.to_container(source_cfg, resolve=True)

    print(f"device={device}")
    print(f"otf_vqvae_checkpoint={otf_vqvae_checkpoint_path if otf_vqvae_checkpoint_path is not None else 'unused'}")
    print(f"dino_model={cfg.dino_model_name} dino_image_size={cfg.dino_image_size}")
    print(f"run_name={run_name}")
    print(f"output_root={output_root}")
    print(f"output_dir={output_dir}")
    print(f"train_dataset_size={len(train_dataset)} num_batches={len(train_loader)}")
    print(f"val_dataset_size={len(val_dataset)} num_batches={len(val_loader)}")
    print(f"motion_input_type={motion_input_type} motion_transform={motion_transform}")
    print(
        f"patch_grid_match={model.patch_grid_match} "
        f"dino_grid={model.dino_grid_size} vq_grid={model.vq_grid_size} "
        f"grid_alignment={model.grid_alignment}"
    )
    print(
        f"mode={cfg.mode} target_mode={cfg.target_mode} jepa_loss_type={cfg.jepa_loss_type} "
        f"vq_finetune_mode={cfg.vq_finetune_mode} "
        f"use_patch_motion_codes_in_predictor={model.use_patch_motion_codes_in_predictor} "
        f"use_global_action_token={model.use_global_action_token} total_steps={total_steps}"
    )
    epoch = start_epoch
    while global_step < total_steps:
        model.train()
        epoch += 1
        for batch in train_loader:
            if global_step >= total_steps:
                break
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            loss = output["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            vq_encoder_grad_norm = None
            codebook_grad_norm = None
            if model.otf_vqvae_extractor is not None and model.otf_vqvae_extractor.encoder_trainable:
                vq_encoder_grad_norm = parameter_grad_norm(model.otf_vqvae_extractor.otf_vqvae.encoder.parameters())
            if model.otf_vqvae_extractor is not None and model.otf_vqvae_extractor.codebook_gradient_trainable:
                codebook_grad_norm = parameter_grad_norm([model.otf_vqvae_extractor.otf_vqvae.quantizer.embedding])
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), float(args.grad_clip_norm))
            optimizer.step()

            global_step += 1
            codebook_update_info = model.update_codebook(global_step)
            token_metrics = jepa_token_metrics(output["pred_tokens"], output["target_tokens"])
            log_values = {
                "train/jepa_loss": float(output["jepa_loss"].item()),
                "train/feature_mse": token_metrics["feature_mse"],
                "train/cos": token_metrics["cosine_similarity"],
                "train/epoch": float(epoch),
                "H_dino_p": float(model.dino_grid_size[0]),
                "W_dino_p": float(model.dino_grid_size[1]),
            }
            if model.use_motion_codes and output["factors"] is not None:
                usage = code_usage_statistics(output["factors"])
                log_values.update(
                    {
                        "train/code_usage_active_ratio": usage["code_usage_active_ratio"],
                        "train/code_usage_entropy": usage["code_usage_entropy"],
                        "train/dead_codes": usage["dead_codes"],
                        "train/avg_active_codes_per_sample": usage["avg_active_codes_per_sample"],
                        "H_vq_p": float(model.vq_grid_size[0]),
                        "W_vq_p": float(model.vq_grid_size[1]),
                    }
                )
            if output.get("z_act") is not None:
                log_values.update(
                    {
                        "train/z_act_norm": float(output["z_act"].detach().norm(dim=-1).mean().item()),
                        "train/action_query_norm": float(output["action_query_norm"].detach().item()),
                        "train/aggregator_output_norm": float(output["aggregator_output_norm"].detach().item()),
                    }
                )
                attn_stats = attention_statistics(output.get("action_query_attn"))
                for key, value in attn_stats.items():
                    log_values[f"train/{key}"] = value
            if vq_encoder_grad_norm is not None:
                log_values["train/vq_encoder_grad_norm"] = vq_encoder_grad_norm
            if codebook_grad_norm is not None:
                log_values["train/codebook_grad_norm"] = codebook_grad_norm
            if output["codebook_orthogonality_loss"] is not None:
                log_values["train/codebook_orthogonality_metric"] = float(output["codebook_orthogonality_loss"].detach().item())
            if codebook_update_info.get("reinitialized_codes", 0):
                log_values["train/reinitialized_codes"] = float(codebook_update_info["reinitialized_codes"])
            log_wandb(wandb_run, log_values, global_step)

            if global_step == 1 or global_step % int(args.log_every_steps) == 0:
                fields = [
                    f"step={global_step:06d}",
                    f"epoch={epoch}",
                    f"jepa_loss={output['jepa_loss'].item():.6f}",
                    f"cos={token_metrics['cosine_similarity']:.6f}",
                    f"feature_mse={token_metrics['feature_mse']:.6f}",
                ]
                if output.get("z_act") is not None:
                    fields.append(f"z_norm={log_values['train/z_act_norm']:.4f}")
                print(" ".join(fields))

            if val_loader is not None and int(args.eval_every_steps) > 0 and global_step % int(args.eval_every_steps) == 0:
                val_metrics = evaluate_jepa(
                    model,
                    val_loader,
                    device,
                    max_batches=args.eval_max_batches,
                )
                log_wandb(
                    wandb_run,
                    {
                        "val/jepa_loss": val_metrics["jepa_loss"],
                        "val/feature_mse": val_metrics["feature_mse"],
                        "val/cos": val_metrics["cosine_similarity"],
                    },
                    global_step,
                )
                print(
                    f"validation step={global_step:06d} "
                    f"jepa_loss={val_metrics['jepa_loss']:.6f} "
                    f"cos={val_metrics['cosine_similarity']:.6f} "
                    f"feature_mse={val_metrics['feature_mse']:.6f}"
                )

            if (
                val_dataset is not None
                and int(args.qual_every_steps) > 0
                and global_step % int(args.qual_every_steps) == 0
            ):
                summary = save_jepa_debug_examples(
                    model,
                    val_dataset,
                    output_dir / "qualitative" / f"step_{global_step:06d}",
                    device,
                    num_examples=int(args.num_qual_examples),
                )
                maybe_log_debug_to_wandb(wandb_run, summary, global_step, "val/qualitative")

            if int(args.checkpoint_every_steps) > 0 and global_step % int(args.checkpoint_every_steps) == 0:
                checkpoint_path = save_dinolam_checkpoint(
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

    final_checkpoint_path = save_dinolam_checkpoint(
        model=model,
        optimizer=optimizer,
        cfg_snapshot=cfg_snapshot,
        otf_vqvae_checkpoint_path=otf_vqvae_checkpoint_path,
        otf_vqvae_cfg_snapshot=otf_vqvae_cfg_snapshot,
        epoch=epoch,
        step=global_step,
        output_dir=output_dir,
        filename_prefix="dinolam_final",
    )
    print(f"saved_final_checkpoint={final_checkpoint_path}")

    if val_dataset is not None:
        final_summary = save_jepa_debug_examples(
            model,
            val_dataset,
            output_dir / "qualitative" / "final",
            device,
            num_examples=int(args.num_qual_examples),
        )
        maybe_log_debug_to_wandb(wandb_run, final_summary, global_step, "val/final_qualitative")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
