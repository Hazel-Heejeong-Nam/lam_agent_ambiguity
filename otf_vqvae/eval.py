from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import model as train_main


BASE_OVERLAY_COLORS: Sequence[tuple[int, int, int]] = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained OTF-VQ-VAE checkpoint by visualizing the learned codebook "
            "and patch-level quantized assignments on dataset samples."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path. If omitted, the newest checkpoint under --checkpoint-root is used.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="/users/hnam16/scratch/otf_vqvae_runs",
        help="Root directory searched when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for evaluation outputs. Defaults to "
            "eval/otf_vqvae_runs/<run_name>/<checkpoint_stem>/."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference. Uses the same semantics as training: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of dataset samples to visualize.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated dataset indices to visualize. Overrides --num-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic evaluation and t-SNE.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=1,
        help="Transparency used for quantized patch overlays.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=None,
        help="Optional t-SNE perplexity. Defaults to an automatic value based on codebook size.",
    )
    return parser.parse_args()


def set_eval_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_checkpoint_step(path: Path) -> int:
    if path.stem.endswith("_final"):
        return 10**18
    match = re.search(r"_step(\d+)$", path.stem)
    if match is None:
        return -1
    return int(match.group(1))


def discover_latest_checkpoint(checkpoint_root: Path) -> Path:
    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"Checkpoint root does not exist: {checkpoint_root}. "
            "Pass --checkpoint explicitly or point --checkpoint-root to the run directory tree."
        )

    candidates = sorted(checkpoint_root.rglob("otf_vqvae_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No OTF-VQ-VAE checkpoints found under {checkpoint_root}")

    return max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime,
            parse_checkpoint_step(path),
            str(path),
        ),
    )


def build_output_dir(checkpoint_path: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir is not None:
        output_dir = Path(explicit_output_dir).expanduser()
    else:
        run_name = (
            checkpoint_path.parent.parent.name
            if checkpoint_path.parent.name == "checkpoints"
            else checkpoint_path.parent.name
        )
        output_dir = Path("eval") / "otf_vqvae_runs" / run_name / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[Dict[str, Any], DictConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a saved config: {checkpoint_path}")
    cfg = OmegaConf.create(checkpoint["config"])
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from checkpoint config, got {type(cfg)}")
    train_main.apply_checkpoint_model_settings(cfg, checkpoint)
    return checkpoint, cfg


def build_frozen_model(cfg: DictConfig, checkpoint: Dict[str, Any], device: torch.device) -> train_main.OTFVQVAE:
    train_main.apply_model_config_defaults(cfg.model)
    model = train_main.OTFVQVAE(cfg.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def parse_sample_indices(value: str | None, dataset_size: int, default_num_samples: int) -> List[int]:
    if value is not None:
        indices = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
        if not indices:
            raise ValueError("--sample-indices was provided but no indices were parsed")
    else:
        num_samples = max(1, min(default_num_samples, dataset_size))
        if dataset_size <= num_samples:
            indices = list(range(dataset_size))
        else:
            indices = np.linspace(0, dataset_size - 1, num=num_samples, dtype=int).tolist()

    unique_indices = []
    seen = set()
    for index in indices:
        if index < 0 or index >= dataset_size:
            raise IndexError(f"Sample index {index} is out of range for dataset of size {dataset_size}")
        if index not in seen:
            unique_indices.append(index)
            seen.add(index)
    return unique_indices


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(tensor.shape)}")
    image = tensor.detach().cpu().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
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
    image = image / (2.0 * max_abs_value) + 0.5
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = (image * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return image


def draw_patch_grid(
    image: Image.Image,
    patch_height: int,
    patch_width: int,
    foreground: tuple[int, int, int, int] = (255, 255, 255, 255),
    background: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> Image.Image:
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    def _draw_lines(color: tuple[int, int, int, int], line_width: int) -> None:
        for x in range(0, width + 1, patch_width):
            draw.line([(x, 0), (x, height)], fill=color, width=line_width)
        for y in range(0, height + 1, patch_height):
            draw.line([(0, y), (width, y)], fill=color, width=line_width)

    _draw_lines(background, 3)
    _draw_lines(foreground, 1)
    return canvas


def save_source_frame_triptych(
    previous_frame: torch.Tensor,
    current_frame: torch.Tensor,
    next_frame: torch.Tensor,
    output_path: Path,
) -> None:
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
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def extend_palette(num_colors: int) -> List[tuple[int, int, int]]:
    if num_colors <= len(BASE_OVERLAY_COLORS):
        return list(BASE_OVERLAY_COLORS[:num_colors])

    palette = list(BASE_OVERLAY_COLORS)
    cmap = plt.get_cmap("tab20", num_colors)
    for color_idx in range(len(BASE_OVERLAY_COLORS), num_colors):
        rgba = cmap(color_idx)
        palette.append(tuple(int(round(channel * 255.0)) for channel in rgba[:3]))
    return palette


def color_to_hex(color: Iterable[int]) -> str:
    red, green, blue = list(color)
    return f"#{red:02x}{green:02x}{blue:02x}"


def make_quantized_overlay_image(
    frame: torch.Tensor,
    assignment_grid: np.ndarray,
    patch_height: int,
    patch_width: int,
    alpha: float,
) -> tuple[Image.Image, List[Dict[str, Any]]]:
    base = Image.fromarray(tensor_to_uint8_image(frame)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    unique_codes = sorted(int(code) for code in np.unique(assignment_grid))
    colors = extend_palette(len(unique_codes))
    metadata = []
    alpha_value = int(round(255.0 * alpha))

    for color, code_id in zip(colors, unique_codes):
        positions = np.argwhere(assignment_grid == code_id)
        for row_idx, col_idx in positions:
            x0 = int(col_idx * patch_width)
            y0 = int(row_idx * patch_height)
            x1 = int((col_idx + 1) * patch_width) - 1
            y1 = int((row_idx + 1) * patch_height) - 1
            draw.rectangle([(x0, y0), (x1, y1)], fill=(*color, alpha_value))
        metadata.append(
            {
                "code_id": code_id,
                "color_rgb": list(color),
                "color_hex": color_to_hex(color),
                "num_patches": int(len(positions)),
            }
        )

    composited = Image.alpha_composite(base, overlay)
    composited = draw_patch_grid(composited, patch_height=patch_height, patch_width=patch_width)
    return composited, metadata


def automatic_tsne_perplexity(num_points: int) -> float:
    if num_points < 2:
        raise ValueError(f"Need at least 2 codebook vectors for t-SNE, got {num_points}")
    base = min(30.0, max(5.0, float(num_points - 1) / 3.0))
    return float(min(base, float(num_points - 1)))


def save_codebook_tsne(
    codebook: torch.Tensor,
    output_dir: Path,
    seed: int,
    perplexity: float | None,
) -> Dict[str, Any]:
    vectors = codebook.detach().cpu().numpy()
    num_codes = int(vectors.shape[0])
    perplexity = automatic_tsne_perplexity(num_codes) if perplexity is None else float(perplexity)
    if not 0.0 < perplexity < num_codes:
        raise ValueError(f"t-SNE perplexity must be in (0, {num_codes}), got {perplexity}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(vectors)

    csv_path = output_dir / "codebook_tsne.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code_id", "tsne_x", "tsne_y"])
        for code_id, (x_coord, y_coord) in enumerate(coords):
            writer.writerow([code_id, float(x_coord), float(y_coord)])

    fig, axis = plt.subplots(figsize=(10, 8))
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.arange(num_codes),
        cmap="tab20",
        s=60,
        edgecolors="black",
        linewidths=0.4,
    )
    for code_id, (x_coord, y_coord) in enumerate(coords):
        axis.text(x_coord, y_coord, str(code_id), fontsize=7, ha="center", va="center")
    axis.set_title(f"Codebook t-SNE ({num_codes} codes)")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=axis, label="Code ID")
    fig.tight_layout()
    png_path = output_dir / "codebook_tsne.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    return {
        "num_codes": num_codes,
        "perplexity": perplexity,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_patch_embedding_tsne(
    patch_embeddings: torch.Tensor,
    code_ids: np.ndarray,
    grid_height: int,
    grid_width: int,
    output_dir: Path,
    file_prefix: str,
    seed: int,
) -> Dict[str, Any]:
    if patch_embeddings.ndim != 2:
        raise ValueError(f"Expected [num_patches, latent_dim] patch embeddings, got {tuple(patch_embeddings.shape)}")

    vectors = patch_embeddings.detach().cpu().numpy()
    num_patches = int(vectors.shape[0])
    perplexity = automatic_tsne_perplexity(num_patches)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    coords = tsne.fit_transform(vectors)

    patch_rows, patch_cols = np.unravel_index(np.arange(num_patches), (grid_height, grid_width))
    csv_path = output_dir / f"{file_prefix}_patch_embedding_tsne.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patch_id", "patch_row", "patch_col", "code_id", "tsne_x", "tsne_y"])
        for patch_id in range(num_patches):
            writer.writerow(
                [
                    patch_id,
                    int(patch_rows[patch_id]),
                    int(patch_cols[patch_id]),
                    int(code_ids[patch_id]),
                    float(coords[patch_id, 0]),
                    float(coords[patch_id, 1]),
                ]
            )

    unique_codes = sorted(int(code) for code in np.unique(code_ids))
    palette = extend_palette(len(unique_codes))
    color_by_code = {code_id: color for code_id, color in zip(unique_codes, palette)}
    point_colors = [np.asarray(color_by_code[int(code_id)], dtype=np.float32) / 255.0 for code_id in code_ids]

    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=40,
        edgecolors="black",
        linewidths=0.35,
    )
    axis.set_title("Patch Embedding t-SNE Before Quantization")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")

    legend_labels = [
        f"code {code_id} ({int((code_ids == code_id).sum())} patches)"
        for code_id in unique_codes
    ]
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=np.asarray(color_by_code[code_id], dtype=np.float32) / 255.0,
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=8,
        )
        for code_id, label in zip(unique_codes, legend_labels)
    ]
    if legend_handles:
        axis.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    png_path = output_dir / f"{file_prefix}_patch_embedding_tsne.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    embedding_path = output_dir / f"{file_prefix}_patch_embeddings.npy"
    np.save(embedding_path, vectors)

    return {
        "num_patches": num_patches,
        "perplexity": perplexity,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
        "embedding_path": str(embedding_path),
    }


def evaluate_samples(
    model: train_main.OTFVQVAE,
    dataset: Any,
    sample_indices: Sequence[int],
    device: torch.device,
    output_dir: Path,
    alpha: float,
    seed: int,
) -> List[Dict[str, Any]]:
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    patch_height = int(model.encoder.patch_height)
    patch_width = int(model.encoder.patch_width)
    grid_height = int(model.encoder.grid_height)
    grid_width = int(model.encoder.grid_width)
    sample_summaries = []

    with torch.inference_mode():
        for sample_rank, dataset_index in enumerate(sample_indices):
            sample = dataset[dataset_index]
            batch = {
                key: value.unsqueeze(0).to(device, non_blocking=True)
                for key, value in sample.items()
            }
            motion = train_main.make_motion_signal(batch, model.motion_input_type, model.motion_transform)
            output = model(
                motion,
                reference_frame=batch.get("reference_frame"),
                use_quantization=True,
            )
            patch_embeddings = output["patch_embeddings"][0]
            assignment_codes = output["indices"][0]
            reconstruction = output["reconstruction"][0]

            assignment_grid = (
                assignment_codes
                .reshape(grid_height, grid_width)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            flat_code_ids = assignment_grid.reshape(-1)

            file_prefix = f"sample_{sample_rank:02d}_dataset_{dataset_index:06d}"
            current_frame = sample["current"]
            grid_image = draw_patch_grid(
                Image.fromarray(tensor_to_uint8_image(current_frame)),
                patch_height=patch_height,
                patch_width=patch_width,
            )
            motion_grid_image = draw_patch_grid(
                Image.fromarray(signed_tensor_to_uint8_image(motion[0])),
                patch_height=patch_height,
                patch_width=patch_width,
            )
            overlay_image, overlay_metadata = make_quantized_overlay_image(
                frame=current_frame,
                assignment_grid=assignment_grid,
                patch_height=patch_height,
                patch_width=patch_width,
                alpha=alpha,
            )
            motion_image = Image.fromarray(signed_tensor_to_uint8_image(motion[0]))
            reconstruction_image = Image.fromarray(signed_tensor_to_uint8_image(reconstruction))
            patch_tsne_summary = save_patch_embedding_tsne(
                patch_embeddings=patch_embeddings,
                code_ids=flat_code_ids,
                grid_height=grid_height,
                grid_width=grid_width,
                output_dir=samples_dir,
                file_prefix=file_prefix,
                seed=seed + sample_rank,
            )

            grid_path = samples_dir / f"{file_prefix}_grid.png"
            motion_grid_path = samples_dir / f"{file_prefix}_motion_grid.png"
            source_frames_path = samples_dir / f"{file_prefix}_source_frames.png"
            overlay_path = samples_dir / f"{file_prefix}_quantized_overlay.png"
            motion_path = samples_dir / f"{file_prefix}_motion.png"
            reconstruction_path = samples_dir / f"{file_prefix}_reconstruction.png"
            assignment_path = samples_dir / f"{file_prefix}_patch_codes.npy"
            sample_metadata_path = samples_dir / f"{file_prefix}_metadata.json"

            grid_image.save(grid_path)
            motion_grid_image.save(motion_grid_path)
            saved_source_frames_path = None
            if "previous" in sample:
                save_source_frame_triptych(
                    previous_frame=sample["previous"],
                    current_frame=sample["current"],
                    next_frame=sample["next"],
                    output_path=source_frames_path,
                )
                saved_source_frames_path = str(source_frames_path)
            overlay_image.save(overlay_path)
            motion_image.save(motion_path)
            reconstruction_image.save(reconstruction_path)
            np.save(assignment_path, assignment_grid)

            summary = {
                "sample_rank": sample_rank,
                "dataset_index": int(dataset_index),
                "motion_signal_formula": train_main.describe_motion_signal(model.motion_input_type),
                "motion_transform": model.motion_transform,
                "motion_visual_max_abs_value": MOTION_VISUAL_MAX_ABS_VALUE,
                "patch_grid_shape": [grid_height, grid_width],
                "patch_size": [patch_height, patch_width],
                "num_patch_tokens": int(model.num_patches),
                "grid_image_path": str(grid_path),
                "motion_grid_image_path": str(motion_grid_path),
                "source_frames_image_path": saved_source_frames_path,
                "overlay_image_path": str(overlay_path),
                "motion_image_path": str(motion_path),
                "reconstruction_image_path": str(reconstruction_path),
                "assignment_path": str(assignment_path),
                "patch_embedding_tsne": patch_tsne_summary,
                "codes": overlay_metadata,
            }
            save_json(sample_metadata_path, summary)
            summary["metadata_path"] = str(sample_metadata_path)
            sample_summaries.append(summary)

    return sample_summaries


def main() -> None:
    args = parse_args()
    set_eval_determinism(args.seed)

    checkpoint_path = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint is not None
        else discover_latest_checkpoint(Path(args.checkpoint_root).expanduser())
    )
    device = train_main.choose_device(args.device)
    output_dir = build_output_dir(checkpoint_path, args.output_dir)

    checkpoint, cfg = load_checkpoint(checkpoint_path, device)
    model = build_frozen_model(cfg, checkpoint, device)
    dataset = train_main.build_dataset(
        cfg.data,
        motion_input_type=train_main.get_motion_input_type(cfg.model),
        motion_transform=train_main.get_motion_transform(cfg.model),
        use_reference_conditioning=train_main.get_use_reference_conditioning(cfg.model),
    )
    sample_indices = parse_sample_indices(args.sample_indices, len(dataset), args.num_samples)

    tsne_summary = save_codebook_tsne(
        codebook=model.quantizer.embedding,
        output_dir=output_dir,
        seed=args.seed,
        perplexity=args.tsne_perplexity,
    )
    sample_summaries = evaluate_samples(
        model=model,
        dataset=dataset,
        sample_indices=sample_indices,
        device=device,
        output_dir=output_dir,
        alpha=args.overlay_alpha,
        seed=args.seed,
    )

    evaluation_summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "device": str(device),
        "seed": int(args.seed),
        "dataset_size": int(len(dataset)),
        "motion_signal_formula": train_main.describe_motion_signal(model.motion_input_type),
        "motion_transform": model.motion_transform,
        "selected_sample_indices": [int(index) for index in sample_indices],
        "tsne": tsne_summary,
        "samples": sample_summaries,
        "resolved_train_config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(output_dir / "evaluation_summary.json", evaluation_summary)

    print(f"checkpoint={checkpoint_path}")
    print(f"output_dir={output_dir}")
    print(f"dataset_size={len(dataset)}")
    print(f"selected_sample_indices={sample_indices}")
    print(f"saved_tsne={output_dir / 'codebook_tsne.png'}")
    print(f"saved_summary={output_dir / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
