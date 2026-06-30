from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import model as train_main


DEFAULT_RUNS_ROOT = "/users/hnam16/scratch/otf_vqvae_runs"
DEFAULT_OUTPUT_ROOT = "/users/hnam16/scratch/otf_vqvae_runs/quantitative_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run quantitative decoder-side factor probing for OTF-VQ-VAE experiments. "
            "By default, each immediate child of --runs-root with checkpoints is evaluated "
            "using its latest checkpoint and the first 500 training trajectories."
        )
    )
    parser.add_argument("--runs-root", "--checkpoint-root", type=str, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Exact output directory. If omitted, --output-root/<run-id> is used.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Evaluate one or more explicit checkpoint paths instead of discovering experiments under --runs-root.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Experiment directory name to include. May be passed multiple times.",
    )
    parser.add_argument("--max-trajectories", "--max-sequences", type=int, default=500)
    parser.add_argument(
        "--samples-per-trajectory",
        "--samples-per-sequence",
        dest="samples_per_trajectory",
        type=int,
        default=1,
        help="Evaluate this many evenly spaced transitions per selected trajectory. Set <=0 to evaluate all transitions.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-intervention-energy", type=float, default=1.0e-8)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional output run id. Defaults to timestamp plus Slurm job id when available.",
    )
    return parser.parse_args()


def set_eval_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_checkpoint_step(path: Path) -> int:
    if path.stem.endswith("_final"):
        return 10**18
    match = re.search(r"_step(\d+)$", path.stem)
    return int(match.group(1)) if match is not None else -1


def checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    return (parse_checkpoint_step(path), path.stat().st_mtime, str(path))


def latest_checkpoint_for_run(run_dir: Path) -> Optional[Path]:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("otf_vqvae_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=checkpoint_sort_key)


def discover_experiments(runs_root: Path, selected_names: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    selected = set(selected_names) if selected_names else None
    experiments: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if selected is not None and run_dir.name not in selected:
            continue
        checkpoint_path = latest_checkpoint_for_run(run_dir)
        if checkpoint_path is None:
            continue
        experiments.append(
            {
                "experiment": run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint_path": str(checkpoint_path),
            }
        )

    if selected is not None:
        found = {item["experiment"] for item in experiments}
        missing = sorted(selected.difference(found))
        if missing:
            raise FileNotFoundError(f"No checkpoints found for selected experiments: {missing}")
    if not experiments:
        raise FileNotFoundError(f"No experiment checkpoints found under {runs_root}")
    return experiments


def experiment_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    return {
        "experiment": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
    }


def make_output_dir(output_root: Path, run_id: Optional[str]) -> Path:
    if run_id is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get("SLURM_JOB_ID")
        run_id = f"{stamp}_job{job_id}" if job_id else stamp
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def mean_and_stderr(values: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.mean().item()), float((tensor.std(unbiased=True) / math.sqrt(len(values))).item())


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def load_checkpoint_and_config(checkpoint_path: Path) -> tuple[Dict[str, Any], DictConfig]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a saved config: {checkpoint_path}")
    cfg = OmegaConf.create(checkpoint["config"])
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected checkpoint config to produce DictConfig, got {type(cfg)}")
    train_main.apply_checkpoint_model_settings(cfg, checkpoint)
    return checkpoint, cfg


def build_model_from_checkpoint(
    cfg: DictConfig,
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> train_main.OTFVQVAE:
    train_main.apply_model_config_defaults(cfg.model)
    model = train_main.OTFVQVAE(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def update_data_limit(cfg: DictConfig, max_trajectories: Optional[int]) -> None:
    if max_trajectories is not None:
        cfg.data.max_sequences = int(max_trajectories)


def count_sequences(dataset: Any) -> Optional[int]:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        frames = getattr(dataset, "frames", None)
        return int(frames.shape[0]) if frames is not None else None
    return len({item[0] for item in indices})


def sorted_sequence_keys(keys: Iterable[Any]) -> List[Any]:
    return sorted(keys, key=lambda value: int(value) if str(value).isdigit() else str(value))


def choose_eval_indices(
    dataset: Any,
    samples_per_trajectory: int,
    max_samples: Optional[int],
) -> List[int]:
    dataset_size = len(dataset)
    if dataset_size == 0:
        return []
    if samples_per_trajectory <= 0 or not hasattr(dataset, "indices"):
        selected = list(range(dataset_size))
    else:
        by_sequence: Dict[Any, List[int]] = {}
        for dataset_index, item in enumerate(dataset.indices):
            sequence_key = item[0] if isinstance(item, tuple) and item else 0
            by_sequence.setdefault(sequence_key, []).append(dataset_index)
        selected = []
        for sequence_key in sorted_sequence_keys(by_sequence.keys()):
            sequence_indices = by_sequence[sequence_key]
            count = min(int(samples_per_trajectory), len(sequence_indices))
            if count == len(sequence_indices):
                selected.extend(sequence_indices)
            else:
                positions = torch.linspace(0, len(sequence_indices) - 1, steps=count).round().long().tolist()
                selected.extend(sequence_indices[int(position)] for position in positions)
    if max_samples is not None:
        selected = selected[: int(max_samples)]
    return selected


def make_loader(
    cfg: DictConfig,
    model: train_main.OTFVQVAE,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    samples_per_trajectory: int,
    max_samples: Optional[int],
) -> tuple[Any, DataLoader, List[int]]:
    dataset = train_main.build_dataset(
        cfg.data,
        motion_input_type=model.motion_input_type,
        motion_transform=model.motion_transform,
        use_reference_conditioning=model.use_reference_conditioning,
    )
    selected_indices = choose_eval_indices(dataset, samples_per_trajectory, max_samples)
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return dataset, loader, selected_indices


def batch_index_batches(indices: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield [int(index) for index in indices[start : start + batch_size]]


def make_reference_features(
    model: train_main.OTFVQVAE,
    batch: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    if not model.use_reference_conditioning:
        return None
    reference_frame = batch.get("reference_frame")
    if reference_frame is None:
        raise ValueError("Batch is missing reference_frame while reference conditioning is enabled")
    return model.reference_encoder(reference_frame)


def decode_from_assignments(
    model: train_main.OTFVQVAE,
    code_embeddings: torch.Tensor,
    weights: torch.Tensor,
    occupancy_maps: torch.Tensor,
    reference_features: Optional[torch.Tensor],
) -> torch.Tensor:
    return model.decoder.decode_factors(code_embeddings, weights, occupancy_maps, reference_features)


def evaluate_probe_batch(
    model: train_main.OTFVQVAE,
    batch: Dict[str, torch.Tensor],
    dataset_indices: Sequence[int],
    *,
    eps: float,
    min_intervention_energy: float,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], List[Dict[str, Any]], List[Dict[str, Any]]]:
    motion = train_main.make_motion_signal(batch, model.motion_input_type, model.motion_transform)
    patch_embeddings = model.encoder(motion)
    vq_output = model.quantizer(patch_embeddings)
    weights, occupancy_maps, active_mask, code_embeddings, _ = model.summarize_assignments(
        vq_output["indices"],
        vq_output["quantized_st"],
    )
    reference_features = make_reference_features(model, batch)
    reconstruction = decode_from_assignments(model, code_embeddings, weights, occupancy_maps, reference_features)

    batch_size, _, height, width = reconstruction.shape
    log_hw = math.log(float(height * width))
    reconstruction_mse = F.mse_loss(reconstruction, motion, reduction="none").flatten(1).mean(dim=1)
    sample_records: List[Dict[str, Any]] = []
    factor_records: List[Dict[str, Any]] = []

    for batch_idx in range(batch_size):
        active_factor_ids = torch.nonzero(active_mask[batch_idx], as_tuple=False).flatten().tolist()
        valid_factor_ids: List[int] = []
        valid_energies: List[float] = []
        valid_aia: List[float] = []
        valid_locality: List[float] = []
        valid_delta_vectors: List[torch.Tensor] = []

        for factor_id in active_factor_ids:
            ablated_occupancy = occupancy_maps[batch_idx : batch_idx + 1].clone()
            ablated_weights = weights[batch_idx : batch_idx + 1].clone()
            ablated_code_embeddings = code_embeddings[batch_idx : batch_idx + 1].clone()
            ablated_occupancy[:, factor_id] = 0.0
            ablated_weights[:, factor_id] = 0.0
            ablated_code_embeddings[:, factor_id] = 0.0
            ref_features = None if reference_features is None else reference_features[batch_idx : batch_idx + 1]
            ablated_reconstruction = decode_from_assignments(
                model,
                ablated_code_embeddings,
                ablated_weights,
                ablated_occupancy,
                ref_features,
            )
            delta = (reconstruction[batch_idx : batch_idx + 1] - ablated_reconstruction).abs().sum(dim=1)[0]
            energy = float(delta.sum().item())
            if energy <= min_intervention_energy:
                continue

            occupancy_up = F.interpolate(
                occupancy_maps[batch_idx : batch_idx + 1, factor_id : factor_id + 1],
                size=(height, width),
                mode="nearest",
            )[0, 0]
            aia = float(((occupancy_up * delta).sum() / (delta.sum() + eps)).item())
            probability = delta / (delta.sum() + eps)
            entropy = float((-(probability * torch.log(probability + eps)).sum() / log_hw).item())
            locality = 1.0 - entropy
            delta_vector = delta.flatten()
            delta_vector = delta_vector / (delta_vector.norm(p=2) + eps)

            valid_factor_ids.append(int(factor_id))
            valid_energies.append(energy)
            valid_aia.append(aia)
            valid_locality.append(locality)
            valid_delta_vectors.append(delta_vector.detach().cpu())
            factor_records.append(
                {
                    "dataset_index": int(dataset_indices[batch_idx]),
                    "factor_id": int(factor_id),
                    "code_activation": float(weights[batch_idx, factor_id].item()),
                    "intervention_energy": energy,
                    "assignment_intervention_alignment": aia,
                    "locality": locality,
                }
            )

        non_redundancy = None
        if len(valid_delta_vectors) >= 2:
            delta_matrix = torch.stack(valid_delta_vectors, dim=0)
            similarity = delta_matrix @ delta_matrix.t()
            n = int(similarity.shape[0])
            mean_off_diag = float(((similarity.sum() - n) / (n * (n - 1))).item())
            non_redundancy = 1.0 - mean_off_diag

        sample_records.append(
            {
                "dataset_index": int(dataset_indices[batch_idx]),
                "active_factor_ids": [int(factor_id) for factor_id in active_factor_ids],
                "valid_factor_ids": valid_factor_ids,
                "code_activations": [float(weights[batch_idx, factor_id].item()) for factor_id in active_factor_ids],
                "intervention_energy": valid_energies,
                "assignment_intervention_alignment_mean": mean_or_none(valid_aia),
                "locality_mean": mean_or_none(valid_locality),
                "non_redundancy": non_redundancy,
                "reconstruction_mse": float(reconstruction_mse[batch_idx].item()),
            }
        )

    probe_tensors = {
        "indices": vq_output["indices"],
        "code_loss": vq_output["code_loss"],
        "commit_loss": vq_output["commit_loss"],
        "orth_loss": model.quantizer.orthogonality_loss(vq_output["indices"], model.orth_active_only),
    }
    return motion, reconstruction, probe_tensors, sample_records, factor_records


def top_code_usage(code_counts: torch.Tensor, limit: int = 10) -> List[Dict[str, Any]]:
    total = int(code_counts.sum().item())
    if total == 0:
        return []
    values, indices = torch.topk(code_counts, k=min(limit, int(code_counts.numel())))
    return [
        {
            "code_id": int(code_id.item()),
            "count": int(count.item()),
            "fraction": float(count.item() / total),
        }
        for count, code_id in zip(values.cpu(), indices.cpu())
        if int(count.item()) > 0
    ]


def write_code_usage_csv(path: Path, code_counts: torch.Tensor) -> None:
    total = int(code_counts.sum().item())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code_id", "count", "fraction"])
        for code_id, count in enumerate(code_counts.cpu().tolist()):
            fraction = float(count / total) if total else 0.0
            writer.writerow([code_id, int(count), fraction])


def write_factor_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_index",
        "factor_id",
        "code_activation",
        "intervention_energy",
        "assignment_intervention_alignment",
        "locality",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def finalize_metrics(
    acc: Dict[str, Any],
    model: train_main.OTFVQVAE,
    sample_records: Sequence[Dict[str, Any]],
    factor_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    num_elements = int(acc["num_elements"])
    num_samples = int(acc["num_samples"])
    if num_elements <= 0 or num_samples <= 0:
        raise RuntimeError("No samples were evaluated")

    sse = float(acc["squared_error_sum"])
    sae = float(acc["absolute_error_sum"])
    target_squares = float(acc["target_squared_sum"])
    target_sum = float(acc["target_sum"])
    target_sst = target_squares - (target_sum * target_sum / float(num_elements))

    mse = sse / float(num_elements)
    normalized_mse = sse / target_squares if target_squares > 0.0 else math.inf
    snr_db = 10.0 * math.log10(target_squares / sse) if target_squares > 0.0 and sse > 0.0 else math.inf
    r2 = 1.0 - (sse / target_sst) if target_sst > 0.0 else math.nan

    code_counts = acc["code_counts"]
    code_total = float(code_counts.sum().item())
    if code_total > 0.0:
        probabilities = code_counts.float() / code_total
        nonzero_probabilities = probabilities[probabilities > 0.0]
        entropy_nats = float(-(nonzero_probabilities * torch.log(nonzero_probabilities)).sum().item())
        entropy_bits = entropy_nats / math.log(2.0)
        perplexity = math.exp(entropy_nats)
    else:
        entropy_bits = 0.0
        perplexity = 0.0

    aia_values = [float(record["assignment_intervention_alignment"]) for record in factor_records]
    locality_values = [float(record["locality"]) for record in factor_records]
    energy_values = [float(record["intervention_energy"]) for record in factor_records]
    nr_values = [float(record["non_redundancy"]) for record in sample_records if record["non_redundancy"] is not None]
    aia_mean, aia_stderr = mean_and_stderr(aia_values)
    locality_mean, locality_stderr = mean_and_stderr(locality_values)
    nr_mean, nr_stderr = mean_and_stderr(nr_values)
    energy_mean, energy_stderr = mean_and_stderr(energy_values)

    return {
        "num_samples": num_samples,
        "num_batches": int(acc["num_batches"]),
        "num_elements": num_elements,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": sae / float(num_elements),
        "normalized_mse": normalized_mse if math.isfinite(normalized_mse) else None,
        "snr_db": snr_db if math.isfinite(snr_db) else None,
        "r2": r2 if math.isfinite(r2) else None,
        "target_mean": target_sum / float(num_elements),
        "target_mean_square": target_squares / float(num_elements),
        "code_loss_mean": float(acc["code_loss_weighted_sum"] / num_samples),
        "commit_loss_mean": float(acc["commit_loss_weighted_sum"] / num_samples),
        "orth_loss_mean": float(acc["orth_loss_weighted_sum"] / num_samples),
        "active_codes_dataset": int((code_counts > 0).sum().item()),
        "codebook_size": int(model.num_codes),
        "code_usage_entropy_bits": entropy_bits,
        "code_usage_perplexity": perplexity,
        "mean_active_codes_per_sample": float(acc["active_codes_per_sample_sum"] / num_samples),
        "top_codes": top_code_usage(code_counts),
        "num_factor_instances": len(factor_records),
        "num_non_redundancy_samples": len(nr_values),
        "assignment_intervention_alignment_mean": aia_mean,
        "assignment_intervention_alignment_stderr": aia_stderr,
        "locality_mean": locality_mean,
        "locality_stderr": locality_stderr,
        "non_redundancy_mean": nr_mean,
        "non_redundancy_stderr": nr_stderr,
        "intervention_energy_mean": energy_mean,
        "intervention_energy_stderr": energy_stderr,
    }


def evaluate_checkpoint(
    experiment: str,
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    start = time.perf_counter()
    checkpoint, cfg = load_checkpoint_and_config(checkpoint_path)
    update_data_limit(cfg, args.max_trajectories)
    model = build_model_from_checkpoint(cfg, checkpoint, device)
    dataset, loader, selected_indices = make_loader(
        cfg=cfg,
        model=model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        samples_per_trajectory=args.samples_per_trajectory,
        max_samples=args.max_samples,
    )

    experiment_dir = output_dir / "experiments" / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    code_counts = torch.zeros(int(model.num_codes), dtype=torch.long, device=device)
    acc: Dict[str, Any] = {
        "num_samples": 0,
        "num_batches": 0,
        "num_elements": 0,
        "squared_error_sum": 0.0,
        "absolute_error_sum": 0.0,
        "target_squared_sum": 0.0,
        "target_sum": 0.0,
        "code_loss_weighted_sum": 0.0,
        "commit_loss_weighted_sum": 0.0,
        "orth_loss_weighted_sum": 0.0,
        "active_codes_per_sample_sum": 0.0,
        "code_counts": code_counts,
    }
    sample_records: List[Dict[str, Any]] = []
    factor_records: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for dataset_index_batch, batch in zip(batch_index_batches(selected_indices, args.batch_size), loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            motion, reconstruction, probe_tensors, batch_sample_records, batch_factor_records = evaluate_probe_batch(
                model,
                batch,
                dataset_index_batch,
                eps=float(args.eps),
                min_intervention_energy=float(args.min_intervention_energy),
            )
            residual = reconstruction - motion
            batch_size = int(motion.shape[0])

            acc["num_samples"] += batch_size
            acc["num_batches"] += 1
            acc["num_elements"] += int(motion.numel())
            acc["squared_error_sum"] += float(residual.square().sum().item())
            acc["absolute_error_sum"] += float(residual.abs().sum().item())
            acc["target_squared_sum"] += float(motion.square().sum().item())
            acc["target_sum"] += float(motion.sum().item())
            acc["code_loss_weighted_sum"] += float(probe_tensors["code_loss"].item()) * batch_size
            acc["commit_loss_weighted_sum"] += float(probe_tensors["commit_loss"].item()) * batch_size
            acc["orth_loss_weighted_sum"] += float(probe_tensors["orth_loss"].item()) * batch_size

            indices = probe_tensors["indices"]
            code_counts += torch.bincount(indices.reshape(-1), minlength=int(model.num_codes))
            assignment_counts = F.one_hot(indices, num_classes=int(model.num_codes)).sum(dim=1)
            active_per_sample = (assignment_counts > 0).sum(dim=1)
            acc["active_codes_per_sample_sum"] += float(active_per_sample.sum().item())
            sample_records.extend(batch_sample_records)
            factor_records.extend(batch_factor_records)

    metrics = finalize_metrics(acc, model, sample_records, factor_records)
    elapsed_seconds = time.perf_counter() - start
    metrics.update(
        {
            "status": "ok",
            "experiment": experiment,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "run_dir": str(checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent),
            "elapsed_seconds": elapsed_seconds,
            "dataset_size": int(len(dataset)),
            "selected_samples": int(len(selected_indices)),
            "trajectories_used": count_sequences(dataset),
            "max_trajectories_requested": args.max_trajectories,
            "samples_per_trajectory": args.samples_per_trajectory,
            "max_samples_requested": args.max_samples,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(device),
            "data_type": str(cfg.data.type),
            "data_path": None if cfg.data.path is None else str(cfg.data.path),
            "motion_input_type": model.motion_input_type,
            "motion_transform": model.motion_transform,
            "use_reference_conditioning": bool(model.use_reference_conditioning),
            "motion_signal_formula": train_main.describe_motion_signal(model.motion_input_type),
            "resolved_train_config": OmegaConf.to_container(cfg, resolve=True),
        }
    )

    save_json(experiment_dir / "selected_dataset_indices.json", [int(index) for index in selected_indices])
    save_json(experiment_dir / "metrics.json", metrics)
    write_code_usage_csv(experiment_dir / "code_usage.csv", code_counts)
    write_factor_csv(experiment_dir / "factor_metrics.csv", factor_records)
    with (experiment_dir / "sample_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for record in sample_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return metrics


SUMMARY_FIELDS = [
    "status",
    "experiment",
    "checkpoint_global_step",
    "motion_input_type",
    "motion_transform",
    "data_type",
    "trajectories_used",
    "dataset_size",
    "selected_samples",
    "num_samples",
    "num_factor_instances",
    "assignment_intervention_alignment_mean",
    "assignment_intervention_alignment_stderr",
    "locality_mean",
    "locality_stderr",
    "non_redundancy_mean",
    "non_redundancy_stderr",
    "mse",
    "rmse",
    "mae",
    "normalized_mse",
    "snr_db",
    "r2",
    "active_codes_dataset",
    "code_usage_perplexity",
    "mean_active_codes_per_sample",
    "elapsed_seconds",
    "checkpoint_path",
    "error",
]


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, rows: Sequence[Dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]

    lines = [
        "# OTF-VQ-VAE Quantitative Factor Probe Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output directory: `{output_dir}`",
        f"- Runs root: `{args.runs_root}`",
        f"- First training trajectories requested: `{args.max_trajectories}`",
        f"- Samples per trajectory: `{args.samples_per_trajectory}`",
        f"- Max samples: `{args.max_samples}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Device: `{args.device}`",
        f"- Experiments evaluated successfully: `{len(ok_rows)}`",
        f"- Experiments failed: `{len(failed_rows)}`",
        "",
        "## Summary",
        "",
        "| Experiment | Step | Motion | Transform | Traj | Samples | Factors | AIA | Locality | Non-red. | MSE | Code PPL | Time s |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ok_rows:
        values = [
            f"`{row['experiment']}`",
            str(row.get("checkpoint_global_step", "n/a")),
            str(row.get("motion_input_type", "n/a")),
            str(row.get("motion_transform", "n/a")),
            str(row.get("trajectories_used", "n/a")),
            str(row.get("num_samples", "n/a")),
            str(row.get("num_factor_instances", "n/a")),
            format_float(row.get("assignment_intervention_alignment_mean")),
            format_float(row.get("locality_mean")),
            format_float(row.get("non_redundancy_mean")),
            format_float(row.get("mse")),
            format_float(row.get("code_usage_perplexity")),
            format_float(row.get("elapsed_seconds"), digits=4),
        ]
        lines.append("| " + " | ".join(values) + " |")

    if failed_rows:
        lines.extend(["", "## Failures", ""])
        for row in failed_rows:
            lines.append(f"- `{row.get('experiment', 'unknown')}`: {row.get('error', 'unknown error')}")

    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- AIA is assignment-intervention alignment: the fraction of a factor's ablation energy that lies inside its upsampled assignment occupancy map.",
            "- Locality is one minus normalized entropy of the ablation difference map.",
            "- Non-red. is non-redundancy: one minus average pairwise cosine similarity between valid factor intervention maps in the same sample.",
            "- MSE compares reconstructed motion tensors against the selected motion target, not future RGB frames.",
            "- Per-experiment JSON metrics, per-sample JSONL, per-factor CSV, selected dataset indices, and code usage CSV are under `experiments/<experiment>/`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_eval_determinism(int(args.seed))

    runs_root = Path(args.runs_root).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else make_output_dir(Path(args.output_root).expanduser(), args.run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = train_main.choose_device(str(args.device))

    if args.checkpoint is not None:
        experiments = [experiment_from_checkpoint(Path(path).expanduser()) for path in args.checkpoint]
    else:
        experiments = discover_experiments(runs_root, args.experiment)

    manifest = {
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "max_trajectories": args.max_trajectories,
        "samples_per_trajectory": args.samples_per_trajectory,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "experiments": experiments,
    }
    save_json(output_dir / "manifest.json", manifest)

    print(f"output_dir={output_dir}", flush=True)
    print(f"num_experiments={len(experiments)}", flush=True)

    rows: List[Dict[str, Any]] = []
    jsonl_path = output_dir / "experiment_results.jsonl"
    total_start = time.perf_counter()
    for index, item in enumerate(experiments, start=1):
        experiment = str(item["experiment"])
        checkpoint_path = Path(str(item["checkpoint_path"])).expanduser()
        print(f"[{index}/{len(experiments)}] experiment={experiment} checkpoint={checkpoint_path}", flush=True)
        try:
            row = evaluate_checkpoint(
                experiment=experiment,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                args=args,
                device=device,
            )
            print(
                " ".join(
                    [
                        f"done experiment={experiment}",
                        f"aia={format_float(row['assignment_intervention_alignment_mean'])}",
                        f"locality={format_float(row['locality_mean'])}",
                        f"non_redundancy={format_float(row['non_redundancy_mean'])}",
                        f"samples={row['num_samples']}",
                        f"factors={row['num_factor_instances']}",
                        f"elapsed_seconds={row['elapsed_seconds']:.2f}",
                    ]
                ),
                flush=True,
            )
        except Exception as exc:
            row = {
                "status": "failed",
                "experiment": experiment,
                "checkpoint_path": str(checkpoint_path),
                "error": repr(exc),
            }
            print(f"failed experiment={experiment} error={repr(exc)}", flush=True)
            if args.fail_fast:
                rows.append(row)
                append_jsonl(jsonl_path, row)
                raise

        rows.append(row)
        append_jsonl(jsonl_path, row)
        write_summary_csv(output_dir / "summary.csv", rows)
        save_json(
            output_dir / "summary.json",
            {
                "total_elapsed_seconds_so_far": time.perf_counter() - total_start,
                "results": rows,
            },
        )
        write_markdown_report(output_dir / "result_report.md", rows, output_dir, args)

    total_elapsed_seconds = time.perf_counter() - total_start
    save_json(
        output_dir / "summary.json",
        {
            "total_elapsed_seconds": total_elapsed_seconds,
            "results": rows,
        },
    )
    write_markdown_report(output_dir / "result_report.md", rows, output_dir, args)
    print(f"total_elapsed_seconds={total_elapsed_seconds:.2f}", flush=True)
    print(f"saved_summary={output_dir / 'summary.csv'}", flush=True)
    print(f"saved_report={output_dir / 'result_report.md'}", flush=True)


if __name__ == "__main__":
    main()
