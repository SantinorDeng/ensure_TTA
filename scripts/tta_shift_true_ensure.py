#!/usr/bin/env python3
"""Run TRUE-ENSURE test-time adaptation on shift-manifest target samples."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_ensure.datasets import StaticShiftSourceENSUREDataset
from cardiac_ensure.models import TemporalNormUnet
from cardiac_ensure.scripts.eval_metrics import to_magnitude
from cardiac_ensure.scripts.train_common import (
    configure_torch,
    ensure_dir,
    maybe_center_crop_batch,
    move_to_device,
    resolve_device,
    save_json,
    set_random_seed,
)
from cardiac_ensure.tta import ENSURETTAResult, run_true_ensure_tta


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "cardiac_ensure"
    / "outputs"
    / "shifts"
    / "main"
    / "modality_shift"
    / "true_ensure_source_r4_w1_auto_unroll_seed7"
    / "best.pt"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "cardiac_ensure" / "manifests" / "shifts" / "main" / "modality_shift.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "cardiac_ensure"
    / "outputs"
    / "tta"
    / "shifts"
    / "main"
    / "modality_shift"
    / "true_ensure_source_r4_w1_auto_unroll_seed7"
)

METRIC_NAMES = ("nmse", "nrmse", "psnr", "frame_diff_nmse", "ssim")
METRICS_FIELDS = [
    "status",
    "method",
    "training_objective",
    "checkpoint",
    "checkpoint_sha256",
    "manifest_sha256",
    "shift_name",
    "experiment_family",
    "experiment_tier",
    "split_role",
    "rank_in_split",
    "sample_id",
    "file",
    "slice",
    "dataset",
    "split",
    "acquisition",
    "source_acquisition",
    "target_acquisition",
    "patient_id",
    "split_group_id",
    "is_in_domain_control",
    "target_key",
    "test_noise_snr_db",
    "test_noise_seed",
    "mask_seed",
    "run_tta",
    "update_mode",
    "num_trainable_params",
    "before_tta_nmse",
    "before_tta_nrmse",
    "before_tta_psnr",
    "before_tta_frame_diff_nmse",
    "before_tta_ssim",
    "after_tta_nmse",
    "after_tta_nrmse",
    "after_tta_psnr",
    "after_tta_frame_diff_nmse",
    "after_tta_ssim",
    "delta_nmse",
    "delta_nrmse",
    "delta_psnr",
    "delta_frame_diff_nmse",
    "delta_ssim",
    "num_tta_steps",
    "early_stop_step",
    "best_step",
    "self_val_best",
    "runtime_sec",
    "adapt_runtime_sec",
    "norm_scale",
    "input_noise_snr_db",
    "input_noise_sigma2",
    "noise_sigma2_total",
    "recon_npz",
    "curve_json",
    "error",
]
SUMMARY_FIELDS = [
    "method",
    "training_objective",
    "shift_name",
    "experiment_family",
    "source_acquisition",
    "target_acquisition",
    "is_in_domain_control",
    "num_samples",
    "before_tta_nmse_mean",
    "before_tta_nmse_std",
    "after_tta_nmse_mean",
    "after_tta_nmse_std",
    "delta_nmse_mean",
    "before_tta_psnr_mean",
    "before_tta_psnr_std",
    "after_tta_psnr_mean",
    "after_tta_psnr_std",
    "delta_psnr_mean",
    "before_tta_ssim_mean",
    "before_tta_ssim_std",
    "after_tta_ssim_mean",
    "after_tta_ssim_std",
    "delta_ssim_mean",
    "runtime_sec_per_slice",
    "adapt_runtime_sec_per_slice",
    "tta_trigger_rate",
    "negative_adaptation_rate",
]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-role", default="target_test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-noise-snr-db", type=float, default=None)
    parser.add_argument("--test-noise-seed", type=int, default=9007)
    parser.add_argument(
        "--target-shift-name",
        dest="target_shift_names",
        action="append",
        default=[],
        help="Only evaluate matching target shift_name rows; repeat to select multiple cells.",
    )
    parser.add_argument(
        "--training-objective",
        default="auto",
        help="Result label; 'auto' reads checkpoint config.",
    )
    parser.add_argument("--tta-steps", type=int, default=250)
    parser.add_argument("--tta-lr", type=float, default=1.0e-5)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--self-val-fraction", type=float, default=0.05)
    parser.add_argument("--early-stop-window", type=int, default=20)
    parser.add_argument("--update-mode", default="all_params", choices=("all_params", "adapter"))
    parser.add_argument(
        "--tta-loss",
        default="l1",
        choices=("ensure", "ensure_data", "self_supervised", "l1"),
        help="Loss used for TTA updates; l1 is an alias for measured-kspace normalized complex L1.",
    )
    parser.add_argument("--run-tta", dest="run_tta", action="store_true", default=True)
    parser.add_argument("--no-run-tta", dest="run_tta", action="store_false")
    parser.add_argument("--include-ssim", action="store_true")
    parser.add_argument("--save-recons", action="store_true")
    parser.add_argument("--save-curves", dest="save_curves", action="store_true", default=True)
    parser.add_argument("--no-save-curves", dest="save_curves", action="store_false")
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument(
        "--acceleration",
        type=float,
        default=None,
        help="Override the checkpoint acceleration for target-test mask generation.",
    )
    parser.add_argument(
        "--sigma-mask",
        type=float,
        default=None,
        help="Override the checkpoint sigma_mask for target-test mask generation.",
    )
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--crop-width", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _path_or_none(value: Any) -> Path | None:
    if value in (None, "", "None"):
        return None
    return Path(str(value))


def _config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[TemporalNormUnet, dict[str, Any]]:
    checkpoint = _torch_load_checkpoint(path)
    config = dict(checkpoint.get("config", {}))
    model = TemporalNormUnet(
        num_frames=int(_config_value(config, "window_size", 1)),
        chans=int(_config_value(config, "chans", 64)),
        num_pools=int(_config_value(config, "num_pools", 4)),
        drop_prob=float(_config_value(config, "drop_prob", 0.0)),
        residual=not bool(_config_value(config, "no_residual", False)),
        output_mode="all_frames",
        num_unrolls=int(_config_value(config, "num_unrolls", 12)),
        denoiser_sharing=str(_config_value(config, "denoiser_sharing", "shared")),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config


def make_dataset(args: argparse.Namespace, config: Mapping[str, Any]) -> StaticShiftSourceENSUREDataset:
    preproc_root = args.preproc_root or _path_or_none(config.get("preproc_root"))
    density_root = args.density_root or _path_or_none(config.get("density_root"))
    return StaticShiftSourceENSUREDataset(
        manifest_csv=args.manifest_csv,
        subset="all",
        preproc_root=preproc_root,
        density_root=density_root,
        source_role=args.split_role,
        val_fraction=float(_config_value(config, "val_fraction", 0.2)),
        split_seed=int(_config_value(config, "split_seed", args.seed)),
        allow_slice_val_fallback=bool(_config_value(config, "allow_slice_val_fallback", True)),
        acceleration=float(args.acceleration if args.acceleration is not None else _config_value(config, "acceleration", 4.0)),
        sigma_mask=float(args.sigma_mask if args.sigma_mask is not None else _config_value(config, "sigma_mask", 0.18)),
        window_size=int(_config_value(config, "window_size", 1)),
        deterministic_masks=True,
        mask_seed=int(args.seed),
        return_target=True,
        # Training may require source-only sidecars. Target TTA must still support
        # unseen acquisitions by estimating maps/noise from their measured volume.
        require_preproc=False,
        test_noise_snr_db=args.test_noise_snr_db,
        test_noise_seed=int(args.test_noise_seed),
        shift_names=args.target_shift_names,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_training_objective(requested: str, config: Mapping[str, Any]) -> str:
    if requested != "auto":
        return str(requested)
    configured = str(config.get("training_objective", "")).strip()
    if configured:
        return configured
    if "cg_l2lam" in config or "divergence_mode" in config:
        return "true_ensure"
    return "unknown_training_objective"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_token(value: object) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar(value[0])
    return value


def _batch_meta(batch: Mapping[str, Any]) -> dict[str, Any]:
    meta = batch.get("meta", {})
    if not isinstance(meta, Mapping):
        return {}
    return {str(key): _scalar(value) for key, value in meta.items()}


def metric_row_base(
    *,
    status: str,
    method: str,
    training_objective: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
    manifest_row: Mapping[str, str],
    run_tta: bool,
    update_mode: str,
    test_noise_snr_db: float | None,
    test_noise_seed: int,
    mask_seed: int,
    error: str = "",
) -> dict[str, object]:
    return {
        "status": status,
        "method": method,
        "training_objective": training_objective,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "shift_name": manifest_row.get("shift_name", ""),
        "experiment_family": manifest_row.get("experiment_family", ""),
        "experiment_tier": manifest_row.get("experiment_tier", ""),
        "split_role": manifest_row.get("split_role", ""),
        "rank_in_split": manifest_row.get("rank_in_split", ""),
        "sample_id": manifest_row.get("sample_id", ""),
        "file": manifest_row.get("path", ""),
        "slice": manifest_row.get("slice_idx", ""),
        "dataset": manifest_row.get("dataset", ""),
        "split": manifest_row.get("split", ""),
        "acquisition": manifest_row.get("acquisition", ""),
        "source_acquisition": manifest_row.get("source_acquisition", ""),
        "target_acquisition": manifest_row.get("target_acquisition", ""),
        "patient_id": manifest_row.get("patient_id", ""),
        "split_group_id": manifest_row.get("split_group_id", ""),
        "is_in_domain_control": manifest_row.get("is_in_domain_control", ""),
        "target_key": manifest_row.get("target_key", ""),
        "test_noise_snr_db": "clean" if test_noise_snr_db is None else test_noise_snr_db,
        "test_noise_seed": int(test_noise_seed),
        "mask_seed": int(mask_seed),
        "run_tta": bool(run_tta),
        "update_mode": update_mode if run_tta else "",
        "num_trainable_params": 0,
        "before_tta_nmse": "",
        "before_tta_nrmse": "",
        "before_tta_psnr": "",
        "before_tta_frame_diff_nmse": "",
        "before_tta_ssim": "",
        "after_tta_nmse": "",
        "after_tta_nrmse": "",
        "after_tta_psnr": "",
        "after_tta_frame_diff_nmse": "",
        "after_tta_ssim": "",
        "delta_nmse": "",
        "delta_nrmse": "",
        "delta_psnr": "",
        "delta_frame_diff_nmse": "",
        "delta_ssim": "",
        "num_tta_steps": "",
        "early_stop_step": "",
        "best_step": "",
        "self_val_best": "",
        "runtime_sec": "",
        "adapt_runtime_sec": "",
        "norm_scale": "",
        "input_noise_snr_db": "",
        "input_noise_sigma2": "",
        "noise_sigma2_total": "",
        "recon_npz": "",
        "curve_json": "",
        "error": error,
    }


def populate_result_fields(row: dict[str, object], result: ENSURETTAResult, meta: Mapping[str, Any]) -> None:
    row["num_trainable_params"] = result.num_trainable_params
    row["num_tta_steps"] = result.num_tta_steps
    row["early_stop_step"] = result.early_stop_step if result.early_stop_step is not None else ""
    row["best_step"] = result.best_step if result.best_step is not None else ""
    row["self_val_best"] = result.self_val_best
    row["runtime_sec"] = result.runtime_sec
    row["adapt_runtime_sec"] = result.adapt_runtime_sec
    row["norm_scale"] = meta.get("norm_scale", "")
    row["input_noise_snr_db"] = meta.get("input_noise_snr_db", "")
    row["input_noise_sigma2"] = meta.get("input_noise_sigma2", "")
    row["noise_sigma2_total"] = meta.get("noise_sigma2_total", "")

    for metric in METRIC_NAMES:
        before = result.before_metrics.get(metric)
        after = result.after_metrics.get(metric)
        if before is not None:
            row[f"before_tta_{metric}"] = before
        if after is not None:
            row[f"after_tta_{metric}"] = after
        if before is not None and after is not None:
            row[f"delta_{metric}"] = after - before


def curve_tag(row: Mapping[str, object]) -> str:
    rank = str(row.get("rank_in_split", "0")).zfill(3)
    sample = _safe_token(row.get("sample_id", "sample"))
    return f"{row.get('method', 'method')}_{rank}_{sample}_slice{row.get('slice', '')}"


def save_curve(path: Path, result: ENSURETTAResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        path,
        {
            "train_losses": result.train_losses,
            "self_val_losses": result.self_val_losses,
            "step_logs": result.step_logs,
            "best_step": result.best_step,
            "early_stop_step": result.early_stop_step,
            "self_val_best": result.self_val_best,
            "adapt_runtime_sec": result.adapt_runtime_sec,
            "num_trainable_params": result.num_trainable_params,
        },
    )


def save_reconstruction_npz(
    path: Path,
    *,
    batch: Mapping[str, Any],
    result: ENSURETTAResult,
    row: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = batch.get("target_rss")
    target_np = target.detach().cpu().float().numpy() if torch.is_tensor(target) else np.asarray([], dtype=np.float32)
    zero_filled = to_magnitude(batch["zf"]).detach().cpu().float().numpy()
    mask = batch["mask"].detach().cpu().float().numpy()
    metadata = {
        key: row.get(key, "")
        for key in (
            "method",
            "sample_id",
            "shift_name",
            "split_role",
            "rank_in_split",
            "file",
            "slice",
            "before_tta_nmse",
            "after_tta_nmse",
            "num_tta_steps",
            "best_step",
            "runtime_sec",
            "adapt_runtime_sec",
            "input_noise_snr_db",
            "input_noise_sigma2",
            "noise_sigma2_total",
        )
    }
    np.savez_compressed(
        path,
        target=target_np.astype(np.float32, copy=False),
        zero_filled=zero_filled.astype(np.float32, copy=False),
        before_tta_rec=np.asarray(result.before_rec, dtype=np.float32),
        after_tta_rec=np.asarray(result.after_rec, dtype=np.float32),
        mask=mask.astype(np.float32, copy=False),
        metadata_json=np.array(json.dumps(metadata)),
    )


def finite(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in ok_rows:
        grouped.setdefault(
            (
                str(row.get("method", "")),
                str(row.get("training_objective", "")),
                str(row.get("shift_name", "")),
            ),
            [],
        ).append(row)

    summaries: list[dict[str, object]] = []
    for (method, training_objective, shift_name), group in sorted(grouped.items()):
        before_nmse = [value for value in (finite(row, "before_tta_nmse") for row in group) if value is not None]
        after_nmse = [value for value in (finite(row, "after_tta_nmse") for row in group) if value is not None]
        delta_nmse = [value for value in (finite(row, "delta_nmse") for row in group) if value is not None]
        before_psnr = [value for value in (finite(row, "before_tta_psnr") for row in group) if value is not None]
        after_psnr = [value for value in (finite(row, "after_tta_psnr") for row in group) if value is not None]
        delta_psnr = [value for value in (finite(row, "delta_psnr") for row in group) if value is not None]
        before_ssim = [value for value in (finite(row, "before_tta_ssim") for row in group) if value is not None]
        after_ssim = [value for value in (finite(row, "after_tta_ssim") for row in group) if value is not None]
        delta_ssim = [value for value in (finite(row, "delta_ssim") for row in group) if value is not None]
        runtime = [value for value in (finite(row, "runtime_sec") for row in group) if value is not None]
        adapt_runtime = [value for value in (finite(row, "adapt_runtime_sec") for row in group) if value is not None]
        trigger_flags = [
            bool(str(row.get("run_tta")).lower() == "true" and (finite(row, "num_tta_steps") or 0.0) > 0.0)
            for row in group
        ]
        neg_flags = [
            (finite(row, "delta_nmse") is not None and float(finite(row, "delta_nmse")) > 0.0)
            for row in group
        ]

        before_nmse_mean, before_nmse_std = mean_std(before_nmse)
        after_nmse_mean, after_nmse_std = mean_std(after_nmse)
        delta_nmse_mean, _ = mean_std(delta_nmse)
        before_psnr_mean, before_psnr_std = mean_std(before_psnr)
        after_psnr_mean, after_psnr_std = mean_std(after_psnr)
        delta_psnr_mean, _ = mean_std(delta_psnr)
        before_ssim_mean, before_ssim_std = mean_std(before_ssim)
        after_ssim_mean, after_ssim_std = mean_std(after_ssim)
        delta_ssim_mean, _ = mean_std(delta_ssim)
        runtime_mean, _ = mean_std(runtime)
        adapt_runtime_mean, _ = mean_std(adapt_runtime)
        summaries.append(
            {
                "method": method,
                "training_objective": training_objective,
                "shift_name": shift_name,
                "experiment_family": group[0].get("experiment_family", ""),
                "source_acquisition": group[0].get("source_acquisition", ""),
                "target_acquisition": group[0].get("target_acquisition", ""),
                "is_in_domain_control": group[0].get("is_in_domain_control", ""),
                "num_samples": len(group),
                "before_tta_nmse_mean": before_nmse_mean,
                "before_tta_nmse_std": before_nmse_std,
                "after_tta_nmse_mean": after_nmse_mean,
                "after_tta_nmse_std": after_nmse_std,
                "delta_nmse_mean": delta_nmse_mean,
                "before_tta_psnr_mean": before_psnr_mean,
                "before_tta_psnr_std": before_psnr_std,
                "after_tta_psnr_mean": after_psnr_mean,
                "after_tta_psnr_std": after_psnr_std,
                "delta_psnr_mean": delta_psnr_mean,
                "before_tta_ssim_mean": before_ssim_mean,
                "before_tta_ssim_std": before_ssim_std,
                "after_tta_ssim_mean": after_ssim_mean,
                "after_tta_ssim_std": after_ssim_std,
                "delta_ssim_mean": delta_ssim_mean,
                "runtime_sec_per_slice": runtime_mean,
                "adapt_runtime_sec_per_slice": adapt_runtime_mean,
                "tta_trigger_rate": float(np.mean(trigger_flags)) if trigger_flags else float("nan"),
                "negative_adaptation_rate": float(np.mean(neg_flags)) if neg_flags else float("nan"),
            }
        )
    return summaries


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    configure_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.csv"
    payload_path = output_dir / "summary.json"

    model, checkpoint_config = load_model_from_checkpoint(args.checkpoint, device)
    training_objective = resolve_training_objective(args.training_objective, checkpoint_config)
    checkpoint_hash = file_sha256(args.checkpoint)
    manifest_hash = file_sha256(args.manifest_csv)
    dataset = make_dataset(args, checkpoint_config)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    max_samples = len(dataset) if args.max_samples is None else min(int(args.max_samples), len(dataset))
    effective_tta_loss = "self_supervised" if args.tta_loss == "l1" else args.tta_loss
    if args.run_tta:
        method_by_loss = {
            "ensure": "true_ensure_tta",
            "ensure_data": "true_ensure_data_tta",
            "self_supervised": "measured_kspace_l1_tta",
        }
        method = method_by_loss[effective_tta_loss]
    else:
        method = "frozen"

    crop_height = args.crop_height if args.crop_height is not None else checkpoint_config.get("crop_height")
    crop_width = args.crop_width if args.crop_width is not None else checkpoint_config.get("crop_width")

    rows: list[dict[str, object]] = []
    print(f"Output: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Training objective: {training_objective}", flush=True)
    print(f"Manifest: {args.manifest_csv} split_role={args.split_role} rows={max_samples}", flush=True)
    print(f"TTA loss: {args.tta_loss} (effective={effective_tta_loss})", flush=True)
    print(f"Test noise SNR dB: {args.test_noise_snr_db}", flush=True)

    progress = tqdm(enumerate(loader), total=max_samples, desc=method)
    for sample_idx, batch in progress:
        if sample_idx >= max_samples:
            break
        manifest_row = dataset.samples[sample_idx].row
        row = metric_row_base(
            status="ok",
            method=method,
            training_objective=training_objective,
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_hash,
            manifest_sha256=manifest_hash,
            manifest_row=manifest_row,
            run_tta=args.run_tta,
            update_mode=args.update_mode,
            test_noise_snr_db=args.test_noise_snr_db,
            test_noise_seed=args.test_noise_seed,
            mask_seed=args.seed,
        )
        try:
            batch = move_to_device(batch, device)
            batch = maybe_center_crop_batch(batch, crop_height=crop_height, crop_width=crop_width)
            meta = _batch_meta(batch)
            result = run_true_ensure_tta(
                model=model,
                batch=batch,
                device=device,
                seed=int(args.seed) + int(sample_idx),
                max_steps=args.tta_steps,
                lr=args.tta_lr,
                weight_decay=args.tta_weight_decay,
                grad_clip=args.grad_clip,
                self_val_fraction=args.self_val_fraction,
                early_stop_window=args.early_stop_window,
                update_mode=args.update_mode,
                run_tta=args.run_tta,
                include_ssim=args.include_ssim,
                cg_l2lam=float(_config_value(checkpoint_config, "cg_l2lam", 1.0e-6)),
                cg_max_iter=int(_config_value(checkpoint_config, "cg_max_iter", 25)),
                cg_tol=float(_config_value(checkpoint_config, "cg_tol", 1.0e-6)),
                divergence_eps=checkpoint_config.get("divergence_eps"),
                divergence_mc_samples=int(_config_value(checkpoint_config, "divergence_mc_samples", 1)),
                divergence_mode=str(_config_value(checkpoint_config, "divergence_mode", "measurement")),
                project_divergence=not bool(_config_value(checkpoint_config, "no_divergence_projection", False)),
                tta_loss=effective_tta_loss,
            )
            populate_result_fields(row, result, meta)
            if args.save_curves and args.run_tta:
                curve_path = output_dir / "curves" / f"{curve_tag(row)}.json"
                save_curve(curve_path, result)
                row["curve_json"] = str(curve_path)
            if args.save_recons:
                recon_path = output_dir / "recons" / f"{curve_tag(row)}.npz"
                save_reconstruction_npz(recon_path, batch=batch, result=result, row=row)
                row["recon_npz"] = str(recon_path)
            rows.append(row)
            write_csv(metrics_path, rows, METRICS_FIELDS)
            progress.set_postfix(
                nmse=row.get("after_tta_nmse", ""),
                steps=row.get("num_tta_steps", ""),
                refresh=False,
            )
        except Exception as exc:  # noqa: BLE001 - experiment script should log per-sample failures.
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            row = metric_row_base(
                status="failed",
                method=method,
                training_objective=training_objective,
                checkpoint=args.checkpoint,
                checkpoint_sha256=checkpoint_hash,
                manifest_sha256=manifest_hash,
                manifest_row=manifest_row,
                run_tta=args.run_tta,
                update_mode=args.update_mode,
                test_noise_snr_db=args.test_noise_snr_db,
                test_noise_seed=args.test_noise_seed,
                mask_seed=args.seed,
                error=error,
            )
            rows.append(row)
            write_csv(metrics_path, rows, METRICS_FIELDS)
            print(f"[failed] sample={sample_idx:03d} {manifest_row.get('sample_id', '')}: {error}", flush=True)
            if args.fail_fast:
                raise

    summary = aggregate(rows)
    write_csv(summary_path, summary, SUMMARY_FIELDS)
    payload = {
        "stage": "true_ensure_tta",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "manifest_csv": str(args.manifest_csv),
        "output_dir": str(output_dir),
        "device": str(device),
        "split_role": args.split_role,
        "method": method,
        "training_objective": training_objective,
        "tta_loss": args.tta_loss,
        "effective_tta_loss": effective_tta_loss,
        "uses_ensure_loss": effective_tta_loss in ("ensure", "ensure_data"),
        "uses_ensure_divergence": effective_tta_loss == "ensure",
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "target_shift_names": list(args.target_shift_names),
        "num_dataset_rows": len(dataset),
        "num_requested_rows": max_samples,
        "num_metric_rows": len(rows),
        "num_ok": len([row for row in rows if row.get("status") == "ok"]),
        "num_failed": len([row for row in rows if row.get("status") == "failed"]),
        "run_tta": args.run_tta,
        "tta_steps": args.tta_steps,
        "tta_lr": args.tta_lr,
        "tta_weight_decay": args.tta_weight_decay,
        "self_val_fraction": args.self_val_fraction,
        "early_stop_window": args.early_stop_window,
        "update_mode": args.update_mode,
        "include_ssim": args.include_ssim,
        "save_recons": args.save_recons,
        "test_noise_snr_db": args.test_noise_snr_db,
        "test_noise_seed": args.test_noise_seed,
        "test_noise_domain": "normalized_full_kspace_before_sampling",
        "test_noise_target": "kspace_fs_only_target_rss_left_clean",
        "adapt_runtime_sec_definition": (
            "CUDA-synchronized time spent in the per-step TTA update and self-validation early-stop path; "
            "excludes before/after metrics, SSIM/GT step metrics, curve logging, and reconstruction saving."
        ),
        "metrics_csv": str(metrics_path),
        "summary_csv": str(summary_path),
    }
    save_json(payload_path, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> int:
    args = build_argparser().parse_args()
    payload = run_experiment(args)
    return 0 if payload["num_ok"] > 0 and payload["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
