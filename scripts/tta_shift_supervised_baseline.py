#!/usr/bin/env python3
"""Run self-supervised TTA on shift-manifest samples for supervised/self-trained models."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import statistics
import sys
import time
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
from cardiac_ensure.ops import dynamic_a_adjoint, dynamic_a_forward
from cardiac_ensure.scripts.eval_metrics import summarize_reconstruction_metrics, to_magnitude
from cardiac_ensure.scripts.train_common import (
    configure_torch,
    ensure_dir,
    maybe_center_crop_batch,
    move_to_device,
    resolve_device,
    save_json,
    set_random_seed,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "cardiac_ensure"
    / "outputs"
    / "shifts"
    / "main"
    / "modality_shift"
    / "uncertainty_joint_source_r4_w1_unroll12_seed7"
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
    / "uncertainty_joint_source_r4_w1_unroll12_seed7"
)

EPS = 1.0e-8
METRIC_NAMES = ("nmse", "nrmse", "psnr", "frame_diff_nmse", "ssim")
METRICS_FIELDS = [
    "status",
    "method",
    "checkpoint",
    "shift_name",
    "experiment_tier",
    "split_role",
    "rank_in_split",
    "sample_id",
    "file",
    "slice",
    "dataset",
    "split",
    "acquisition",
    "target_key",
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
    "recon_npz",
    "curve_json",
    "error",
]
SUMMARY_FIELDS = [
    "method",
    "shift_name",
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


@dataclass
class SelfSupervisedTTAResult:
    before_rec: np.ndarray
    after_rec: np.ndarray
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]
    train_losses: list[float]
    self_val_losses: list[float]
    step_logs: list[dict[str, float | int | None]]
    num_tta_steps: int
    early_stop_step: int | None
    self_val_best: float
    runtime_sec: float
    adapt_runtime_sec: float
    best_step: int | None
    num_trainable_params: int


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
    parser.add_argument("--tta-steps", type=int, default=250)
    parser.add_argument("--tta-lr", type=float, default=1.0e-5)
    parser.add_argument("--tta-weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--self-val-fraction", type=float, default=0.05)
    parser.add_argument("--early-stop-window", type=int, default=20)
    parser.add_argument("--update-mode", default="all_params", choices=("all_params",))
    parser.add_argument("--run-tta", dest="run_tta", action="store_true", default=True)
    parser.add_argument("--no-run-tta", dest="run_tta", action="store_false")
    parser.add_argument("--include-ssim", dest="include_ssim", action="store_true", default=True)
    parser.add_argument("--no-include-ssim", dest="include_ssim", action="store_false")
    parser.add_argument("--save-recons", dest="save_recons", action="store_true", default=True)
    parser.add_argument("--no-save-recons", dest="save_recons", action="store_false")
    parser.add_argument("--save-curves", dest="save_curves", action="store_true", default=True)
    parser.add_argument("--no-save-curves", dest="save_curves", action="store_false")
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
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
        acceleration=float(_config_value(config, "acceleration", 4.0)),
        sigma_mask=float(_config_value(config, "sigma_mask", 0.18)),
        window_size=int(_config_value(config, "window_size", 1)),
        deterministic_masks=True,
        mask_seed=int(args.seed),
        return_target=True,
        require_preproc=bool(_config_value(config, "require_preproc", False)),
    )


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state_dict_on_device(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()}, strict=True)


def _target_or_none(batch: Mapping[str, Any]) -> torch.Tensor | None:
    target = batch.get("target_rss")
    return target if torch.is_tensor(target) else None


def _center_crop_to_shape(x: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = int(spatial_shape[0]), int(spatial_shape[1])
    if x.shape[-2] < target_h or x.shape[-1] < target_w:
        raise ValueError(
            f"Cannot center-crop tensor with spatial shape {tuple(x.shape[-2:])} to larger target {spatial_shape}"
        )
    h_start = (x.shape[-2] - target_h) // 2
    w_start = (x.shape[-1] - target_w) // 2
    return x[..., h_start : h_start + target_h, w_start : w_start + target_w]


def _align_prediction_to_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction
    return _center_crop_to_shape(prediction, tuple(int(v) for v in target.shape[-2:]))


def _compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor | None,
    *,
    include_ssim: bool,
) -> dict[str, float]:
    if target is None:
        return {}
    return summarize_reconstruction_metrics(
        _align_prediction_to_target(prediction, target),
        target,
        include_ssim=include_ssim,
    )


def _prediction_np(prediction: torch.Tensor) -> np.ndarray:
    return to_magnitude(prediction).detach().cpu().float().numpy()


def _as_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().real.item())
    return float(value)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_trainable(parameters: list[torch.nn.Parameter]) -> int:
    return int(sum(param.numel() for param in parameters if param.requires_grad))


def _masked_kspace(kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return kspace * mask.to(device=kspace.device, dtype=kspace.real.dtype)


def _normalized_complex_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    numerator = torch.sum(torch.abs(prediction - target))
    denominator = torch.sum(torch.abs(target.detach())).clamp_min(float(eps))
    return numerator / denominator


def _self_validation_loss(
    model: torch.nn.Module,
    *,
    train_zf: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    kspace: torch.Tensor,
    maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model(train_zf, maps=maps, mask=train_mask)
    val_prediction = dynamic_a_forward(prediction, maps, val_mask)
    val_target = _masked_kspace(kspace, val_mask)
    return _normalized_complex_l1(val_prediction, val_target), prediction


def _dc_weight_stats(model: torch.nn.Module) -> tuple[float | None, float | None]:
    values = [
        float(block.dc_weight_value().detach().cpu())
        for block in getattr(model, "dc_blocks", [])
        if hasattr(block, "dc_weight_value")
    ]
    if not values:
        return None, None
    return min(values), max(values)


def _center_slice(width: int, center_fraction: float) -> slice:
    num_cols = max(1, int(round(float(width) * float(center_fraction))))
    start = max(0, (int(width) - num_cols) // 2)
    return slice(start, start + num_cols)


def split_self_validation_mask(
    mask: torch.Tensor,
    *,
    fraction: float = 0.05,
    seed: int = 0,
    preserve_center: bool = True,
    center_fraction: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sampled phase-encode lines into train and held-out self-val masks."""

    if not (0.0 < float(fraction) < 1.0):
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if mask.ndim != 5 or mask.shape[2] != 1:
        raise ValueError(f"Expected mask shape [B, T, 1, H, W], got {tuple(mask.shape)}")

    train_mask = mask.detach().clone()
    val_mask = torch.zeros_like(train_mask)
    mask_cpu = mask.detach().cpu()
    batch_size, num_frames, _, _, width = mask_cpu.shape

    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            frame_mask = mask_cpu[batch_idx, frame_idx, 0]
            measured_cols = torch.any(frame_mask > 0, dim=0).numpy()
            candidates = measured_cols.copy()
            if preserve_center and width > 0:
                candidates[_center_slice(width, center_fraction)] = False
                if not np.any(candidates):
                    candidates = measured_cols.copy()
            candidate_cols = np.flatnonzero(candidates)
            if candidate_cols.size == 0:
                raise ValueError("Cannot create self-validation mask from an empty sampled mask")

            measured_count = int(np.count_nonzero(measured_cols))
            num_val = max(1, int(round(measured_count * float(fraction))))
            num_val = min(num_val, int(candidate_cols.size))
            rng = np.random.default_rng(int(seed) + 1009 * batch_idx + 9176 * frame_idx)
            val_cols = rng.choice(candidate_cols, size=num_val, replace=False)
            col_index = torch.as_tensor(val_cols, device=mask.device, dtype=torch.long)
            train_mask[batch_idx, frame_idx, 0, :, col_index] = 0.0
            val_mask[batch_idx, frame_idx, 0, :, col_index] = mask[batch_idx, frame_idx, 0, :, col_index]

    return train_mask, val_mask


def run_self_supervised_tta(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    seed: int,
    max_steps: int = 250,
    lr: float = 1.0e-5,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    self_val_fraction: float = 0.05,
    early_stop_window: int = 20,
    update_mode: str = "all_params",
    run_tta: bool = True,
    include_ssim: bool = False,
) -> SelfSupervisedTTAResult:
    """Adapt with the original self-supervised measured-kspace TTT objective."""

    tic = time.time()
    if update_mode != "all_params":
        raise NotImplementedError("Cardiac supervised-baseline TTA currently supports update_mode='all_params' only")

    kspace = batch["kspace_us"]
    zf = batch["zf"]
    maps = batch["maps"]
    mask = batch["mask"]
    target = _target_or_none(batch)

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        before_prediction = model(zf, maps=maps, mask=mask)
    model.train(model_was_training)
    before_metrics = _compute_metrics(before_prediction, target, include_ssim=include_ssim)
    before_rec = _prediction_np(before_prediction)

    if not run_tta or int(max_steps) <= 0:
        runtime = time.time() - tic
        return SelfSupervisedTTAResult(
            before_rec=before_rec,
            after_rec=before_rec,
            before_metrics=before_metrics,
            after_metrics=before_metrics,
            train_losses=[],
            self_val_losses=[],
            step_logs=[],
            num_tta_steps=0,
            early_stop_step=None,
            self_val_best=float("nan"),
            runtime_sec=runtime,
            adapt_runtime_sec=0.0,
            best_step=None,
            num_trainable_params=0,
        )

    train_mask, val_mask = split_self_validation_mask(mask, fraction=self_val_fraction, seed=seed)
    train_kspace = _masked_kspace(kspace, train_mask)
    train_zf = dynamic_a_adjoint(train_kspace, maps, train_mask)

    model_tta = copy.deepcopy(model).to(device)
    model_tta.train()
    params = [param for param in model_tta.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable model parameters selected for TTA")
    num_trainable_params = _count_trainable(params)
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    train_losses: list[float] = []
    self_val_losses: list[float] = []
    step_logs: list[dict[str, float | int | None]] = []
    best_state = _clone_state_dict(model_tta)
    best_val_loss = float("inf")
    best_step: int | None = None
    early_stop_step: int | None = None
    adapt_runtime_sec = 0.0

    for step_idx in range(int(max_steps)):
        _sync_if_cuda(device)
        step_tic = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = model_tta(train_zf, maps=maps, mask=train_mask)
        train_prediction = dynamic_a_forward(prediction, maps, train_mask)
        loss = _normalized_complex_l1(train_prediction, train_kspace)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"Non-finite TTA loss at step {step_idx + 1}: {float(loss.detach())}")

        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
        optimizer.step()

        with torch.no_grad():
            val_loss, val_prediction = _self_validation_loss(
                model_tta,
                train_zf=train_zf,
                train_mask=train_mask,
                val_mask=val_mask,
                kspace=kspace,
                maps=maps,
            )

        train_loss_value = _as_float(loss)
        val_loss_value = _as_float(val_loss)
        train_losses.append(train_loss_value)
        self_val_losses.append(val_loss_value)

        if np.isfinite(val_loss_value) and val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_step = step_idx + 1
            best_state = _clone_state_dict(model_tta)

        should_stop = False
        if int(early_stop_window) > 0 and step_idx + 1 > 3 * int(early_stop_window):
            window = int(early_stop_window)
            curr = float(np.mean(self_val_losses[-window:]))
            prev = float(np.mean(self_val_losses[-2 * window : -window]))
            if np.isfinite(curr) and np.isfinite(prev) and curr > prev:
                early_stop_step = step_idx + 1
                should_stop = True

        _sync_if_cuda(device)
        adapt_runtime_sec += time.perf_counter() - step_tic

        with torch.no_grad():
            full_prediction_kspace = dynamic_a_forward(val_prediction, maps, mask)
            dc_residual = _normalized_complex_l1(full_prediction_kspace, _masked_kspace(kspace, mask))

        dc_min, dc_max = _dc_weight_stats(model_tta)
        log_row: dict[str, float | int | None] = {
            "step": step_idx + 1,
            "train_loss": train_loss_value,
            "self_val_loss": val_loss_value,
            "dc_residual": _as_float(dc_residual),
            "dc_weight_min": dc_min,
            "dc_weight_max": dc_max,
        }
        if target is not None:
            metrics_step = _compute_metrics(val_prediction, target, include_ssim=include_ssim)
            if "nmse" in metrics_step:
                log_row["score_if_gt_available_nmse"] = float(metrics_step["nmse"])
            if "ssim" in metrics_step:
                log_row["score_if_gt_available_ssim"] = float(metrics_step["ssim"])
        step_logs.append(log_row)

        if should_stop:
            break

    model_tta.eval()
    if best_step is not None:
        _load_state_dict_on_device(model_tta, best_state, device)
    with torch.no_grad():
        after_prediction = model_tta(zf, maps=maps, mask=mask)
    after_metrics = _compute_metrics(after_prediction, target, include_ssim=include_ssim)
    after_rec = _prediction_np(after_prediction)
    runtime = time.time() - tic

    return SelfSupervisedTTAResult(
        before_rec=before_rec,
        after_rec=after_rec,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        train_losses=train_losses,
        self_val_losses=self_val_losses,
        step_logs=step_logs,
        num_tta_steps=len(train_losses),
        early_stop_step=early_stop_step,
        self_val_best=best_val_loss if self_val_losses else float("nan"),
        runtime_sec=runtime,
        adapt_runtime_sec=adapt_runtime_sec,
        best_step=best_step,
        num_trainable_params=num_trainable_params,
    )


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
    checkpoint: Path,
    manifest_row: Mapping[str, str],
    run_tta: bool,
    update_mode: str,
    error: str = "",
) -> dict[str, object]:
    return {
        "status": status,
        "method": method,
        "checkpoint": str(checkpoint),
        "shift_name": manifest_row.get("shift_name", ""),
        "experiment_tier": manifest_row.get("experiment_tier", ""),
        "split_role": manifest_row.get("split_role", ""),
        "rank_in_split": manifest_row.get("rank_in_split", ""),
        "sample_id": manifest_row.get("sample_id", ""),
        "file": manifest_row.get("path", ""),
        "slice": manifest_row.get("slice_idx", ""),
        "dataset": manifest_row.get("dataset", ""),
        "split": manifest_row.get("split", ""),
        "acquisition": manifest_row.get("acquisition", ""),
        "target_key": manifest_row.get("target_key", ""),
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
        "recon_npz": "",
        "curve_json": "",
        "error": error,
    }


def populate_result_fields(row: dict[str, object], result: SelfSupervisedTTAResult, meta: Mapping[str, Any]) -> None:
    row["num_trainable_params"] = result.num_trainable_params
    row["num_tta_steps"] = result.num_tta_steps
    row["early_stop_step"] = result.early_stop_step if result.early_stop_step is not None else ""
    row["best_step"] = result.best_step if result.best_step is not None else ""
    row["self_val_best"] = result.self_val_best
    row["runtime_sec"] = result.runtime_sec
    row["adapt_runtime_sec"] = result.adapt_runtime_sec
    row["norm_scale"] = meta.get("norm_scale", "")

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


def save_curve(path: Path, result: SelfSupervisedTTAResult) -> None:
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
            "tta_loss": "simple_measured_kspace_l1",
        },
    )


def save_reconstruction_npz(
    path: Path,
    *,
    batch: Mapping[str, Any],
    result: SelfSupervisedTTAResult,
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
            "delta_nmse",
            "before_tta_psnr",
            "after_tta_psnr",
            "delta_psnr",
            "before_tta_ssim",
            "after_tta_ssim",
            "delta_ssim",
            "num_tta_steps",
            "best_step",
            "runtime_sec",
            "adapt_runtime_sec",
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
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in ok_rows:
        grouped.setdefault((str(row.get("method", "")), str(row.get("shift_name", ""))), []).append(row)

    summaries: list[dict[str, object]] = []
    for (method, shift_name), group in sorted(grouped.items()):
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
                "shift_name": shift_name,
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
    method = "uncertainty_joint_self_tta" if args.run_tta else "uncertainty_joint_frozen"

    crop_height = args.crop_height if args.crop_height is not None else checkpoint_config.get("crop_height")
    crop_width = args.crop_width if args.crop_width is not None else checkpoint_config.get("crop_width")

    rows: list[dict[str, object]] = []
    print(f"Output: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Manifest: {args.manifest_csv} split_role={args.split_role} rows={max_samples}", flush=True)
    print("TTA loss: simple measured-kspace normalized L1 (no ENSURE loss)", flush=True)

    progress = tqdm(enumerate(loader), total=max_samples, desc=method)
    for sample_idx, batch in progress:
        if sample_idx >= max_samples:
            break
        manifest_row = dataset.samples[sample_idx].row
        row = metric_row_base(
            status="ok",
            method=method,
            checkpoint=args.checkpoint,
            manifest_row=manifest_row,
            run_tta=args.run_tta,
            update_mode=args.update_mode,
        )
        try:
            batch = move_to_device(batch, device)
            batch = maybe_center_crop_batch(batch, crop_height=crop_height, crop_width=crop_width)
            meta = _batch_meta(batch)
            result = run_self_supervised_tta(
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
                checkpoint=args.checkpoint,
                manifest_row=manifest_row,
                run_tta=args.run_tta,
                update_mode=args.update_mode,
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
        "stage": "supervised_baseline_self_supervised_tta",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "manifest_csv": str(args.manifest_csv),
        "output_dir": str(output_dir),
        "device": str(device),
        "split_role": args.split_role,
        "method": method,
        "tta_loss": "simple_measured_kspace_normalized_l1",
        "uses_ensure_loss": False,
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
        "adapt_runtime_sec_definition": (
            "CUDA-synchronized time spent in the per-step TTA update and self-validation early-stop path; "
            "excludes before/after metrics, SSIM/GT step metrics, diagnostic logging, curve logging, and reconstruction saving."
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
