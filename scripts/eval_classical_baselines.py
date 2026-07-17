#!/usr/bin/env python3
"""Evaluate classical reconstruction baselines on shift-manifest samples."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
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

from cardiac_ensure.baselines import bart_pics_from_batch, zero_filled_from_batch
from cardiac_ensure.datasets import StaticShiftSourceENSUREDataset, load_bart
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
from cardiac_ensure.scripts.tta_shift_true_ensure import (
    METRIC_NAMES,
    METRICS_FIELDS,
    SUMMARY_FIELDS,
    _batch_meta,
    aggregate,
    curve_tag,
    file_sha256,
    metric_row_base,
    write_csv,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "cardiac_ensure" / "manifests" / "shifts" / "main" / "modality_shift.csv"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("zero_filled", "bart_pics_cs"), required=True)
    parser.add_argument("--manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--sigma-mask", type=float, default=0.18)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--crop-width", type=int, default=None)
    parser.add_argument("--include-ssim", action="store_true")
    parser.add_argument("--save-recons", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--bart-lambda", type=float, default=0.01)
    parser.add_argument("--bart-command", default=None)
    parser.add_argument("--bart-toolbox-path", default=None)
    parser.add_argument("--bart-python-path", default=None)
    return parser


def make_dataset(args: argparse.Namespace) -> StaticShiftSourceENSUREDataset:
    return StaticShiftSourceENSUREDataset(
        manifest_csv=args.manifest_csv,
        subset="all",
        preproc_root=args.preproc_root,
        density_root=args.density_root,
        source_role=args.split_role,
        val_fraction=0.2,
        split_seed=int(args.seed),
        allow_slice_val_fallback=True,
        acceleration=float(args.acceleration),
        sigma_mask=float(args.sigma_mask),
        window_size=1,
        deterministic_masks=True,
        mask_seed=int(args.seed),
        return_target=True,
        require_preproc=False,
        test_noise_snr_db=args.test_noise_snr_db,
        test_noise_seed=int(args.test_noise_seed),
        shift_names=args.target_shift_names,
    )


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


def _populate_classical_fields(
    row: dict[str, object],
    *,
    metrics: Mapping[str, float],
    runtime_sec: float,
    meta: Mapping[str, Any],
) -> None:
    row["num_trainable_params"] = 0
    row["num_tta_steps"] = 0
    row["early_stop_step"] = ""
    row["best_step"] = ""
    row["self_val_best"] = ""
    row["runtime_sec"] = runtime_sec
    row["adapt_runtime_sec"] = 0.0
    row["norm_scale"] = meta.get("norm_scale", "")
    row["input_noise_snr_db"] = meta.get("input_noise_snr_db", "")
    row["input_noise_sigma2"] = meta.get("input_noise_sigma2", "")
    row["noise_sigma2_total"] = meta.get("noise_sigma2_total", "")
    for metric in METRIC_NAMES:
        value = metrics.get(metric)
        if value is not None:
            row[f"before_tta_{metric}"] = value
            row[f"after_tta_{metric}"] = value
            row[f"delta_{metric}"] = 0.0


def save_reconstruction_npz(
    path: Path,
    *,
    batch: Mapping[str, Any],
    prediction: torch.Tensor,
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
            "after_tta_nmse",
            "runtime_sec",
            "input_noise_snr_db",
            "input_noise_sigma2",
            "noise_sigma2_total",
        )
    }
    np.savez_compressed(
        path,
        target=target_np.astype(np.float32, copy=False),
        zero_filled=zero_filled.astype(np.float32, copy=False),
        recon=to_magnitude(prediction).detach().cpu().float().numpy().astype(np.float32, copy=False),
        mask=mask.astype(np.float32, copy=False),
        metadata_json=np.array(json.dumps(metadata)),
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    configure_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.csv"
    payload_path = output_dir / "summary.json"

    manifest_hash = file_sha256(args.manifest_csv)
    dataset = make_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    max_samples = len(dataset) if args.max_samples is None else min(int(args.max_samples), len(dataset))

    bart_fn = None
    if args.method == "bart_pics_cs":
        bart_fn = load_bart(
            bart_toolbox_path=args.bart_toolbox_path,
            bart_python_path=args.bart_python_path,
        )

    rows: list[dict[str, object]] = []
    print(f"Output: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Method: {args.method}", flush=True)
    print(f"Manifest: {args.manifest_csv} split_role={args.split_role} rows={max_samples}", flush=True)
    if args.method == "bart_pics_cs":
        print(f"BART lambda: {args.bart_lambda}", flush=True)

    progress = tqdm(enumerate(loader), total=max_samples, desc=args.method)
    for sample_idx, batch in progress:
        if sample_idx >= max_samples:
            break
        manifest_row = dataset.samples[sample_idx].row
        row = metric_row_base(
            status="ok",
            method=args.method,
            training_objective="classical",
            checkpoint=Path("none"),
            checkpoint_sha256="",
            manifest_sha256=manifest_hash,
            manifest_row=manifest_row,
            run_tta=False,
            update_mode="",
            test_noise_snr_db=args.test_noise_snr_db,
            test_noise_seed=args.test_noise_seed,
            mask_seed=args.seed,
        )
        try:
            batch = move_to_device(batch, device)
            batch = maybe_center_crop_batch(batch, crop_height=args.crop_height, crop_width=args.crop_width)
            meta = _batch_meta(batch)
            tic = time.time()
            if args.method == "zero_filled":
                prediction = zero_filled_from_batch(batch)
            else:
                if bart_fn is None:
                    raise RuntimeError("BART function was not loaded")
                prediction = bart_pics_from_batch(
                    batch,
                    bart_fn=bart_fn,
                    lamda=float(args.bart_lambda),
                    command=args.bart_command,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_sec = time.time() - tic

            metrics = _compute_metrics(
                prediction,
                batch.get("target_rss"),
                include_ssim=args.include_ssim,
            )
            _populate_classical_fields(row, metrics=metrics, runtime_sec=runtime_sec, meta=meta)
            if args.save_recons:
                recon_path = output_dir / "recons" / f"{curve_tag(row)}.npz"
                save_reconstruction_npz(recon_path, batch=batch, prediction=prediction, row=row)
                row["recon_npz"] = str(recon_path)
            rows.append(row)
            write_csv(metrics_path, rows, METRICS_FIELDS)
            progress.set_postfix(nmse=row.get("after_tta_nmse", ""), refresh=False)
        except Exception as exc:  # noqa: BLE001 - experiment script should log per-sample failures.
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            row = metric_row_base(
                status="failed",
                method=args.method,
                training_objective="classical",
                checkpoint=Path("none"),
                checkpoint_sha256="",
                manifest_sha256=manifest_hash,
                manifest_row=manifest_row,
                run_tta=False,
                update_mode="",
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
        "stage": "classical_baseline_eval",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "manifest_csv": str(args.manifest_csv),
        "output_dir": str(output_dir),
        "device": str(device),
        "split_role": args.split_role,
        "method": args.method,
        "training_objective": "classical",
        "manifest_sha256": manifest_hash,
        "target_shift_names": list(args.target_shift_names),
        "num_dataset_rows": len(dataset),
        "num_requested_rows": max_samples,
        "num_metric_rows": len(rows),
        "num_ok": len([row for row in rows if row.get("status") == "ok"]),
        "num_failed": len([row for row in rows if row.get("status") == "failed"]),
        "include_ssim": args.include_ssim,
        "save_recons": args.save_recons,
        "test_noise_snr_db": args.test_noise_snr_db,
        "test_noise_seed": args.test_noise_seed,
        "bart_lambda": args.bart_lambda if args.method == "bart_pics_cs" else None,
        "bart_command": args.bart_command,
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
