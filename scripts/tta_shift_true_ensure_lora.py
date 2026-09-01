#!/usr/bin/env python3
"""Run parameter-efficient Conv-LoRA TTA for a TRUE-ENSURE checkpoint.

This is an independent experiment entry point.  It reuses the dataset,
reporting, and measured-kspace self-validation protocol from
``tta_shift_true_ensure.py`` without modifying that baseline script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_ensure.models import TemporalNormUnet
from cardiac_ensure.scripts import tta_shift_true_ensure as baseline
from cardiac_ensure.scripts.train_common import (
    configure_torch,
    ensure_dir,
    maybe_center_crop_batch,
    move_to_device,
    resolve_device,
    save_json,
    set_random_seed,
)
from cardiac_ensure.tta import run_true_ensure_tta


class ConvLoRA2d(nn.Module):
    """Frozen Conv2d plus a trainable low-rank convolutional residual.

    The adapter factorizes a flattened k x k convolution update into a k x k
    rank-reduction convolution followed by a 1 x 1 rank-expansion convolution.
    The expansion is initialized to zero, so inserting the adapter preserves the
    checkpoint output exactly while still giving the expansion weights a
    non-zero first-step gradient.
    """

    def __init__(self, base_conv: nn.Conv2d, *, rank: int, alpha: float) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if base_conv.groups != 1:
            raise ValueError("ConvLoRA2d currently supports groups=1 only")

        self.base_conv = base_conv
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.base_conv.requires_grad_(False)

        self.lora_down = nn.Conv2d(
            in_channels=base_conv.in_channels,
            out_channels=self.rank,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            groups=1,
            bias=False,
            padding_mode=base_conv.padding_mode,
        ).to(device=base_conv.weight.device, dtype=base_conv.weight.dtype)
        self.lora_up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ).to(device=base_conv.weight.device, dtype=base_conv.weight.dtype)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        base = self.base_conv(image)
        update = self.lora_up(self.lora_down(image))
        return base + self.scale * update


def _named_denoiser_conv_layers(
    model: TemporalNormUnet,
) -> list[tuple[str, list[nn.Module]]]:
    if model.denoiser_sharing == "shared":
        return [("norm_unet", list(model.norm_unet.cnn.layers))]
    if model.denoiser_sharing == "independent":
        return [
            (f"norm_unets.{index}", list(denoiser.cnn.layers))
            for index, denoiser in enumerate(model.norm_unets)
        ]
    raise ValueError(f"Unsupported denoiser_sharing={model.denoiser_sharing!r}")


def inject_lora_adapters(
    model: TemporalNormUnet,
    *,
    rank: int,
    alpha: float,
    layer_indices: list[int],
    adapt_dc: bool = False,
) -> list[str]:
    """Freeze the checkpoint and insert LoRA into selected PaperCnn layers."""

    model.requires_grad_(False)
    denoisers = _named_denoiser_conv_layers(model)
    selected = sorted(set(int(index) for index in layer_indices))
    if not selected:
        raise ValueError("At least one --lora-layer must be selected")

    adapter_names: list[str] = []
    for denoiser_name, layers in denoisers:
        for index in selected:
            if index < 0 or index >= len(layers):
                raise IndexError(f"LoRA layer index {index} is outside [0, {len(layers)})")
            conv_layer = layers[index]
            if not hasattr(conv_layer, "layers") or not isinstance(conv_layer.layers[0], nn.Conv2d):
                raise TypeError(
                    f"Expected PaperCnn ConvLayer at index {index}, got {type(conv_layer).__name__}"
                )
            adapter = ConvLoRA2d(conv_layer.layers[0], rank=rank, alpha=alpha)
            conv_layer.layers[0] = adapter
            adapter_names.extend(
                [
                    f"{denoiser_name}.cnn.layers.{index}.layers.0.lora_down.weight",
                    f"{denoiser_name}.cnn.layers.{index}.layers.0.lora_up.weight",
                ]
            )

    if adapt_dc:
        for block in model.dc_blocks:
            block.dc_weight.requires_grad_(True)
        adapter_names.extend(f"dc_blocks.{index}.dc_weight" for index in range(len(model.dc_blocks)))

    actual_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if actual_trainable != adapter_names:
        raise RuntimeError(
            "Unexpected trainable parameter selection: "
            f"expected={adapter_names}, actual={actual_trainable}"
        )
    return actual_trainable


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def select_dc_only_parameters(model: TemporalNormUnet) -> list[str]:
    """Freeze the reconstruction network and select only cascade DC scalars."""

    model.requires_grad_(False)
    trainable_names = [
        f"dc_blocks.{index}.dc_weight" for index in range(len(model.dc_blocks))
    ]
    for block in model.dc_blocks:
        block.dc_weight.requires_grad_(True)

    actual_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if actual_trainable != trainable_names:
        raise RuntimeError(
            "Unexpected DC-only parameter selection: "
            f"expected={trainable_names}, actual={actual_trainable}"
        )
    return trainable_names


def build_argparser() -> argparse.ArgumentParser:
    parser = baseline.build_argparser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=(
            baseline.DEFAULT_OUTPUT_DIR.parent
            / "true_ensure_source_r4_w1_auto_unroll_seed7_lora"
        ),
        update_mode="adapter",
        tta_loss="l1",
    )
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Zero-based dataset offset for deterministic multi-GPU sharding.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA scaling numerator; defaults to rank so the effective scale is 1.",
    )
    parser.add_argument(
        "--lora-layers",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Zero-based PaperCnn layer indices. Phase 1 defaults to the three 64->64 layers.",
    )
    parser.add_argument(
        "--adapt-dc",
        action="store_true",
        help="Also adapt the per-cascade scalar DC weights (off for the LoRA-only phase-1 experiment).",
    )
    parser.add_argument(
        "--dc-only",
        action="store_true",
        help="Adapt only the per-cascade scalar DC weights without inserting LoRA adapters.",
    )
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.update_mode != "adapter":
        raise ValueError("This entry point only supports --update-mode adapter")
    if args.tta_loss not in ("l1", "self_supervised"):
        raise ValueError("Phase-1 Conv-LoRA experiments use measured-kspace L1 TTA only")
    if args.dc_only and args.adapt_dc:
        raise ValueError("--dc-only and --adapt-dc are mutually exclusive")

    configure_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.csv"
    payload_path = output_dir / "summary.json"

    model, checkpoint_config = baseline.load_model_from_checkpoint(args.checkpoint, device)
    lora_alpha = float(args.lora_rank if args.lora_alpha is None else args.lora_alpha)
    if args.dc_only:
        trainable_names = select_dc_only_parameters(model)
    else:
        trainable_names = inject_lora_adapters(
            model,
            rank=args.lora_rank,
            alpha=lora_alpha,
            layer_indices=args.lora_layers,
            adapt_dc=args.adapt_dc,
        )
    num_trainable = count_trainable_parameters(model)
    num_total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_fraction = float(num_trainable / num_total)
    model.eval()

    training_objective = baseline.resolve_training_objective(args.training_objective, checkpoint_config)
    checkpoint_hash = baseline.file_sha256(args.checkpoint)
    manifest_hash = baseline.file_sha256(args.manifest_csv)
    dataset = baseline.make_dataset(args, checkpoint_config)
    start_sample = int(args.start_sample)
    if start_sample < 0 or start_sample >= len(dataset):
        raise ValueError(f"--start-sample must be in [0, {len(dataset)}), got {start_sample}")
    stop_sample = (
        len(dataset)
        if args.max_samples is None
        else min(start_sample + int(args.max_samples), len(dataset))
    )
    sample_indices = range(start_sample, stop_sample)
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    num_requested_samples = stop_sample - start_sample
    method = f"true_ensure_convlora_r{int(args.lora_rank)}_tta" if args.run_tta else "frozen"
    if args.dc_only and args.run_tta:
        method = "true_ensure_dc_only_tta"
    elif args.adapt_dc and args.run_tta:
        method = f"true_ensure_convlora_r{int(args.lora_rank)}_dc_tta"

    crop_height = args.crop_height if args.crop_height is not None else checkpoint_config.get("crop_height")
    crop_width = args.crop_width if args.crop_width is not None else checkpoint_config.get("crop_width")

    print(f"Output: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(
        f"Manifest: {args.manifest_csv} split_role={args.split_role} "
        f"sample_range=[{start_sample}, {stop_sample}) rows={num_requested_samples}",
        flush=True,
    )
    print("TTA loss: measured-kspace normalized complex L1", flush=True)
    if args.dc_only:
        print("Adapter selection: DC-only", flush=True)
    else:
        print(
            f"Conv-LoRA: rank={args.lora_rank} alpha={lora_alpha:g} "
            f"layers={sorted(set(args.lora_layers))} adapt_dc={args.adapt_dc}",
            flush=True,
        )
    print(
        f"Trainable parameters: {num_trainable}/{num_total} ({100.0 * trainable_fraction:.3f}%)",
        flush=True,
    )
    for name in trainable_names:
        print(f"  trainable: {name}", flush=True)

    rows: list[dict[str, object]] = []
    progress = tqdm(enumerate(loader), total=num_requested_samples, desc=method)
    for local_idx, batch in progress:
        sample_idx = start_sample + local_idx
        manifest_row = dataset.samples[sample_idx].row
        row = baseline.metric_row_base(
            status="ok",
            method=method,
            training_objective=training_objective,
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_hash,
            manifest_sha256=manifest_hash,
            manifest_row=manifest_row,
            run_tta=args.run_tta,
            update_mode="adapter",
            test_noise_snr_db=args.test_noise_snr_db,
            test_noise_seed=args.test_noise_seed,
            mask_seed=args.seed,
        )
        try:
            batch = move_to_device(batch, device)
            batch = maybe_center_crop_batch(batch, crop_height=crop_height, crop_width=crop_width)
            meta = baseline._batch_meta(batch)
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
                # The core routine selects parameters by requires_grad.  The
                # new entry point has already frozen everything except LoRA.
                update_mode="all_params",
                run_tta=args.run_tta,
                include_ssim=args.include_ssim,
                cg_l2lam=float(baseline._config_value(checkpoint_config, "cg_l2lam", 1.0e-6)),
                cg_max_iter=int(baseline._config_value(checkpoint_config, "cg_max_iter", 25)),
                cg_tol=float(baseline._config_value(checkpoint_config, "cg_tol", 1.0e-6)),
                divergence_eps=checkpoint_config.get("divergence_eps"),
                divergence_mc_samples=int(
                    baseline._config_value(checkpoint_config, "divergence_mc_samples", 1)
                ),
                divergence_mode=str(
                    baseline._config_value(checkpoint_config, "divergence_mode", "measurement")
                ),
                project_divergence=not bool(
                    baseline._config_value(checkpoint_config, "no_divergence_projection", False)
                ),
                tta_loss="self_supervised",
            )
            if result.num_trainable_params != num_trainable:
                raise RuntimeError(
                    f"TTA selected {result.num_trainable_params} parameters, expected {num_trainable}"
                )
            baseline.populate_result_fields(row, result, meta)
            if args.save_curves and args.run_tta:
                curve_path = output_dir / "curves" / f"{baseline.curve_tag(row)}.json"
                baseline.save_curve(curve_path, result)
                row["curve_json"] = str(curve_path)
            if args.save_recons:
                recon_path = output_dir / "recons" / f"{baseline.curve_tag(row)}.npz"
                baseline.save_reconstruction_npz(recon_path, batch=batch, result=result, row=row)
                row["recon_npz"] = str(recon_path)
            rows.append(row)
            baseline.write_csv(metrics_path, rows, baseline.METRICS_FIELDS)
            progress.set_postfix(
                nmse=row.get("after_tta_nmse", ""),
                steps=row.get("num_tta_steps", ""),
                refresh=False,
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-sample experiment logging.
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            row = baseline.metric_row_base(
                status="failed",
                method=method,
                training_objective=training_objective,
                checkpoint=args.checkpoint,
                checkpoint_sha256=checkpoint_hash,
                manifest_sha256=manifest_hash,
                manifest_row=manifest_row,
                run_tta=args.run_tta,
                update_mode="adapter",
                test_noise_snr_db=args.test_noise_snr_db,
                test_noise_seed=args.test_noise_seed,
                mask_seed=args.seed,
                error=error,
            )
            rows.append(row)
            baseline.write_csv(metrics_path, rows, baseline.METRICS_FIELDS)
            print(f"[failed] sample={sample_idx:03d}: {error}", flush=True)
            if args.fail_fast:
                raise

    summary = baseline.aggregate(rows)
    baseline.write_csv(summary_path, summary, baseline.SUMMARY_FIELDS)
    payload = {
        "stage": "true_ensure_convlora_tta",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "manifest_csv": str(args.manifest_csv),
        "output_dir": str(output_dir),
        "device": str(device),
        "split_role": args.split_role,
        "method": method,
        "training_objective": training_objective,
        "tta_loss": "measured_kspace_normalized_complex_l1",
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "target_shift_names": list(args.target_shift_names),
        "num_dataset_rows": len(dataset),
        "start_sample": start_sample,
        "stop_sample": stop_sample,
        "num_requested_rows": num_requested_samples,
        "num_metric_rows": len(rows),
        "num_ok": len([row for row in rows if row.get("status") == "ok"]),
        "num_failed": len([row for row in rows if row.get("status") == "failed"]),
        "run_tta": args.run_tta,
        "tta_steps": args.tta_steps,
        "tta_lr": args.tta_lr,
        "tta_weight_decay": args.tta_weight_decay,
        "self_val_fraction": args.self_val_fraction,
        "early_stop_window": args.early_stop_window,
        "update_mode": "adapter",
        "adapter_mode": "dc_only" if args.dc_only else "convlora",
        "lora_rank": None if args.dc_only else int(args.lora_rank),
        "lora_alpha": None if args.dc_only else lora_alpha,
        "lora_layers": [] if args.dc_only else sorted(set(int(index) for index in args.lora_layers)),
        "adapt_dc": bool(args.adapt_dc),
        "dc_only": bool(args.dc_only),
        "trainable_parameter_names": trainable_names,
        "num_trainable_params": num_trainable,
        "num_total_params_with_adapters": num_total,
        "trainable_parameter_fraction": trainable_fraction,
        "include_ssim": args.include_ssim,
        "save_recons": args.save_recons,
        "test_noise_snr_db": args.test_noise_snr_db,
        "test_noise_seed": args.test_noise_seed,
        "test_noise_domain": "normalized_full_kspace_before_sampling",
        "test_noise_target": "kspace_fs_only_target_rss_left_clean",
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
