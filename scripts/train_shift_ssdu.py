#!/usr/bin/env python3
"""Train SSDU on source-domain static samples from shift manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cardiac_ensure.datasets import StaticShiftSourceENSUREDataset
from cardiac_ensure.datasets.cardiac_cine_dataset import stable_seed
from cardiac_ensure.losses import compute_ssdu_loss
from cardiac_ensure.models import TemporalNormUnet
from cardiac_ensure.scripts.eval_metrics import summarize_reconstruction_metrics
from cardiac_ensure.scripts.train_common import (
    RunningStats,
    configure_torch,
    count_trainable_parameters,
    ensure_dir,
    input_noise_stats_from_batch,
    maybe_center_crop_batch,
    move_to_device,
    resolve_device,
    save_checkpoint,
    save_json,
    set_random_seed,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument("--source-role", type=str, default="source_train")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--allow-slice-val-fallback", action="store_true", default=True)
    parser.add_argument("--no-allow-slice-val-fallback", dest="allow_slice_val_fallback", action="store_false")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--sigma-mask", type=float, default=0.18)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--chans", type=int, default=64)
    parser.add_argument("--num-pools", type=int, default=4)
    parser.add_argument("--num-unrolls", type=int, default=12)
    parser.add_argument(
        "--denoiser-sharing",
        choices=("shared", "independent"),
        default="shared",
        help="Reuse one denoiser across unrolls or train one denoiser per unroll.",
    )
    parser.add_argument("--drop-prob", type=float, default=0.0)
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--include-ssim", action="store_true")
    parser.add_argument("--train-deterministic-masks", dest="train_deterministic_masks", action="store_true", default=True)
    parser.add_argument("--no-train-deterministic-masks", dest="train_deterministic_masks", action="store_false")
    parser.add_argument("--val-deterministic-masks", dest="val_deterministic_masks", action="store_true", default=True)
    parser.add_argument("--no-val-deterministic-masks", dest="val_deterministic_masks", action="store_false")
    parser.add_argument("--train-noise-snr-db-min", type=float, default=None)
    parser.add_argument("--train-noise-snr-db-max", type=float, default=None)
    parser.add_argument("--train-noise-seed", type=int, default=7007)
    parser.add_argument("--val-noise-snr-db", type=float, default=None)
    parser.add_argument("--val-noise-seed", type=int, default=8007)
    parser.add_argument("--best-val-metric", choices=("val_nmse", "noisy_nmse"), default="val_nmse")
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--crop-width", type=int, default=None)
    parser.add_argument("--require-preproc", action="store_true")
    parser.add_argument("--ssdu-rho", type=float, default=0.4)
    parser.add_argument("--ssdu-mask-type", choices=("gaussian", "uniform"), default="gaussian")
    parser.add_argument("--ssdu-small-acs-block", nargs=2, type=int, default=(4, 4), metavar=("ROWS", "COLS"))
    parser.add_argument("--ssdu-seed", type=int, default=17017)
    parser.add_argument("--ssdu-gaussian-std-scale", type=float, default=4.0)
    return parser


def _make_dataset(
    args: argparse.Namespace,
    subset: str,
    deterministic_masks: bool,
    return_target: bool,
) -> StaticShiftSourceENSUREDataset:
    if subset == "train":
        noise_kwargs = {
            "input_noise_snr_db_min": args.train_noise_snr_db_min,
            "input_noise_snr_db_max": args.train_noise_snr_db_max,
            "input_noise_seed": args.train_noise_seed,
        }
    else:
        noise_kwargs = {
            "input_noise_snr_db": args.val_noise_snr_db,
            "input_noise_seed": args.val_noise_seed,
        }
    return StaticShiftSourceENSUREDataset(
        manifest_csv=args.manifest_csv,
        subset=subset,
        preproc_root=args.preproc_root,
        density_root=args.density_root,
        source_role=args.source_role,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        allow_slice_val_fallback=args.allow_slice_val_fallback,
        acceleration=args.acceleration,
        sigma_mask=args.sigma_mask,
        window_size=args.window_size,
        deterministic_masks=deterministic_masks,
        mask_seed=args.seed,
        return_target=return_target,
        require_preproc=args.require_preproc,
        **noise_kwargs,
    )


def _make_loader(dataset: StaticShiftSourceENSUREDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _make_model(args: argparse.Namespace, device: torch.device) -> TemporalNormUnet:
    return TemporalNormUnet(
        num_frames=args.window_size,
        chans=args.chans,
        num_pools=args.num_pools,
        drop_prob=args.drop_prob,
        residual=not args.no_residual,
        output_mode="all_frames",
        num_unrolls=args.num_unrolls,
        denoiser_sharing=args.denoiser_sharing,
    ).to(device)


def _meta_values(meta: Mapping[str, Any], key: str, batch_size: int) -> list[Any]:
    value = meta.get(key, "")
    if isinstance(value, (list, tuple)):
        values = list(value)
    elif torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    else:
        values = [value] * batch_size
    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    if len(values) != batch_size:
        raise ValueError(f"meta[{key!r}] has {len(values)} values for batch_size={batch_size}")
    return values


def _sample_seeds(batch: Mapping[str, Any], args: argparse.Namespace) -> list[int]:
    kspace = batch["kspace_us"]
    batch_size = int(kspace.shape[0]) if kspace.ndim == 5 else 1
    meta = batch.get("meta", {})
    if not isinstance(meta, Mapping):
        return [stable_seed(args.ssdu_seed, idx) for idx in range(batch_size)]
    sample_ids = _meta_values(meta, "sample_id", batch_size)
    volume_ids = _meta_values(meta, "volume_id", batch_size)
    slice_ids = _meta_values(meta, "slice_id", batch_size)
    return [
        stable_seed(args.ssdu_seed, sample_ids[idx], volume_ids[idx], slice_ids[idx])
        for idx in range(batch_size)
    ]


def _compute_train_loss(
    model: TemporalNormUnet,
    batch: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    return compute_ssdu_loss(
        model,
        batch,
        rho=args.ssdu_rho,
        mask_type=args.ssdu_mask_type,
        small_acs_block=tuple(int(v) for v in args.ssdu_small_acs_block),
        seed=args.ssdu_seed,
        sample_seeds=_sample_seeds(batch, args),
        gaussian_std_scale=args.ssdu_gaussian_std_scale,
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


def train_one_epoch(
    model: TemporalNormUnet,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    stats = RunningStats()
    noise_stats = RunningStats()
    observed_target_in_train = False

    progress = tqdm(loader, desc="train", leave=False)
    for step_idx, batch in enumerate(progress):
        if args.max_train_steps is not None and step_idx >= args.max_train_steps:
            break
        batch = move_to_device(batch, device)
        batch = maybe_center_crop_batch(batch, crop_height=args.crop_height, crop_width=args.crop_width)
        observed_target_in_train = observed_target_in_train or ("target_rss" in batch)
        batch_noise_stats = input_noise_stats_from_batch(batch, "train_input_noise")
        if batch_noise_stats:
            noise_stats.update(batch_noise_stats)

        loss_dict = _compute_train_loss(model, batch, args)
        loss = loss_dict["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach())
        l1_value = float(loss_dict["ssdu_l1"].detach())
        l2_value = float(loss_dict["ssdu_l2"].detach())
        frac_value = float(loss_dict["loss_mask_fraction"].detach())
        stats.update(
            {
                "loss": loss_value,
                "ssdu_l1": l1_value,
                "ssdu_l2": l2_value,
                "loss_mask_fraction": frac_value,
            }
        )
        progress.set_postfix(
            loss=f"{loss_value:.6f}",
            avg_loss=f"{stats.totals['loss'] / stats.count:.6f}",
            l1=f"{l1_value:.6f}",
            l2=f"{l2_value:.6f}",
            refresh=False,
        )

    out = stats.averages()
    out.update(noise_stats.averages())
    out["num_steps"] = stats.count
    out["observed_target_in_train"] = float(observed_target_in_train)
    return out


def validate(
    model: TemporalNormUnet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    stats = RunningStats()
    noise_stats = RunningStats()

    with torch.no_grad():
        progress = tqdm(loader, desc="val", leave=False)
        for batch_idx, batch in enumerate(progress):
            if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                break
            batch = move_to_device(batch, device)
            batch = maybe_center_crop_batch(batch, crop_height=args.crop_height, crop_width=args.crop_width)
            if "target_rss" not in batch:
                raise KeyError("Validation requires source_val target_rss.")
            batch_noise_stats = input_noise_stats_from_batch(batch, "val_input_noise")
            if batch_noise_stats:
                noise_stats.update(batch_noise_stats)
            prediction = model(batch["zf"], maps=batch.get("maps"), mask=batch.get("mask"))
            target = batch["target_rss"]
            metrics = summarize_reconstruction_metrics(
                _align_prediction_to_target(prediction, target),
                target,
                include_ssim=args.include_ssim,
            )
            stats.update(metrics)
            progress.set_postfix(
                nmse=f"{metrics['nmse']:.6f}",
                avg_nmse=f"{stats.totals['nmse'] / stats.count:.6f}",
                refresh=False,
            )

    out = stats.averages()
    out.update(noise_stats.averages())
    out["num_batches"] = stats.count
    return out


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.window_size != 1:
        raise ValueError("Static shift SSDU training requires --window-size 1")
    if args.best_val_metric == "noisy_nmse" and args.val_noise_snr_db is None:
        raise ValueError("--best-val-metric noisy_nmse requires --val-noise-snr-db.")

    configure_torch()
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_dataset = _make_dataset(args, "train", args.train_deterministic_masks, return_target=False)
    val_dataset = _make_dataset(args, "val", args.val_deterministic_masks, return_target=True)
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    val_loader = _make_loader(val_dataset, args, shuffle=False)
    save_json(
        output_dir / "source_split.json",
        {
            "train": train_dataset.split_summary(),
            "val": val_dataset.split_summary(),
        },
    )

    model = _make_model(args, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config["device"] = str(device)
    config["training_objective"] = "ssdu"
    config["train_dataset_len"] = len(train_dataset)
    config["val_dataset_len"] = len(val_dataset)
    config["num_parameters"] = count_trainable_parameters(model)
    config["ssdu_small_acs_block"] = [int(v) for v in args.ssdu_small_acs_block]
    save_json(output_dir / "config.json", config)

    history: list[Dict[str, Any]] = []
    best_val_nmse = float("inf")
    best_val_score = float("inf")

    for epoch_idx in range(args.epochs):
        train_dataset.set_noise_epoch(epoch_idx)
        train_stats = train_one_epoch(model, train_loader, optimizer, device, args)
        val_stats = validate(model, val_loader, device, args)
        current_val_score = float(val_stats["nmse"])

        epoch_record = {
            "epoch": epoch_idx,
            "train_loss": train_stats["loss"],
            "train_ssdu_l1": train_stats["ssdu_l1"],
            "train_ssdu_l2": train_stats["ssdu_l2"],
            "train_loss_mask_fraction": train_stats["loss_mask_fraction"],
            "val_nmse": val_stats["nmse"],
            "val_nrmse": val_stats["nrmse"],
            "val_psnr": val_stats["psnr"],
            "val_frame_diff_nmse": val_stats["frame_diff_nmse"],
            "train_steps": train_stats["num_steps"],
            "val_batches": val_stats["num_batches"],
            "observed_target_in_train": bool(train_stats["observed_target_in_train"]),
            "num_unrolls": int(args.num_unrolls),
            "best_val_metric": args.best_val_metric,
            "best_val_score": current_val_score,
        }
        for key, value in train_stats.items():
            if key.startswith("train_input_noise_"):
                epoch_record[key] = value
        for key, value in val_stats.items():
            if key.startswith("val_input_noise_"):
                epoch_record[key] = value
        if "ssim" in val_stats:
            epoch_record["val_ssim"] = val_stats["ssim"]
        history.append(epoch_record)

        print(
            f"[shift-ssdu] epoch={epoch_idx} "
            f"train_loss={epoch_record['train_loss']:.6f} "
            f"l1={epoch_record['train_ssdu_l1']:.6f} "
            f"l2={epoch_record['train_ssdu_l2']:.6f} "
            f"val_nmse={epoch_record['val_nmse']:.6f} "
            f"best_metric={args.best_val_metric}:{current_val_score:.6f} "
            f"unrolls={int(args.num_unrolls)}"
        )

        if current_val_score < best_val_score:
            best_val_score = current_val_score
            best_val_nmse = val_stats["nmse"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_idx,
                history=history,
                config=config,
                extra={
                    "best_val_nmse": best_val_nmse,
                    "best_val_score": best_val_score,
                    "best_val_metric": args.best_val_metric,
                },
            )

        if (epoch_idx + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch_idx:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_idx,
                history=history,
                config=config,
                extra={
                    "best_val_nmse": best_val_nmse,
                    "best_val_score": best_val_score,
                    "best_val_metric": args.best_val_metric,
                },
            )

        save_json(
            output_dir / "history.json",
            {
                "history": history,
                "best_val_nmse": best_val_nmse,
                "best_val_score": best_val_score,
                "best_val_metric": args.best_val_metric,
            },
        )

    summary = {
        "best_val_nmse": best_val_nmse,
        "best_val_score": best_val_score,
        "best_val_metric": args.best_val_metric,
        "history": history,
        "output_dir": str(output_dir),
        "training_objective": "ssdu",
        "ssdu_rho": float(args.ssdu_rho),
        "ssdu_mask_type": args.ssdu_mask_type,
        "ssdu_small_acs_block": [int(v) for v in args.ssdu_small_acs_block],
        "num_unrolls": int(args.num_unrolls),
        "train_dataset_returns_target": bool(train_dataset.return_target),
        "val_dataset_returns_target": bool(val_dataset.return_target),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
