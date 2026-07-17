#!/usr/bin/env python3
"""Generate or run main-shift commands for zero-filled, BART, and SSDU baselines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
MAIN_SHIFTS = ("acceleration_shift", "modality_shift", "anatomy_shift")
TRAIN_NOISE_TAG = "train_noise_snr15_25_val20_seed7007"


def _token(value: float | int | None) -> str:
    if value is None:
        return "clean"
    return f"{float(value):g}".replace(".", "p")


def _format_command(command: Sequence[object]) -> str:
    return " ".join(str(part) for part in command)


def run_command(command: Sequence[object], *, dry_run: bool) -> int:
    print(_format_command(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run([str(part) for part in command], check=True).returncode


def manifest_path(shift: str, tier: str) -> Path:
    return ROOT / "manifests" / "shifts" / tier / f"{shift}.csv"


def preproc_root(tier: str) -> Path:
    return ROOT / "preproc" / "shifts" / tier


def density_root(tier: str) -> Path:
    return ROOT / "density_stats" / "shifts" / tier


def classical_output_dir(method: str, shift: str, tier: str, *, seed: int, snr_db: float | None, lamda: float) -> Path:
    noise = _token(snr_db)
    if method == "zero_filled":
        tag = f"zero_filled_maskseed{seed}_{noise}"
    else:
        tag = f"bart_pics_lam{_token(lamda)}_maskseed{seed}_{noise}"
    return ROOT / "outputs" / "baselines" / "shifts" / tier / shift / tag


def ssdu_train_output_dir(shift: str, tier: str) -> Path:
    return ROOT / "outputs" / "shifts" / tier / shift / f"ssdu_source_r4_w1_unroll12_seed7_{TRAIN_NOISE_TAG}"


def ssdu_tta_output_dir(shift: str, tier: str, *, snr_db: float | None) -> Path:
    return (
        ROOT
        / "outputs"
        / "tta"
        / "shifts"
        / tier
        / shift
        / f"ssdu_source_r4_w1_unroll12_seed7_{TRAIN_NOISE_TAG}_l1_noise_{_token(snr_db)}"
    )


def classical_command(args: argparse.Namespace, shift: str, method: str) -> list[object]:
    command: list[object] = [
        sys.executable,
        ROOT / "scripts" / "eval_classical_baselines.py",
        "--method", method,
        "--manifest-csv", manifest_path(shift, args.tier),
        "--output-dir", classical_output_dir(
            method,
            shift,
            args.tier,
            seed=args.seed,
            snr_db=args.test_noise_snr_db,
            lamda=args.bart_lambda,
        ),
        "--split-role", "target_test",
        "--preproc-root", preproc_root(args.tier),
        "--density-root", density_root(args.tier),
        "--device", args.device,
        "--seed", args.seed,
        "--test-noise-seed", args.test_noise_seed,
        "--include-ssim",
        "--save-recons",
        "--num-workers", args.num_workers,
    ]
    if args.test_noise_snr_db is not None:
        command.extend(["--test-noise-snr-db", args.test_noise_snr_db])
    if method == "bart_pics_cs":
        command.extend(["--bart-lambda", args.bart_lambda])
        if args.bart_toolbox_path:
            command.extend(["--bart-toolbox-path", args.bart_toolbox_path])
        if args.bart_python_path:
            command.extend(["--bart-python-path", args.bart_python_path])
    if args.max_samples is not None:
        command.extend(["--max-samples", args.max_samples])
    return command


def ssdu_train_command(args: argparse.Namespace, shift: str) -> list[object]:
    command: list[object] = [
        sys.executable,
        ROOT / "scripts" / "train_shift_ssdu.py",
        "--manifest-csv", manifest_path(shift, args.tier),
        "--preproc-root", preproc_root(args.tier),
        "--density-root", density_root(args.tier),
        "--output-dir", ssdu_train_output_dir(shift, args.tier),
        "--epochs", args.epochs,
        "--batch-size", "1",
        "--acceleration", "4",
        "--sigma-mask", "0.18",
        "--window-size", "1",
        "--num-unrolls", "12",
        "--denoiser-sharing", args.denoiser_sharing,
        "--chans", "64",
        "--num-pools", "4",
        "--lr", "1e-4",
        "--weight-decay", "1e-4",
        "--grad-clip", "1",
        "--device", args.device,
        "--seed", "7",
        "--split-seed", "7",
        "--include-ssim",
        "--train-noise-snr-db-min", "15",
        "--train-noise-snr-db-max", "25",
        "--train-noise-seed", "7007",
        "--val-noise-snr-db", "20",
        "--val-noise-seed", "8007",
        "--best-val-metric", "noisy_nmse",
        "--ssdu-rho", "0.4",
        "--ssdu-mask-type", "gaussian",
        "--num-workers", args.num_workers,
        "--require-preproc",
    ]
    if args.max_train_steps is not None:
        command.extend(["--max-train-steps", args.max_train_steps])
    if args.max_val_batches is not None:
        command.extend(["--max-val-batches", args.max_val_batches])
    return command


def ssdu_tta_command(args: argparse.Namespace, shift: str) -> list[object]:
    command: list[object] = [
        sys.executable,
        ROOT / "scripts" / "tta_shift_true_ensure.py",
        "--checkpoint", ssdu_train_output_dir(shift, args.tier) / "best.pt",
        "--manifest-csv", manifest_path(shift, args.tier),
        "--output-dir", ssdu_tta_output_dir(shift, args.tier, snr_db=args.test_noise_snr_db),
        "--split-role", "target_test",
        "--device", args.device,
        "--seed", args.seed,
        "--test-noise-seed", args.test_noise_seed,
        "--training-objective", "ssdu",
        "--tta-loss", "l1",
        "--tta-steps", "250",
        "--tta-lr", "1e-5",
        "--tta-weight-decay", "0",
        "--grad-clip", "1",
        "--self-val-fraction", "0.05",
        "--early-stop-window", "20",
        "--update-mode", "all_params",
        "--run-tta",
        "--include-ssim",
        "--save-recons",
        "--save-curves",
        "--num-workers", args.num_workers,
    ]
    if args.test_noise_snr_db is not None:
        command.extend(["--test-noise-snr-db", args.test_noise_snr_db])
    if args.max_samples is not None:
        command.extend(["--max-samples", args.max_samples])
    return command


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("debug", "pilot", "main"), default="main")
    parser.add_argument("--shift", choices=(*MAIN_SHIFTS, "all"), default="all")
    parser.add_argument("--stage", choices=("classical", "train-ssdu", "tta-ssdu", "all"), default="classical")
    parser.add_argument(
        "--method",
        action="append",
        choices=("zero_filled", "bart_pics_cs", "ssdu"),
        default=[],
        help="Repeat to select methods; default runs zero-filled and BART for classical, SSDU for SSDU stages.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-noise-snr-db", type=float, default=20.0)
    parser.add_argument("--test-noise-seed", type=int, default=9007)
    parser.add_argument("--bart-lambda", type=float, default=0.01)
    parser.add_argument("--bart-toolbox-path", default=None)
    parser.add_argument("--bart-python-path", default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--denoiser-sharing", choices=("shared", "independent"), default="shared")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def selected_shifts(value: str) -> tuple[str, ...]:
    return MAIN_SHIFTS if value == "all" else (value,)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    methods = tuple(args.method)
    if not methods:
        methods = ("zero_filled", "bart_pics_cs") if args.stage == "classical" else ("ssdu",)

    for shift in selected_shifts(args.shift):
        if args.stage in {"classical", "all"}:
            for method in methods:
                if method in {"zero_filled", "bart_pics_cs"}:
                    run_command(classical_command(args, shift, method), dry_run=args.dry_run)
        if args.stage in {"train-ssdu", "all"} and "ssdu" in methods:
            run_command(ssdu_train_command(args, shift), dry_run=args.dry_run)
        if args.stage in {"tta-ssdu", "all"} and "ssdu" in methods:
            run_command(ssdu_tta_command(args, shift), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
