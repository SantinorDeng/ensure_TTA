#!/usr/bin/env python3
"""Prepare, train, or run L1-TTA for one fastMRI modality-matrix job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "manifests" / "shifts" / "modality_matrix"
PREPROC_ROOT = ROOT / "preproc" / "shifts" / "modality_matrix"
DENSITY_ROOT = ROOT / "density_stats" / "shifts" / "modality_matrix"
TRAIN_OUTPUT_ROOT = ROOT / "outputs" / "shifts" / "modality_matrix"
TTA_OUTPUT_ROOT = ROOT / "outputs" / "tta" / "shifts" / "modality_matrix"

SOURCES = {
    "brain": ("axflair", "axt1", "axt1pre", "axt1post", "axt2"),
    "knee": ("pd", "pdfs"),
}
TRAINING_OBJECTIVES = {
    "ensure": "true_ensure",
    "traditional": "normalized_l1_supervised_plus_measured_kspace_self_supervision",
}
TRAIN_TAG = "r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007"


def manifest_path(dataset: str, source: str, tier: str = "main") -> Path:
    validate_dataset_source(dataset, source)
    return MANIFEST_ROOT / tier / dataset / f"{source}.csv"


def train_output_dir(dataset: str, source: str, method: str, tier: str = "main") -> Path:
    return TRAIN_OUTPUT_ROOT / tier / dataset / source / f"{method}_{TRAIN_TAG}"


def _snr_tag(value: float | None) -> str:
    if value is None:
        return "clean"
    return f"snr{float(value):g}".replace(".", "p")


def tta_output_dir(
    dataset: str,
    source: str,
    method: str,
    *,
    tier: str,
    snr_db: float | None,
    eval_seed: int,
    noise_seed: int,
    target_shift_names: Sequence[str],
) -> Path:
    tag = f"{method}_l1_{_snr_tag(snr_db)}_maskseed{eval_seed}_noiseseed{noise_seed}"
    if target_shift_names:
        digest = hashlib.sha256("\n".join(sorted(target_shift_names)).encode("utf-8")).hexdigest()[:8]
        tag = f"{tag}_subset{digest}"
    return TTA_OUTPUT_ROOT / tier / dataset / source / tag


def validate_dataset_source(dataset: str, source: str) -> None:
    if dataset not in SOURCES:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(SOURCES)}")
    if source not in SOURCES[dataset]:
        raise ValueError(f"Unknown source {dataset}:{source}; choose from {SOURCES[dataset]}")


def validate_cuda_device(device: str) -> None:
    match = re.fullmatch(r"cuda:(\d+)", str(device))
    if match is None:
        raise ValueError("Formal modality jobs require --device cuda:N")
    import torch

    index = int(match.group(1))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to fall back to CPU")
    if index >= torch.cuda.device_count():
        raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible")


def _format_command(command: Sequence[object]) -> str:
    return " ".join(str(part) for part in command)


def run_command(
    command: Sequence[object],
    *,
    log_path: Path | None,
    dry_run: bool,
) -> int:
    printable = _format_command(command)
    print(printable, flush=True)
    if dry_run:
        return 0
    if log_path is None:
        return subprocess.run([str(part) for part in command], check=True).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        return subprocess.run(
            [str(part) for part in command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        ).returncode


def completed_training(output_dir: Path) -> bool:
    return (output_dir / "best.pt").is_file() and (output_dir / "summary.json").is_file()


def completed_tta(output_dir: Path) -> bool:
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.csv"
    if not summary_path.is_file() or not metrics_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(summary.get("num_ok", 0)) > 0 and int(summary.get("num_failed", 1)) == 0


def prepare(args: argparse.Namespace) -> int:
    build_cmd = [
        sys.executable,
        ROOT / "scripts" / "build_shift_source_manifests.py",
        "--tiers", args.tier,
        "--modality-matrices", "brain_modality_matrix", "knee_modality_matrix",
        "--only-modality-matrices",
    ]
    run_command(build_cmd, log_path=None, dry_run=args.dry_run)

    density_cmd = [
        sys.executable,
        ROOT / "datasets" / "precompute_density_stats.py",
        "--output-root", DENSITY_ROOT,
        "--shape", "640x320", "640x368",
        "--accelerations", "4",
        "--sigma-mask", "0.18",
        "--num-samples", "1024",
        "--seed", "0",
    ]
    if args.overwrite_density:
        density_cmd.append("--overwrite")
    run_command(density_cmd, log_path=None, dry_run=args.dry_run)

    if args.skip_preproc:
        return 0
    for dataset, sources in SOURCES.items():
        for source in sources:
            preproc_cmd = [
                sys.executable,
                ROOT / "datasets" / "preprocess_shift_manifest_source.py",
                "--manifest-csv", manifest_path(dataset, source, args.tier),
                "--output-root", PREPROC_ROOT,
                "--source-role", "source_train",
                "--map-method", "rss",
                "--norm-source", "reconstruction_rss",
                "--norm-percentile", "99",
                "--corner-fraction", "0.08",
            ]
            if args.overwrite_preproc:
                preproc_cmd.append("--overwrite")
            log_path = ROOT / "outputs" / "logs" / "modality_matrix" / "prepare" / f"{dataset}_{source}.log"
            run_command(preproc_cmd, log_path=log_path, dry_run=args.dry_run)
    return 0


def train(args: argparse.Namespace) -> int:
    validate_dataset_source(args.dataset, args.source)
    output_dir = train_output_dir(args.dataset, args.source, args.method, args.tier)
    if args.resume and completed_training(output_dir):
        print(f"[skip complete] {output_dir}")
        return 0
    if not args.dry_run:
        validate_cuda_device(args.device)
        if not manifest_path(args.dataset, args.source, args.tier).is_file():
            raise FileNotFoundError("Missing modality manifest; run the prepare subcommand first")

    script = "train_shift_true_ensure.py" if args.method == "ensure" else "train_supervised_baseline.py"
    command: list[object] = [
        sys.executable,
        ROOT / "scripts" / script,
        "--manifest-csv", manifest_path(args.dataset, args.source, args.tier),
        "--preproc-root", PREPROC_ROOT,
        "--density-root", DENSITY_ROOT,
        "--output-dir", output_dir,
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
        "--num-workers", args.num_workers,
        "--require-preproc",
    ]
    return run_command(command, log_path=output_dir / "run.log", dry_run=args.dry_run)


def tta(args: argparse.Namespace) -> int:
    validate_dataset_source(args.dataset, args.source)
    output_dir = tta_output_dir(
        args.dataset,
        args.source,
        args.method,
        tier=args.tier,
        snr_db=args.test_noise_snr_db,
        eval_seed=args.eval_seed,
        noise_seed=args.test_noise_seed,
        target_shift_names=args.target_shift_names,
    )
    if args.resume and completed_tta(output_dir):
        print(f"[skip complete] {output_dir}")
        return 0
    checkpoint = train_output_dir(args.dataset, args.source, args.method, args.tier) / "best.pt"
    if not args.dry_run:
        validate_cuda_device(args.device)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    command: list[object] = [
        sys.executable,
        ROOT / "scripts" / "tta_shift_true_ensure.py",
        "--checkpoint", checkpoint,
        "--manifest-csv", manifest_path(args.dataset, args.source, args.tier),
        "--output-dir", output_dir,
        "--split-role", "target_test",
        "--device", args.device,
        "--seed", args.eval_seed,
        "--test-noise-seed", args.test_noise_seed,
        "--training-objective", TRAINING_OBJECTIVES[args.method],
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
    for shift_name in args.target_shift_names:
        command.extend(["--target-shift-name", shift_name])
    if args.max_samples is not None:
        command.extend(["--max-samples", args.max_samples])
    return run_command(command, log_path=output_dir / "run.log", dry_run=args.dry_run)


def parse_snr(value: str) -> float | None:
    if value.lower() == "clean":
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("SNR must be positive or 'clean'")
    return parsed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build manifests, density statistics, and source sidecars.")
    prepare_parser.add_argument("--tier", choices=("debug", "pilot", "main"), default="main")
    prepare_parser.add_argument("--skip-preproc", action="store_true")
    prepare_parser.add_argument("--overwrite-density", action="store_true")
    prepare_parser.add_argument("--overwrite-preproc", action="store_true")
    prepare_parser.add_argument("--dry-run", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    for name, function in (("train", train), ("tta", tta)):
        sub = subparsers.add_parser(name)
        sub.add_argument("--dataset", choices=sorted(SOURCES), required=True)
        sub.add_argument("--source", required=True)
        sub.add_argument("--method", choices=sorted(TRAINING_OBJECTIVES), required=True)
        sub.add_argument("--device", required=True, help="Explicit physical CUDA device, e.g. cuda:3")
        sub.add_argument("--tier", choices=("debug", "pilot", "main"), default="main")
        sub.add_argument("--num-workers", type=int, default=0)
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--dry-run", action="store_true")
        sub.set_defaults(func=function)

    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument(
        "--denoiser-sharing",
        choices=("shared", "independent"),
        default="shared",
        help="Reuse one denoiser across unrolls or train one denoiser per unroll.",
    )

    tta_parser = subparsers.choices["tta"]
    tta_parser.add_argument("--test-noise-snr-db", type=parse_snr, default=20.0)
    tta_parser.add_argument("--test-noise-seed", type=int, default=9007)
    tta_parser.add_argument("--eval-seed", type=int, default=7)
    tta_parser.add_argument("--target-shift-name", dest="target_shift_names", action="append", default=[])
    tta_parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
