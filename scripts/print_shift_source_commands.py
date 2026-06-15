#!/usr/bin/env python3
"""Print source-only shift ENSURE command templates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


SHIFT_NAMES = ("anatomy_shift", "dataset_shift", "modality_shift", "acceleration_shift")


def _cmd(parts: Sequence[object]) -> str:
    return " ".join(str(part) for part in parts)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/home/hulabdl/Deng_proj"))
    parser.add_argument("--tier", choices=("debug", "pilot", "main"), default="debug")
    parser.add_argument("--env-name", default="uncertainty_tta")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--shifts", nargs="+", choices=SHIFT_NAMES, default=list(SHIFT_NAMES))
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--skip-build-command", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cardiac_root = args.project_root / "cardiac_ensure"
    manifest_root = cardiac_root / "manifests" / "shifts" / args.tier
    preproc_root = cardiac_root / "preproc" / "shifts" / args.tier
    density_root = cardiac_root / "density_stats" / "shifts" / args.tier
    output_root = cardiac_root / "outputs" / "shifts" / args.tier
    epochs = args.epochs if args.epochs is not None else (1 if args.tier == "debug" else 50)

    if not args.skip_build_command:
        print("# build cardiac_ensure manifests")
        print(
            _cmd(
                [
                    "conda run -n",
                    args.env_name,
                    "python",
                    cardiac_root / "scripts" / "build_shift_source_manifests.py",
                    "--tiers",
                    args.tier,
                ]
            )
        )
        print()

    for shift_name in args.shifts:
        manifest_csv = manifest_root / f"{shift_name}.csv"
        train_cmd = [
            "conda run -n",
            args.env_name,
            "python",
            cardiac_root / "scripts" / "train_shift_true_ensure.py",
            "--manifest-csv",
            manifest_csv,
            "--preproc-root",
            preproc_root,
            "--density-root",
            density_root,
            "--output-dir",
            output_root / shift_name / "true_ensure_source_r4_w1_auto_unroll_seed7",
            "--epochs",
            epochs,
            "--num-unrolls 12",
            "--run-memory-probe",
            "--probe-unrolls 12 10 8 6 3",
            "--device",
            args.device,
        ]
        if args.max_train_steps is not None:
            train_cmd.extend(["--max-train-steps", args.max_train_steps])
        if args.max_val_batches is not None:
            train_cmd.extend(["--max-val-batches", args.max_val_batches])

        print(f"# {shift_name}")
        print(
            _cmd(
                [
                    "conda run -n",
                    args.env_name,
                    "python",
                    cardiac_root / "datasets" / "preprocess_shift_manifest_source.py",
                    "--manifest-csv",
                    manifest_csv,
                    "--output-root",
                    preproc_root,
                    "--map-method rss",
                    "--compression None",
                    "--overwrite",
                ]
            )
        )
        print(
            _cmd(
                [
                    "conda run -n",
                    args.env_name,
                    "python",
                    cardiac_root / "datasets" / "precompute_density_stats.py",
                    "--output-root",
                    density_root,
                    "--manifest-csv",
                    manifest_csv,
                    "--accelerations 4.0",
                    "--sigma-mask 0.18",
                    "--num-samples 1024",
                    "--overwrite",
                ]
            )
        )
        print(
            _cmd(train_cmd)
        )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
