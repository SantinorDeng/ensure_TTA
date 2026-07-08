#!/usr/bin/env python3
"""Queue modality-matrix train or L1-TTA jobs on user-selected GPUs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = ROOT / "scripts" / "run_modality_matrix_job.py"
SOURCES = {
    "brain": ("axflair", "axt1", "axt1pre", "axt1post", "axt2"),
    "knee": ("pd", "pdfs"),
}


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--devices must be comma-separated GPU indices") from exc
    if not devices or len(devices) != len(set(devices)) or min(devices) < 0:
        raise argparse.ArgumentTypeError("--devices must contain unique non-negative GPU indices")
    return devices


def selected_sources(tokens: Sequence[str], datasets: Sequence[str]) -> list[tuple[str, str]]:
    if not tokens:
        return [(dataset, source) for dataset in datasets for source in SOURCES[dataset]]
    selected: list[tuple[str, str]] = []
    for token in tokens:
        try:
            dataset, source = token.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid --source {token!r}; use dataset:source") from exc
        if dataset not in SOURCES or source not in SOURCES[dataset]:
            raise ValueError(f"Invalid source {token!r}")
        if dataset not in datasets:
            raise ValueError(f"Source {token!r} is outside --datasets")
        selected.append((dataset, source))
    return selected


def free_memory_mib() -> dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result: dict[int, int] = {}
    for line in output.splitlines():
        index, free = [token.strip() for token in line.split(",", 1)]
        result[int(index)] = int(free)
    return result


def job_commands(args: argparse.Namespace) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for dataset, source in selected_sources(args.sources, args.datasets):
        for method in args.methods:
            base: list[object] = [
                sys.executable,
                JOB_SCRIPT,
                args.stage,
                "--dataset", dataset,
                "--source", source,
                "--method", method,
                "--tier", args.tier,
                "--num-workers", args.num_workers,
            ]
            if args.resume:
                base.append("--resume")
            if args.stage == "train":
                base.extend(
                    [
                        "--epochs", args.epochs,
                        "--denoiser-sharing", args.denoiser_sharing,
                    ]
                )
            else:
                base.extend(
                    [
                        "--test-noise-snr-db", args.test_noise_snr_db,
                        "--test-noise-seed", args.test_noise_seed,
                        "--eval-seed", args.eval_seed,
                    ]
                )
                for shift_name in args.target_shift_names:
                    base.extend(["--target-shift-name", shift_name])
                if args.max_samples is not None:
                    base.extend(["--max-samples", args.max_samples])
            jobs.append(
                {
                    "id": f"{args.stage}:{dataset}:{source}:{method}",
                    "dataset": dataset,
                    "source": source,
                    "method": method,
                    "base_command": [str(part) for part in base],
                    "status": "pending",
                }
            )
    return jobs


def write_state(path: Path, jobs: Sequence[dict[str, object]], devices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "devices": list(devices),
        "jobs": list(jobs),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_command(command: Sequence[str]) -> str:
    return " ".join(command)


def run_queue(args: argparse.Namespace) -> int:
    devices = parse_devices(args.devices)
    jobs = job_commands(args)
    state_path = args.state_file or (
        ROOT / "outputs" / "job_state" / f"modality_{args.stage}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    if args.dry_run:
        for index, job in enumerate(jobs):
            device = devices[index % len(devices)]
            command = [*job["base_command"], "--device", f"cuda:{device}", "--dry-run"]
            print(format_command(command))
        return 0

    running: dict[int, tuple[subprocess.Popen[bytes], dict[str, object]]] = {}
    threshold_mib = int(round(float(args.min_free_memory_gb) * 1024.0))
    failed = False
    write_state(state_path, jobs, devices)

    while any(job["status"] == "pending" for job in jobs) or running:
        for device, (process, job) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            job["returncode"] = int(returncode)
            job["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
            job["status"] = "completed" if returncode == 0 else "failed"
            failed = failed or returncode != 0
            del running[device]
            write_state(state_path, jobs, devices)

        free = free_memory_mib()
        pending = [job for job in jobs if job["status"] == "pending"]
        for device in devices:
            if not pending or device in running:
                continue
            if device not in free:
                raise RuntimeError(f"GPU {device} is not reported by nvidia-smi")
            if free[device] < threshold_mib:
                continue
            job = pending.pop(0)
            command = [*job["base_command"], "--device", f"cuda:{device}"]
            print(f"[launch gpu={device} free={free[device]} MiB] {format_command(command)}", flush=True)
            process = subprocess.Popen(command)
            job["status"] = "running"
            job["device"] = f"cuda:{device}"
            job["pid"] = int(process.pid)
            job["started_at"] = dt.datetime.now().isoformat(timespec="seconds")
            running[device] = (process, job)
            write_state(state_path, jobs, devices)

        if any(job["status"] == "pending" for job in jobs) or running:
            time.sleep(max(float(args.poll_seconds), 1.0))
    print(f"State: {state_path}")
    return 1 if failed else 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "tta"), required=True)
    parser.add_argument("--devices", required=True, help="Physical GPU indices, e.g. 0,2,5,7")
    parser.add_argument("--datasets", nargs="+", choices=sorted(SOURCES), default=list(SOURCES))
    parser.add_argument("--source", dest="sources", action="append", default=[], help="dataset:source; repeat as needed")
    parser.add_argument("--methods", nargs="+", choices=("ensure", "traditional"), default=["ensure", "traditional"])
    parser.add_argument("--tier", choices=("debug", "pilot", "main"), default="main")
    parser.add_argument("--min-free-memory-gb", type=float, default=18.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--denoiser-sharing",
        choices=("shared", "independent"),
        default="shared",
        help="Training-only denoiser sharing mode.",
    )
    parser.add_argument("--test-noise-snr-db", default="20")
    parser.add_argument("--test-noise-seed", type=int, default=9007)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--target-shift-name", dest="target_shift_names", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.min_free_memory_gb < 0:
        raise ValueError("--min-free-memory-gb must be non-negative")
    return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())
