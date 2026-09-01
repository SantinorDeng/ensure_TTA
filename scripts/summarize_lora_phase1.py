#!/usr/bin/env python3
"""Summarize the clean R4 Conv-LoRA phase-1 experiment against full TTA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


DEFAULT_ROOT = Path(
    "outputs/tta/shifts/main/modality_shift/lora_phase1_clean_r4_lr1e-3"
)
DEFAULT_FULL_TTA = Path(
    "outputs/tta/shifts/main/modality_shift/"
    "true_ensure_source_r4_w1_auto_unroll_seed7_self_supervised_loss_strict_runtime"
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--full-tta-dir", type=Path, default=DEFAULT_FULL_TTA)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{path} contains {len(failed)} failed rows")
    return rows


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def median(rows: list[dict[str, str]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = build_argparser().parse_args()
    output = args.output or args.experiment_root / "phase1_comparison.md"
    methods = {
        "Full-parameter TTA": args.full_tta_dir,
        "Conv-LoRA R1": args.experiment_root / "convlora_r1",
        "Conv-LoRA R2": args.experiment_root / "convlora_r2",
        "Conv-LoRA R4": args.experiment_root / "convlora_r4",
        "Conv-LoRA R2 + DC": args.experiment_root / "convlora_r2_dc",
    }

    rows_by_method = {name: read_rows(path / "metrics.csv") for name, path in methods.items()}
    payloads: dict[str, dict[str, Any]] = {
        name: json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for name, path in methods.items()
    }
    # Older baseline outputs predate hash columns.  Hash their recorded files
    # directly and compare them with the hashes emitted by the new runs.
    baseline_payload = payloads["Full-parameter TTA"]
    checkpoint_hashes = {
        file_sha256(Path(baseline_payload["checkpoint"])),
        *(
            row["checkpoint_sha256"]
            for name, rows in rows_by_method.items()
            if name != "Full-parameter TTA"
            for row in rows
        ),
    }
    manifest_hashes = {
        file_sha256(Path(baseline_payload["manifest_csv"])),
        *(
            row["manifest_sha256"]
            for name, rows in rows_by_method.items()
            if name != "Full-parameter TTA"
            for row in rows
        ),
    }
    if len(checkpoint_hashes) != 1 or len(manifest_hashes) != 1:
        raise RuntimeError("Compared runs do not share the same checkpoint and manifest hashes")

    base_rows = rows_by_method["Full-parameter TTA"]
    base_by_id = {row["sample_id"]: row for row in base_rows}
    base_params = int(base_rows[0]["num_trainable_params"])

    lines = [
        "# Conv-LoRA phase-1: modality R4 clean",
        "",
        "Protocol: fastMRI brain AXT2 → AXT1PRE, 100 target slices, nominal R=4, clean k-space, "
        "TRUE-ENSURE source checkpoint, measured-kspace normalized complex L1 TTA, 250 maximum steps, "
        "5% held-out self-validation, seed 7.",
        "",
        f"Checkpoint SHA256: `{next(iter(checkpoint_hashes))}`",
        "",
        f"Manifest SHA256: `{next(iter(manifest_hashes))}`",
        "",
        "## Main results",
        "",
        "| method | trainable params | vs full | mean NMSE | median NMSE | PSNR | SSIM | ΔPSNR | mean steps | adapt sec/slice | negative adaptation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rows in rows_by_method.items():
        params = int(rows[0]["num_trainable_params"])
        negative = statistics.mean(
            float(row["after_tta_nmse"]) > float(row["before_tta_nmse"]) for row in rows
        )
        lines.append(
            f"| {name} | {params:,} | {100.0 * params / base_params:.2f}% | "
            f"{mean(rows, 'after_tta_nmse'):.6f} | {median(rows, 'after_tta_nmse'):.6f} | "
            f"{mean(rows, 'after_tta_psnr'):.4f} | {mean(rows, 'after_tta_ssim'):.4f} | "
            f"{mean(rows, 'delta_psnr'):+.4f} | {mean(rows, 'num_tta_steps'):.1f} | "
            f"{mean(rows, 'adapt_runtime_sec'):.2f} | {100.0 * negative:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Paired comparison against full-parameter TTA",
            "",
            "| method | same before max abs diff | NMSE win rate | PSNR win rate | mean NMSE difference | mean PSNR difference |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, rows in list(rows_by_method.items())[1:]:
        current = {row["sample_id"]: row for row in rows}
        ids = sorted(base_by_id.keys() & current.keys())
        if len(ids) != 100:
            raise RuntimeError(f"{name} has only {len(ids)} samples paired with full TTA")
        before_diff = max(
            abs(float(current[key]["before_tta_nmse"]) - float(base_by_id[key]["before_tta_nmse"]))
            for key in ids
        )
        nmse_win = statistics.mean(
            float(current[key]["after_tta_nmse"]) < float(base_by_id[key]["after_tta_nmse"])
            for key in ids
        )
        psnr_win = statistics.mean(
            float(current[key]["after_tta_psnr"]) > float(base_by_id[key]["after_tta_psnr"])
            for key in ids
        )
        nmse_diff = statistics.mean(
            float(current[key]["after_tta_nmse"]) - float(base_by_id[key]["after_tta_nmse"])
            for key in ids
        )
        psnr_diff = statistics.mean(
            float(current[key]["after_tta_psnr"]) - float(base_by_id[key]["after_tta_psnr"])
            for key in ids
        )
        lines.append(
            f"| {name} | {before_diff:.3g} | {100.0 * nmse_win:.1f}% | {100.0 * psnr_win:.1f}% | "
            f"{nmse_diff:+.6f} | {psnr_diff:+.4f} dB |"
        )

    r2 = {row["sample_id"]: row for row in rows_by_method["Conv-LoRA R2"]}
    r2_dc = {row["sample_id"]: row for row in rows_by_method["Conv-LoRA R2 + DC"]}
    paired_ids = sorted(r2.keys() & r2_dc.keys())
    lines.extend(
        [
            "",
            "## Effect of adapting the 12 DC scalars (R2 + DC versus R2)",
            "",
            f"- NMSE win rate: {100.0 * statistics.mean(float(r2_dc[key]['after_tta_nmse']) < float(r2[key]['after_tta_nmse']) for key in paired_ids):.1f}%",
            f"- PSNR win rate: {100.0 * statistics.mean(float(r2_dc[key]['after_tta_psnr']) > float(r2[key]['after_tta_psnr']) for key in paired_ids):.1f}%",
            f"- Mean NMSE difference: {statistics.mean(float(r2_dc[key]['after_tta_nmse']) - float(r2[key]['after_tta_nmse']) for key in paired_ids):+.6f}",
            f"- Median NMSE difference: {statistics.median(float(r2_dc[key]['after_tta_nmse']) - float(r2[key]['after_tta_nmse']) for key in paired_ids):+.6f}",
            f"- Mean PSNR difference: {statistics.mean(float(r2_dc[key]['after_tta_psnr']) - float(r2[key]['after_tta_psnr']) for key in paired_ids):+.4f} dB",
            "",
            "## Validation",
            "",
            "| method | metric rows | recon files | curve files | checkpoint/manifest match |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name, path in methods.items():
        lines.append(
            f"| {name} | {len(rows_by_method[name])} | {count_files(path / 'recons', '*.npz')} | "
            f"{count_files(path / 'curves', '*.json')} | yes |"
        )

    lines.extend(["", "## Negative-adaptation samples", ""])
    for name, rows in rows_by_method.items():
        negative_ids = [
            row["sample_id"]
            for row in rows
            if float(row["after_tta_nmse"]) > float(row["before_tta_nmse"])
        ]
        rendered = ", ".join(f"`{sample_id}`" for sample_id in negative_ids) or "none"
        lines.append(f"- {name}: {rendered}")

    report = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
