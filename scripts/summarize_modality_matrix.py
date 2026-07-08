#!/usr/bin/env python3
"""Aggregate modality-matrix L1-TTA metrics with paired hierarchical bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BRAIN = ("axflair", "axt1", "axt1pre", "axt1post", "axt2")
KNEE = ("pd", "pdfs")
METRICS = ("before_tta_nmse", "after_tta_nmse", "delta_nmse")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: Mapping[str, object], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def objective_key(value: object) -> str:
    text = str(value).lower()
    if text == "true_ensure" or "ensure" in text and "supervised" not in text:
        return "ensure"
    if "supervised" in text or "traditional" in text:
        return "traditional"
    return text or "unknown"


def snr_matches(row: Mapping[str, str], requested: str) -> bool:
    value = str(row.get("test_noise_snr_db", "")).lower()
    if requested.lower() == "clean":
        return value == "clean"
    try:
        return abs(float(value) - float(requested)) < 1e-8
    except ValueError:
        return False


def load_metric_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    deduplicated: dict[tuple[str, ...], dict[str, str]] = {}
    failed: list[dict[str, str]] = []
    for path in sorted(args.input_root.rglob("metrics.csv")):
        for row in read_csv(path):
            if row.get("method") != "measured_kspace_l1_tta":
                continue
            if not snr_matches(row, args.test_noise_snr_db):
                continue
            if int(float(row.get("mask_seed", -1))) != int(args.eval_seed):
                continue
            if int(float(row.get("test_noise_seed", -1))) != int(args.test_noise_seed):
                continue
            if row.get("status") != "ok":
                failed.append(row)
                continue
            row["training_objective_key"] = objective_key(row.get("training_objective", ""))
            key = (
                row["training_objective_key"],
                row.get("shift_name", ""),
                row.get("sample_id", ""),
                str(row.get("test_noise_snr_db", "")),
                str(row.get("mask_seed", "")),
                str(row.get("test_noise_seed", "")),
            )
            deduplicated[key] = row
    return list(deduplicated.values()), failed


def mean(values: Iterable[float | None]) -> float:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def volume_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in rows:
        group_id = row.get("split_group_id") or row.get("patient_id") or row.get("file", "")
        key = (
            row.get("training_objective_key", ""),
            row.get("experiment_family", ""),
            row.get("shift_name", ""),
            row.get("source_acquisition", ""),
            row.get("target_acquisition", ""),
            row.get("is_in_domain_control", ""),
            group_id,
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        objective, family, shift, source, target, control, group_id = key
        item: dict[str, object] = {
            "training_objective": objective,
            "experiment_family": family,
            "shift_name": shift,
            "source_acquisition": source,
            "target_acquisition": target,
            "is_in_domain_control": str(control).lower() == "true",
            "split_group_id": group_id,
            "num_slices": len(group),
        }
        for metric in METRICS:
            item[metric] = mean(as_float(row, metric) for row in group)
        item["before_tta_psnr"] = mean(as_float(row, "before_tta_psnr") for row in group)
        item["after_tta_psnr"] = mean(as_float(row, "after_tta_psnr") for row in group)
        item["before_tta_ssim"] = mean(as_float(row, "before_tta_ssim") for row in group)
        item["after_tta_ssim"] = mean(as_float(row, "after_tta_ssim") for row in group)
        item["negative_adaptation"] = bool(float(item["delta_nmse"]) > 0.0)
        out.append(item)
    return out


def cell_summary(volumes: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in volumes:
        key = (
            str(row["training_objective"]),
            str(row["experiment_family"]),
            str(row["shift_name"]),
            str(row["source_acquisition"]),
            str(row["target_acquisition"]),
            str(row["is_in_domain_control"]),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        objective, family, shift, source, target, control = key
        item: dict[str, object] = {
            "training_objective": objective,
            "experiment_family": family,
            "shift_name": shift,
            "source_acquisition": source,
            "target_acquisition": target,
            "is_in_domain_control": control,
            "num_volumes": len(group),
            "negative_adaptation_rate": float(np.mean([bool(row["negative_adaptation"]) for row in group])),
        }
        for metric in (*METRICS, "before_tta_psnr", "after_tta_psnr", "before_tta_ssim", "after_tta_ssim"):
            item[f"{metric}_mean"] = mean(float(row[metric]) for row in group)
        summaries.append(item)
    return summaries


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_cells(
    volumes: Sequence[Mapping[str, object]],
    *,
    resamples: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, np.ndarray]]]:
    by_key: dict[tuple[str, str, str], dict[str, Mapping[str, object]]] = {}
    for row in volumes:
        key = (str(row["experiment_family"]), str(row["shift_name"]), str(row["split_group_id"]))
        by_key.setdefault(key, {})[str(row["training_objective"])] = row
    cell_diffs: dict[tuple[str, str], dict[str, list[float]]] = {}
    metadata: dict[tuple[str, str], Mapping[str, object]] = {}
    for (family, shift, _), objectives in by_key.items():
        if set(objectives) != {"ensure", "traditional"}:
            continue
        cell = (family, shift)
        metadata[cell] = objectives["ensure"]
        diff = cell_diffs.setdefault(cell, {metric: [] for metric in METRICS})
        for metric in METRICS:
            diff[metric].append(float(objectives["ensure"][metric]) - float(objectives["traditional"][metric]))

    rng = np.random.default_rng(seed)
    summaries: list[dict[str, object]] = []
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for cell, metric_values in sorted(cell_diffs.items()):
        meta = metadata[cell]
        arrays[cell] = {}
        item: dict[str, object] = {
            "experiment_family": cell[0],
            "shift_name": cell[1],
            "source_acquisition": meta["source_acquisition"],
            "target_acquisition": meta["target_acquisition"],
            "is_in_domain_control": meta["is_in_domain_control"],
            "num_paired_volumes": len(metric_values[METRICS[0]]),
        }
        for metric, values in metric_values.items():
            values_array = np.asarray(values, dtype=np.float64)
            arrays[cell][metric] = values_array
            indices = rng.integers(0, len(values_array), size=(resamples, len(values_array)))
            boot = values_array[indices].mean(axis=1)
            low, high = percentile_ci(boot)
            item[f"ensure_minus_traditional_{metric}_mean"] = float(values_array.mean())
            item[f"ensure_minus_traditional_{metric}_ci_low"] = low
            item[f"ensure_minus_traditional_{metric}_ci_high"] = high
        summaries.append(item)
    return summaries, arrays


def macro_bootstrap(
    arrays: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    scopes = {
        "brain": [cell for cell in arrays if cell[0] == "brain_modality_matrix"],
        "knee": [cell for cell in arrays if cell[0] == "knee_modality_matrix"],
        "overall": list(arrays),
    }
    out: list[dict[str, object]] = []
    for scope, cells_all in scopes.items():
        cells = [
            cell for cell in cells_all
            if str(cell[1]).split("_to_")[0].rsplit("_", 1)[-1]
            != str(cell[1]).split("_to_")[-1]
        ]
        if not cells:
            continue
        for metric in METRICS:
            observed = float(np.mean([arrays[cell][metric].mean() for cell in cells]))
            boot = np.empty(resamples, dtype=np.float64)
            for sample_idx in range(resamples):
                sampled_cells = rng.choice(len(cells), size=len(cells), replace=True)
                cell_means = []
                for cell_index in sampled_cells:
                    values = arrays[cells[int(cell_index)]][metric]
                    picked = rng.integers(0, len(values), size=len(values))
                    cell_means.append(float(values[picked].mean()))
                boot[sample_idx] = float(np.mean(cell_means))
            low, high = percentile_ci(boot)
            out.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "num_ood_cells": len(cells),
                    "ensure_minus_traditional_macro_mean": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_resamples": resamples,
                    "bootstrap_seed": seed,
                }
            )
    return out


def expected_cells() -> set[tuple[str, str]]:
    expected = set()
    for source in BRAIN:
        for target in BRAIN:
            expected.add(("ensure", f"brain_modality_matrix_{source}_to_{target}"))
            expected.add(("traditional", f"brain_modality_matrix_{source}_to_{target}"))
    for source in KNEE:
        for target in KNEE:
            expected.add(("ensure", f"knee_modality_matrix_{source}_to_{target}"))
            expected.add(("traditional", f"knee_modality_matrix_{source}_to_{target}"))
    return expected


def write_heatmaps(summaries: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warning] matplotlib unavailable; skipping heatmaps")
        return
    for family, labels in (("brain_modality_matrix", BRAIN), ("knee_modality_matrix", KNEE)):
        for objective in ("ensure", "traditional"):
            for metric in METRICS:
                matrix = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
                for row in summaries:
                    if row["experiment_family"] != family or row["training_objective"] != objective:
                        continue
                    shift = str(row["shift_name"])
                    source_slug, target_slug = shift.split("_to_", 1)
                    source_slug = source_slug.rsplit("_", 1)[-1]
                    if source_slug in labels and target_slug in labels:
                        matrix[labels.index(source_slug), labels.index(target_slug)] = float(row[f"{metric}_mean"])
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                image = ax.imshow(matrix, cmap="viridis")
                ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
                ax.set_yticks(range(len(labels)), labels=labels)
                ax.set_xlabel("Target acquisition")
                ax.set_ylabel("Source acquisition")
                ax.set_title(f"{family} | {objective} | {metric}")
                fig.colorbar(image, ax=ax)
                fig.tight_layout()
                fig.savefig(output_dir / f"heatmap_{family}_{objective}_{metric}.png", dpi=180)
                plt.close(fig)


def robustness_directions(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    brain_scores: dict[str, list[float]] = {}
    knee_shifts: set[str] = set()
    for row in summaries:
        if str(row["is_in_domain_control"]).lower() == "true":
            continue
        family = str(row["experiment_family"])
        shift = str(row["shift_name"])
        if family == "brain_modality_matrix":
            brain_scores.setdefault(shift, []).append(float(row["before_tta_nmse_mean"]))
        elif family == "knee_modality_matrix":
            knee_shifts.add(shift)
    averaged = {shift: float(np.mean(values)) for shift, values in brain_scores.items()}
    ordered = sorted(averaged, key=lambda shift: (averaged[shift], shift))
    return {
        "selection_metric": "mean frozen volume-level NMSE across training objectives",
        "brain_easiest": ordered[0] if ordered else None,
        "brain_hardest": ordered[-1] if ordered else None,
        "brain_scores": averaged,
        "knee_directions": sorted(knee_shifts),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs" / "tta" / "shifts" / "modality_matrix")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "tta" / "shifts" / "modality_matrix" / "analysis")
    parser.add_argument("--test-noise-snr-db", default="20")
    parser.add_argument("--test-noise-seed", type=int, default=9007)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260702)
    parser.add_argument("--strict-completeness", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, failed_rows = load_metric_rows(args)
    volumes = volume_rows(rows)
    summaries = cell_summary(volumes)
    paired, arrays = paired_cells(
        volumes,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    macro = macro_bootstrap(
        arrays,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    volume_fields = [
        "training_objective", "experiment_family", "shift_name", "source_acquisition",
        "target_acquisition", "is_in_domain_control", "split_group_id", "num_slices",
        *METRICS, "before_tta_psnr", "after_tta_psnr", "before_tta_ssim", "after_tta_ssim",
        "negative_adaptation",
    ]
    cell_fields = list(summaries[0]) if summaries else []
    paired_fields = list(paired[0]) if paired else []
    macro_fields = list(macro[0]) if macro else []
    write_csv(args.output_dir / "volume_metrics.csv", volumes, volume_fields)
    write_csv(args.output_dir / "cell_summary.csv", summaries, cell_fields)
    write_csv(args.output_dir / "paired_cell_comparison.csv", paired, paired_fields)
    write_csv(args.output_dir / "macro_bootstrap.csv", macro, macro_fields)

    observed = {(str(row["training_objective"]), str(row["shift_name"])) for row in summaries}
    missing = sorted(expected_cells() - observed)
    sample_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["training_objective_key"]), str(row["shift_name"]))
        sample_counts[key] = sample_counts.get(key, 0) + 1
    incomplete_sample_counts = {
        f"{objective}:{shift}": sample_counts.get((objective, shift), 0)
        for objective, shift in sorted(expected_cells())
        if sample_counts.get((objective, shift), 0) != 100
    }
    completeness = {
        "metric_rows": len(rows),
        "failed_metric_rows": len(failed_rows),
        "volume_rows": len(volumes),
        "observed_cells": len(observed),
        "expected_cells": len(expected_cells()),
        "missing_cells": [list(item) for item in missing],
        "incomplete_sample_counts": incomplete_sample_counts,
        "complete": not missing and not incomplete_sample_counts and not failed_rows,
        "test_noise_snr_db": args.test_noise_snr_db,
        "test_noise_seed": args.test_noise_seed,
        "eval_seed": args.eval_seed,
    }
    (args.output_dir / "completeness.json").write_text(
        json.dumps(completeness, indent=2, sort_keys=True), encoding="utf-8"
    )
    directions = robustness_directions(summaries)
    (args.output_dir / "robustness_directions.json").write_text(
        json.dumps(directions, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_heatmaps(summaries, args.output_dir)
    print(json.dumps(completeness, indent=2), flush=True)
    if args.strict_completeness and not completeness["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
