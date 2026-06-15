#!/usr/bin/env python3
"""Build cardiac_ensure shift manifests with larger source-train splits."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTTA_ROOT = PROJECT_ROOT.parent / "Uncertainty_TTA"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FIELDS = [
    "sample_id",
    "dataset",
    "split",
    "path",
    "filename",
    "volume_id",
    "slice_idx",
    "num_slices",
    "is_middle_slice",
    "anatomy",
    "sequence_or_contrast",
    "acquisition",
    "coil_count",
    "kspace_shape",
    "matrix_size",
    "target_key",
    "has_target",
    "source_or_target_role",
    "kspace_key",
    "h5_format",
    "has_top_level_kspace",
]


SHIFT_FIELDS = [
    "shift_name",
    "experiment_tier",
    "split_role",
    "rank_in_split",
    "source_domain",
    "target_domain",
    "source_model_keys",
    "target_reference_model_keys",
    "eval_mask_path",
    "eval_mask_shape",
    "eval_mask_actual_acceleration",
    "role_nominal_acceleration",
    "selection_rule",
    *FIELDS,
]


TIERS = {
    "debug": {"source_slices": 5, "target_slices": 5, "slices_per_volume": 1},
    "pilot": {"source_slices": 75, "target_slices": 30, "slices_per_volume": 5},
    "main": {"source_slices": 440, "target_slices": 100, "slices_per_volume": 5},
}


SHIFT_SPECS = {
    "anatomy_shift": {
        "source_domain": "fastMRI knee PD",
        "target_domain": "fastMRI brain AXT2",
        "source_filter": {"dataset": "fastmri_knee", "splits": ["train"], "sequence": "PD"},
        "target_filter": {
            "dataset": "fastmri_brain",
            "splits": ["val"],
            "sequence": "AXT2",
            "matrix_size": (768, 396),
        },
        "source_model_keys": ["knee_sup", "knee_self"],
        "target_reference_model_keys": ["brain_sup", "brain_self"],
        "eval_mask_path": UTTA_ROOT / "varnet" / "test_data" / "anatomy_shift" / "mask2d",
        "source_nominal_acceleration": "R=4",
        "target_nominal_acceleration": "R=4",
    },
    "dataset_shift": {
        "source_domain": "fastMRI knee PDFS",
        "target_domain": "Stanford knee PDFS-like",
        "source_filter": {"dataset": "fastmri_knee", "splits": ["train"], "sequence": "PDFS"},
        "target_filter": {"dataset": "stanford_knee", "splits": ["test"], "sequence": "FSE", "matrix_size": (320, 320)},
        "source_model_keys": ["fs_sup", "fs_self"],
        "target_reference_model_keys": ["stanford_sup", "stanford_self"],
        "eval_mask_path": UTTA_ROOT / "varnet" / "test_data" / "dataset_shift" / "mask2d",
        "source_nominal_acceleration": "R=4",
        "target_nominal_acceleration": "R=4",
    },
    "modality_shift": {
        "source_domain": "fastMRI brain AXT2",
        "target_domain": "fastMRI brain AXT1PRE",
        "source_filter": {"dataset": "fastmri_brain", "splits": ["train"], "sequence": "AXT2"},
        "target_filter": {
            "dataset": "fastmri_brain",
            "splits": ["val"],
            "sequence": "AXT1PRE",
            "matrix_size": (640, 320),
        },
        "source_model_keys": ["t2_sup", "t2_self"],
        "target_reference_model_keys": ["t1pre_sup", "t1pre_self"],
        "eval_mask_path": UTTA_ROOT / "varnet" / "test_data" / "modality_shift" / "mask2d",
        "source_nominal_acceleration": "R=4",
        "target_nominal_acceleration": "R=4",
    },
    "acceleration_shift": {
        "source_domain": "fastMRI knee PD R=4",
        "target_domain": "fastMRI knee PD R=2",
        "source_filter": {"dataset": "fastmri_knee", "splits": ["train"], "sequence": "PD"},
        "target_filter": {
            "dataset": "fastmri_knee",
            "splits": ["val"],
            "sequence": "PD",
            "matrix_size": (640, 372),
        },
        "source_model_keys": ["knee_sup", "knee_self"],
        "target_reference_model_keys": ["2x_sup", "2x_self"],
        "eval_mask_path": UTTA_ROOT / "varnet" / "test_data" / "acceleration_shift" / "mask2x",
        "source_nominal_acceleration": "R=4",
        "target_nominal_acceleration": "R=2",
    },
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def parse_matrix_size(row: Mapping[str, object]) -> tuple[int, int]:
    value = row["matrix_size"]
    loaded = json.loads(value) if isinstance(value, str) else value
    return int(loaded[0]), int(loaded[1])


def row_matches(row: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    if row["dataset"] != filters["dataset"]:
        return False
    splits = filters.get("splits")
    if splits and row["split"] not in set(splits):
        return False
    sequence = filters.get("sequence")
    if sequence and row["sequence_or_contrast"] != sequence:
        return False
    matrix_size = filters.get("matrix_size")
    if matrix_size and parse_matrix_size(row) != tuple(matrix_size):
        return False
    return True


def middle_slice_indices(num_slices: int, count: int) -> set[int]:
    count = min(max(int(count), 1), int(num_slices))
    if count <= 1:
        return {int(num_slices) // 2}
    center = int(num_slices) // 2
    half = count // 2
    start = max(0, center - half)
    end = min(int(num_slices), start + count)
    start = max(0, end - count)
    return set(range(start, end))


def select_middle_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    target_count: int,
    slices_per_volume: int,
) -> List[Dict[str, object]]:
    by_volume: Dict[str, List[Mapping[str, object]]] = {}
    for row in sorted(rows, key=lambda item: (str(item["path"]), int(item["slice_idx"]))):
        by_volume.setdefault(str(row["path"]), []).append(row)

    selected: List[Dict[str, object]] = []
    selected_ids: set[str] = set()
    for volume_rows in by_volume.values():
        num_slices = int(volume_rows[0]["num_slices"])
        allowed = middle_slice_indices(num_slices, slices_per_volume)
        for row in volume_rows:
            if int(row["slice_idx"]) not in allowed:
                continue
            selected.append(dict(row))
            selected_ids.add(str(row["sample_id"]))
            if len(selected) >= int(target_count):
                return selected

    if len(selected) >= int(target_count):
        return selected

    fallback_rows: List[Mapping[str, object]] = []
    for volume_rows in by_volume.values():
        center = int(volume_rows[0]["num_slices"]) // 2
        fallback_rows.extend(
            sorted(
                volume_rows,
                key=lambda row: (abs(int(row["slice_idx"]) - center), str(row["path"]), int(row["slice_idx"])),
            )
        )
    for row in fallback_rows:
        sample_id = str(row["sample_id"])
        if sample_id in selected_ids:
            continue
        selected.append(dict(row))
        selected_ids.add(sample_id)
        if len(selected) >= int(target_count):
            return selected
    return selected


def mask_metadata(path: Path) -> tuple[tuple[int, int], float]:
    with path.open("rb") as handle:
        mask = np.asarray(pickle.load(handle)).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask at {path}, got {mask.shape}")
    sampled = int(mask.sum())
    acceleration = float(mask.size / sampled) if sampled else float("inf")
    return (int(mask.shape[0]), int(mask.shape[1])), acceleration


def shift_rows_for_role(
    *,
    base_rows: Sequence[Mapping[str, object]],
    spec: Mapping[str, object],
    shift_name: str,
    tier_name: str,
    split_role: str,
    target_count: int,
    slices_per_volume: int,
    mask_shape: tuple[int, int],
    mask_acceleration: float,
) -> List[Dict[str, object]]:
    role_filter = spec["source_filter"] if split_role == "source_train" else spec["target_filter"]
    nominal_acceleration = (
        spec["source_nominal_acceleration"] if split_role == "source_train" else spec["target_nominal_acceleration"]
    )
    matches = [row for row in base_rows if row_matches(row, role_filter)]
    selected = select_middle_rows(matches, target_count=target_count, slices_per_volume=slices_per_volume)
    selection_rule = "middle slice only" if slices_per_volume == 1 else f"middle {slices_per_volume} slices"

    role_rows: List[Dict[str, object]] = []
    for rank, row in enumerate(selected):
        role_row = {
            "shift_name": shift_name,
            "experiment_tier": tier_name,
            "split_role": split_role,
            "rank_in_split": rank,
            "source_domain": spec["source_domain"],
            "target_domain": spec["target_domain"],
            "source_model_keys": json.dumps(spec["source_model_keys"]),
            "target_reference_model_keys": json.dumps(spec["target_reference_model_keys"]),
            "eval_mask_path": str(spec["eval_mask_path"]),
            "eval_mask_shape": json.dumps(list(mask_shape)),
            "eval_mask_actual_acceleration": f"{mask_acceleration:.6g}",
            "role_nominal_acceleration": nominal_acceleration,
            "selection_rule": selection_rule,
        }
        role_row.update(row)
        role_rows.append(role_row)
    return role_rows


def build_dataset_rows(manifest_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for name in ("fastmri_brain", "fastmri_knee", "stanford_knee"):
        path = manifest_root / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing source dataset manifest: {path}")
        rows.extend(read_csv(path))
    return rows


def build_shift_manifests(
    *,
    input_manifest_root: Path,
    output_dir: Path,
    tiers: Sequence[str],
) -> Dict[str, object]:
    dataset_rows = build_dataset_rows(input_manifest_root)
    summary: Dict[str, object] = {}
    for tier_name in tiers:
        tier = TIERS[tier_name]
        for shift_name, spec in SHIFT_SPECS.items():
            mask_shape, mask_acceleration = mask_metadata(spec["eval_mask_path"])
            source_rows = shift_rows_for_role(
                base_rows=dataset_rows,
                spec=spec,
                shift_name=shift_name,
                tier_name=tier_name,
                split_role="source_train",
                target_count=tier["source_slices"],
                slices_per_volume=tier["slices_per_volume"],
                mask_shape=mask_shape,
                mask_acceleration=mask_acceleration,
            )
            target_rows = shift_rows_for_role(
                base_rows=dataset_rows,
                spec=spec,
                shift_name=shift_name,
                tier_name=tier_name,
                split_role="target_test",
                target_count=tier["target_slices"],
                slices_per_volume=tier["slices_per_volume"],
                mask_shape=mask_shape,
                mask_acceleration=mask_acceleration,
            )
            rows = source_rows + target_rows
            path = output_dir / "shifts" / tier_name / f"{shift_name}.csv"
            write_csv(path, rows, fields=SHIFT_FIELDS)
            summary[f"{tier_name}/{shift_name}"] = {
                "path": str(path),
                "source_train_rows": len(source_rows),
                "target_test_rows": len(target_rows),
                "source_domain": spec["source_domain"],
                "target_domain": spec["target_domain"],
                "eval_mask_shape": list(mask_shape),
                "eval_mask_actual_acceleration": mask_acceleration,
                "selection_rule": rows[0]["selection_rule"] if rows else "",
            }
            print(f"[ok] {tier_name}/{shift_name}: source_train={len(source_rows)} target_test={len(target_rows)}")
    write_csv(output_dir / "summary.csv", [{"key": key, **value} for key, value in summary.items()], fields=["key", "path", "source_train_rows", "target_test_rows", "source_domain", "target_domain", "eval_mask_shape", "eval_mask_actual_acceleration", "selection_rule"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest-root", type=Path, default=UTTA_ROOT / "manifests")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIERS), default=list(TIERS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    build_shift_manifests(
        input_manifest_root=args.input_manifest_root,
        output_dir=args.output_dir,
        tiers=args.tiers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
