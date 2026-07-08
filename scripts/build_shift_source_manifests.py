#!/usr/bin/env python3
"""Build cardiac_ensure shift manifests with larger source-train splits."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import h5py
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


MATRIX_METADATA_FIELDS = [
    "experiment_family",
    "source_acquisition",
    "target_acquisition",
    "patient_id",
    "split_group_id",
    "is_in_domain_control",
    "sampling_family",
    "sampling_acceleration",
    "sampling_sigma_mask",
    "sampling_seed_policy",
]


MATRIX_SHIFT_FIELDS = [
    *SHIFT_FIELDS[:13],
    *MATRIX_METADATA_FIELDS,
    *SHIFT_FIELDS[13:],
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


MODALITY_MATRIX_SPECS = {
    "brain_modality_matrix": {
        "dataset": "fastmri_brain",
        "output_name": "brain",
        "matrix_size": (640, 320),
        "modalities": {
            "axflair": "AXFLAIR",
            "axt1": "AXT1",
            "axt1pre": "AXT1PRE",
            "axt1post": "AXT1POST",
            "axt2": "AXT2",
        },
        "paired_patients": False,
    },
    "knee_modality_matrix": {
        "dataset": "fastmri_knee",
        "output_name": "knee",
        "matrix_size": (640, 368),
        "modalities": {
            "pd": "CORPD_FBK",
            "pdfs": "CORPDFS_FBK",
        },
        "paired_patients": True,
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
    acquisition = filters.get("acquisition")
    if acquisition and row["acquisition"] != acquisition:
        return False
    matrix_size = filters.get("matrix_size")
    if matrix_size and parse_matrix_size(row) != tuple(matrix_size):
        return False
    return True


def _h5_patient_id(path: Path, cache: MutableMapping[str, str]) -> str:
    key = str(path)
    cached = cache.get(key)
    if cached is not None:
        return cached
    with h5py.File(path, "r") as handle:
        value = handle.attrs.get("patient_id", path.stem)
    if isinstance(value, bytes):
        patient_id = value.decode("utf-8", errors="replace")
    else:
        patient_id = str(value)
    cache[key] = patient_id
    return patient_id


def _rows_by_volume(rows: Sequence[Mapping[str, object]]) -> Dict[str, List[Mapping[str, object]]]:
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    for row in sorted(rows, key=lambda item: (str(item["path"]), int(item["slice_idx"]))):
        grouped.setdefault(str(row["path"]), []).append(row)
    return grouped


def _candidate_volumes_by_patient(
    rows: Sequence[Mapping[str, object]],
    *,
    patient_cache: MutableMapping[str, str],
) -> Dict[str, List[List[Mapping[str, object]]]]:
    grouped: Dict[str, List[List[Mapping[str, object]]]] = {}
    for path, volume_rows in _rows_by_volume(rows).items():
        patient_id = _h5_patient_id(Path(path), patient_cache)
        grouped.setdefault(patient_id, []).append(volume_rows)
    for patient_id in grouped:
        grouped[patient_id].sort(key=lambda volume: str(volume[0]["path"]))
    return grouped


def _rotate(values: Sequence[str], offset: int) -> List[str]:
    out = list(values)
    if not out:
        return out
    normalized = int(offset) % len(out)
    return out[normalized:] + out[:normalized]


def _middle_rows_from_volume(
    volume_rows: Sequence[Mapping[str, object]],
    *,
    slices_per_volume: int,
) -> List[Dict[str, object]]:
    num_slices = int(volume_rows[0]["num_slices"])
    allowed = middle_slice_indices(num_slices, slices_per_volume)
    return [dict(row) for row in volume_rows if int(row["slice_idx"]) in allowed]


def _eligible_for_full_middle_selection(
    rows: Sequence[Mapping[str, object]],
    slices_per_volume: int,
) -> List[Mapping[str, object]]:
    minimum = max(int(slices_per_volume), 1)
    return [row for row in rows if int(row.get("num_slices", 0)) >= minimum]


def select_paired_patient_rows(
    candidates_by_acquisition: Mapping[str, Dict[str, List[List[Mapping[str, object]]]]],
    *,
    target_count: int,
    slices_per_volume: int,
    patient_offset: int,
) -> tuple[Dict[str, List[Dict[str, object]]], List[str]]:
    patient_sets = [set(candidates) for candidates in candidates_by_acquisition.values()]
    common_patients = sorted(set.intersection(*patient_sets)) if patient_sets else []
    ordered_patients = _rotate(common_patients, patient_offset)
    patient_count = int(np.ceil(int(target_count) / max(int(slices_per_volume), 1)))
    selected_patients = ordered_patients[:patient_count]
    selected: Dict[str, List[Dict[str, object]]] = {}
    for acquisition, candidates in candidates_by_acquisition.items():
        rows: List[Dict[str, object]] = []
        for patient_id in selected_patients:
            volume_rows = candidates[patient_id][0]
            rows.extend(
                _middle_rows_from_volume(volume_rows, slices_per_volume=slices_per_volume)
            )
        selected[acquisition] = rows[: int(target_count)]
    return selected, selected_patients


def _enrich_patient_metadata(
    rows: Sequence[Mapping[str, object]],
    patient_cache: MutableMapping[str, str],
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        patient_id = _h5_patient_id(Path(str(row["path"])), patient_cache)
        copied["patient_id"] = patient_id
        copied["split_group_id"] = patient_id
        enriched.append(copied)
    return enriched


def _matrix_role_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    experiment_family: str,
    tier_name: str,
    split_role: str,
    source_slug: str,
    source_acquisition: str,
    target_slug: str,
    target_acquisition: str,
    matrix_size: tuple[int, int],
    selection_rule: str,
) -> List[Dict[str, object]]:
    source_domain = f"fastMRI {experiment_family.split('_')[0]} {source_acquisition}"
    target_domain = f"fastMRI {experiment_family.split('_')[0]} {target_acquisition}"
    shift_name = f"{experiment_family}_{source_slug}_to_{target_slug}"
    if split_role == "source_train":
        shift_name = f"{experiment_family}_{source_slug}_source"
        target_domain = f"fastMRI {experiment_family.split('_')[0]} modality matrix"
    role_rows: List[Dict[str, object]] = []
    for rank, row in enumerate(rows):
        role_row = {
            "shift_name": shift_name,
            "experiment_tier": tier_name,
            "split_role": split_role,
            "rank_in_split": rank,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "source_model_keys": json.dumps([f"{source_slug}_ensure", f"{source_slug}_traditional"]),
            "target_reference_model_keys": json.dumps([]),
            "eval_mask_path": "",
            "eval_mask_shape": json.dumps(list(matrix_size)),
            "eval_mask_actual_acceleration": "4",
            "role_nominal_acceleration": "R=4",
            "selection_rule": selection_rule,
            "experiment_family": experiment_family,
            "source_acquisition": source_acquisition,
            "target_acquisition": target_acquisition,
            "patient_id": row.get("patient_id", ""),
            "split_group_id": row.get("split_group_id", row.get("patient_id", "")),
            "is_in_domain_control": str(source_acquisition == target_acquisition),
            "sampling_family": "bernoulli_gaussian_lines",
            "sampling_acceleration": "4",
            "sampling_sigma_mask": "0.18",
            "sampling_seed_policy": "stable(sample_id, mask_seed)",
        }
        role_row.update({key: value for key, value in row.items() if key not in role_row})
        role_rows.append(role_row)
    return role_rows


def build_modality_matrix_manifests(
    *,
    input_manifest_root: Path,
    output_dir: Path,
    tiers: Sequence[str],
    matrices: Sequence[str],
    source_volume_offset: int,
    target_volume_offset: int,
) -> Dict[str, object]:
    base_by_dataset = {
        name: read_csv(input_manifest_root / f"{name}.csv")
        for name in ("fastmri_brain", "fastmri_knee")
    }
    patient_cache: Dict[str, str] = {}
    summary: Dict[str, object] = {}
    matrix_root = output_dir / "shifts" / "modality_matrix"

    for tier_name in tiers:
        tier = TIERS[tier_name]
        for matrix_name in matrices:
            spec = MODALITY_MATRIX_SPECS[matrix_name]
            dataset = str(spec["dataset"])
            matrix_size = tuple(spec["matrix_size"])
            modalities = dict(spec["modalities"])
            base_rows = base_by_dataset[dataset]
            selected_source: Dict[str, List[Dict[str, object]]] = {}
            selected_target: Dict[str, List[Dict[str, object]]] = {}

            if bool(spec["paired_patients"]):
                train_candidates: Dict[str, Dict[str, List[List[Mapping[str, object]]]]] = {}
                val_candidates: Dict[str, Dict[str, List[List[Mapping[str, object]]]]] = {}
                for acquisition in modalities.values():
                    common_filter = {
                        "dataset": dataset,
                        "acquisition": acquisition,
                        "matrix_size": matrix_size,
                    }
                    train_matches = [
                        row for row in base_rows
                        if row_matches(row, {**common_filter, "splits": ["train"]})
                    ]
                    val_matches = [
                        row for row in base_rows
                        if row_matches(row, {**common_filter, "splits": ["val"]})
                    ]
                    train_matches = _eligible_for_full_middle_selection(
                        train_matches, int(tier["slices_per_volume"])
                    )
                    val_matches = _eligible_for_full_middle_selection(
                        val_matches, int(tier["slices_per_volume"])
                    )
                    train_candidates[acquisition] = _candidate_volumes_by_patient(
                        train_matches, patient_cache=patient_cache
                    )
                    val_candidates[acquisition] = _candidate_volumes_by_patient(
                        val_matches, patient_cache=patient_cache
                    )
                selected_source, train_patients = select_paired_patient_rows(
                    train_candidates,
                    target_count=int(tier["source_slices"]),
                    slices_per_volume=int(tier["slices_per_volume"]),
                    patient_offset=source_volume_offset,
                )
                selected_target, target_patients = select_paired_patient_rows(
                    val_candidates,
                    target_count=int(tier["target_slices"]),
                    slices_per_volume=int(tier["slices_per_volume"]),
                    patient_offset=target_volume_offset,
                )
                if set(train_patients) & set(target_patients):
                    raise ValueError(f"Patient leakage in {matrix_name}/{tier_name}")
                for acquisition in modalities.values():
                    selected_source[acquisition] = _enrich_patient_metadata(
                        selected_source[acquisition], patient_cache
                    )
                    selected_target[acquisition] = _enrich_patient_metadata(
                        selected_target[acquisition], patient_cache
                    )
            else:
                for acquisition in modalities.values():
                    common_filter = {
                        "dataset": dataset,
                        "acquisition": acquisition,
                        "matrix_size": matrix_size,
                    }
                    train_matches = [
                        row for row in base_rows
                        if row_matches(row, {**common_filter, "splits": ["train"]})
                    ]
                    val_matches = [
                        row for row in base_rows
                        if row_matches(row, {**common_filter, "splits": ["val"]})
                    ]
                    train_matches = _eligible_for_full_middle_selection(
                        train_matches, int(tier["slices_per_volume"])
                    )
                    val_matches = _eligible_for_full_middle_selection(
                        val_matches, int(tier["slices_per_volume"])
                    )
                    selected_source[acquisition] = _enrich_patient_metadata(
                        select_middle_rows(
                            train_matches,
                            target_count=int(tier["source_slices"]),
                            slices_per_volume=int(tier["slices_per_volume"]),
                            volume_offset=source_volume_offset,
                        ),
                        patient_cache,
                    )
                    selected_target[acquisition] = _enrich_patient_metadata(
                        select_middle_rows(
                            val_matches,
                            target_count=int(tier["target_slices"]),
                            slices_per_volume=int(tier["slices_per_volume"]),
                            volume_offset=target_volume_offset,
                        ),
                        patient_cache,
                    )

            selection_rule = (
                f"one volume per paired patient, middle {tier['slices_per_volume']} slices"
                if bool(spec["paired_patients"])
                else f"middle {tier['slices_per_volume']} slices"
            )
            expected_source = int(tier["source_slices"])
            expected_target = int(tier["target_slices"])
            for acquisition in modalities.values():
                if len(selected_source[acquisition]) != expected_source:
                    raise ValueError(
                        f"{matrix_name}/{tier_name}/{acquisition}: expected "
                        f"{expected_source} source rows, got {len(selected_source[acquisition])}"
                    )
                if len(selected_target[acquisition]) != expected_target:
                    raise ValueError(
                        f"{matrix_name}/{tier_name}/{acquisition}: expected "
                        f"{expected_target} target rows, got {len(selected_target[acquisition])}"
                    )
            for source_slug, source_acquisition in modalities.items():
                source_rows = _matrix_role_rows(
                    rows=selected_source[source_acquisition],
                    experiment_family=matrix_name,
                    tier_name=tier_name,
                    split_role="source_train",
                    source_slug=source_slug,
                    source_acquisition=source_acquisition,
                    target_slug="all",
                    target_acquisition="all",
                    matrix_size=matrix_size,
                    selection_rule=selection_rule,
                )
                target_rows: List[Dict[str, object]] = []
                target_counts: Dict[str, int] = {}
                for target_slug, target_acquisition in modalities.items():
                    cell_rows = _matrix_role_rows(
                        rows=selected_target[target_acquisition],
                        experiment_family=matrix_name,
                        tier_name=tier_name,
                        split_role="target_test",
                        source_slug=source_slug,
                        source_acquisition=source_acquisition,
                        target_slug=target_slug,
                        target_acquisition=target_acquisition,
                        matrix_size=matrix_size,
                        selection_rule=selection_rule,
                    )
                    target_rows.extend(cell_rows)
                    target_counts[target_slug] = len(cell_rows)
                path = matrix_root / tier_name / str(spec["output_name"]) / f"{source_slug}.csv"
                write_csv(path, source_rows + target_rows, fields=MATRIX_SHIFT_FIELDS)
                key = f"modality_matrix/{tier_name}/{spec['output_name']}/{source_slug}"
                summary[key] = {
                    "path": str(path),
                    "matrix": matrix_name,
                    "source_acquisition": source_acquisition,
                    "matrix_size": list(matrix_size),
                    "source_train_rows": len(source_rows),
                    "target_test_rows": len(target_rows),
                    "target_counts": target_counts,
                    "paired_patients": bool(spec["paired_patients"]),
                    "selection_rule": selection_rule,
                }
                print(
                    f"[ok] {key}: source_train={len(source_rows)} "
                    f"target_test={len(target_rows)}"
                )

    summary_rows = [
        {
            "key": key,
            **{name: json.dumps(value) if isinstance(value, (dict, list)) else value for name, value in item.items()},
        }
        for key, item in summary.items()
    ]
    summary_fields = [
        "key", "path", "matrix", "source_acquisition", "matrix_size",
        "source_train_rows", "target_test_rows", "target_counts",
        "paired_patients", "selection_rule",
    ]
    write_csv(matrix_root / "summary.csv", summary_rows, fields=summary_fields)
    (matrix_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


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
    volume_offset: int = 0,
) -> List[Dict[str, object]]:
    by_volume: Dict[str, List[Mapping[str, object]]] = {}
    for row in sorted(rows, key=lambda item: (str(item["path"]), int(item["slice_idx"]))):
        by_volume.setdefault(str(row["path"]), []).append(row)

    selected: List[Dict[str, object]] = []
    selected_ids: set[str] = set()
    volume_groups = list(by_volume.values())
    if volume_groups:
        offset = int(volume_offset) % len(volume_groups)
        volume_groups = volume_groups[offset:] + volume_groups[:offset]
    for volume_rows in volume_groups:
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
    volume_offset: int,
    mask_shape: tuple[int, int],
    mask_acceleration: float,
) -> List[Dict[str, object]]:
    role_filter = spec["source_filter"] if split_role == "source_train" else spec["target_filter"]
    nominal_acceleration = (
        spec["source_nominal_acceleration"] if split_role == "source_train" else spec["target_nominal_acceleration"]
    )
    matches = [row for row in base_rows if row_matches(row, role_filter)]
    selected = select_middle_rows(
        matches,
        target_count=target_count,
        slices_per_volume=slices_per_volume,
        volume_offset=volume_offset,
    )
    selection_rule = "middle slice only" if slices_per_volume == 1 else f"middle {slices_per_volume} slices"
    if volume_offset:
        selection_rule = f"{selection_rule}, volume offset {int(volume_offset)}"

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
    shifts: Sequence[str],
    tier_output_name: str | None,
    source_volume_offset: int,
    target_volume_offset: int,
) -> Dict[str, object]:
    dataset_rows = build_dataset_rows(input_manifest_root)
    summary: Dict[str, object] = {}
    for tier_name in tiers:
        tier = TIERS[tier_name]
        output_tier = tier_output_name or tier_name
        for shift_name in shifts:
            spec = SHIFT_SPECS[shift_name]
            mask_shape, mask_acceleration = mask_metadata(spec["eval_mask_path"])
            source_rows = shift_rows_for_role(
                base_rows=dataset_rows,
                spec=spec,
                shift_name=shift_name,
                tier_name=tier_name,
                split_role="source_train",
                target_count=tier["source_slices"],
                slices_per_volume=tier["slices_per_volume"],
                volume_offset=source_volume_offset,
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
                volume_offset=target_volume_offset,
                mask_shape=mask_shape,
                mask_acceleration=mask_acceleration,
            )
            rows = source_rows + target_rows
            path = output_dir / "shifts" / output_tier / f"{shift_name}.csv"
            write_csv(path, rows, fields=SHIFT_FIELDS)
            summary[f"{output_tier}/{shift_name}"] = {
                "path": str(path),
                "source_train_rows": len(source_rows),
                "target_test_rows": len(target_rows),
                "source_domain": spec["source_domain"],
                "target_domain": spec["target_domain"],
                "eval_mask_shape": list(mask_shape),
                "eval_mask_actual_acceleration": mask_acceleration,
                "selection_rule": rows[0]["selection_rule"] if rows else "",
            }
            print(f"[ok] {output_tier}/{shift_name}: source_train={len(source_rows)} target_test={len(target_rows)}")
    write_csv(output_dir / "summary.csv", [{"key": key, **value} for key, value in summary.items()], fields=["key", "path", "source_train_rows", "target_test_rows", "source_domain", "target_domain", "eval_mask_shape", "eval_mask_actual_acceleration", "selection_rule"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest-root", type=Path, default=UTTA_ROOT / "manifests")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--tiers", nargs="+", choices=sorted(TIERS), default=list(TIERS))
    parser.add_argument("--shifts", nargs="+", choices=sorted(SHIFT_SPECS), default=list(SHIFT_SPECS))
    parser.add_argument(
        "--modality-matrices",
        nargs="+",
        choices=sorted(MODALITY_MATRIX_SPECS),
        default=[],
        help="Build source-specific all-target fastMRI modality matrix manifests.",
    )
    parser.add_argument(
        "--only-modality-matrices",
        action="store_true",
        help="Skip the legacy shift manifests and only build --modality-matrices.",
    )
    parser.add_argument(
        "--tier-output-name",
        default=None,
        help="Directory name under output-dir/shifts. CSV experiment_tier values still use --tiers.",
    )
    parser.add_argument("--source-volume-offset", type=int, default=0)
    parser.add_argument("--target-volume-offset", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.only_modality_matrices:
        build_shift_manifests(
            input_manifest_root=args.input_manifest_root,
            output_dir=args.output_dir,
            tiers=args.tiers,
            shifts=args.shifts,
            tier_output_name=args.tier_output_name,
            source_volume_offset=args.source_volume_offset,
            target_volume_offset=args.target_volume_offset,
        )
    if args.modality_matrices:
        build_modality_matrix_manifests(
            input_manifest_root=args.input_manifest_root,
            output_dir=args.output_dir,
            tiers=args.tiers,
            matrices=args.modality_matrices,
            source_volume_offset=args.source_volume_offset,
            target_volume_offset=args.target_volume_offset,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
