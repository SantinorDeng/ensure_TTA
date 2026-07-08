#!/usr/bin/env python3
"""Manifest-driven static source-domain ENSURE dataset for shift experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .cardiac_cine_dataset import (
        EPS,
        adjoint_image,
        bernoulli_gaussian_line_prob,
        ensure_complex,
        estimate_maps_rss,
        estimate_noise_sigma2,
        load_density_stats,
        make_mask_from_line_prob,
        stable_seed,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from cardiac_ensure.datasets.cardiac_cine_dataset import (
        EPS,
        adjoint_image,
        bernoulli_gaussian_line_prob,
        ensure_complex,
        estimate_maps_rss,
        estimate_noise_sigma2,
        load_density_stats,
        make_mask_from_line_prob,
        stable_seed,
    )


DEFAULT_SOURCE_ROLE = "source_train"


@dataclass(frozen=True)
class StaticShiftSample:
    row: Dict[str, str]
    preproc_path: Optional[Path]
    sample_id: str
    volume_key: str
    source_path: Path
    volume_id: str
    slice_id: int
    num_slices: int
    num_coils: int
    height: int
    width: int


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest_rows(manifest_csv: str | Path) -> List[Dict[str, str]]:
    with Path(manifest_csv).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def source_manifest_rows(
    manifest_csv: str | Path,
    source_role: str = DEFAULT_SOURCE_ROLE,
    shift_names: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    rows = read_manifest_rows(manifest_csv)
    allowed = set(shift_names or [])
    return [
        row
        for row in rows
        if row.get("split_role", "") == source_role
        and (not allowed or row.get("shift_name", "") in allowed)
    ]


def parse_json_tuple(value: object) -> Tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected a JSON list/tuple, got {value!r}")
    return tuple(int(item) for item in value)


def shape_from_manifest_row(row: Mapping[str, str]) -> Tuple[int, int]:
    matrix_size = row.get("matrix_size", "")
    if matrix_size:
        loaded = parse_json_tuple(matrix_size)
        if len(loaded) >= 2:
            return int(loaded[-2]), int(loaded[-1])
    kspace_shape = row.get("kspace_shape", "")
    if kspace_shape:
        loaded = parse_json_tuple(kspace_shape)
        if len(loaded) >= 2:
            return int(loaded[-2]), int(loaded[-1])
    raise ValueError(f"Could not infer spatial shape for sample_id={row.get('sample_id', '')}")


def coil_count_from_row(row: Mapping[str, str]) -> int:
    value = row.get("coil_count", "")
    if value:
        return int(value)
    kshape = parse_json_tuple(row["kspace_shape"])
    if len(kshape) >= 4:
        return int(kshape[-3])
    return 1


def volume_key_for_row(row: Mapping[str, str]) -> str:
    return f"{Path(row['path']).resolve()}::{row.get('volume_id') or Path(row['path']).stem}"


def split_group_key_for_row(row: Mapping[str, str]) -> str:
    value = str(row.get("split_group_id", "")).strip()
    return value or volume_key_for_row(row)


def sample_id_for_row(row: Mapping[str, str]) -> str:
    sample_id = row.get("sample_id", "")
    if sample_id:
        return sample_id
    return f"{row.get('dataset', 'unknown')}:{row.get('volume_id') or Path(row['path']).stem}:slice{int(row['slice_idx']):04d}"


def _safe_token(value: object) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def preproc_path_for_manifest_row(preproc_root: str | Path | None, row: Mapping[str, str]) -> Optional[Path]:
    if preproc_root is None:
        return None
    source_path = Path(row["path"])
    digest = hashlib.sha256(str(source_path.resolve()).encode("utf-8")).hexdigest()[:16]
    tier = _safe_token(row.get("experiment_tier", "unknown_tier"))
    shift = _safe_token(row.get("shift_name", "unknown_shift"))
    dataset = _safe_token(row.get("dataset", "unknown_dataset"))
    volume = _safe_token(row.get("volume_id") or source_path.stem)
    root = Path(preproc_root)
    if root.name == shift:
        base = root
    elif root.name == tier:
        base = root / shift
    else:
        base = root / tier / shift
    return base / dataset / f"{volume}_{digest}.preproc.h5"


def split_source_rows(
    rows: Sequence[Dict[str, str]],
    subset: str,
    val_fraction: float = 0.2,
    seed: int = 7,
    allow_slice_val_fallback: bool = False,
) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
    if subset not in {"train", "val", "all"}:
        raise ValueError(f"subset must be train, val, or all; got {subset!r}")
    if not (0.0 <= float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    volume_keys = sorted({split_group_key_for_row(row) for row in rows})
    shuffled = list(volume_keys)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(shuffled)
    fallback_reason = ""
    if len(shuffled) <= 1 or float(val_fraction) == 0.0:
        val_count = 0
    else:
        val_count = max(1, int(round(len(shuffled) * float(val_fraction))))
        val_count = min(val_count, len(shuffled) - 1)

    val_keys = set(shuffled[:val_count])
    train_keys = set(shuffled[val_count:])
    fallback_train_ids: set[str] = set()
    fallback_val_ids: set[str] = set()
    if val_count == 0 and allow_slice_val_fallback and len(rows) > 1 and float(val_fraction) > 0.0:
        fallback_reason = "slice_level_fallback_because_only_one_source_volume"
        indexed = list(range(len(rows)))
        rng.shuffle(indexed)
        row_val_count = max(1, int(round(len(indexed) * float(val_fraction))))
        row_val_count = min(row_val_count, len(indexed) - 1)
        fallback_val_indices = set(indexed[:row_val_count])
        for row_idx, row in enumerate(rows):
            sample_id = sample_id_for_row(row)
            if row_idx in fallback_val_indices:
                fallback_val_ids.add(sample_id)
            else:
                fallback_train_ids.add(sample_id)

    if fallback_reason:
        if subset == "train":
            selected = [row for row in rows if sample_id_for_row(row) in fallback_train_ids]
        elif subset == "val":
            selected = [row for row in rows if sample_id_for_row(row) in fallback_val_ids]
        else:
            selected = list(rows)
    else:
        if subset == "train":
            selected_keys = train_keys
        elif subset == "val":
            selected_keys = val_keys
        else:
            selected_keys = set(volume_keys)
        selected = [row for row in rows if split_group_key_for_row(row) in selected_keys]

    summary = {
        "subset": subset,
        "val_fraction": float(val_fraction),
        "split_seed": int(seed),
        "allow_slice_val_fallback": bool(allow_slice_val_fallback),
        "fallback_reason": fallback_reason,
        "num_source_rows": len(rows),
        "num_source_volumes": len(volume_keys),
        "num_train_volumes": len(train_keys),
        "num_val_volumes": len(val_keys),
        "num_fallback_train_rows": len(fallback_train_ids),
        "num_fallback_val_rows": len(fallback_val_ids),
        "num_selected_rows": len(selected),
        "train_volume_keys": sorted(train_keys),
        "val_volume_keys": sorted(val_keys),
        "selected_sample_ids": [sample_id_for_row(row) for row in selected],
        "split_group_key": "split_group_id_or_volume_key",
    }
    return selected, summary


def discover_manifest_shapes(
    manifest_csv: str | Path,
    source_role: str = DEFAULT_SOURCE_ROLE,
) -> List[Tuple[int, int]]:
    rows = source_manifest_rows(manifest_csv, source_role=source_role)
    return sorted({shape_from_manifest_row(row) for row in rows})


def _read_h5_dataset(handle: h5py.File, key: str) -> h5py.Dataset:
    if not key or key == "none":
        raise KeyError("Empty h5 dataset key")
    if key not in handle:
        raise KeyError(f"Missing h5 dataset key {key!r}")
    obj = handle[key]
    if not isinstance(obj, h5py.Dataset):
        raise KeyError(f"h5 key {key!r} is not a dataset")
    return obj


def _full_static_kspace_from_h5(handle: h5py.File, row: Mapping[str, str], slice_id: int) -> np.ndarray:
    kspace_key = row.get("kspace_key") or "kspace"
    if kspace_key != "kspace":
        raise ValueError(
            f"Static source training currently supports fastMRI-style top-level kspace only; got {kspace_key!r}"
        )
    dset = _read_h5_dataset(handle, kspace_key)
    shape = dset.shape
    if len(shape) == 5:
        return ensure_complex(dset[:, slice_id]).astype(np.complex64, copy=False)
    if len(shape) == 4:
        return ensure_complex(dset[slice_id])[None].astype(np.complex64, copy=False)
    if len(shape) == 3:
        return ensure_complex(dset[slice_id][None, ...])[None].astype(np.complex64, copy=False)
    raise ValueError(f"Unsupported kspace shape in {row.get('path')}: {shape}")


def _select_static_frame(kspace_tchw: np.ndarray) -> np.ndarray:
    if kspace_tchw.shape[0] == 1:
        return kspace_tchw
    center = int(kspace_tchw.shape[0] // 2)
    return kspace_tchw[center : center + 1]


def _target_key(row: Mapping[str, str], handle: h5py.File) -> Optional[str]:
    key = row.get("target_key", "")
    if key and key != "none" and key in handle:
        return key
    if "reconstruction_rss" in handle:
        return "reconstruction_rss"
    if "reconstruction_esc" in handle:
        return "reconstruction_esc"
    return None


def _read_static_target(handle: h5py.File, row: Mapping[str, str], slice_id: int, scale: float) -> Optional[np.ndarray]:
    key = _target_key(row, handle)
    if key is None:
        return None
    target = np.asarray(handle[key][slice_id], dtype=np.float32)
    if target.ndim == 3:
        target = target[target.shape[0] // 2]
    target = target / max(float(scale), EPS)
    return target[None, None, ...].astype(np.float32)


def _complex_noise_for_snr(
    kspace: np.ndarray,
    *,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    signal_power = float(np.mean(np.abs(kspace) ** 2))
    if not np.isfinite(signal_power) or signal_power <= 0.0:
        return np.zeros_like(kspace, dtype=np.complex64), 0.0
    noise_sigma2 = signal_power / (10.0 ** (float(snr_db) / 10.0))
    component_std = float(np.sqrt(noise_sigma2 / 2.0))
    real = rng.normal(loc=0.0, scale=component_std, size=kspace.shape).astype(np.float32)
    imag = rng.normal(loc=0.0, scale=component_std, size=kspace.shape).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64, copy=False), float(noise_sigma2)


def _validate_positive_snr(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive when provided, got {value}")
    return out


class StaticShiftSourceENSUREDataset(Dataset):
    """Return static source-domain samples from shift manifests."""

    def __init__(
        self,
        manifest_csv: str | Path,
        subset: str = "train",
        preproc_root: str | Path | None = None,
        density_root: str | Path | None = None,
        source_role: str = DEFAULT_SOURCE_ROLE,
        val_fraction: float = 0.2,
        split_seed: int = 7,
        allow_slice_val_fallback: bool = False,
        acceleration: float = 4.0,
        sigma_mask: float = 0.18,
        window_size: int = 1,
        deterministic_masks: bool = True,
        mask_seed: int = 0,
        max_prob: float = 0.95,
        normalize: bool = True,
        return_target: bool = True,
        require_preproc: bool = False,
        cache_maps: bool = True,
        input_noise_snr_db: float | None = None,
        input_noise_snr_db_min: float | None = None,
        input_noise_snr_db_max: float | None = None,
        input_noise_seed: int = 0,
        test_noise_snr_db: float | None = None,
        test_noise_seed: int = 0,
        shift_names: Sequence[str] | None = None,
    ) -> None:
        if int(window_size) != 1:
            raise ValueError("StaticShiftSourceENSUREDataset is source-static and currently requires window_size=1")
        explicit_input_noise = any(
            value is not None
            for value in (input_noise_snr_db, input_noise_snr_db_min, input_noise_snr_db_max)
        )
        if explicit_input_noise and test_noise_snr_db is not None:
            raise ValueError("Use either input_noise_* or test_noise_snr_db, not both.")

        fixed_snr = _validate_positive_snr(input_noise_snr_db, "input_noise_snr_db")
        min_snr = _validate_positive_snr(input_noise_snr_db_min, "input_noise_snr_db_min")
        max_snr = _validate_positive_snr(input_noise_snr_db_max, "input_noise_snr_db_max")
        if (min_snr is None) != (max_snr is None):
            raise ValueError("input_noise_snr_db_min and input_noise_snr_db_max must be provided together.")
        if fixed_snr is not None and min_snr is not None:
            raise ValueError("Use either fixed input_noise_snr_db or an input_noise_snr_db_min/max range, not both.")
        if min_snr is not None and max_snr is not None and min_snr > max_snr:
            raise ValueError(
                f"input_noise_snr_db_min must be <= input_noise_snr_db_max, got {min_snr} > {max_snr}"
            )
        test_snr = _validate_positive_snr(test_noise_snr_db, "test_noise_snr_db")
        if test_snr is not None:
            fixed_snr = test_snr
            input_noise_seed = int(test_noise_seed)
        self.manifest_csv = Path(manifest_csv)
        self.subset = subset
        self.preproc_root = Path(preproc_root) if preproc_root is not None else None
        self.density_root = Path(density_root) if density_root is not None else None
        self.source_role = source_role
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.allow_slice_val_fallback = bool(allow_slice_val_fallback)
        self.acceleration = float(acceleration)
        self.sigma_mask = float(sigma_mask)
        self.window_size = int(window_size)
        self.deterministic_masks = bool(deterministic_masks)
        self.mask_seed = int(mask_seed)
        self.max_prob = float(max_prob)
        self.normalize = bool(normalize)
        self.return_target = bool(return_target)
        self.require_preproc = bool(require_preproc)
        self.cache_maps = bool(cache_maps)
        self.input_noise_snr_db = fixed_snr
        self.input_noise_snr_db_min = min_snr
        self.input_noise_snr_db_max = max_snr
        self.input_noise_seed = int(input_noise_seed)
        self.input_noise_epoch = 0
        self.shift_names = tuple(str(name) for name in (shift_names or ()))
        self._input_noise_seed_token = "test_noise" if test_snr is not None else "input_noise"
        self._input_noise_include_epoch = test_snr is None
        self._map_cache: Dict[Tuple[Path, int], np.ndarray] = {}
        self._density_cache: Dict[Tuple[int, int, float, float], Dict[str, np.ndarray]] = {}

        source_rows = source_manifest_rows(
            self.manifest_csv,
            source_role=self.source_role,
            shift_names=self.shift_names,
        )
        if not source_rows:
            raise ValueError(f"No rows with split_role={self.source_role!r} in {self.manifest_csv}")
        selected_rows, split_summary = split_source_rows(
            source_rows,
            subset=self.subset,
            val_fraction=self.val_fraction,
            seed=self.split_seed,
            allow_slice_val_fallback=self.allow_slice_val_fallback,
        )
        if not selected_rows:
            raise ValueError(
                f"No rows selected for subset={self.subset!r}; "
                f"source_rows={len(source_rows)}, val_fraction={self.val_fraction}"
            )
        self._all_source_rows = source_rows
        self._split_summary = split_summary
        self.samples = [self._row_to_sample(row) for row in selected_rows]

    def set_noise_epoch(self, epoch: int) -> None:
        self.input_noise_epoch = int(epoch)

    def _row_to_sample(self, row: Dict[str, str]) -> StaticShiftSample:
        height, width = shape_from_manifest_row(row)
        path = Path(row["path"])
        num_slices = int(row.get("num_slices") or 0)
        slice_id = int(row["slice_idx"])
        if num_slices <= 0:
            num_slices = slice_id + 1
        preproc_path = preproc_path_for_manifest_row(self.preproc_root, row)
        if self.require_preproc and (preproc_path is None or not preproc_path.exists()):
            raise FileNotFoundError(f"Missing preprocessing sidecar for {sample_id_for_row(row)}: {preproc_path}")
        return StaticShiftSample(
            row=dict(row),
            preproc_path=preproc_path,
            sample_id=sample_id_for_row(row),
            volume_key=volume_key_for_row(row),
            source_path=path,
            volume_id=row.get("volume_id") or path.stem,
            slice_id=slice_id,
            num_slices=num_slices,
            num_coils=coil_count_from_row(row),
            height=height,
            width=width,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def split_summary(self) -> Dict[str, object]:
        out = dict(self._split_summary)
        out.update(
            {
                "manifest_csv": str(self.manifest_csv),
                "manifest_sha256": manifest_sha256(self.manifest_csv),
                "source_role": self.source_role,
                "dataset_len": len(self),
                "input_noise_snr_db": self.input_noise_snr_db,
                "input_noise_snr_db_min": self.input_noise_snr_db_min,
                "input_noise_snr_db_max": self.input_noise_snr_db_max,
                "input_noise_seed": int(self.input_noise_seed),
                "input_noise_epoch": int(self.input_noise_epoch),
                "shift_names": list(self.shift_names),
            }
        )
        return out

    def _fallback_scale(self, handle: h5py.File, sample: StaticShiftSample) -> float:
        key = _target_key(sample.row, handle)
        if key is not None:
            target = np.asarray(handle[key][sample.slice_id], dtype=np.float32)
            scale = float(np.percentile(np.abs(target), 99.0))
            if np.isfinite(scale) and scale > 0:
                return scale
        attrs_max = float(handle.attrs.get("max", 0.0)) if "max" in handle.attrs else 0.0
        return attrs_max if attrs_max > 0 else 1.0

    def _read_preproc_values(self, sample: StaticShiftSample) -> Tuple[Optional[np.ndarray], float, float, Dict[str, object], bool]:
        maps = None
        scale = 1.0
        noise_sigma2 = 0.0
        meta: Dict[str, object] = {}
        path = sample.preproc_path
        if path is None or not path.exists():
            return maps, scale, noise_sigma2, meta, False

        with h5py.File(str(path), "r") as handle:
            processed = True
            if "processed_slice_mask" in handle:
                processed = bool(np.asarray(handle["processed_slice_mask"][sample.slice_id]).item())
            if "norm_scale" in handle and processed:
                value = float(handle["norm_scale"][sample.slice_id])
                if np.isfinite(value) and value > 0:
                    scale = value
            if "noise_sigma2" in handle and processed:
                value = float(handle["noise_sigma2"][sample.slice_id])
                if np.isfinite(value) and value >= 0:
                    noise_sigma2 = value
            if "maps" in handle and processed:
                maps = np.asarray(handle["maps"][sample.slice_id], dtype=np.complex64)
            meta["preproc_path"] = str(path)
            meta["map_method"] = str(handle.attrs.get("map_method", "unknown"))
            if not processed:
                meta["preproc_slice_processed"] = False
        return maps, scale, noise_sigma2, meta, processed

    def _get_maps(
        self,
        sample: StaticShiftSample,
        full_slice_kspace_tchw: np.ndarray,
        preproc_maps: Optional[np.ndarray],
    ) -> np.ndarray:
        if preproc_maps is not None:
            return preproc_maps.astype(np.complex64, copy=False)
        cache_key = (sample.source_path, sample.slice_id)
        if self.cache_maps and cache_key in self._map_cache:
            return self._map_cache[cache_key]
        maps = estimate_maps_rss(full_slice_kspace_tchw)
        if self.cache_maps:
            self._map_cache[cache_key] = maps
        return maps

    def _get_density_stats(self, height: int, width: int) -> Dict[str, np.ndarray]:
        key = (height, width, self.acceleration, self.sigma_mask)
        if key not in self._density_cache:
            self._density_cache[key] = load_density_stats(
                density_root=self.density_root,
                height=height,
                width=width,
                acceleration=self.acceleration,
                sigma_mask=self.sigma_mask,
            )
        return self._density_cache[key]

    def _rng_for_sample(self, sample: StaticShiftSample, idx: int) -> np.random.Generator:
        if self.deterministic_masks:
            seed = stable_seed(
                self.mask_seed,
                sample.sample_id,
                sample.volume_id,
                sample.slice_id,
                self.acceleration,
                self.sigma_mask,
            )
            return np.random.default_rng(seed)
        return np.random.default_rng()

    def _rng_for_input_noise(self, sample: StaticShiftSample) -> np.random.Generator:
        seed_parts: list[object] = [
            self.input_noise_seed,
            sample.sample_id,
            sample.volume_id,
            sample.slice_id,
            self._input_noise_seed_token,
        ]
        if self._input_noise_include_epoch:
            seed_parts.append(self.input_noise_epoch)
        seed = stable_seed(*seed_parts)
        return np.random.default_rng(seed)

    def _sample_input_noise_snr(self, rng: np.random.Generator) -> float | None:
        if self.input_noise_snr_db is not None:
            return float(self.input_noise_snr_db)
        if self.input_noise_snr_db_min is not None and self.input_noise_snr_db_max is not None:
            return float(rng.uniform(self.input_noise_snr_db_min, self.input_noise_snr_db_max))
        return None

    def __getitem__(self, idx: int) -> Mapping[str, object]:
        sample = self.samples[int(idx)]
        preproc_maps, scale, noise_sigma2, meta_from_preproc, preproc_processed = self._read_preproc_values(sample)

        with h5py.File(str(sample.source_path), "r") as handle:
            full_slice_kspace = _full_static_kspace_from_h5(handle, sample.row, sample.slice_id)
            kspace_fs = _select_static_frame(full_slice_kspace)
            if self.normalize and not preproc_processed:
                scale = self._fallback_scale(handle, sample)
            target = _read_static_target(handle, sample.row, sample.slice_id, scale) if self.return_target else None

        maps = self._get_maps(sample, full_slice_kspace, preproc_maps)
        if not preproc_processed:
            noise_sigma2 = estimate_noise_sigma2(full_slice_kspace, scale=scale)

        if self.normalize:
            kspace_fs = kspace_fs / max(float(scale), EPS)

        input_noise_sigma2 = 0.0
        input_noise_snr_db = float("nan")
        rng_noise = self._rng_for_input_noise(sample)
        sampled_snr_db = self._sample_input_noise_snr(rng_noise)
        if sampled_snr_db is not None:
            input_noise_snr_db = float(sampled_snr_db)
            noise, input_noise_sigma2 = _complex_noise_for_snr(
                kspace_fs,
                snr_db=sampled_snr_db,
                rng=rng_noise,
            )
            kspace_fs = (kspace_fs + noise).astype(np.complex64, copy=False)
            noise_sigma2 = float(noise_sigma2) + float(input_noise_sigma2)

        density_stats = self._get_density_stats(sample.height, sample.width)
        if "density_line_prob" in density_stats:
            line_prob = density_stats["density_line_prob"]
        else:
            line_prob = bernoulli_gaussian_line_prob(
                width=sample.width,
                acceleration=self.acceleration,
                sigma_mask=self.sigma_mask,
                max_prob=self.max_prob,
            )

        rng = self._rng_for_sample(sample, int(idx))
        mask, mask_prob = make_mask_from_line_prob(
            line_prob=line_prob,
            height=sample.height,
            num_frames=self.window_size,
            rng=rng,
        )
        kspace_us = kspace_fs * mask
        zf = adjoint_image(kspace_us, maps)

        meta: Dict[str, object] = {
            "sample_id": sample.sample_id,
            "volume_id": sample.volume_id,
            "volume_key": sample.volume_key,
            "slice_id": int(sample.slice_id),
            "source_file": str(sample.source_path),
            "manifest_csv": str(self.manifest_csv),
            "subset": self.subset,
            "split_role": sample.row.get("split_role", ""),
            "shift_name": sample.row.get("shift_name", ""),
            "experiment_tier": sample.row.get("experiment_tier", ""),
            "dataset": sample.row.get("dataset", ""),
            "norm_scale": float(scale),
            "acceleration": float(self.acceleration),
            "sigma_mask": float(self.sigma_mask),
            "window_size": int(self.window_size),
            "input_noise_enabled": sampled_snr_db is not None,
            "input_noise_snr_db": input_noise_snr_db,
            "input_noise_snr_db_min": (
                float(self.input_noise_snr_db_min) if self.input_noise_snr_db_min is not None else float("nan")
            ),
            "input_noise_snr_db_max": (
                float(self.input_noise_snr_db_max) if self.input_noise_snr_db_max is not None else float("nan")
            ),
            "input_noise_sigma2": float(input_noise_sigma2),
            "noise_sigma2_total": float(noise_sigma2),
            "input_noise_seed": int(self.input_noise_seed),
            "input_noise_epoch": int(self.input_noise_epoch),
        }
        meta.update(meta_from_preproc)
        if "density_stats_path" in density_stats:
            meta["density_stats_path"] = str(density_stats["density_stats_path"])

        out: Dict[str, object] = {
            "kspace_fs": torch.from_numpy(kspace_fs.astype(np.complex64, copy=False)),
            "kspace_us": torch.from_numpy(kspace_us.astype(np.complex64, copy=False)),
            "mask": torch.from_numpy(mask),
            "mask_prob": torch.from_numpy(mask_prob),
            "maps": torch.from_numpy(maps.astype(np.complex64, copy=False)),
            "zf": torch.from_numpy(zf),
            "noise_sigma2": torch.tensor(float(noise_sigma2), dtype=torch.float32),
            "meta": meta,
        }
        if target is not None:
            out["target_rss"] = torch.from_numpy(target)
        if "empirical_density" in density_stats:
            out["empirical_density"] = torch.from_numpy(density_stats["empirical_density"])
        if "inv_sqrt_density" in density_stats:
            out["inv_sqrt_density"] = torch.from_numpy(density_stats["inv_sqrt_density"])
        return out


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test StaticShiftSourceENSUREDataset.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--subset", choices=("train", "val", "all"), default="train")
    parser.add_argument("--preproc-root", type=Path, default=None)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument("--source-role", default=DEFAULT_SOURCE_ROLE)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--allow-slice-val-fallback", action="store_true")
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--sigma-mask", type=float, default=0.18)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--no-target", action="store_true")
    parser.add_argument("--input-noise-snr-db", type=float, default=None)
    parser.add_argument("--input-noise-snr-db-min", type=float, default=None)
    parser.add_argument("--input-noise-snr-db-max", type=float, default=None)
    parser.add_argument("--input-noise-seed", type=int, default=0)
    parser.add_argument("--input-noise-epoch", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    dataset = StaticShiftSourceENSUREDataset(
        manifest_csv=args.manifest_csv,
        subset=args.subset,
        preproc_root=args.preproc_root,
        density_root=args.density_root,
        source_role=args.source_role,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        allow_slice_val_fallback=args.allow_slice_val_fallback,
        acceleration=args.acceleration,
        sigma_mask=args.sigma_mask,
        return_target=not args.no_target,
        input_noise_snr_db=args.input_noise_snr_db,
        input_noise_snr_db_min=args.input_noise_snr_db_min,
        input_noise_snr_db_max=args.input_noise_snr_db_max,
        input_noise_seed=args.input_noise_seed,
    )
    dataset.set_noise_epoch(args.input_noise_epoch)
    sample = dataset[args.index]
    print(json.dumps(dataset.split_summary(), indent=2))
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
