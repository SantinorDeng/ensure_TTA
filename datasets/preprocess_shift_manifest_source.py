#!/usr/bin/env python3
"""Build preprocessing sidecars for source rows in shift manifests."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import h5py
import numpy as np

try:
    from .preprocess_raw_cine import (
        EPS,
        compute_noise_stats,
        create_dataset_kwargs,
        ensure_complex,
        estimate_maps_bart,
        estimate_maps_rss,
        load_bart,
    )
    from .shift_manifest_dataset import (
        DEFAULT_SOURCE_ROLE,
        preproc_path_for_manifest_row,
        source_manifest_rows,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from cardiac_ensure.datasets.preprocess_raw_cine import (
        EPS,
        compute_noise_stats,
        create_dataset_kwargs,
        ensure_complex,
        estimate_maps_bart,
        estimate_maps_rss,
        load_bart,
    )
    from cardiac_ensure.datasets.shift_manifest_dataset import (
        DEFAULT_SOURCE_ROLE,
        preproc_path_for_manifest_row,
        source_manifest_rows,
    )


def _full_slice_kspace(handle: h5py.File, slice_idx: int) -> np.ndarray:
    if "kspace" not in handle:
        raise KeyError("Expected fastMRI-style top-level kspace")
    shape = handle["kspace"].shape
    if len(shape) == 5:
        return ensure_complex(handle["kspace"][:, slice_idx]).astype(np.complex64, copy=False)
    if len(shape) == 4:
        return ensure_complex(handle["kspace"][slice_idx])[None].astype(np.complex64, copy=False)
    if len(shape) == 3:
        return ensure_complex(handle["kspace"][slice_idx][None, ...])[None].astype(np.complex64, copy=False)
    raise ValueError(f"Unsupported kspace shape: {shape}")


def _source_shape(handle: h5py.File) -> tuple[int, int, int, int, int]:
    if "kspace" not in handle:
        raise KeyError("Expected fastMRI-style top-level kspace")
    shape = handle["kspace"].shape
    if len(shape) == 5:
        num_t, num_slices, num_coils, height, width = map(int, shape)
    elif len(shape) == 4:
        num_slices, num_coils, height, width = map(int, shape)
        num_t = 1
    elif len(shape) == 3:
        num_slices, height, width = map(int, shape)
        num_t = 1
        num_coils = 1
    else:
        raise ValueError(f"Unsupported kspace shape: {shape}")
    return num_t, num_slices, num_coils, height, width


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(shifted, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(image, axes=(-2, -1))


def compute_static_norm_scale(
    handle: h5py.File,
    slice_idx: int,
    kspace_tchw: np.ndarray,
    norm_source: str = "reconstruction_rss",
    percentile: float = 99.0,
) -> tuple[float, str]:
    if norm_source == "reconstruction_rss" and "reconstruction_rss" in handle:
        dset = handle["reconstruction_rss"]
        if len(dset.shape) == 4:
            rss = np.asarray(dset[:, slice_idx], dtype=np.float32)
        elif len(dset.shape) == 3:
            rss = np.asarray(dset[slice_idx], dtype=np.float32)
        else:
            rss = None
        if rss is not None:
            scale = float(np.percentile(np.abs(rss), percentile))
            return max(scale, EPS), "reconstruction_rss"

    coil_images = ifft2c(kspace_tchw)
    rss_from_kspace = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=1))
    scale = float(np.percentile(np.abs(rss_from_kspace), percentile))
    return max(scale, EPS), "rss_from_kspace"


def _volume_groups(rows: Iterable[Mapping[str, str]]) -> Dict[Path, List[Mapping[str, str]]]:
    grouped: Dict[Path, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if (row.get("kspace_key") or "kspace") != "kspace":
            raise ValueError(
                f"Source preprocessing currently supports fastMRI-style top-level kspace only; "
                f"sample_id={row.get('sample_id', '')} kspace_key={row.get('kspace_key', '')!r}"
            )
        grouped[Path(row["path"])].append(row)
    return dict(sorted(grouped.items(), key=lambda item: str(item[0])))


def process_manifest_volume(
    *,
    source_path: Path,
    rows: Sequence[Mapping[str, str]],
    output_root: Path,
    map_method: str = "rss",
    norm_source: str = "reconstruction_rss",
    norm_percentile: float = 99.0,
    corner_fraction: float = 0.08,
    compression: str | None = "gzip",
    overwrite: bool = False,
    bart_fn: Callable | None = None,
    bart_ecalib_crop: float = 0.0,
    bart_ecalib_cmd: str | None = None,
) -> Path:
    if not rows:
        raise ValueError("rows must not be empty")
    out_path = preproc_path_for_manifest_row(output_root, rows[0])
    if out_path is None:
        raise ValueError("Could not resolve output sidecar path")
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = sorted({int(row["slice_idx"]) for row in rows})
    dset_kwargs = create_dataset_kwargs(compression)
    with h5py.File(str(source_path), "r") as handle:
        num_t, num_slices, num_coils, height, width = _source_shape(handle)
        processed_slice_mask = np.zeros((num_slices,), dtype=np.uint8)
        for slice_idx in selected:
            if slice_idx < 0 or slice_idx >= num_slices:
                raise IndexError(f"slice_idx={slice_idx} out of range for {source_path} with {num_slices} slices")
            processed_slice_mask[slice_idx] = 1

        with h5py.File(str(out_path), "w") as out:
            if map_method != "none":
                maps_ds = out.create_dataset(
                    "maps",
                    shape=(num_slices, num_coils, height, width),
                    dtype=np.complex64,
                    chunks=(1, num_coils, height, width),
                    **dset_kwargs,
                )
            else:
                maps_ds = None

            norm_scale = out.create_dataset("norm_scale", shape=(num_slices,), dtype=np.float32)
            noise_sigma2 = out.create_dataset("noise_sigma2", shape=(num_slices,), dtype=np.float32)
            noise_sigma2_raw = out.create_dataset("noise_sigma2_raw", shape=(num_slices,), dtype=np.float32)
            noise_var_real = out.create_dataset("noise_var_real", shape=(num_slices,), dtype=np.float32)
            noise_var_imag = out.create_dataset("noise_var_imag", shape=(num_slices,), dtype=np.float32)
            norm_source_used = out.create_dataset(
                "norm_source_used",
                shape=(num_slices,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            sample_ids = out.create_dataset(
                "source_manifest_sample_ids",
                shape=(num_slices,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            out.create_dataset("processed_slice_mask", data=processed_slice_mask)

            norm_scale[...] = np.nan
            noise_sigma2[...] = np.nan
            noise_sigma2_raw[...] = np.nan
            noise_var_real[...] = np.nan
            noise_var_imag[...] = np.nan
            norm_source_used[...] = ""
            sample_ids[...] = ""

            for key, value in handle.attrs.items():
                try:
                    out.attrs[f"source_attr_{key}"] = value
                except TypeError:
                    out.attrs[f"source_attr_{key}"] = str(value)
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_shape"] = np.asarray(handle["kspace"].shape, dtype=np.int32)
            out.attrs["source_num_t"] = int(num_t)
            out.attrs["map_method"] = map_method
            out.attrs["norm_source_requested"] = norm_source
            out.attrs["norm_percentile"] = float(norm_percentile)
            out.attrs["noise_sigma2_definition"] = "var(real) + var(imag), normalized by norm_scale**2"
            out.attrs["preproc_kind"] = "static_shift_source"
            out.attrs["preproc_version"] = 2
            out.attrs["norm_shape_handling"] = "fastMRI [slice,H,W] and cine [time,slice,H,W]"
            out.attrs["manifest_role"] = rows[0].get("split_role", "")
            out.attrs["manifest_shift_name"] = rows[0].get("shift_name", "")
            out.attrs["manifest_experiment_tier"] = rows[0].get("experiment_tier", "")

            rows_by_slice = {int(row["slice_idx"]): row for row in rows}
            for slice_idx in selected:
                kspace_tchw = _full_slice_kspace(handle, slice_idx)
                scale, used_source = compute_static_norm_scale(
                    handle=handle,
                    slice_idx=slice_idx,
                    kspace_tchw=kspace_tchw,
                    norm_source=norm_source,
                    percentile=norm_percentile,
                )
                sigma2_raw, var_r_raw, var_i_raw = compute_noise_stats(
                    kspace_tchw=kspace_tchw,
                    corner_fraction=corner_fraction,
                )
                scale2 = max(scale * scale, EPS)

                norm_scale[slice_idx] = np.float32(scale)
                noise_sigma2_raw[slice_idx] = np.float32(sigma2_raw)
                noise_var_real[slice_idx] = np.float32(var_r_raw / scale2)
                noise_var_imag[slice_idx] = np.float32(var_i_raw / scale2)
                noise_sigma2[slice_idx] = np.float32(sigma2_raw / scale2)
                norm_source_used[slice_idx] = used_source
                sample_ids[slice_idx] = rows_by_slice[slice_idx].get("sample_id", "")

                if maps_ds is not None:
                    if map_method == "rss":
                        maps = estimate_maps_rss(kspace_tchw)
                    elif map_method == "bart":
                        if bart_fn is None:
                            raise RuntimeError("map_method='bart' requested but BART was not loaded")
                        maps = estimate_maps_bart(
                            kspace_tchw=kspace_tchw,
                            bart_fn=bart_fn,
                            ecalib_crop=bart_ecalib_crop,
                            ecalib_cmd=bart_ecalib_cmd,
                        )
                    else:
                        raise ValueError(f"Unsupported map_method: {map_method}")
                    maps_ds[slice_idx] = maps

    return out_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-role", default=DEFAULT_SOURCE_ROLE)
    parser.add_argument("--map-method", choices=("rss", "bart", "none"), default="rss")
    parser.add_argument("--norm-source", choices=("reconstruction_rss", "rss_from_kspace"), default="reconstruction_rss")
    parser.add_argument("--norm-percentile", type=float, default=99.0)
    parser.add_argument("--corner-fraction", type=float, default=0.08)
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--bart-toolbox-path", default=None)
    parser.add_argument("--bart-python-path", default=None)
    parser.add_argument("--bart-ecalib-crop", type=float, default=0.0)
    parser.add_argument("--bart-ecalib-cmd", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    bart_fn = None
    if args.map_method == "bart":
        bart_fn = load_bart(
            bart_toolbox_path=args.bart_toolbox_path,
            bart_python_path=args.bart_python_path,
        )

    rows = source_manifest_rows(args.manifest_csv, source_role=args.source_role)
    if not rows:
        raise SystemExit(f"No rows with split_role={args.source_role!r} in {args.manifest_csv}")
    grouped = list(_volume_groups(rows).items())
    if args.max_volumes is not None:
        grouped = grouped[: int(args.max_volumes)]
    if not grouped:
        raise SystemExit(f"No source volumes found in {args.manifest_csv}")

    for idx, (source_path, volume_rows) in enumerate(grouped, start=1):
        out_path = process_manifest_volume(
            source_path=source_path,
            rows=volume_rows,
            output_root=args.output_root,
            map_method=args.map_method,
            norm_source=args.norm_source,
            norm_percentile=args.norm_percentile,
            corner_fraction=args.corner_fraction,
            compression=args.compression,
            overwrite=args.overwrite,
            bart_fn=bart_fn,
            bart_ecalib_crop=args.bart_ecalib_crop,
            bart_ecalib_cmd=args.bart_ecalib_cmd,
        )
        print(f"[{idx}/{len(grouped)}] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
