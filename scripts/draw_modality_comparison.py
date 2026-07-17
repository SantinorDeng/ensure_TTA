#!/usr/bin/env python3
"""Summarize and visualize modality-shift reconstruction comparisons.

This script is intentionally result-driven: it reads the already saved
summary/metrics/recon files for the three brain shifts and creates compact
per-shift tables plus paper-style qualitative figures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "figures" / "modality_comparison"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    result_dir: Path
    is_tta: bool

    @property
    def summary_csv(self) -> Path:
        return self.result_dir / "summary.csv"

    @property
    def metrics_csv(self) -> Path:
        return self.result_dir / "metrics.csv"


@dataclass(frozen=True)
class ShiftSpec:
    key: str
    label: str
    methods: tuple[MethodSpec, ...]


def rel(path: str) -> Path:
    return ROOT / path


def make_shifts(result_noise_tag: str) -> tuple[ShiftSpec, ...]:
    """Build result paths for clean or noisy comparison runs."""
    if result_noise_tag == "clean":
        baseline_tag = "clean"
        tta_tag = "clean"
    else:
        baseline_tag = result_noise_tag
        tta_tag = result_noise_tag

    return (
        ShiftSpec(
            key="t1_to_flair",
            label="AXT1 -> AXFLAIR",
            methods=(
                MethodSpec(
                    "zero_filled",
                    "Zero-filled",
                    rel(f"outputs/baselines/shifts/modality_matrix/main/brain/axflair/zero_filled_{baseline_tag}_maskseed7_noiseseed9007"),
                    False,
                ),
                MethodSpec(
                    "pics",
                    "PICS",
                    rel(f"outputs/baselines/shifts/modality_matrix/main/brain/axflair/bart_pics_tvxy_lam0p05_{baseline_tag}_maskseed7_noiseseed9007"),
                    False,
                ),
                MethodSpec(
                    "ssdu",
                    "SSDU + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt1/ssdu_l1_{tta_tag}_maskseed7_noiseseed9007_subset97e69ab1"),
                    True,
                ),
                MethodSpec(
                    "traditional",
                    "Traditional + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt1/traditional_l1_{tta_tag}_maskseed7_noiseseed9007_subset97e69ab1"),
                    True,
                ),
                MethodSpec(
                    "ensure",
                    "ENSURE + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt1/ensure_l1_{tta_tag}_maskseed7_noiseseed9007_subset97e69ab1"),
                    True,
                ),
            ),
        ),
        ShiftSpec(
            key="t2_to_flair",
            label="AXT2 -> AXFLAIR",
            methods=(
                MethodSpec(
                    "zero_filled",
                    "Zero-filled",
                    rel(f"outputs/baselines/shifts/modality_matrix/main/brain/axflair/zero_filled_{baseline_tag}_maskseed7_noiseseed9007"),
                    False,
                ),
                MethodSpec(
                    "pics",
                    "PICS",
                    rel(f"outputs/baselines/shifts/modality_matrix/main/brain/axflair/bart_pics_tvxy_lam0p05_{baseline_tag}_maskseed7_noiseseed9007"),
                    False,
                ),
                MethodSpec(
                    "ssdu",
                    "SSDU + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt2/ssdu_l1_{tta_tag}_maskseed7_noiseseed9007_subset9d20678e"),
                    True,
                ),
                MethodSpec(
                    "traditional",
                    "Traditional + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt2/traditional_l1_{tta_tag}_maskseed7_noiseseed9007_subset9d20678e"),
                    True,
                ),
                MethodSpec(
                    "ensure",
                    "ENSURE + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt2/ensure_l1_{tta_tag}_maskseed7_noiseseed9007_subset9d20678e"),
                    True,
                ),
            ),
        ),
        ShiftSpec(
            key="pre_to_post",
            label="AXT1PRE -> AXT1POST",
            methods=(
                MethodSpec(
                    "zero_filled",
                    "Zero-filled",
                    rel(f"outputs/baselines/shifts/modality_matrix/main/brain/axt1post/zero_filled_{baseline_tag}_maskseed7_noiseseed9007"),
                    False,
                ),
                MethodSpec(
                    "pics",
                    "PICS",
                    rel(f"outputs/baselines/shifts/modality_matrix/main/brain/axt1post/bart_pics_tvxy_lam0p05_{baseline_tag}_maskseed7_noiseseed9007"),
                    False,
                ),
                MethodSpec(
                    "ssdu",
                    "SSDU + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt1pre/ssdu_l1_{tta_tag}_maskseed7_noiseseed9007_subset5ea2ce8d"),
                    True,
                ),
                MethodSpec(
                    "traditional",
                    "Traditional + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt1pre/traditional_l1_{tta_tag}_maskseed7_noiseseed9007_subset5ea2ce8d"),
                    True,
                ),
                MethodSpec(
                    "ensure",
                    "ENSURE + TTA",
                    rel(f"outputs/tta/shifts/modality_matrix/main/brain/axt1pre/ensure_l1_{tta_tag}_maskseed7_noiseseed9007_subset5ea2ce8d"),
                    True,
                ),
            ),
        ),
    )

SELECTED_SAMPLE_OVERRIDES: dict[str, str] = {
    "t1_to_flair": "fastmri_brain:file_brain_AXFLAIR_200_6002560:slice0006",
    "t2_to_flair": "fastmri_brain:file_brain_AXFLAIR_200_6002584:slice0010",
}

ZOOM_ROIS: dict[str, tuple[int, int, int, int]] = {
    "pre_to_post": (145, 55, 55, 40),
    "t1_to_flair": (182, 120, 62, 44),
    "t2_to_flair": (112, 92, 58, 42),
}


TABLE_COLUMNS = [
    "shift",
    "method",
    "num_samples",
    "before_nmse",
    "after_nmse",
    "before_psnr",
    "after_psnr",
    "before_ssim",
    "after_ssim",
    "runtime_sec_per_slice",
]


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in headers) + " |")
    return "\n".join(lines) + "\n"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def read_summary_row(shift: ShiftSpec, method: MethodSpec) -> dict[str, object]:
    require_file(method.summary_csv)
    raw = pd.read_csv(method.summary_csv).iloc[0].to_dict()
    row: dict[str, object] = {
        "shift": shift.label,
        "method": method.label,
        "num_samples": int(raw["num_samples"]),
        "after_nmse": float(raw["after_tta_nmse_mean"]),
        "after_psnr": float(raw["after_tta_psnr_mean"]),
        "after_ssim": float(raw["after_tta_ssim_mean"]),
        "runtime_sec_per_slice": float(raw["runtime_sec_per_slice"]),
    }
    if method.is_tta:
        row.update(
            {
                "before_nmse": float(raw["before_tta_nmse_mean"]),
                "before_psnr": float(raw["before_tta_psnr_mean"]),
                "before_ssim": float(raw["before_tta_ssim_mean"]),
            }
        )
    else:
        row.update(
            {
                "before_nmse": np.nan,
                "before_psnr": np.nan,
                "before_ssim": np.nan,
            }
        )
    return {column: row[column] for column in TABLE_COLUMNS}


def build_tables(shifts: tuple[ShiftSpec, ...], output_dir: Path) -> pd.DataFrame:
    rows = []
    for shift in shifts:
        shift_rows = [read_summary_row(shift, method) for method in shift.methods]
        shift_df = pd.DataFrame(shift_rows, columns=TABLE_COLUMNS)
        shift_df.to_csv(output_dir / f"{shift.key}_summary_table.csv", index=False)
        (output_dir / f"{shift.key}_summary_table.md").write_text(
            dataframe_to_markdown(shift_df),
            encoding="utf-8",
        )
        rows.extend(shift_rows)
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def read_metrics(method: MethodSpec) -> pd.DataFrame:
    require_file(method.metrics_csv)
    df = pd.read_csv(method.metrics_csv)
    df = df[df["status"] == "ok"].copy()
    df["method_key"] = method.key
    df["method_label"] = method.label
    df["metric_nmse"] = df["after_tta_nmse"].astype(float)
    df["metric_ssim"] = df["after_tta_ssim"].astype(float)
    return df


def choose_sample(shift: ShiftSpec) -> dict[str, object]:
    metrics = {method.key: read_metrics(method) for method in shift.methods}
    ensure = metrics["ensure"][["sample_id", "metric_nmse", "metric_ssim"]].rename(
        columns={"metric_nmse": "ensure_nmse", "metric_ssim": "ensure_ssim"}
    )
    merged = ensure.copy()
    other_keys = [method.key for method in shift.methods if method.key != "ensure"]
    for key in other_keys:
        other = metrics[key][["sample_id", "metric_nmse", "metric_ssim"]].rename(
            columns={"metric_nmse": f"{key}_nmse", "metric_ssim": f"{key}_ssim"}
        )
        merged = merged.merge(other, on="sample_id", how="inner")

    other_nmse_cols = [f"{key}_nmse" for key in other_keys]
    other_ssim_cols = [f"{key}_ssim" for key in other_keys]
    merged["best_other_nmse"] = merged[other_nmse_cols].min(axis=1)
    merged["best_other_ssim"] = merged[other_ssim_cols].max(axis=1)
    merged["worst_other_nmse"] = merged[other_nmse_cols].max(axis=1)
    # Positive score means ENSURE is better than the nearest competitor. The
    # relative term avoids only selecting extremely easy slices.
    merged["score"] = (
        (merged["best_other_nmse"] - merged["ensure_nmse"]) / merged["best_other_nmse"].clip(lower=1.0e-8)
        + 0.35 * (merged["ensure_ssim"] - merged["best_other_ssim"])
        + 0.15 * (merged["worst_other_nmse"] - merged["ensure_nmse"]) / merged["worst_other_nmse"].clip(lower=1.0e-8)
    )
    sample_override = SELECTED_SAMPLE_OVERRIDES.get(shift.key)
    if sample_override is not None:
        matches = merged[merged["sample_id"].astype(str) == sample_override]
        if matches.empty:
            raise ValueError(f"Selected sample override for {shift.key} was not found: {sample_override}")
        selected = matches.iloc[0]
    else:
        selected = merged.sort_values("score", ascending=False).iloc[0]
    sample_id = str(selected["sample_id"])
    return {
        "sample_id": sample_id,
        "score": float(selected["score"]),
        "ensure_nmse": float(selected["ensure_nmse"]),
        "best_other_nmse": float(selected["best_other_nmse"]),
        "ensure_ssim": float(selected["ensure_ssim"]),
        "best_other_ssim": float(selected["best_other_ssim"]),
        "rows": {key: metrics[key][metrics[key]["sample_id"] == sample_id].iloc[0] for key in metrics},
    }


def squeeze_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    arr = np.abs(arr)
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(np.float32, copy=False)


def center_crop_to(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = image.shape[-2:]
    out_h, out_w = shape
    start_h = max((h - out_h) // 2, 0)
    start_w = max((w - out_w) // 2, 0)
    return image[start_h : start_h + out_h, start_w : start_w + out_w]


def foreground_crop(images: Iterable[np.ndarray], gt: np.ndarray, margin: int = 18) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    threshold = max(float(np.percentile(gt, 82)) * 0.20, float(gt.max()) * 0.03)
    ys, xs = np.where(gt > threshold)
    if len(ys) == 0 or len(xs) == 0:
        return list(images), (0, gt.shape[0], 0, gt.shape[1])

    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, gt.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, gt.shape[1])
    height = y1 - y0
    width = x1 - x0
    side = min(max(height, width), min(gt.shape))
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = max(min(cy - side // 2, gt.shape[0] - side), 0)
    x0 = max(min(cx - side // 2, gt.shape[1] - side), 0)
    y1 = y0 + side
    x1 = x0 + side
    return [image[y0:y1, x0:x1] for image in images], (y0, y1, x0, x1)


def load_recon(row: pd.Series, method: MethodSpec) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(row["recon_npz"]))
    if not path.is_absolute():
        path = ROOT / path
    require_file(path)
    data = np.load(path)
    target = squeeze_image(data["target"])
    if method.is_tta:
        recon = squeeze_image(data["after_tta_rec"])
    else:
        recon = squeeze_image(data["recon"])
    recon = center_crop_to(recon, target.shape)
    return recon, target


def clamp_roi(roi: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, width, height = roi
    image_h, image_w = image_shape
    width = max(1, min(int(width), image_w))
    height = max(1, min(int(height), image_h))
    x0 = max(0, min(int(x0), image_w - width))
    y0 = max(0, min(int(y0), image_h - height))
    return x0, y0, width, height


def add_zoom_box(
    ax: plt.Axes,
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    x0, y0, width, height = clamp_roi(roi, image.shape)
    edge_color = "white"
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=edge_color,
            linewidth=1.0,
        )
    )

    inset_ax = ax.inset_axes([0.68, 0.68, 0.30, 0.30])
    inset = image[y0 : y0 + height, x0 : x0 + width]
    inset_ax.imshow(inset, cmap=cmap, vmin=vmin, vmax=vmax)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(1.0)


def plot_shift(shift: ShiftSpec, selection: dict[str, object], output_dir: Path, *, dpi: int) -> Path:
    rows = selection["rows"]
    images: list[np.ndarray] = []
    labels: list[str] = []
    target: np.ndarray | None = None
    for method in shift.methods:
        recon, gt = load_recon(rows[method.key], method)
        images.append(recon)
        labels.append(method.label)
        target = gt
    if target is None:
        raise RuntimeError("No target loaded")
    images.append(target)
    labels.append("Ground truth")

    cropped, crop = foreground_crop(images, target)
    gt_crop = cropped[-1]
    vmax = float(np.percentile(gt_crop, 99.5))
    vmax = max(vmax, 1.0e-6)
    display_images = [np.clip(image / vmax, 0, 1) for image in cropped]
    errors = [np.abs(image - gt_crop) / vmax for image in cropped[:-1]]
    error_vmax = 0.5
    zoom_roi = ZOOM_ROIS.get(shift.key)

    ncols = len(labels)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(2.0 * ncols, 4.3),
        gridspec_kw={"wspace": 0.02, "hspace": 0.02},
        constrained_layout=False,
    )
    fig.suptitle(
        f"{shift.label} | selected {selection['sample_id']} | "
        f"ENSURE NMSE {selection['ensure_nmse']:.4f} vs best other {selection['best_other_nmse']:.4f}",
        fontsize=11,
        y=0.98,
    )
    for col, (image, label) in enumerate(zip(display_images, labels)):
        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(label, fontsize=9)
        if zoom_roi is not None:
            add_zoom_box(axes[0, col], image, zoom_roi, cmap="gray", vmin=0, vmax=1)
        axes[0, col].axis("off")
        error_mappable = None
        if col < ncols - 1:
            err = errors[col]
            error_mappable = axes[1, col].imshow(err, cmap="jet", vmin=0, vmax=error_vmax)
            if zoom_roi is not None:
                add_zoom_box(axes[1, col], err, zoom_roi, cmap="jet", vmin=0, vmax=error_vmax)
        else:
            error_mappable = axes[1, col].imshow(np.zeros_like(gt_crop), cmap="jet", vmin=0, vmax=error_vmax)
        axes[1, col].axis("off")
    if error_mappable is not None:
        cbar = fig.colorbar(error_mappable, ax=axes[1, -1], fraction=0.046, pad=0.02)
        cbar.set_ticks([0.0, 0.25, 0.5])
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Error", fontsize=8)
    axes[0, 0].text(
        -0.09,
        0.5,
        "Recon",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
    )
    axes[1, 0].text(
        -0.09,
        0.5,
        "Error",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=9,
    )
    out_path = output_dir / f"{shift.key}_comparison.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / f"{shift.key}_comparison.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    metadata = {
        "shift": shift.key,
        "label": shift.label,
        "sample_id": selection["sample_id"],
        "score": selection["score"],
        "ensure_nmse": selection["ensure_nmse"],
        "best_other_nmse": selection["best_other_nmse"],
        "ensure_ssim": selection["ensure_ssim"],
        "best_other_ssim": selection["best_other_ssim"],
        "crop_y0_y1_x0_x1": crop,
        "zoom_roi_x0_y0_width_height": zoom_roi,
    }
    (output_dir / f"{shift.key}_selected_sample.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-noise-tag", choices=("clean", "snr15"), default="clean")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shifts = make_shifts(args.result_noise_tag)

    table = build_tables(shifts, output_dir)
    figure_paths = []
    for shift in shifts:
        selection = choose_sample(shift)
        figure_paths.append(plot_shift(shift, selection, output_dir, dpi=args.dpi))

    print(f"Wrote table rows: {len(table)}")
    print(f"Output dir: {output_dir}")
    for path in figure_paths:
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
