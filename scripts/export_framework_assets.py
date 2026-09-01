#!/usr/bin/env python3
"""Export provenance-tracked image assets for the paper framework figure."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.shift_manifest_dataset import StaticShiftSourceENSUREDataset  # noqa: E402


OUTPUT_DIR = ROOT / "outputs" / "figures" / "framework_asset_pack"
MANIFEST = ROOT / "manifests" / "shifts" / "modality_matrix" / "main" / "brain" / "axt1.csv"
PREPROC_ROOT = ROOT / "preproc" / "shifts" / "modality_matrix"
DENSITY_ROOT = ROOT / "density_stats" / "shifts" / "modality_matrix"

SOURCE_SAMPLE_ID = "fastmri_brain:file_brain_AXT1_201_6002688:slice0008"

SELECTED_RECONS = {
    "t2": (
        "fastmri_brain:file_brain_AXT2_202_2020088:slice0009",
        ROOT
        / "outputs/tta/other_modality/t1_to_t2/"
        "ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007/recons/"
        "true_ensure_convlora_r2_dc_tta_033_fastmri_brain_file_brain_"
        "AXT2_202_2020088_slice0009_slice9.npz",
    ),
    "flair": (
        "fastmri_brain:file_brain_AXFLAIR_200_6002560:slice0006",
        ROOT
        / "outputs/tta/other_modality/t1_to_flair/"
        "ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007/recons/"
        "true_ensure_convlora_r2_dc_tta_040_fastmri_brain_file_brain_"
        "AXFLAIR_200_6002560_slice0006_slice6.npz",
    ),
    "post": (
        "fastmri_brain:file_brain_AXT1POST_200_6002198:slice0007",
        ROOT
        / "outputs/tta/other_modality/t1_to_post/"
        "ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007/recons/"
        "true_ensure_convlora_r2_dc_tta_086_fastmri_brain_file_brain_"
        "AXT1POST_200_6002198_slice0007_slice7.npz",
    ),
}


def _frame(array: np.ndarray) -> np.ndarray:
    image = np.asarray(array).squeeze()
    if image.ndim != 2:
        raise ValueError(f"Expected a single 2D frame, got {image.shape}")
    if np.iscomplexobj(image):
        image = np.abs(image)
    return image.astype(np.float32, copy=False)


def _center_crop(image: np.ndarray, height: int = 320, width: int = 320) -> np.ndarray:
    y0 = max(0, (image.shape[-2] - height) // 2)
    x0 = max(0, (image.shape[-1] - width) // 2)
    return image[y0 : y0 + height, x0 : x0 + width]


def _display_limit(reference: np.ndarray) -> float:
    values = np.asarray(reference, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.maximum(finite, 0.0), 99.5))
    return max(limit, float(np.finfo(np.float32).eps))


def _save_grayscale(path: Path, image: np.ndarray, limit: float) -> None:
    scaled = np.clip(np.asarray(image, dtype=np.float32) / limit, 0.0, 1.0)
    pixels = np.round(scaled * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def _source_case() -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    dataset = StaticShiftSourceENSUREDataset(
        manifest_csv=MANIFEST,
        subset="all",
        preproc_root=PREPROC_ROOT,
        density_root=DENSITY_ROOT,
        source_role="source_train",
        acceleration=4.0,
        sigma_mask=0.18,
        window_size=1,
        deterministic_masks=True,
        mask_seed=7,
        return_target=True,
        require_preproc=False,
        shift_names=["brain_modality_matrix_axt1_source"],
    )
    index = next(
        idx for idx, sample in enumerate(dataset.samples) if sample.sample_id == SOURCE_SAMPLE_ID
    )
    item = dataset[index]
    metadata = dict(item["meta"])
    target = _frame(item["target_rss"].numpy())
    zero_filled = _frame(item["zf"].numpy())
    mask = _frame(item["mask"].numpy())
    return metadata, target, zero_filled, mask


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, object] = {
        "display": {
            "format": "8-bit grayscale PNG",
            "crop": "center 320x320 for anatomy/reconstruction images",
            "window": "[0, modality-specific reference 99.5th percentile]",
        },
        "experiment": {
            "source_modality": "AXT1",
            "target_modalities": ["AXT2", "AXFLAIR", "AXT1POST"],
            "method": "TRUE-ENSURE + Conv-LoRA r=2 + 12 trainable DC scalars",
            "tta_loss": "measured-kspace normalized complex L1",
            "nominal_acceleration": 4.0,
            "sigma_mask": 0.18,
            "mask_seed": 7,
            "input_noise": None,
        },
        "assets": {},
    }

    source_meta, source_target, source_zf, source_mask = _source_case()
    source_target = _center_crop(source_target)
    source_zf = _center_crop(source_zf)
    source_limit = _display_limit(source_target)
    _save_grayscale(OUTPUT_DIR / "01_source_T1_reference.png", source_target, source_limit)
    _save_grayscale(OUTPUT_DIR / "02_source_T1_zero_filled_R4.png", source_zf, source_limit)
    Image.fromarray(np.round(source_mask * 255.0).astype(np.uint8), mode="L").save(
        OUTPUT_DIR / "06_actual_sampling_mask_R4.png"
    )
    np.save(OUTPUT_DIR / "06_actual_sampling_mask_R4.npy", source_mask.astype(np.uint8))
    provenance["assets"]["source_t1"] = {
        "sample_id": SOURCE_SAMPLE_ID,
        "reference_png": "01_source_T1_reference.png",
        "zero_filled_png": "02_source_T1_zero_filled_R4.png",
        "source_file": source_meta.get("source_file"),
        "slice_id": source_meta.get("slice_id"),
    }
    provenance["assets"]["sampling_mask"] = {
        "png": "06_actual_sampling_mask_R4.png",
        "numpy_array": "06_actual_sampling_mask_R4.npy",
        "shape": list(source_mask.shape),
        "sampled_fraction": float(source_mask.mean()),
        "actual_acceleration": float(1.0 / source_mask.mean()),
        "generation": "deterministic Bernoulli-Gaussian Cartesian line mask",
    }

    target_filenames = {
        "t2": "03_target_T2_reference.png",
        "flair": "04_target_FLAIR_reference.png",
        "post": "05_target_POST_reference.png",
    }
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for modality, (sample_id, npz_path) in SELECTED_RECONS.items():
        with np.load(npz_path) as archive:
            arrays = {
                "target": _center_crop(_frame(archive["target"])),
                "before_tta_rec": _center_crop(_frame(archive["before_tta_rec"])),
                "after_tta_rec": _center_crop(_frame(archive["after_tta_rec"])),
            }
        loaded[modality] = arrays
        limit = _display_limit(arrays["target"])
        _save_grayscale(OUTPUT_DIR / target_filenames[modality], arrays["target"], limit)
        provenance["assets"][f"target_{modality}"] = {
            "sample_id": sample_id,
            "png": target_filenames[modality],
            "source_npz": str(npz_path),
        }

    # POST is used because it is a requested challenging target and exhibits a
    # clear improvement in this selected experiment (37.17 -> 39.53 dB).
    post_limit = _display_limit(loaded["post"]["target"])
    _save_grayscale(
        OUTPUT_DIR / "07_proposed_LoRA_TTA_POST_reconstruction.png",
        loaded["post"]["after_tta_rec"],
        post_limit,
    )
    _save_grayscale(
        OUTPUT_DIR / "08_optional_POST_before_TTA.png",
        loaded["post"]["before_tta_rec"],
        post_limit,
    )
    provenance["assets"]["final_reconstruction"] = {
        "png": "07_proposed_LoRA_TTA_POST_reconstruction.png",
        "optional_before_tta_png": "08_optional_POST_before_TTA.png",
        "sample_id": SELECTED_RECONS["post"][0],
        "before_tta": {"nmse": 0.004148, "psnr_db": 37.172932, "ssim": 0.934554},
        "after_tta": {"nmse": 0.002409, "psnr_db": 39.532883, "ssim": 0.961798},
        "tta_steps": 250,
    }

    (OUTPUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        """# Framework figure asset pack

Use `02_source_T1_zero_filled_R4.png` for the source-domain input, and optionally
place `01_source_T1_reference.png` nearby. Use files 03--05 as the three target
modality examples. File 06 is the exact realized mask for the displayed source
case (the `.npy` file is the binary Python array). File 07 is the proposed
TRUE-ENSURE + Conv-LoRA r=2 + DC test-time-adapted POST reconstruction. File 08
is supplied only if a before/after inset is useful.

POST means axial T1 post-contrast (`AXT1POST`). All exported cases use clean
R=4 evaluation with deterministic mask seed 7. See `provenance.json` for sample
IDs, source paths, mask statistics, and reconstruction metrics.
""",
        encoding="utf-8",
    )

    print(f"Exported framework assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
