from __future__ import annotations

from cardiac_ensure.datasets.shift_manifest_dataset import split_source_rows
from cardiac_ensure.scripts.build_shift_source_manifests import (
    row_matches,
    select_paired_patient_rows,
)
from cardiac_ensure.scripts.run_modality_matrix_job import (
    build_argparser as build_job_argparser,
    manifest_path,
    parse_snr,
)
from cardiac_ensure.scripts.tta_shift_true_ensure import build_argparser as build_tta_argparser


def _row(path: str, sample_id: str, slice_idx: int, *, acquisition: str = "AXT1POST") -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "dataset": "fastmri_brain",
        "split": "train",
        "path": path,
        "volume_id": path,
        "slice_idx": str(slice_idx),
        "num_slices": "5",
        "sequence_or_contrast": "AXT1",
        "acquisition": acquisition,
        "matrix_size": "[640, 320]",
    }


def test_exact_acquisition_filter_keeps_axt1post_distinct() -> None:
    row = _row("volume", "sample", 0)

    assert row_matches(
        row,
        {
            "dataset": "fastmri_brain",
            "splits": ["train"],
            "acquisition": "AXT1POST",
            "matrix_size": (640, 320),
        },
    )
    assert not row_matches(row, {"dataset": "fastmri_brain", "acquisition": "AXT1"})


def test_paired_patient_selection_uses_same_order_for_both_acquisitions() -> None:
    candidates = {}
    for acquisition in ("CORPD_FBK", "CORPDFS_FBK"):
        candidates[acquisition] = {}
        for patient_index in range(4):
            patient_id = f"patient-{patient_index}"
            volume = [
                {
                    **_row(
                        f"{acquisition}-{patient_id}",
                        f"{acquisition}-{patient_id}-{slice_idx}",
                        slice_idx,
                        acquisition=acquisition,
                    ),
                    "num_slices": "5",
                }
                for slice_idx in range(5)
            ]
            candidates[acquisition][patient_id] = [volume]

    selected, patients = select_paired_patient_rows(
        candidates,
        target_count=10,
        slices_per_volume=5,
        patient_offset=1,
    )

    assert patients == ["patient-1", "patient-2"]
    assert all(len(rows) == 10 for rows in selected.values())
    for rows in selected.values():
        assert {row["path"].split("patient-")[-1] for row in rows} == {"1", "2"}


def test_source_split_prefers_patient_split_group() -> None:
    rows = []
    for patient_index in range(5):
        for volume_index in range(2):
            row = _row(
                f"volume-{patient_index}-{volume_index}",
                f"sample-{patient_index}-{volume_index}",
                0,
            )
            row["split_group_id"] = f"patient-{patient_index}"
            rows.append(row)

    train, train_summary = split_source_rows(rows, subset="train", val_fraction=0.2, seed=7)
    val, val_summary = split_source_rows(rows, subset="val", val_fraction=0.2, seed=7)

    train_patients = {row["split_group_id"] for row in train}
    val_patients = {row["split_group_id"] for row in val}
    assert train_patients.isdisjoint(val_patients)
    assert len(val_patients) == 1
    assert train_summary["split_group_key"] == "split_group_id_or_volume_key"
    assert val_summary["split_group_key"] == "split_group_id_or_volume_key"


def test_l1_alias_is_the_new_tta_default() -> None:
    args = build_tta_argparser().parse_args([])
    assert args.tta_loss == "l1"


def test_single_job_cli_requires_explicit_device() -> None:
    parser = build_job_argparser()
    args = parser.parse_args(
        [
            "train",
            "--dataset", "knee",
            "--source", "pd",
            "--method", "ensure",
            "--device", "cuda:6",
            "--dry-run",
        ]
    )
    assert args.device == "cuda:6"
    assert manifest_path("knee", "pd").name == "pd.csv"
    assert parse_snr("clean") is None
    assert parse_snr("10") == 10.0
