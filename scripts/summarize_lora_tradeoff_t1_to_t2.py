#!/usr/bin/env python3
"""Validate and summarize the independent-denoiser T1-to-T2 LoRA trade-off."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADEOFF_ROOT = PROJECT_ROOT / (
    "outputs/tta/other_modality/t1_to_t2/lora_tradeoff_independent_clean_r4"
)
FULL_ROOT = PROJECT_ROOT / (
    "outputs/tta/other_modality/t1_to_t2/"
    "ensure_l1_clean_r4_maskseed7_noiseseed9007"
)
R2_DC_ROOT = PROJECT_ROOT / (
    "outputs/tta/other_modality/t1_to_t2/"
    "ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007"
)

EXPECTED_PARAMS = {
    "dc_only": 12,
    "r1": 23_040,
    "r2": 46_080,
    "r4": 92_160,
    "r8": 184_320,
    "r2_dc": 46_092,
    "full": 1_354_764,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_sharded(method: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    shard_root = TRADEOFF_ROOT / method / "shards"
    shard_dirs = sorted(path for path in shard_root.glob("*") if path.is_dir())
    if len(shard_dirs) != 5:
        raise RuntimeError(f"{method}: expected 5 shard directories, found {len(shard_dirs)}")
    rows: list[dict[str, str]] = []
    payloads: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        payload_path = shard_dir / "summary.json"
        metrics_path = shard_dir / "metrics.csv"
        if not payload_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(f"{method}: incomplete shard {shard_dir}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if payload["num_ok"] != 20 or payload["num_failed"] != 0:
            raise RuntimeError(f"{method}: failed/incomplete payload in {shard_dir}")
        payloads.append(payload)
        rows.extend(read_csv(metrics_path))
    return validate_rows(method, rows), payloads


def validate_rows(method: str, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{method}: {len(failed)} failed metric rows")
    rows.sort(key=lambda row: int(row["rank_in_split"]))
    ids = [row["sample_id"] for row in rows]
    if len(rows) != 100 or len(set(ids)) != 100:
        raise RuntimeError(f"{method}: expected 100 unique samples, got {len(rows)}/{len(set(ids))}")
    params = {int(row["num_trainable_params"]) for row in rows}
    if params != {EXPECTED_PARAMS[method]}:
        raise RuntimeError(f"{method}: unexpected trainable parameter counts {params}")
    return rows


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def median(rows: list[dict[str, str]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def paired_win_rate(
    rows: list[dict[str, str]],
    reference: list[dict[str, str]],
    key: str,
    *,
    higher_is_better: bool,
) -> float:
    current = {row["sample_id"]: row for row in rows}
    baseline = {row["sample_id"]: row for row in reference}
    if current.keys() != baseline.keys():
        raise RuntimeError("Paired comparison received different sample sets")
    if higher_is_better:
        return statistics.fmean(
            float(current[sample_id][key]) > float(baseline[sample_id][key])
            for sample_id in current
        )
    return statistics.fmean(
        float(current[sample_id][key]) < float(baseline[sample_id][key])
        for sample_id in current
    )


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_merged_metrics(method: str, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    write_rows(TRADEOFF_ROOT / "merged_metrics" / f"{method}.csv", rows, fieldnames)


def main() -> int:
    full_payload = json.loads((FULL_ROOT / "summary.json").read_text(encoding="utf-8"))
    r2_dc_payload = json.loads((R2_DC_ROOT / "summary.json").read_text(encoding="utf-8"))
    rows_by_key: dict[str, list[dict[str, str]]] = {
        "full": validate_rows("full", read_csv(FULL_ROOT / "metrics.csv")),
        "r2_dc": validate_rows("r2_dc", read_csv(R2_DC_ROOT / "metrics.csv")),
    }
    shard_payloads: dict[str, list[dict[str, Any]]] = {}
    for key in ("r1", "r2", "r4", "r8", "dc_only"):
        rows_by_key[key], shard_payloads[key] = read_sharded(key)

    full_ids = {row["sample_id"] for row in rows_by_key["full"]}
    for key, rows in rows_by_key.items():
        if {row["sample_id"] for row in rows} != full_ids:
            raise RuntimeError(f"{key}: sample IDs differ from full TTA")
        full_before = {row["sample_id"]: row for row in rows_by_key["full"]}
        max_before_diff = max(
            abs(float(row["before_tta_nmse"]) - float(full_before[row["sample_id"]]["before_tta_nmse"]))
            for row in rows
        )
        if max_before_diff != 0.0:
            raise RuntimeError(f"{key}: before-TTA NMSE mismatch {max_before_diff}")

    checkpoint_hashes = {
        full_payload["checkpoint_sha256"],
        r2_dc_payload["checkpoint_sha256"],
        *(payload["checkpoint_sha256"] for payloads in shard_payloads.values() for payload in payloads),
    }
    manifest_hashes = {
        full_payload["manifest_sha256"],
        r2_dc_payload["manifest_sha256"],
        *(payload["manifest_sha256"] for payloads in shard_payloads.values() for payload in payloads),
    }
    if len(checkpoint_hashes) != 1 or len(manifest_hashes) != 1:
        raise RuntimeError("Checkpoint or manifest hashes differ across methods")

    for key, rows in rows_by_key.items():
        write_merged_metrics(key, rows)

    display = [
        ("dc_only", "DC-only"),
        ("r1", "Conv-LoRA r=1"),
        ("r2", "Conv-LoRA r=2"),
        ("r2_dc", "Conv-LoRA r=2 + 12 DC"),
        ("r4", "Conv-LoRA r=4"),
        ("r8", "Conv-LoRA r=8"),
        ("full", "Full-parameter TTA"),
    ]
    full_params = EXPECTED_PARAMS["full"]
    full_rows = rows_by_key["full"]
    result_rows: list[dict[str, Any]] = []
    for key, label in display:
        rows = rows_by_key[key]
        result_rows.append(
            {
                "method": label,
                "trainable_parameters": EXPECTED_PARAMS[key],
                "fraction_of_full": EXPECTED_PARAMS[key] / full_params,
                "psnr_db": mean(rows, "after_tta_psnr"),
                "psnr_gap_vs_full_db": mean(rows, "after_tta_psnr")
                - mean(full_rows, "after_tta_psnr"),
                "ssim": mean(rows, "after_tta_ssim"),
                "mean_nmse": mean(rows, "after_tta_nmse"),
                "median_nmse": median(rows, "after_tta_nmse"),
                "delta_psnr_db": mean(rows, "delta_psnr"),
                "mean_tta_steps": mean(rows, "num_tta_steps"),
                "adapt_sec_per_slice": mean(rows, "adapt_runtime_sec"),
                "total_sec_per_slice": mean(rows, "runtime_sec"),
                "negative_adaptation_rate": statistics.fmean(
                    float(row["after_tta_nmse"]) > float(row["before_tta_nmse"])
                    for row in rows
                ),
                "psnr_win_rate_vs_full": paired_win_rate(
                    rows, full_rows, "after_tta_psnr", higher_is_better=True
                ),
                "num_samples": len(rows),
            }
        )

    frozen = {
        "method": "Frozen / no TTA",
        "trainable_parameters": 0,
        "fraction_of_full": 0.0,
        "psnr_db": mean(full_rows, "before_tta_psnr"),
        "psnr_gap_vs_full_db": mean(full_rows, "before_tta_psnr")
        - mean(full_rows, "after_tta_psnr"),
        "ssim": mean(full_rows, "before_tta_ssim"),
        "mean_nmse": mean(full_rows, "before_tta_nmse"),
        "median_nmse": median(full_rows, "before_tta_nmse"),
        "delta_psnr_db": 0.0,
        "mean_tta_steps": 0.0,
        "adapt_sec_per_slice": 0.0,
        "total_sec_per_slice": "",
        "negative_adaptation_rate": 0.0,
        "psnr_win_rate_vs_full": 0.0,
        "num_samples": len(full_rows),
    }
    result_rows.insert(0, frozen)
    fieldnames = list(result_rows[0])
    write_rows(TRADEOFF_ROOT / "tradeoff_table.csv", result_rows, fieldnames)

    by_key = {key: rows_by_key[key] for key, _ in display}
    r2_dc_vs_r2_psnr_win = paired_win_rate(
        by_key["r2_dc"], by_key["r2"], "after_tta_psnr", higher_is_better=True
    )
    r2_dc_vs_dc_psnr_win = paired_win_rate(
        by_key["r2_dc"], by_key["dc_only"], "after_tta_psnr", higher_is_better=True
    )
    r2_dc_vs_r2_nmse_win = paired_win_rate(
        by_key["r2_dc"], by_key["r2"], "after_tta_nmse", higher_is_better=False
    )
    r2_dc_vs_dc_nmse_win = paired_win_rate(
        by_key["r2_dc"], by_key["dc_only"], "after_tta_nmse", higher_is_better=False
    )

    lines = [
        "# T1 to T2 LoRA accuracy-efficiency trade-off",
        "",
        "Protocol: AXT1 source to AXT2 target, independent denoisers in 12 unrolls, "
        "100 paired target slices, clean k-space, R_acc=4, measured-k-space L1 TTA, "
        "250 maximum steps, 5% held-out self-validation, mask seed 7, test-noise seed 9007. "
        "LoRA/DC use lr=1e-3; the established full-parameter baseline uses lr=1e-5.",
        "",
        f"Checkpoint SHA256: `{next(iter(checkpoint_hashes))}`",
        "",
        f"Manifest SHA256: `{next(iter(manifest_hashes))}`",
        "",
        "| method | trainable params | vs full | PSNR | gap to full | SSIM | mean NMSE | median NMSE | ΔPSNR | mean steps | observed adapt s/slice† | observed total s/slice† | negative adaptation | PSNR win vs full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        total = "—" if row["total_sec_per_slice"] == "" else f"{row['total_sec_per_slice']:.2f}"
        lines.append(
            f"| {row['method']} | {row['trainable_parameters']:,} | "
            f"{100.0 * row['fraction_of_full']:.3f}% | {row['psnr_db']:.4f} | "
            f"{row['psnr_gap_vs_full_db']:+.4f} | {row['ssim']:.4f} | "
            f"{row['mean_nmse']:.6f} | {row['median_nmse']:.6f} | "
            f"{row['delta_psnr_db']:+.4f} | {row['mean_tta_steps']:.1f} | "
            f"{row['adapt_sec_per_slice']:.2f} | {total} | "
            f"{100.0 * row['negative_adaptation_rate']:.1f}% | "
            f"{100.0 * row['psnr_win_rate_vs_full']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "† Runtime caveat: the new rank/DC runs used parallel sharding and some shards were "
            "rerouted as GPU availability changed; the established Full TTA and r=2+DC runs were "
            "recorded separately. These are observed wall-clock values, not a controlled same-GPU "
            "strict-runtime benchmark. Use trainable parameters as the primary efficiency axis.",
            "",
            "## DC collaboration at r=2",
            "",
            f"- r=2+DC versus r=2: mean PSNR difference "
            f"{mean(by_key['r2_dc'], 'after_tta_psnr') - mean(by_key['r2'], 'after_tta_psnr'):+.4f} dB; "
            f"PSNR/NMSE paired win rates {100.0 * r2_dc_vs_r2_psnr_win:.1f}%/"
            f"{100.0 * r2_dc_vs_r2_nmse_win:.1f}%.",
            f"- r=2+DC versus DC-only: mean PSNR difference "
            f"{mean(by_key['r2_dc'], 'after_tta_psnr') - mean(by_key['dc_only'], 'after_tta_psnr'):+.4f} dB; "
            f"PSNR/NMSE paired win rates {100.0 * r2_dc_vs_dc_psnr_win:.1f}%/"
            f"{100.0 * r2_dc_vs_dc_nmse_win:.1f}%.",
            "",
            "## Validation",
            "",
            "All methods contain exactly 100 successful rows with identical sample IDs, checkpoint hash, "
            "manifest hash, and bit-identical before-TTA NMSE values.",
        ]
    )
    (TRADEOFF_ROOT / "tradeoff_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = {
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "manifest_sha256": next(iter(manifest_hashes)),
        "num_samples_per_method": {key: len(rows) for key, rows in rows_by_key.items()},
        "expected_trainable_parameters": EXPECTED_PARAMS,
        "all_sample_ids_match": True,
        "all_before_tta_nmse_match": True,
        "r2_dc_vs_r2_psnr_win_rate": r2_dc_vs_r2_psnr_win,
        "r2_dc_vs_r2_nmse_win_rate": r2_dc_vs_r2_nmse_win,
        "r2_dc_vs_dc_only_psnr_win_rate": r2_dc_vs_dc_psnr_win,
        "r2_dc_vs_dc_only_nmse_win_rate": r2_dc_vs_dc_nmse_win,
    }
    (TRADEOFF_ROOT / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"Saved: {TRADEOFF_ROOT / 'tradeoff_table.csv'}")
    print(f"Saved: {TRADEOFF_ROOT / 'tradeoff_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
