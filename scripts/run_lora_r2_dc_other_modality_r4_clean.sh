#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

CHECKPOINT="outputs/shifts/modality_matrix/main/brain/axt1/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt"
MANIFEST="manifests/shifts/modality_matrix/main/brain/axt1.csv"
COMMON=(
  --checkpoint "${CHECKPOINT}"
  --manifest-csv "${MANIFEST}"
  --split-role target_test
  --preproc-root preproc/shifts/modality_matrix
  --density-root density_stats/shifts/modality_matrix
  --seed 7
  --test-noise-seed 9007
  --training-objective true_ensure
  --tta-loss l1
  --tta-lr 1e-3
  --tta-weight-decay 0
  --grad-clip 1
  --self-val-fraction 0.05
  --early-stop-window 20
  --update-mode adapter
  --acceleration 4
  --lora-rank 2
  --lora-layers 1 2 3
  --adapt-dc
  --run-tta
  --include-ssim
  --save-recons
  --save-curves
  --num-workers 0
  --fail-fast
)

run_one() {
  local name="$1"
  local device="$2"
  local shift_name="$3"
  local output_dir="$4"
  local start_sample="$5"
  local max_samples="$6"
  local steps="$7"

  if [[ -f "${output_dir}/summary.json" ]]; then
    printf '[skip] %s: summary exists\n' "${name}"
    return 0
  fi
  mkdir -p "${output_dir}"
  printf '[start] %s device=%s shift=%s range=[%s,%s) steps=%s\n' \
    "${name}" "${device}" "${shift_name}" "${start_sample}" "$((start_sample + max_samples))" "${steps}"
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure_lora.py \
    "${COMMON[@]}" \
    --device "${device}" \
    --target-shift-name "${shift_name}" \
    --output-dir "${output_dir}" \
    --start-sample "${start_sample}" \
    --max-samples "${max_samples}" \
    --tta-steps "${steps}" \
    > "${output_dir}/run.log" 2>&1
  printf '[done] %s\n' "${name}"
}

run_smoke() {
  local root="outputs/tta/other_modality/_smoke_lora_r2_dc_clean_r4"
  run_one flair_smoke cuda:1 brain_modality_matrix_axt1_to_axflair "${root}/t1_to_flair" 0 1 3 &
  local pid1=$!
  run_one t2_smoke cuda:2 brain_modality_matrix_axt1_to_axt2 "${root}/t1_to_t2" 0 1 3 &
  local pid2=$!
  run_one post_smoke cuda:3 brain_modality_matrix_axt1_to_axt1post "${root}/t1_to_post" 0 1 3 &
  local pid3=$!
  wait "${pid1}" "${pid2}" "${pid3}"
}

run_main() {
  local flair="outputs/tta/other_modality/t1_to_flair/ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007"
  local t2="outputs/tta/other_modality/t1_to_t2/ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007"
  local post="outputs/tta/other_modality/t1_to_post/ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007"

  run_one flair cuda:1 brain_modality_matrix_axt1_to_axflair "${flair}" 0 100 250 &
  local pid1=$!
  run_one t2 cuda:2 brain_modality_matrix_axt1_to_axt2 "${t2}" 0 100 250 &
  local pid2=$!
  run_one post cuda:3 brain_modality_matrix_axt1_to_axt1post "${post}" 0 100 250 &
  local pid3=$!
  wait "${pid1}" "${pid2}" "${pid3}"
  printf '[done] all three modality scenarios\n'
}

case "${1:-}" in
  smoke)
    run_smoke
    ;;
  main)
    run_main
    ;;
  *)
    printf 'Usage: %s {smoke|main}\n' "$0" >&2
    exit 2
    ;;
esac
