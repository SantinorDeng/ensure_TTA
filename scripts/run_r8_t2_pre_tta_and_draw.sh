#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_tta() {
  local name="$1"
  local device="$2"
  local checkpoint="$3"
  local manifest="$4"
  local output_dir="$5"
  local shift_name="$6"
  local objective="$7"

  mkdir -p "${output_dir}"
  log "START ${name} on ${device}"
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure.py \
    --checkpoint "${checkpoint}" \
    --manifest-csv "${manifest}" \
    --output-dir "${output_dir}" \
    --split-role target_test \
    --device "${device}" \
    --seed 7 \
    --test-noise-seed 9007 \
    --target-shift-name "${shift_name}" \
    --training-objective "${objective}" \
    --tta-loss l1 \
    --tta-steps 250 \
    --tta-lr 1e-5 \
    --tta-weight-decay 0 \
    --grad-clip 1 \
    --self-val-fraction 0.05 \
    --early-stop-window 20 \
    --update-mode all_params \
    --acceleration 8 \
    --run-tta \
    --include-ssim \
    --save-recons \
    --save-curves \
    --num-workers 0 \
    --fail-fast \
    > "${output_dir}/run.log" 2>&1
  log "DONE ${name}"
}

require_summary() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "Missing required summary: ${path}"
    return 1
  fi
}

log "Checking reusable R8 classical baselines"
require_summary outputs/baselines/shifts/modality_matrix/main/brain/axflair/zero_filled_clean_r8_maskseed7_noiseseed9007/summary.csv
require_summary outputs/baselines/shifts/modality_matrix/main/brain/axflair/bart_pics_tvxy_lam0p05_clean_r8_maskseed7_noiseseed9007/summary.csv
require_summary outputs/baselines/shifts/modality_matrix/main/brain/axt1post/zero_filled_clean_r8_maskseed7_noiseseed9007/summary.csv
require_summary outputs/baselines/shifts/modality_matrix/main/brain/axt1post/bart_pics_tvxy_lam0p05_clean_r8_maskseed7_noiseseed9007/summary.csv

log "Batch 1/2: t2->flair all methods plus pre->post SSDU"
run_tta \
  r8_clean_axt2_ssdu_flair_cuda0 \
  cuda:0 \
  outputs/shifts/modality_matrix/main/brain/axt2/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt2/ssdu_l1_clean_r8_maskseed7_noiseseed9007_subset9d20678e \
  brain_modality_matrix_axt2_to_axflair \
  ssdu &
pid1=$!

run_tta \
  r8_clean_axt2_trad_flair_cuda1 \
  cuda:1 \
  outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt2/traditional_l1_clean_r8_maskseed7_noiseseed9007_subset9d20678e \
  brain_modality_matrix_axt2_to_axflair \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid2=$!

run_tta \
  r8_clean_axt2_ensure_flair_cuda2 \
  cuda:2 \
  outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt2/ensure_l1_clean_r8_maskseed7_noiseseed9007_subset9d20678e \
  brain_modality_matrix_axt2_to_axflair \
  true_ensure &
pid3=$!

run_tta \
  r8_clean_pre_ssdu_post_cuda3 \
  cuda:3 \
  outputs/shifts/modality_matrix/main/brain/axt1pre/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1pre/ssdu_l1_clean_r8_maskseed7_noiseseed9007_subset5ea2ce8d \
  brain_modality_matrix_axt1pre_to_axt1post \
  ssdu &
pid4=$!

wait "${pid1}" "${pid2}" "${pid3}" "${pid4}"
log "Batch 1/2 complete"

log "Batch 2/2: pre->post traditional and ENSURE"
run_tta \
  r8_clean_pre_trad_post_cuda0 \
  cuda:0 \
  outputs/shifts/modality_matrix/main/brain/axt1pre/traditional_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1pre/traditional_l1_clean_r8_maskseed7_noiseseed9007_subset5ea2ce8d \
  brain_modality_matrix_axt1pre_to_axt1post \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid5=$!

run_tta \
  r8_clean_pre_ensure_post_cuda1 \
  cuda:1 \
  outputs/shifts/modality_matrix/main/brain/axt1pre/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1pre/ensure_l1_clean_r8_maskseed7_noiseseed9007_subset5ea2ce8d \
  brain_modality_matrix_axt1pre_to_axt1post \
  true_ensure &
pid6=$!

wait "${pid5}" "${pid6}"
log "Batch 2/2 complete"

log "Generating R8 t2->flair figure and table"
conda run -n uncertainty_tta python scripts/draw_modality_comparison.py \
  --result-noise-tag clean_r8 \
  --shift-key t2_to_flair \
  --output-dir outputs/figures/modality_comparison_clean_r8_t2_to_flair \
  --dpi 300

log "Generating R8 pre->post figure and table"
conda run -n uncertainty_tta python scripts/draw_modality_comparison.py \
  --result-noise-tag clean_r8 \
  --shift-key pre_to_post \
  --output-dir outputs/figures/modality_comparison_clean_r8_pre_to_post \
  --dpi 300

log "All R8 t2/pre tasks and figures complete"
