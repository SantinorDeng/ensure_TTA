#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_classical() {
  local name="$1"
  local cuda_visible="$2"
  local method="$3"
  local manifest="$4"
  local output_dir="$5"
  local shift_name="$6"

  if [[ -f "${output_dir}/summary.csv" ]]; then
    log "SKIP ${name}: summary exists"
    return 0
  fi

  mkdir -p "${output_dir}"
  log "START ${name}"
  if [[ -n "${cuda_visible}" ]]; then
    CUDA_VISIBLE_DEVICES="${cuda_visible}" conda run -n uncertainty_tta python scripts/eval_classical_baselines.py \
      --method "${method}" \
      --manifest-csv "${manifest}" \
      --output-dir "${output_dir}" \
      --split-role target_test \
      --target-shift-name "${shift_name}" \
      --preproc-root preproc/shifts/modality_matrix \
      --density-root density_stats/shifts/modality_matrix \
      --device cpu \
      --seed 7 \
      --acceleration 8 \
      --include-ssim \
      --save-recons \
      --num-workers 0 \
      --test-noise-snr-db 15 \
      --test-noise-seed 9007 \
      --bart-lambda 0.05 \
      --bart-command "pics -g -d0 -S -R T:3:0:0.05" \
      --fail-fast \
      > "${output_dir}/run.log" 2>&1
  else
    conda run -n uncertainty_tta python scripts/eval_classical_baselines.py \
      --method "${method}" \
      --manifest-csv "${manifest}" \
      --output-dir "${output_dir}" \
      --split-role target_test \
      --target-shift-name "${shift_name}" \
      --preproc-root preproc/shifts/modality_matrix \
      --density-root density_stats/shifts/modality_matrix \
      --device cpu \
      --seed 7 \
      --acceleration 8 \
      --include-ssim \
      --save-recons \
      --num-workers 0 \
      --test-noise-snr-db 15 \
      --test-noise-seed 9007 \
      --fail-fast \
      > "${output_dir}/run.log" 2>&1
  fi
  log "DONE ${name}"
}

run_tta() {
  local name="$1"
  local device="$2"
  local checkpoint="$3"
  local manifest="$4"
  local output_dir="$5"
  local shift_name="$6"
  local objective="$7"

  if [[ -f "${output_dir}/summary.csv" ]]; then
    log "SKIP ${name}: summary exists"
    return 0
  fi

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
    --test-noise-snr-db 15 \
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

wait_batch() {
  local label="$1"
  shift
  log "WAIT ${label}: $*"
  wait "$@"
  log "DONE ${label}"
}

log "Stage 1/5: R8 snr15 classical baselines"
run_classical \
  snr15_r8_zf_axflair \
  "" \
  zero_filled \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/baselines/shifts/modality_matrix/main/brain/axflair/zero_filled_snr15_r8_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axflair &
pid1=$!

run_classical \
  snr15_r8_zf_axt1post \
  "" \
  zero_filled \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/baselines/shifts/modality_matrix/main/brain/axt1post/zero_filled_snr15_r8_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1pre_to_axt1post &
pid2=$!

run_classical \
  snr15_r8_bart_axflair \
  5 \
  bart_pics_cs \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/baselines/shifts/modality_matrix/main/brain/axflair/bart_pics_tvxy_lam0p05_snr15_r8_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axflair &
pid3=$!

run_classical \
  snr15_r8_bart_axt1post \
  6 \
  bart_pics_cs \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/baselines/shifts/modality_matrix/main/brain/axt1post/bart_pics_tvxy_lam0p05_snr15_r8_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1pre_to_axt1post &
pid4=$!

wait_batch "classical baselines" "${pid1}" "${pid2}" "${pid3}" "${pid4}"

log "Stage 2/5: TTA batch 1/3"
run_tta \
  snr15_r8_axt1_ssdu_flair_cuda0 \
  cuda:0 \
  outputs/shifts/modality_matrix/main/brain/axt1/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1/ssdu_l1_snr15_r8_maskseed7_noiseseed9007_subset97e69ab1 \
  brain_modality_matrix_axt1_to_axflair \
  ssdu &
pid1=$!

run_tta \
  snr15_r8_axt1_trad_flair_cuda5 \
  cuda:5 \
  outputs/shifts/modality_matrix/main/brain/axt1/traditional_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1/traditional_l1_snr15_r8_maskseed7_noiseseed9007_subset97e69ab1 \
  brain_modality_matrix_axt1_to_axflair \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid2=$!

run_tta \
  snr15_r8_axt1_ensure_flair_cuda6 \
  cuda:6 \
  outputs/shifts/modality_matrix/main/brain/axt1/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1/ensure_l1_snr15_r8_maskseed7_noiseseed9007_subset97e69ab1 \
  brain_modality_matrix_axt1_to_axflair \
  true_ensure &
pid3=$!

run_tta \
  snr15_r8_axt2_ssdu_flair_cuda7 \
  cuda:7 \
  outputs/shifts/modality_matrix/main/brain/axt2/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt2/ssdu_l1_snr15_r8_maskseed7_noiseseed9007_subset9d20678e \
  brain_modality_matrix_axt2_to_axflair \
  ssdu &
pid4=$!

wait_batch "TTA batch 1/3" "${pid1}" "${pid2}" "${pid3}" "${pid4}"

log "Stage 3/5: TTA batch 2/3"
run_tta \
  snr15_r8_axt2_trad_flair_cuda0 \
  cuda:0 \
  outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt2/traditional_l1_snr15_r8_maskseed7_noiseseed9007_subset9d20678e \
  brain_modality_matrix_axt2_to_axflair \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid1=$!

run_tta \
  snr15_r8_axt2_ensure_flair_cuda5 \
  cuda:5 \
  outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt2/ensure_l1_snr15_r8_maskseed7_noiseseed9007_subset9d20678e \
  brain_modality_matrix_axt2_to_axflair \
  true_ensure &
pid2=$!

run_tta \
  snr15_r8_pre_ssdu_post_cuda6 \
  cuda:6 \
  outputs/shifts/modality_matrix/main/brain/axt1pre/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1pre/ssdu_l1_snr15_r8_maskseed7_noiseseed9007_subset5ea2ce8d \
  brain_modality_matrix_axt1pre_to_axt1post \
  ssdu &
pid3=$!

run_tta \
  snr15_r8_pre_trad_post_cuda7 \
  cuda:7 \
  outputs/shifts/modality_matrix/main/brain/axt1pre/traditional_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1pre/traditional_l1_snr15_r8_maskseed7_noiseseed9007_subset5ea2ce8d \
  brain_modality_matrix_axt1pre_to_axt1post \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid4=$!

wait_batch "TTA batch 2/3" "${pid1}" "${pid2}" "${pid3}" "${pid4}"

log "Stage 4/5: TTA batch 3/3"
run_tta \
  snr15_r8_pre_ensure_post_cuda0 \
  cuda:0 \
  outputs/shifts/modality_matrix/main/brain/axt1pre/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1pre.csv \
  outputs/tta/shifts/modality_matrix/main/brain/axt1pre/ensure_l1_snr15_r8_maskseed7_noiseseed9007_subset5ea2ce8d \
  brain_modality_matrix_axt1pre_to_axt1post \
  true_ensure

log "Stage 5/5: draw final snr15 R8 comparison directory"
conda run -n uncertainty_tta python scripts/draw_modality_comparison.py \
  --result-noise-tag snr15_r8 \
  --output-dir outputs/figures/modality_comparison_snr15_r8 \
  --dpi 300

log "All snr15 R8 comparisons complete"
