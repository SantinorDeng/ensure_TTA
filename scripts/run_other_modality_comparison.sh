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
  if [[ "${method}" == "bart_pics_cs" ]]; then
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
      --acceleration 4 \
      --include-ssim \
      --save-recons \
      --num-workers 0 \
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
      --acceleration 4 \
      --include-ssim \
      --save-recons \
      --num-workers 0 \
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
    --acceleration 4 \
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

log "Stage 1/5: other-modality R4 clean classical baselines"
run_classical \
  other_t1_to_t2_zero_filled \
  "" \
  zero_filled \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/baselines/other_modality/t1_to_t2/zero_filled_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt2 &
pid1=$!

run_classical \
  other_t1_to_t2_bart_pics \
  5 \
  bart_pics_cs \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/baselines/other_modality/t1_to_t2/bart_pics_tvxy_lam0p05_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt2 &
pid2=$!

run_classical \
  other_t1_to_post_zero_filled \
  "" \
  zero_filled \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/baselines/other_modality/t1_to_post/zero_filled_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt1post &
pid3=$!

run_classical \
  other_t1_to_post_bart_pics \
  6 \
  bart_pics_cs \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/baselines/other_modality/t1_to_post/bart_pics_tvxy_lam0p05_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt1post &
pid4=$!

run_classical \
  other_t2_to_post_zero_filled \
  "" \
  zero_filled \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/baselines/other_modality/t2_to_post/zero_filled_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt2_to_axt1post &
pid5=$!

run_classical \
  other_t2_to_post_bart_pics \
  7 \
  bart_pics_cs \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/baselines/other_modality/t2_to_post/bart_pics_tvxy_lam0p05_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt2_to_axt1post &
pid6=$!

wait_batch "classical baselines" "${pid1}" "${pid2}" "${pid3}" "${pid4}" "${pid5}" "${pid6}"

log "Stage 2/5: TTA t1->t2"
run_tta \
  other_t1_to_t2_ssdu_cuda0 \
  cuda:0 \
  outputs/shifts/modality_matrix/main/brain/axt1/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/other_modality/t1_to_t2/ssdu_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt2 \
  ssdu &
pid1=$!

run_tta \
  other_t1_to_t2_traditional_cuda5 \
  cuda:5 \
  outputs/shifts/modality_matrix/main/brain/axt1/traditional_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/other_modality/t1_to_t2/traditional_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt2 \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid2=$!

run_tta \
  other_t1_to_t2_ensure_cuda6 \
  cuda:6 \
  outputs/shifts/modality_matrix/main/brain/axt1/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/other_modality/t1_to_t2/ensure_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt2 \
  true_ensure &
pid3=$!

wait_batch "TTA t1->t2" "${pid1}" "${pid2}" "${pid3}"

log "Stage 3/5: TTA t1->post"
run_tta \
  other_t1_to_post_ssdu_cuda0 \
  cuda:0 \
  outputs/shifts/modality_matrix/main/brain/axt1/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/other_modality/t1_to_post/ssdu_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt1post \
  ssdu &
pid1=$!

run_tta \
  other_t1_to_post_traditional_cuda5 \
  cuda:5 \
  outputs/shifts/modality_matrix/main/brain/axt1/traditional_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/other_modality/t1_to_post/traditional_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt1post \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid2=$!

run_tta \
  other_t1_to_post_ensure_cuda6 \
  cuda:6 \
  outputs/shifts/modality_matrix/main/brain/axt1/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt1.csv \
  outputs/tta/other_modality/t1_to_post/ensure_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt1_to_axt1post \
  true_ensure &
pid3=$!

wait_batch "TTA t1->post" "${pid1}" "${pid2}" "${pid3}"

log "Stage 4/5: TTA t2->post"
run_tta \
  other_t2_to_post_ssdu_cuda0 \
  cuda:0 \
  outputs/shifts/modality_matrix/main/brain/axt2/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/other_modality/t2_to_post/ssdu_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt2_to_axt1post \
  ssdu &
pid1=$!

run_tta \
  other_t2_to_post_traditional_cuda5 \
  cuda:5 \
  outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/other_modality/t2_to_post/traditional_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt2_to_axt1post \
  normalized_l1_supervised_plus_measured_kspace_self_supervision &
pid2=$!

run_tta \
  other_t2_to_post_ensure_cuda6 \
  cuda:6 \
  outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
  manifests/shifts/modality_matrix/main/brain/axt2.csv \
  outputs/tta/other_modality/t2_to_post/ensure_l1_clean_r4_maskseed7_noiseseed9007 \
  brain_modality_matrix_axt2_to_axt1post \
  true_ensure &
pid3=$!

wait_batch "TTA t2->post" "${pid1}" "${pid2}" "${pid3}"

log "Stage 5/5: draw other modality comparison"
conda run -n uncertainty_tta python scripts/draw_modality_comparison.py \
  --comparison-set other_modality \
  --result-noise-tag clean \
  --output-dir outputs/figures/other_modality_comparison \
  --dpi 300

log "All other modality comparisons complete"
