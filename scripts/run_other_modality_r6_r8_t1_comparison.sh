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
  local acceleration="$4"
  local tag="$5"
  local shift_key="$6"
  local shift_name="$7"

  local manifest="manifests/shifts/modality_matrix/main/brain/axt1.csv"
  local output_dir
  if [[ "${method}" == "zero_filled" ]]; then
    output_dir="outputs/baselines/other_modality/${shift_key}/zero_filled_${tag}_maskseed7_noiseseed9007"
  else
    output_dir="outputs/baselines/other_modality/${shift_key}/bart_pics_tvxy_lam0p05_${tag}_maskseed7_noiseseed9007"
  fi

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
      --acceleration "${acceleration}" \
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
      --acceleration "${acceleration}" \
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
  local method_key="$3"
  local checkpoint="$4"
  local objective="$5"
  local acceleration="$6"
  local tag="$7"
  local shift_key="$8"
  local shift_name="$9"

  local manifest="manifests/shifts/modality_matrix/main/brain/axt1.csv"
  local output_dir="outputs/tta/other_modality/${shift_key}/${method_key}_l1_${tag}_maskseed7_noiseseed9007"

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
    --acceleration "${acceleration}" \
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

run_shift_tta_batch() {
  local acceleration="$1"
  local tag="$2"
  local shift_key="$3"
  local shift_name="$4"

  run_tta \
    "${tag}_${shift_key}_ssdu_cuda0" \
    cuda:0 \
    ssdu \
    outputs/shifts/modality_matrix/main/brain/axt1/ssdu_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
    ssdu \
    "${acceleration}" \
    "${tag}" \
    "${shift_key}" \
    "${shift_name}" &
  pid1=$!

  run_tta \
    "${tag}_${shift_key}_traditional_cuda5" \
    cuda:5 \
    traditional \
    outputs/shifts/modality_matrix/main/brain/axt1/traditional_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
    normalized_l1_supervised_plus_measured_kspace_self_supervision \
    "${acceleration}" \
    "${tag}" \
    "${shift_key}" \
    "${shift_name}" &
  pid2=$!

  run_tta \
    "${tag}_${shift_key}_ensure_cuda6" \
    cuda:6 \
    ensure \
    outputs/shifts/modality_matrix/main/brain/axt1/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt \
    true_ensure \
    "${acceleration}" \
    "${tag}" \
    "${shift_key}" \
    "${shift_name}" &
  pid3=$!

  wait_batch "TTA ${tag} ${shift_key}" "${pid1}" "${pid2}" "${pid3}"
}

run_acceleration() {
  local acceleration="$1"
  local tag="$2"
  local figure_dir="$3"

  log "Stage ${tag}: classical baselines"
  run_classical "${tag}_t1_to_t2_zero_filled" "" zero_filled "${acceleration}" "${tag}" t1_to_t2 brain_modality_matrix_axt1_to_axt2 &
  pid1=$!
  run_classical "${tag}_t1_to_t2_bart" 7 bart_pics_cs "${acceleration}" "${tag}" t1_to_t2 brain_modality_matrix_axt1_to_axt2 &
  pid2=$!
  run_classical "${tag}_t1_to_post_zero_filled" "" zero_filled "${acceleration}" "${tag}" t1_to_post brain_modality_matrix_axt1_to_axt1post &
  pid3=$!
  run_classical "${tag}_t1_to_post_bart" 6 bart_pics_cs "${acceleration}" "${tag}" t1_to_post brain_modality_matrix_axt1_to_axt1post &
  pid4=$!
  wait_batch "classical ${tag}" "${pid1}" "${pid2}" "${pid3}" "${pid4}"

  log "Stage ${tag}: TTA t1->t2"
  run_shift_tta_batch "${acceleration}" "${tag}" t1_to_t2 brain_modality_matrix_axt1_to_axt2

  log "Stage ${tag}: TTA t1->post"
  run_shift_tta_batch "${acceleration}" "${tag}" t1_to_post brain_modality_matrix_axt1_to_axt1post

  log "Stage ${tag}: draw ${figure_dir}"
  conda run -n uncertainty_tta python scripts/draw_modality_comparison.py \
    --comparison-set other_modality \
    --result-noise-tag "${tag}" \
    --shift-key t1_to_t2 \
    --shift-key t1_to_post \
    --output-dir "${figure_dir}" \
    --dpi 300
}

run_acceleration 6 clean_r6 outputs/figures/other_modality_comparison_r6
run_acceleration 8 clean_r8 outputs/figures/other_modality_comparison_r8

log "All other-modality R6/R8 t1-source comparisons complete"
