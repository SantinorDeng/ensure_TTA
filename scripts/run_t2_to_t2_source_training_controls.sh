#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

SUPERVISED_DEVICE="${1:-cuda:5}"
MANIFEST="manifests/shifts/modality_matrix/main/brain/axt2.csv"
SHIFT="brain_modality_matrix_axt2_to_axt2"
SUPERVISED_ROOT="outputs/shifts/modality_matrix/main/brain/axt2/supervised_only_r4_w1_unroll12_independent_seed7_clean"
SUPERVISED_CHECKPOINT="${SUPERVISED_ROOT}/best.pt"
OUTPUT_ROOT="outputs/tta/other_modality/t2_to_t2"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_no_tta() {
  local name="$1"
  local checkpoint="$2"
  local objective="$3"
  local device="$4"
  local output_dir="${OUTPUT_ROOT}/${name}"

  if [[ -f "${output_dir}/summary.csv" && -f "${output_dir}/summary.json" ]]; then
    log "SKIP ${name}: completed summary exists"
    return 0
  fi

  mkdir -p "${output_dir}"
  log "START ${name} on ${device}"
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure.py \
    --checkpoint "${checkpoint}" \
    --manifest-csv "${MANIFEST}" \
    --output-dir "${output_dir}" \
    --split-role target_test \
    --target-shift-name "${SHIFT}" \
    --training-objective "${objective}" \
    --device "${device}" \
    --seed 7 \
    --test-noise-seed 9007 \
    --tta-loss l1 \
    --tta-steps 250 \
    --tta-lr 1e-5 \
    --tta-weight-decay 0 \
    --grad-clip 1 \
    --self-val-fraction 0.05 \
    --early-stop-window 20 \
    --update-mode all_params \
    --acceleration 4 \
    --sigma-mask 0.18 \
    --preproc-root preproc/shifts/modality_matrix \
    --density-root density_stats/shifts/modality_matrix \
    --no-run-tta \
    --include-ssim \
    --save-recons \
    --no-save-curves \
    --num-workers 0 \
    --fail-fast \
    > "${output_dir}/run.log" 2>&1
  log "DONE ${name}"
}

train_and_run_supervised_control() {
  if [[ ! -f "${SUPERVISED_CHECKPOINT}" || ! -f "${SUPERVISED_ROOT}/summary.json" ]]; then
    mkdir -p "${SUPERVISED_ROOT}"
    log "START pure supervised AXT2 training on ${SUPERVISED_DEVICE}"
    conda run -n uncertainty_tta python scripts/train_supervised_baseline.py \
      --manifest-csv "${MANIFEST}" \
      --preproc-root preproc/shifts/modality_matrix \
      --density-root density_stats/shifts/modality_matrix \
      --output-dir "${SUPERVISED_ROOT}" \
      --source-role source_train \
      --epochs 25 \
      --batch-size 1 \
      --acceleration 4 \
      --sigma-mask 0.18 \
      --window-size 1 \
      --num-unrolls 12 \
      --denoiser-sharing independent \
      --chans 64 \
      --num-pools 4 \
      --lr 1e-4 \
      --weight-decay 1e-4 \
      --grad-clip 1 \
      --device "${SUPERVISED_DEVICE}" \
      --seed 7 \
      --split-seed 7 \
      --include-ssim \
      --no-self-loss \
      --supervised-loss-weight 1 \
      --best-val-metric val_nmse \
      --num-workers 0 \
      --require-preproc \
      > "${SUPERVISED_ROOT}/run.log" 2>&1
    log "DONE pure supervised AXT2 training"
  else
    log "SKIP pure supervised training: completed checkpoint exists"
  fi

  if ! grep -q '"denoiser_sharing": "independent"' "${SUPERVISED_ROOT}/config.json"; then
    log "ERROR: supervised checkpoint is not independent-denoiser"
    return 1
  fi
  if ! grep -q '"self_loss": false' "${SUPERVISED_ROOT}/config.json"; then
    log "ERROR: supervised checkpoint unexpectedly uses self-loss"
    return 1
  fi
  if ! grep -q '"training_objective": "normalized_l1_supervised"' "${SUPERVISED_ROOT}/config.json"; then
    log "ERROR: supervised checkpoint has an incorrect objective label"
    return 1
  fi

  run_no_tta \
    supervised_only_no_tta_clean_r4_maskseed7_noiseseed9007 \
    "${SUPERVISED_CHECKPOINT}" \
    normalized_l1_supervised \
    "${SUPERVISED_DEVICE}"
}

train_and_run_supervised_control
log "T2->T2 supervised source-training control complete"
