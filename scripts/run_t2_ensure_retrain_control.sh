#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

DEVICE="${1:-cuda:4}"
MANIFEST="manifests/shifts/modality_matrix/main/brain/axt2.csv"
SHIFT="brain_modality_matrix_axt2_to_axt2"
TRAIN_ROOT="outputs/shifts/modality_matrix/main/brain/axt2/ensure_retrained_r4_w1_unroll12_independent_seed7_clean"
CHECKPOINT="${TRAIN_ROOT}/best.pt"
EVAL_ROOT="outputs/tta/other_modality/t2_to_t2/ensure_retrained_clean_no_tta_r4_maskseed7_noiseseed9007"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if [[ ! -f "${CHECKPOINT}" || ! -f "${TRAIN_ROOT}/summary.json" ]]; then
  mkdir -p "${TRAIN_ROOT}"
  log "START TRUE-ENSURE AXT2 training on ${DEVICE}"
  conda run -n uncertainty_tta python scripts/train_shift_true_ensure.py \
    --manifest-csv "${MANIFEST}" \
    --preproc-root preproc/shifts/modality_matrix \
    --density-root density_stats/shifts/modality_matrix \
    --output-dir "${TRAIN_ROOT}" \
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
    --device "${DEVICE}" \
    --seed 7 \
    --split-seed 7 \
    --include-ssim \
    --best-val-metric val_nmse \
    --num-workers 0 \
    --require-preproc \
    > "${TRAIN_ROOT}/run.log" 2>&1
  log "DONE TRUE-ENSURE AXT2 training"
else
  log "SKIP TRUE-ENSURE training: completed checkpoint exists"
fi

if ! grep -q '"denoiser_sharing": "independent"' "${TRAIN_ROOT}/config.json"; then
  log "ERROR: ENSURE checkpoint is not independent-denoiser"
  exit 1
fi
if ! grep -q '"train_dataset_len": 350' "${TRAIN_ROOT}/config.json" || \
   ! grep -q '"val_dataset_len": 90' "${TRAIN_ROOT}/config.json"; then
  log "ERROR: ENSURE checkpoint used an unexpected source split"
  exit 1
fi

if [[ ! -f "${EVAL_ROOT}/summary.csv" || ! -f "${EVAL_ROOT}/summary.json" ]]; then
  mkdir -p "${EVAL_ROOT}"
  log "START retrained ENSURE T2->T2 no-TTA evaluation on ${DEVICE}"
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure.py \
    --checkpoint "${CHECKPOINT}" \
    --manifest-csv "${MANIFEST}" \
    --output-dir "${EVAL_ROOT}" \
    --split-role target_test \
    --target-shift-name "${SHIFT}" \
    --training-objective true_ensure \
    --device "${DEVICE}" \
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
    > "${EVAL_ROOT}/run.log" 2>&1
  log "DONE retrained ENSURE T2->T2 no-TTA evaluation"
else
  log "SKIP ENSURE no-TTA evaluation: completed summary exists"
fi

log "TRUE-ENSURE retrain control complete"
