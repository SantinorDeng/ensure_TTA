#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

TRAIN_DEVICE_INDEX="${1:-5}"
FULL_DEVICE="cuda:${TRAIN_DEVICE_INDEX}"
LORA_DEVICE_INDEX="${2:-7}"
LORA_DEVICE="cuda:${LORA_DEVICE_INDEX}"
CHECKPOINT="outputs/shifts/modality_matrix/main/brain/axt2/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt"
CONFIG="outputs/shifts/modality_matrix/main/brain/axt2/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/config.json"
MANIFEST="manifests/shifts/modality_matrix/main/brain/axt2.csv"
SHIFT="brain_modality_matrix_axt2_to_axt2"
OUTPUT_ROOT="outputs/tta/other_modality/t2_to_t2"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_eval() {
  local name="$1"
  local device="$2"
  shift 2
  local output_dir="${OUTPUT_ROOT}/${name}"
  if [[ -f "${output_dir}/summary.csv" && -f "${output_dir}/summary.json" ]]; then
    log "SKIP ${name}: completed summary exists"
    return 0
  fi

  mkdir -p "${output_dir}"
  log "START ${name} on ${device}"
  "$@" --output-dir "${output_dir}" > "${output_dir}/run.log" 2>&1
  log "DONE ${name}"
}

log "Queue matched AXT2 TRUE-ENSURE training on GPU ${TRAIN_DEVICE_INDEX}"
conda run -n uncertainty_tta python scripts/launch_modality_matrix_jobs.py \
  --stage train \
  --devices "${TRAIN_DEVICE_INDEX}" \
  --min-free-memory-gb 22 \
  --datasets brain \
  --source brain:axt2 \
  --methods ensure \
  --denoiser-sharing independent \
  --resume

if [[ ! -f "${CHECKPOINT}" || ! -f "${CONFIG}" ]]; then
  log "ERROR: matched AXT2 checkpoint/config was not produced"
  exit 1
fi
if ! grep -q '"denoiser_sharing": "independent"' "${CONFIG}"; then
  log "ERROR: ${CONFIG} is not an independent-denoiser checkpoint"
  exit 1
fi

COMMON_ARGS=(
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure.py
  --checkpoint "${CHECKPOINT}"
  --manifest-csv "${MANIFEST}"
  --split-role target_test
  --target-shift-name "${SHIFT}"
  --training-objective true_ensure
  --device "${FULL_DEVICE}"
  --seed 7
  --test-noise-seed 9007
  --tta-loss l1
  --tta-steps 250
  --tta-lr 1e-5
  --tta-weight-decay 0
  --grad-clip 1
  --self-val-fraction 0.05
  --early-stop-window 20
  --update-mode all_params
  --acceleration 4
  --sigma-mask 0.18
  --preproc-root preproc/shifts/modality_matrix
  --density-root density_stats/shifts/modality_matrix
  --include-ssim
  --save-recons
  --save-curves
  --num-workers 0
  --fail-fast
)

run_eval \
  ensure_l1_clean_r4_maskseed7_noiseseed9007 \
  "${FULL_DEVICE}" \
  "${COMMON_ARGS[@]}" \
  --run-tta &
full_pid=$!

run_eval \
  ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007 \
  "${LORA_DEVICE}" \
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure_lora.py \
  --checkpoint "${CHECKPOINT}" \
  --manifest-csv "${MANIFEST}" \
  --split-role target_test \
  --target-shift-name "${SHIFT}" \
  --training-objective true_ensure \
  --device "${LORA_DEVICE}" \
  --seed 7 \
  --test-noise-seed 9007 \
  --tta-loss l1 \
  --tta-steps 250 \
  --tta-lr 1e-3 \
  --tta-weight-decay 0 \
  --grad-clip 1 \
  --self-val-fraction 0.05 \
  --early-stop-window 20 \
  --update-mode adapter \
  --lora-rank 2 \
  --lora-layers 1 2 3 \
  --adapt-dc \
  --acceleration 4 \
  --sigma-mask 0.18 \
  --preproc-root preproc/shifts/modality_matrix \
  --density-root density_stats/shifts/modality_matrix \
  --run-tta \
  --include-ssim \
  --save-recons \
  --save-curves \
  --num-workers 0 \
  --fail-fast &
lora_pid=$!

log "WAIT full TTA pid=${full_pid}, LoRA TTA pid=${lora_pid}"
wait "${full_pid}"
wait "${lora_pid}"

log "T2->T2 in-domain experiment complete"
