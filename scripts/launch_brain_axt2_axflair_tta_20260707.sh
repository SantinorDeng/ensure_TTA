#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
ENV_NAME="uncertainty_tta"

MANIFEST_ROOT="${ROOT}/manifests/shifts/modality_matrix/main/brain"
PREPROC_ROOT="${ROOT}/preproc/shifts/modality_matrix"
DENSITY_ROOT="${ROOT}/density_stats/shifts/modality_matrix"
OUT_ROOT="${ROOT}/outputs/tta/shifts/modality_matrix/main/brain"

T2_ENSURE_CKPT="${ROOT}/outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_train_noise_snr15_25_val20_seed7007/best.pt"
T2_TRAD_CKPT="${ROOT}/outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt"
AXFLAIR_ENSURE_CKPT="${ROOT}/outputs/shifts/modality_matrix/main/brain/axflair/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt"
AXFLAIR_TRAD_CKPT="${ROOT}/outputs/shifts/modality_matrix/main/brain/axflair/traditional_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt"

usage() {
  echo "Usage: $0 {axt2_ensure|axt2_traditional|axflair_ensure|axflair_traditional|launch_all}" >&2
}

run_tta() {
  local device="$1"
  local checkpoint="$2"
  local manifest="$3"
  local output_dir="$4"
  local training_objective="$5"
  local target_shift_name="$6"

  mkdir -p "${output_dir}"
  exec > >(tee -a "${output_dir}/run.log") 2>&1

  echo "[start] $(date -Is)"
  echo "[device] ${device}"
  echo "[checkpoint] ${checkpoint}"
  echo "[manifest] ${manifest}"
  echo "[output] ${output_dir}"
  echo "[target-shift] ${target_shift_name}"

  conda run -n "${ENV_NAME}" python "${ROOT}/scripts/tta_shift_true_ensure.py" \
    --checkpoint "${checkpoint}" \
    --manifest-csv "${manifest}" \
    --output-dir "${output_dir}" \
    --split-role target_test \
    --device "${device}" \
    --seed 7 \
    --test-noise-snr-db 20 \
    --test-noise-seed 9007 \
    --training-objective "${training_objective}" \
    --tta-loss l1 \
    --tta-steps 250 \
    --tta-lr 1e-5 \
    --tta-weight-decay 0 \
    --grad-clip 1 \
    --self-val-fraction 0.05 \
    --early-stop-window 20 \
    --update-mode all_params \
    --run-tta \
    --include-ssim \
    --save-recons \
    --save-curves \
    --num-workers 0 \
    --preproc-root "${PREPROC_ROOT}" \
    --density-root "${DENSITY_ROOT}" \
    --target-shift-name "${target_shift_name}"

  echo "[done] $(date -Is)"
}

case "${1:-}" in
  axt2_ensure)
    run_tta \
      "cuda:4" \
      "${T2_ENSURE_CKPT}" \
      "${MANIFEST_ROOT}/axt2.csv" \
      "${OUT_ROOT}/axt2/ensure_l1_snr20_maskseed7_noiseseed9007_subset9d20678e" \
      "true_ensure" \
      "brain_modality_matrix_axt2_to_axflair"
    ;;
  axt2_traditional)
    run_tta \
      "cuda:5" \
      "${T2_TRAD_CKPT}" \
      "${MANIFEST_ROOT}/axt2.csv" \
      "${OUT_ROOT}/axt2/traditional_l1_snr20_maskseed7_noiseseed9007_subset9d20678e" \
      "normalized_l1_supervised_plus_measured_kspace_self_supervision" \
      "brain_modality_matrix_axt2_to_axflair"
    ;;
  axflair_ensure)
    run_tta \
      "cuda:6" \
      "${AXFLAIR_ENSURE_CKPT}" \
      "${MANIFEST_ROOT}/axflair.csv" \
      "${OUT_ROOT}/axflair/ensure_l1_snr20_maskseed7_noiseseed9007_subset8c7e88d2" \
      "true_ensure" \
      "brain_modality_matrix_axflair_to_axt2"
    ;;
  axflair_traditional)
    run_tta \
      "cuda:7" \
      "${AXFLAIR_TRAD_CKPT}" \
      "${MANIFEST_ROOT}/axflair.csv" \
      "${OUT_ROOT}/axflair/traditional_l1_snr20_maskseed7_noiseseed9007_subset8c7e88d2" \
      "normalized_l1_supervised_plus_measured_kspace_self_supervision" \
      "brain_modality_matrix_axflair_to_axt2"
    ;;
  launch_all)
    SESSION="brain_axt2_axflair_tta_20260707"
    tmux new-session -d -s "${SESSION}" -n "axt2_ensure" "$0 axt2_ensure"
    tmux new-window -t "${SESSION}" -n "axt2_trad" "$0 axt2_traditional"
    tmux new-window -t "${SESSION}" -n "axflair_ensure" "$0 axflair_ensure"
    tmux new-window -t "${SESSION}" -n "axflair_trad" "$0 axflair_traditional"
    echo "Launched tmux session: ${SESSION}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
