#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

CHECKPOINT="outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt"
MANIFEST="manifests/shifts/main/modality_shift.csv"
COMMON=(
  --checkpoint "${CHECKPOINT}"
  --manifest-csv "${MANIFEST}"
  --split-role target_test
  --seed 7
  --test-noise-seed 9007
  --training-objective true_ensure
  --tta-loss l1
  --tta-weight-decay 0
  --grad-clip 1
  --self-val-fraction 0.05
  --early-stop-window 20
  --update-mode adapter
  --acceleration 4
  --lora-layers 1 2 3
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
  local output_dir="$3"
  local rank="$4"
  local lr="$5"
  local steps="$6"
  local max_samples="$7"
  local adapt_dc="$8"

  if [[ -f "${output_dir}/summary.csv" ]]; then
    printf '[skip] %s: summary exists\n' "${name}"
    return 0
  fi
  mkdir -p "${output_dir}"
  local extra=()
  if [[ "${max_samples}" != "all" ]]; then
    extra+=(--max-samples "${max_samples}")
  fi
  if [[ "${adapt_dc}" == "yes" ]]; then
    extra+=(--adapt-dc)
  fi
  printf '[start] %s device=%s rank=%s lr=%s steps=%s samples=%s dc=%s\n' \
    "${name}" "${device}" "${rank}" "${lr}" "${steps}" "${max_samples}" "${adapt_dc}"
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure_lora.py \
    "${COMMON[@]}" \
    --output-dir "${output_dir}" \
    --device "${device}" \
    --lora-rank "${rank}" \
    --tta-lr "${lr}" \
    --tta-steps "${steps}" \
    "${extra[@]}" \
    > "${output_dir}/run.log" 2>&1
  printf '[done] %s\n' "${name}"
}

run_pilot() {
  local root="outputs/tta/shifts/main/modality_shift/lora_phase1_pilot_clean_r4"
  run_one lr1e-5 cuda:0 "${root}/r2_lr1e-5_steps100_n5" 2 1e-5 100 5 no &
  local pid0=$!
  run_one lr1e-4 cuda:5 "${root}/r2_lr1e-4_steps100_n5" 2 1e-4 100 5 no &
  local pid5=$!
  run_one lr3e-4 cuda:6 "${root}/r2_lr3e-4_steps100_n5" 2 3e-4 100 5 no &
  local pid6=$!
  run_one lr1e-3 cuda:7 "${root}/r2_lr1e-3_steps100_n5" 2 1e-3 100 5 no &
  local pid7=$!
  wait "${pid0}" "${pid5}" "${pid6}" "${pid7}"
}

run_main() {
  if [[ $# -ne 1 ]]; then
    printf 'Usage: %s main <learning-rate>\n' "$0" >&2
    return 2
  fi
  local lr="$1"
  local lr_tag="${lr//./p}"
  local root="outputs/tta/shifts/main/modality_shift/lora_phase1_clean_r4_lr${lr_tag}"
  run_one rank1 cuda:0 "${root}/convlora_r1" 1 "${lr}" 250 all no &
  local pid0=$!
  run_one rank2 cuda:5 "${root}/convlora_r2" 2 "${lr}" 250 all no &
  local pid5=$!
  run_one rank4 cuda:6 "${root}/convlora_r4" 4 "${lr}" 250 all no &
  local pid6=$!
  run_one rank2_dc cuda:7 "${root}/convlora_r2_dc" 2 "${lr}" 250 all yes &
  local pid7=$!
  wait "${pid0}" "${pid5}" "${pid6}" "${pid7}"
}

case "${1:-}" in
  pilot)
    run_pilot
    ;;
  main)
    shift
    run_main "$@"
    ;;
  *)
    printf 'Usage: %s {pilot|main <learning-rate>}\n' "$0" >&2
    exit 2
    ;;
esac
