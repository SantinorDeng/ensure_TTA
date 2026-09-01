#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
cd "${ROOT}"

CHECKPOINT="outputs/shifts/modality_matrix/main/brain/axt1/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/best.pt"
MANIFEST="manifests/shifts/modality_matrix/main/brain/axt1.csv"
OUTPUT_ROOT="outputs/tta/other_modality/t1_to_t2/lora_tradeoff_independent_clean_r4"
COMMON=(
  --checkpoint "${CHECKPOINT}"
  --manifest-csv "${MANIFEST}"
  --split-role target_test
  --preproc-root preproc/shifts/modality_matrix
  --density-root density_stats/shifts/modality_matrix
  --target-shift-name brain_modality_matrix_axt1_to_axt2
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
  --lora-layers 1 2 3
  --run-tta
  --include-ssim
  --save-recons
  --save-curves
  --num-workers 0
  --fail-fast
)

run_task() {
  local method="$1"
  local device="$2"
  local start_sample="$3"
  local max_samples="$4"
  local steps="$5"
  local output_dir="$6"
  local extra=()

  if [[ -f "${output_dir}/summary.json" ]]; then
    printf '[skip] method=%s range=[%s,%s): summary exists\n' \
      "${method}" "${start_sample}" "$((start_sample + max_samples))"
    return 0
  fi
  if [[ "${method}" == "dc_only" ]]; then
    extra+=(--dc-only)
  else
    extra+=(--lora-rank "${method#r}")
  fi

  mkdir -p "${output_dir}"
  printf '[start] method=%s device=%s range=[%s,%s) steps=%s\n' \
    "${method}" "${device}" "${start_sample}" "$((start_sample + max_samples))" "${steps}"
  conda run -n uncertainty_tta python scripts/tta_shift_true_ensure_lora.py \
    "${COMMON[@]}" \
    --device "${device}" \
    --output-dir "${output_dir}" \
    --start-sample "${start_sample}" \
    --max-samples "${max_samples}" \
    --tta-steps "${steps}" \
    "${extra[@]}" \
    > "${output_dir}/run.log" 2>&1
  printf '[done] method=%s device=%s range=[%s,%s)\n' \
    "${method}" "${device}" "${start_sample}" "$((start_sample + max_samples))"
}

run_smoke() {
  local smoke_root="${OUTPUT_ROOT}/_smoke"
  run_task r8 cuda:0 0 1 3 "${smoke_root}/r8" &
  local pid0=$!
  run_task dc_only cuda:1 0 1 3 "${smoke_root}/dc_only" &
  local pid1=$!
  wait "${pid0}" "${pid1}"
}

run_main() {
  local methods=(r1 r2 r4 r8 dc_only)
  local task_methods=()
  local task_starts=()
  local method
  local start
  for start in 0 20 40 60 80; do
    for method in "${methods[@]}"; do
      task_methods+=("${method}")
      task_starts+=("${start}")
    done
  done

  worker() {
    local gpu="$1"
    local index
    local task_method
    local task_start
    local shard
    for ((index = gpu; index < ${#task_methods[@]}; index += 8)); do
      task_method="${task_methods[index]}"
      task_start="${task_starts[index]}"
      shard=$(printf '%03d_%03d' "${task_start}" "$((task_start + 20))")
      run_task \
        "${task_method}" \
        "cuda:${gpu}" \
        "${task_start}" \
        20 \
        250 \
        "${OUTPUT_ROOT}/${task_method}/shards/${shard}"
    done
  }

  local pids=()
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    worker "${gpu}" &
    pids+=("$!")
  done
  wait "${pids[@]}"
  printf '[done] all T1-to-T2 LoRA trade-off shards\n'
}

case "${1:-}" in
  smoke)
    run_smoke
    ;;
  main)
    run_main
    ;;
  task)
    if [[ $# -ne 5 ]]; then
      printf 'Usage: %s task <method> <device> <start-sample> <max-samples>\n' "$0" >&2
      exit 2
    fi
    run_task "$2" "$3" "$4" "$5" 250 \
      "${OUTPUT_ROOT}/$2/shards/$(printf '%03d_%03d' "$4" "$(( $4 + $5 ))")"
    ;;
  *)
    printf 'Usage: %s {smoke|main|task <method> <device> <start-sample> <max-samples>}\n' "$0" >&2
    exit 2
    ;;
esac
