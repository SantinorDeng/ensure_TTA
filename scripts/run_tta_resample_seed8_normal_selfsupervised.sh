#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {acc_ensure|acc_traditional|mod_ensure|mod_traditional}" >&2
    exit 2
fi

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
EXP="$1"
TTA_STEPS="${TTA_STEPS:-250}"
TTA_LR="${TTA_LR:-1e-5}"
SELF_VAL_FRACTION="${SELF_VAL_FRACTION:-0.05}"
EARLY_STOP_WINDOW="${EARLY_STOP_WINDOW:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"

case "$EXP" in
    acc_ensure)
        DEVICE="cuda:0"
        SCRIPT="$ROOT/scripts/tta_shift_true_ensure.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main_resample_seed8/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main_resample_seed8/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7_self_supervised_tta_normal_data"
        EXTRA_ARGS="--tta-loss self_supervised"
        ;;
    acc_traditional)
        DEVICE="cuda:1"
        SCRIPT="$ROOT/scripts/tta_shift_supervised_baseline.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main_resample_seed8/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main_resample_seed8/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_self_supervised_tta_normal_data"
        EXTRA_ARGS=""
        ;;
    mod_ensure)
        DEVICE="cuda:2"
        SCRIPT="$ROOT/scripts/tta_shift_true_ensure.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main_resample_seed8/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main_resample_seed8/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_self_supervised_tta_normal_data"
        EXTRA_ARGS="--tta-loss self_supervised"
        ;;
    mod_traditional)
        DEVICE="cuda:3"
        SCRIPT="$ROOT/scripts/tta_shift_supervised_baseline.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main_resample_seed8/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main_resample_seed8/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_self_supervised_tta_normal_data"
        EXTRA_ARGS=""
        ;;
    *)
        echo "Unknown experiment: $EXP" >&2
        exit 2
        ;;
esac

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run.log"

{
    echo "experiment=$EXP"
    echo "device=$DEVICE"
    echo "normal_data=true"
    echo "output_dir=$OUTPUT_DIR"
    date
    conda run -n uncertainty_tta python "$SCRIPT" \
        --checkpoint "$CHECKPOINT" \
        --manifest-csv "$MANIFEST" \
        --output-dir "$OUTPUT_DIR" \
        --split-role target_test \
        --device "$DEVICE" \
        --seed 7 \
        --tta-steps "$TTA_STEPS" \
        --tta-lr "$TTA_LR" \
        --tta-weight-decay 0.0 \
        --grad-clip 1.0 \
        --self-val-fraction "$SELF_VAL_FRACTION" \
        --early-stop-window "$EARLY_STOP_WINDOW" \
        --run-tta \
        --include-ssim \
        --save-recons \
        --save-curves \
        --num-workers "$NUM_WORKERS" \
        $EXTRA_ARGS
    date
} > "$LOG" 2>&1
