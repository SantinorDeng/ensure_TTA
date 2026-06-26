#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {acc_ensure|acc_traditional|mod_ensure|mod_traditional|acc_ensure_train_noise|acc_traditional_train_noise|mod_ensure_train_noise|mod_traditional_train_noise}" >&2
    exit 2
fi

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
EXP="$1"
NOISE_SNR_DB="${NOISE_SNR_DB:-20}"
NOISE_SEED="${NOISE_SEED:-9007}"
TRAIN_NOISE_TAG="${TRAIN_NOISE_TAG:-train_noise_snr15_25_val20_seed7007}"
TTA_STEPS="${TTA_STEPS:-250}"
TTA_LR="${TTA_LR:-1e-5}"
SELF_VAL_FRACTION="${SELF_VAL_FRACTION:-0.05}"
EARLY_STOP_WINDOW="${EARLY_STOP_WINDOW:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NOISE_TAG=$(printf "%s" "$NOISE_SNR_DB" | tr "." "p")

case "$EXP" in
    acc_ensure)
        DEVICE="cuda:4"
        SCRIPT="$ROOT/scripts/tta_shift_true_ensure.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7_noise_snr${NOISE_TAG}"
        EXTRA_ARGS="--tta-loss ensure"
        ;;
    acc_traditional)
        DEVICE="cuda:5"
        SCRIPT="$ROOT/scripts/tta_shift_supervised_baseline.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_noise_snr${NOISE_TAG}"
        EXTRA_ARGS=""
        ;;
    mod_ensure)
        DEVICE="cuda:6"
        SCRIPT="$ROOT/scripts/tta_shift_true_ensure.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_noise_snr${NOISE_TAG}"
        EXTRA_ARGS="--tta-loss ensure"
        ;;
    mod_traditional)
        DEVICE="cuda:7"
        SCRIPT="$ROOT/scripts/tta_shift_supervised_baseline.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_noise_snr${NOISE_TAG}"
        EXTRA_ARGS=""
        ;;
    acc_ensure_train_noise)
        DEVICE="${DEVICE:-cuda:4}"
        SCRIPT="$ROOT/scripts/tta_shift_true_ensure.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7_${TRAIN_NOISE_TAG}/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7_${TRAIN_NOISE_TAG}_noise_snr${NOISE_TAG}"
        EXTRA_ARGS="--tta-loss ensure"
        ;;
    acc_traditional_train_noise)
        DEVICE="${DEVICE:-cuda:5}"
        SCRIPT="$ROOT/scripts/tta_shift_supervised_baseline.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_${TRAIN_NOISE_TAG}/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_${TRAIN_NOISE_TAG}_noise_snr${NOISE_TAG}"
        EXTRA_ARGS=""
        ;;
    mod_ensure_train_noise)
        DEVICE="${DEVICE:-cuda:6}"
        SCRIPT="$ROOT/scripts/tta_shift_true_ensure.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_${TRAIN_NOISE_TAG}/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_${TRAIN_NOISE_TAG}_noise_snr${NOISE_TAG}"
        EXTRA_ARGS="--tta-loss ensure"
        ;;
    mod_traditional_train_noise)
        DEVICE="${DEVICE:-cuda:7}"
        SCRIPT="$ROOT/scripts/tta_shift_supervised_baseline.py"
        CHECKPOINT="$ROOT/outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_${TRAIN_NOISE_TAG}/best.pt"
        MANIFEST="$ROOT/manifests/shifts/main/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/tta/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_${TRAIN_NOISE_TAG}_noise_snr${NOISE_TAG}"
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
    echo "noise_snr_db=$NOISE_SNR_DB"
    echo "noise_seed=$NOISE_SEED"
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
        --test-noise-snr-db "$NOISE_SNR_DB" \
        --test-noise-seed "$NOISE_SEED" \
        --num-workers "$NUM_WORKERS" \
        $EXTRA_ARGS
    date
} > "$LOG" 2>&1
