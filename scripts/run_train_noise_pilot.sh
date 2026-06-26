#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {acc_ensure|acc_traditional|mod_ensure|mod_traditional}" >&2
    exit 2
fi

ROOT="/home/hulabdl/Deng_proj/cardiac_ensure"
EXP="$1"
EPOCHS="${EPOCHS:-25}"
TRAIN_NOISE_SNR_DB_MIN="${TRAIN_NOISE_SNR_DB_MIN:-15}"
TRAIN_NOISE_SNR_DB_MAX="${TRAIN_NOISE_SNR_DB_MAX:-25}"
TRAIN_NOISE_SEED="${TRAIN_NOISE_SEED:-7007}"
VAL_NOISE_SNR_DB="${VAL_NOISE_SNR_DB:-20}"
VAL_NOISE_SEED="${VAL_NOISE_SEED:-8007}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TAG_MIN=$(printf "%s" "$TRAIN_NOISE_SNR_DB_MIN" | tr "." "p")
TAG_MAX=$(printf "%s" "$TRAIN_NOISE_SNR_DB_MAX" | tr "." "p")
VAL_TAG=$(printf "%s" "$VAL_NOISE_SNR_DB" | tr "." "p")
NOISE_TAG="train_noise_snr${TAG_MIN}_${TAG_MAX}_val${VAL_TAG}_seed${TRAIN_NOISE_SEED}"

PREPROC_ROOT="$ROOT/preproc/shifts/main"
DENSITY_ROOT="$ROOT/density_stats/shifts/main"

case "$EXP" in
    acc_ensure)
        DEVICE="${DEVICE:-cuda:4}"
        SCRIPT="$ROOT/scripts/train_shift_true_ensure.py"
        MANIFEST="$ROOT/manifests/shifts/main/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7_${NOISE_TAG}"
        EXTRA_ARGS=""
        ;;
    acc_traditional)
        DEVICE="${DEVICE:-cuda:5}"
        SCRIPT="$ROOT/scripts/train_supervised_baseline.py"
        MANIFEST="$ROOT/manifests/shifts/main/acceleration_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/shifts/main/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_${NOISE_TAG}"
        EXTRA_ARGS=""
        ;;
    mod_ensure)
        DEVICE="${DEVICE:-cuda:6}"
        SCRIPT="$ROOT/scripts/train_shift_true_ensure.py"
        MANIFEST="$ROOT/manifests/shifts/main/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_${NOISE_TAG}"
        EXTRA_ARGS=""
        ;;
    mod_traditional)
        DEVICE="${DEVICE:-cuda:7}"
        SCRIPT="$ROOT/scripts/train_supervised_baseline.py"
        MANIFEST="$ROOT/manifests/shifts/main/modality_shift.csv"
        OUTPUT_DIR="$ROOT/outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_${NOISE_TAG}"
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
    echo "train_noise_snr_db_min=$TRAIN_NOISE_SNR_DB_MIN"
    echo "train_noise_snr_db_max=$TRAIN_NOISE_SNR_DB_MAX"
    echo "train_noise_seed=$TRAIN_NOISE_SEED"
    echo "val_noise_snr_db=$VAL_NOISE_SNR_DB"
    echo "val_noise_seed=$VAL_NOISE_SEED"
    echo "epochs=$EPOCHS"
    echo "output_dir=$OUTPUT_DIR"
    date
    conda run -n uncertainty_tta python "$SCRIPT" \
        --manifest-csv "$MANIFEST" \
        --preproc-root "$PREPROC_ROOT" \
        --density-root "$DENSITY_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --acceleration 4.0 \
        --sigma-mask 0.18 \
        --window-size 1 \
        --num-unrolls 12 \
        --device "$DEVICE" \
        --seed 7 \
        --include-ssim \
        --train-noise-snr-db-min "$TRAIN_NOISE_SNR_DB_MIN" \
        --train-noise-snr-db-max "$TRAIN_NOISE_SNR_DB_MAX" \
        --train-noise-seed "$TRAIN_NOISE_SEED" \
        --val-noise-snr-db "$VAL_NOISE_SNR_DB" \
        --val-noise-seed "$VAL_NOISE_SEED" \
        --best-val-metric noisy_nmse \
        --num-workers "$NUM_WORKERS" \
        $EXTRA_ARGS
    date
} > "$LOG" 2>&1
