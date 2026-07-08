#!/bin/sh
set -eu

ROOT=/home/hulabdl/Deng_proj/cardiac_ensure
RUNNER="$ROOT/scripts/run_modality_matrix_job.py"
COMMON="--test-noise-snr-db 20 --test-noise-seed 9007 --eval-seed 7 --resume"

cd "$ROOT"

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axt1pre --method ensure --device cuda:4 \
  --target-shift-name brain_modality_matrix_axt1pre_to_axt1post $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axt1pre --method traditional --device cuda:5 \
  --target-shift-name brain_modality_matrix_axt1pre_to_axt1post $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axt1post --method ensure --device cuda:6 \
  --target-shift-name brain_modality_matrix_axt1post_to_axt1pre $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axt1post --method traditional --device cuda:7 \
  --target-shift-name brain_modality_matrix_axt1post_to_axt1pre $COMMON &

wait
