#!/bin/sh
set -eu

ROOT=/home/hulabdl/Deng_proj/cardiac_ensure
RUNNER="$ROOT/scripts/run_modality_matrix_job.py"
COMMON="--test-noise-snr-db 20 --test-noise-seed 9007 --eval-seed 7 --resume"

cd "$ROOT"

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axflair --method ensure --device cuda:4 \
  --target-shift-name brain_modality_matrix_axflair_to_axt1 $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axflair --method traditional --device cuda:5 \
  --target-shift-name brain_modality_matrix_axflair_to_axt1 $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axt1 --method ensure --device cuda:6 \
  --target-shift-name brain_modality_matrix_axt1_to_axflair $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset brain --source axt1 --method traditional --device cuda:7 \
  --target-shift-name brain_modality_matrix_axt1_to_axflair $COMMON &

wait
