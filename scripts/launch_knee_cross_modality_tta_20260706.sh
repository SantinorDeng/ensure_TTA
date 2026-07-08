#!/bin/sh
set -eu

ROOT=/home/hulabdl/Deng_proj/cardiac_ensure
RUNNER="$ROOT/scripts/run_modality_matrix_job.py"
COMMON="--test-noise-snr-db 20 --test-noise-seed 9007 --eval-seed 7 --resume"

cd "$ROOT"

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset knee --source pd --method ensure --device cuda:4 \
  --target-shift-name knee_modality_matrix_pd_to_pdfs $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset knee --source pd --method traditional --device cuda:5 \
  --target-shift-name knee_modality_matrix_pd_to_pdfs $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset knee --source pdfs --method ensure --device cuda:6 \
  --target-shift-name knee_modality_matrix_pdfs_to_pd $COMMON &

conda run -n uncertainty_tta python "$RUNNER" tta \
  --dataset knee --source pdfs --method traditional --device cuda:7 \
  --target-shift-name knee_modality_matrix_pdfs_to_pd $COMMON &

wait
