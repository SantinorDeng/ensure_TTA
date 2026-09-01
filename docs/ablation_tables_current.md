# Current ablation and efficiency tables

Updated: 2026-08-30

## A. Recommended main ablation table: cross-domain T1 to T2

Protocol: clean target test data, acceleration `R_acc=4`, 100 AXT2 test slices, source acquisition AXT1. The AXT1 source checkpoints in this table were trained with SNR 15--25 input augmentation and validated at SNR 20. All TTA rows use measured-k-space L1, at most 250 steps, and no target ground truth.

| Source training | Target-time method | Adapted parameters | Trainable parameters | Fraction of full ENSURE | PSNR (dB) | SSIM | NMSE | PSNR gain (dB) | Adaptation time (s/slice) | Total time (s/slice) | Negative adaptation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRUE-ENSURE | Frozen / no TTA | None | 0 | 0% | 33.5482 | 0.8741 | 0.217368 | 0.0000 | 0.00 | not independently timed | 0% by definition |
| TRUE-ENSURE | Full-parameter TTA | Full network | 1,354,764 | 100% | **37.0916** | **0.9592** | **0.004667** | **+3.5434** | 171.77 | 180.29 | 1% |
| TRUE-ENSURE | Conv-LoRA r=2 + 12 DC scalars | LoRA + DC | 46,092 | 3.40% | 36.6681 | 0.9547 | 0.005457 | +3.1199 | **45.84** | **48.22** | 4% |
| SSDU | Full-parameter TTA | Full network | 112,908 | 100% of SSDU model | 34.4353 | 0.9153 | 0.144093 | +4.1143 | 87.59 | 101.40 | 7% |
| Supervised + self-supervised source training | Full-parameter TTA | Full network | 1,354,764 | 100% | 35.8783 | 0.9399 | 0.006135 | +8.2082 | 102.53 | 115.35 | 1% |

Interpretation:

- Full TRUE-ENSURE TTA has the best reconstruction quality in this representative T1 to T2 experiment.
- LoRA r=2 + DC retains most of that quality while using 29.4 times fewer trainable parameters, 3.75 times less adaptation time, and 3.74 times less total runtime.
- Relative to full TRUE-ENSURE TTA, LoRA r=2 + DC changes PSNR by -0.4235 dB, SSIM by -0.00446, and NMSE by +0.000789.
- The T1 to T2 negative-adaptation rate is 4% for LoRA versus 1% for full TTA. The lower 1% LoRA rate observed in the separate mixed-modality rank study must not be reported as the T1 to T2 result.
- SSDU uses a shared denoiser and therefore has only 112,908 full-model parameters; its parameter count is not directly comparable to the independent-denoiser TRUE-ENSURE and supervised models.

## B. Clean in-domain sanity check: T2 to T2

Protocol: both source training and target testing are clean; `R_acc=4`; 100 AXT2 test slices; frozen inference with no TTA. The source-train and target-test volumes do not overlap.

| Source training | Domain | TTA | Trainable parameters | PSNR (dB) | SSIM | NMSE | Total time (s/slice) | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| TRUE-ENSURE, clean T2 training | T2 to T2 | No | 0 | 30.5019 | 0.7940 | 0.194285 | 0.0776 | Complete, 100/100 |
| Pure supervised, clean T2 training | T2 to T2 | No | 0 | **35.2070** | **0.9015** | **0.068196** | 0.0789 | Complete, 100/100 |

This is a sanity/control table, not a replacement for the cross-domain ablation. It demonstrates the expected advantage of direct supervised in-domain training when target-domain labels are available.

## C. Existing T2 to T2 adaptation results with the older noise-augmented T2 checkpoint

These rows are complete, but their source checkpoint was trained with SNR 15--25 augmentation. They must not be merged with Table B, whose checkpoints were trained without noise.

| Source checkpoint | Target-time method | Trainable parameters | PSNR (dB) | SSIM | NMSE | Adaptation time (s/slice) | Total time (s/slice) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRUE-ENSURE, noise-augmented T2 training | Frozen / no TTA | 0 | 32.6405 | 0.8599 | 0.075635 | 0.00 | 0.0601 |
| TRUE-ENSURE, noise-augmented T2 training | Full-parameter TTA | 1,354,764 | **36.2121** | **0.9425** | **0.005672** | 49.02 | 52.47 |
| TRUE-ENSURE, noise-augmented T2 training | Conv-LoRA r=2 + 12 DC scalars | 46,092 | 35.8821 | 0.9403 | 0.006176 | 74.40 | 79.47 |

## D. Separate LoRA capacity trade-off (do not duplicate in the main table)

Protocol: the existing 100-slice main mixed-modality shift experiment, clean `R_acc=4`, shared-denoiser TRUE-ENSURE checkpoint. This is a different evaluation set/model topology from Table A.

| Adaptation method | Trainable parameters | PSNR (dB) | SSIM | NMSE | Adaptation time (s/slice) | Total time (s/slice) | Negative adaptation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-parameter TTA | 112,908 | **36.8407** | 0.9339 | 0.004493 | 39.30 | 41.88 | 4% |
| Conv-LoRA r=1 | 1,920 | 36.4320 | 0.9299 | 0.004702 | 44.96 | 47.40 | 1% |
| Conv-LoRA r=2 | 3,840 | 36.5805 | 0.9307 | 0.007343 | 39.43 | 41.65 | 1% |
| Conv-LoRA r=4 | 7,680 | 36.6425 | 0.9303 | 0.007466 | **36.15** | **38.21** | 2% |
| Conv-LoRA r=2 + 12 DC scalars | 3,852 | 36.7987 | **0.9360** | **0.003588** | 38.40 | 40.55 | 1% |

## Recommended paper organization

- Main ablation/efficiency table: Table A only.
- In-domain sanity check: Table B, preferably in the appendix or robustness section.
- LoRA-rank trade-off: Table D as its own compact table or figure.
- Do not include a DC-only row unless that experiment is deliberately added later.
- Do not mix Table C with the clean in-domain Table B because their source-training noise protocols differ.

## Result provenance

- Table A: `outputs/tta/other_modality/t1_to_t2/*clean_r4*/summary.csv` and corresponding `metrics.csv` files.
- Table B: `outputs/tta/other_modality/t2_to_t2/ensure_retrained_clean_no_tta_r4_maskseed7_noiseseed9007` and `supervised_only_no_tta_clean_r4_maskseed7_noiseseed9007`.
- Table C: `outputs/tta/other_modality/t2_to_t2/ensure_no_tta_clean_r4_maskseed7_noiseseed9007`, `ensure_l1_clean_r4_maskseed7_noiseseed9007`, and `ensure_convlora_r2_dc_l1_clean_r4_lr1e-3_maskseed7_noiseseed9007`.
- Table D: `outputs/tta/shifts/main/modality_shift/lora_phase1_clean_r4_lr1e-3/*/summary.csv` plus the strict-runtime full-TTA result.
