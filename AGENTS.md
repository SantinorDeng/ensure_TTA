# AGENTS.md

这份文件给后续进入本项目的 Codex/Agent 使用。开始任何修改、实验或排查前，先读本文件，再读相关源码和现有文档。

## 项目定位

`/home/hulabdl/Deng_proj/cardiac_ensure` 是 cardiac cine MRI reconstruction / TRUE-ENSURE / TTA 实验项目。当前重点包含：

- supervised baseline 和 TRUE-ENSURE 源域训练。
- shift setting 下的 acceleration / modality / anatomy / dataset 实验。
- test-time adaptation (TTA)，包含 TRUE-ENSURE loss 和传统 measured-kspace self-supervised L1 对照。
- 输出指标通常关注 `PSNR / SSIM / NMSE`、TTA 前后变化、runtime、negative adaptation。

## 运行环境

- Conda 环境：`uncertainty_tta`。
- 默认从项目根目录运行命令：

```sh
cd /home/hulabdl/Deng_proj/cardiac_ensure
conda run -n uncertainty_tta python ...
```

- 实验必须优先使用 GPU，不要静默退回 CPU。跑实验前确认 CUDA 可见：

```sh
conda run -n uncertainty_tta python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

- 无沙箱环境应能看到 8 张 NVIDIA RTX 4090。若只看到 CPU、看不到 8 张 GPU，或脚本日志里 `Device: cpu`，应先停止并向用户说明，不要继续正式实验。
- 代码里的 `resolve_device()` 在不传 `--device` 时会自动选 `cuda` 或 `cpu`。正式实验请显式传 `--device cuda:N`。

## 后台实验习惯

长实验一般用 `tmux` 后台运行，避免前台会话断开影响实验。示例：

```sh
tmux new -s tta_snr20
cd /home/hulabdl/Deng_proj/cardiac_ensure
sh scripts/run_tta_noise_snr20.sh acc_ensure
```

或在已有 tmux session 中分别跑：

```sh
NOISE_SNR_DB=20 sh scripts/run_tta_noise_snr20.sh acc_ensure
NOISE_SNR_DB=20 sh scripts/run_tta_noise_snr20.sh acc_traditional
NOISE_SNR_DB=20 sh scripts/run_tta_noise_snr20.sh mod_ensure
NOISE_SNR_DB=20 sh scripts/run_tta_noise_snr20.sh mod_traditional
```

`scripts/run_tta_noise_snr20.sh` 当前默认分配：

- `acc_ensure` -> `cuda:4`
- `acc_traditional` -> `cuda:5`
- `mod_ensure` -> `cuda:6`
- `mod_traditional` -> `cuda:7`

启动前用 `nvidia-smi` 或等价命令检查显存占用，避免抢占用户正在跑的任务。

## TTA 实验约束

- TTA 实验需要保存 recon。正式跑 TTA 时保留 `--save-recons`，不要加 `--no-save-recons`。
- 通常保留 `--include-ssim` 和 `--save-curves`，便于后续分析。
- 默认 TTA 参数常见为：
  - `--tta-steps 250`
  - `--tta-lr 1e-5`
  - `--tta-weight-decay 0.0`
  - `--grad-clip 1.0`
  - `--self-val-fraction 0.05`
  - `--early-stop-window 20`
  - `--seed 7`
  - `--test-noise-seed 9007`
- TRUE-ENSURE TTA 入口用 `scripts/tta_shift_true_ensure.py`，并明确 `--tta-loss ensure`。
- 传统 TTA baseline 入口用 `scripts/tta_shift_supervised_baseline.py`。
- `tta_shift_true_ensure.py` 的 `--save-recons` 默认不是开启；脚本模板已经显式加了，手写命令时不要漏。
- `tta_shift_supervised_baseline.py` 默认保存 recon，但正式命令仍建议显式保留 `--save-recons`。
- 跑完整实验前可以先用 `--max-samples` 做 smoke test；正式结果不要混淆 smoke output 和 main output。

## 目录约定

- `datasets/`：数据集、预处理、shift manifest dataset、density 统计生成。
- `models/`：动态 MRI 重建模型，主要是 `TemporalNormUnet`。
- `ops/`：MRI forward/adjoint/normal operator 和 CG solver。
- `losses/`：TRUE-ENSURE loss。
- `scripts/`：训练、TTA、评估、manifest 构建入口。
- `manifests/`：实验清单。
  - `manifests/shifts/main/*.csv` 是 main shift 实验常用 manifest。
  - `manifests/shifts/debug/*.csv`、`pilot/*.csv` 只用于调试/小规模试跑。
- `preproc/`：预处理 sidecar 数据。
- `density_stats/`：density / sampling 统计文件。
- `outputs/shifts/main/.../best.pt`：shift 源域训练 checkpoint。
- `outputs/tta/shifts/main/.../{metrics.csv,summary.csv,run.log}`：TTA 结果、汇总和日志。

## 常用 checkpoint 和 manifest

Acceleration shift:

- TRUE-ENSURE checkpoint：`outputs/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt`
- Traditional checkpoint：`outputs/shifts/main/acceleration_shift/uncertainty_joint_source_r4_w1_unroll12_seed7/best.pt`
- Manifest：`manifests/shifts/main/acceleration_shift.csv`

Modality shift:

- TRUE-ENSURE checkpoint：`outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt`
- Traditional checkpoint：`outputs/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7/best.pt`
- Manifest：`manifests/shifts/main/modality_shift.csv`

Anatomy shift:

- TRUE-ENSURE checkpoint：`outputs/shifts/main/anatomy_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt`
- Traditional checkpoint：`outputs/shifts/main/anatomy_shift/uncertainty_joint_source_r4_w1_unroll12_seed7/best.pt`
- Manifest：`manifests/shifts/main/anatomy_shift.csv`

## 输出命名

保持输出路径能直接读出实验条件。已有约定示例：

- `outputs/tta/shifts/main/acceleration_shift/true_ensure_source_r4_w1_auto_unroll_seed7_noise_snr20`
- `outputs/tta/shifts/main/modality_shift/uncertainty_joint_source_r4_w1_unroll12_seed7_noise_snr20`
- `outputs/tta/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7_ensure_loss_strict_runtime`

不要把不同 noise level、loss、runtime setting 或 smoke/main 结果写进同一个 output directory。

## 代码修改注意事项

- 工作树可能已有用户未提交修改。不要随意 `git checkout --`、`git reset --hard` 或回滚不属于本次任务的变化。
- 修改前先看 `git status --short`，只编辑与任务相关的文件。
- 尽量沿用现有风格和参数命名，避免大范围重构。
- 对核心数学逻辑（`ensure_loss.py`、`cg_solver.py`、`mri_ops.py`、`tta.py`）的改动要格外保守；改前先定位 shape、dtype、device、complex layout。
- 本项目大量 tensor 使用 complex 表示或最后一维 size 2 表示复数，改 shape 逻辑前先读相邻 helper。
- 不要把大型 checkpoint、recon、`.pt`、`.h5` 等实验产物加入 git。

## 验证建议

小改动优先做快速验证：

```sh
conda run -n uncertainty_tta python -m compileall datasets losses models ops scripts tta.py
```

TTA 逻辑改动建议先 smoke test：

```sh
conda run -n uncertainty_tta python scripts/tta_shift_true_ensure.py \
  --checkpoint outputs/shifts/main/modality_shift/true_ensure_source_r4_w1_auto_unroll_seed7/best.pt \
  --manifest-csv manifests/shifts/main/modality_shift.csv \
  --output-dir outputs/tta/shifts/main/modality_shift/_smoke_agent \
  --split-role target_test \
  --device cuda:0 \
  --max-samples 2 \
  --tta-loss ensure \
  --run-tta \
  --include-ssim \
  --save-recons \
  --num-workers 0
```

正式实验请用空闲 GPU，并用独立 output directory。完成后检查：

- `run.log` 中的 device、checkpoint、manifest、noise setting。
- `metrics.csv` 是否逐 slice/样本写出。
- `summary.csv` 是否包含 before/after 和 runtime 汇总。
- recon 文件是否实际保存。
