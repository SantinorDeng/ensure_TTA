# fastMRI Brain + Knee Modality Matrix 运行指导书

本文档对应 Brain 五对比度和 Knee PD/PDFS 实验。所有 TTA 都使用 measured-kspace normalized complex L1；`ensure` 和 `traditional` 只表示训练初始化不同。

## 1. 环境与 GPU 检查

从项目根目录运行：

```sh
cd /home/hulabdl/Deng_proj/cardiac_ensure
conda run -n uncertainty_tta python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```

正式任务必须显式指定 `cuda:N`。任务脚本不会自动退回 CPU。

## 2. 构建 manifest、density 和预处理 sidecar

首次运行：

```sh
conda run -n uncertainty_tta python scripts/run_modality_matrix_job.py prepare
```

七个 source 的完整 RSS sidecar 预计占约 100GB；开始前请先确认磁盘空间。若只想先生成 manifest 和 density，可使用 `prepare --skip-preproc`，但正式训练的 `--require-preproc` 会在启动时拒绝缺少 sidecar 的数据。

它会生成：

- `manifests/shifts/modality_matrix/main/brain/{axflair,axt1,axt1pre,axt1post,axt2}.csv`
- `manifests/shifts/modality_matrix/main/knee/{pd,pdfs}.csv`
- `density_stats/shifts/modality_matrix/` 下的 640×320 和 640×368 density
- `preproc/shifts/modality_matrix/main/...` 下的 RSS map、normalization 和 noise sidecar

只预览命令、不执行：

```sh
conda run -n uncertainty_tta python scripts/run_modality_matrix_job.py prepare --dry-run
```

若 manifests/density 已有，只想稍后生成 sidecar，可直接重跑；默认会复用现有文件。不要无故使用 `--overwrite-preproc`。

## 3. 单卡执行一个训练任务

例如在 GPU 3 上训练 Brain AXT2 的 TRUE-ENSURE 初始化：

```sh
conda run -n uncertainty_tta python scripts/run_modality_matrix_job.py train \
  --dataset brain \
  --source axt2 \
  --method ensure \
  --denoiser-sharing independent \
  --device cuda:3 \
  --resume
```

传统初始化只需改为：

```sh
conda run -n uncertainty_tta python scripts/run_modality_matrix_job.py train \
  --dataset brain \
  --source axt2 \
  --method traditional \
  --denoiser-sharing independent \
  --device cuda:3 \
  --resume
```

`--denoiser-sharing independent` 表示 12 个 unroll 各自使用独立 denoiser。正式启动后请检查 `config.json` 中
`denoiser_sharing` 为 `independent`；不要把 `shared` checkpoint 混入同一组统计。

合法 source：

- Brain：`axflair axt1 axt1pre axt1post axt2`
- Knee：`pd pdfs`

`--resume` 仅在 `best.pt` 和 `summary.json` 都存在时跳过训练。

## 4. 在自己选择的多张 GPU 上排队训练

以下命令只使用 GPU 0、2、5、7；每张卡同时最多一个任务，空闲显存低于 18 GB 时等待：

```sh
conda run -n uncertainty_tta python scripts/launch_modality_matrix_jobs.py \
  --stage train \
  --devices 0,2,5,7 \
  --min-free-memory-gb 18 \
  --datasets brain knee \
  --methods ensure traditional \
  --denoiser-sharing independent \
  --resume
```

只运行部分 source：

```sh
conda run -n uncertainty_tta python scripts/launch_modality_matrix_jobs.py \
  --stage train \
  --devices 1,6 \
  --datasets brain knee \
  --source brain:axt2 \
  --source knee:pd \
  --methods ensure traditional \
  --denoiser-sharing independent \
  --resume
```

启动前检查完整任务命令：

```sh
conda run -n uncertainty_tta python scripts/launch_modality_matrix_jobs.py \
  --stage train \
  --devices 1,6 \
  --datasets brain knee \
  --denoiser-sharing independent \
  --dry-run
```

队列状态写入 `outputs/job_state/modality_train_*.json`。退出启动终端会影响前台队列，因此建议放入 tmux：

```sh
tmux new -s modality_train
```

进入 tmux 后执行队列命令；`Ctrl-b d` 可离开，之后用 `tmux attach -t modality_train` 返回。

## 5. 单卡执行统一 L1-TTA

例如在 GPU 4 上评估 AXT2 source 的 ENSURE-trained checkpoint：

```sh
conda run -n uncertainty_tta python scripts/run_modality_matrix_job.py tta \
  --dataset brain \
  --source axt2 \
  --method ensure \
  --device cuda:4 \
  --test-noise-snr-db 20 \
  --test-noise-seed 9007 \
  --eval-seed 7 \
  --resume
```

该任务会一次评估 AXT2 到五种 Brain target 的 500 slices，并保存全部 recon 和 curves。传统初始化使用相同命令，仅将 `--method` 改为 `traditional`。

## 6. 多 GPU 主 TTA

```sh
conda run -n uncertainty_tta python scripts/launch_modality_matrix_jobs.py \
  --stage tta \
  --devices 0,2,5,7 \
  --min-free-memory-gb 18 \
  --datasets brain knee \
  --methods ensure traditional \
  --test-noise-snr-db 20 \
  --test-noise-seed 9007 \
  --eval-seed 7 \
  --resume
```

## 7. 鲁棒性补跑

主实验汇总后，`robustness_directions.json` 会给出 Brain 最难/最易方向以及两个 Knee OOD 方向。使用 `--target-shift-name` 只跑一个 cell，例如：

```sh
conda run -n uncertainty_tta python scripts/run_modality_matrix_job.py tta \
  --dataset brain \
  --source axt2 \
  --method ensure \
  --device cuda:6 \
  --target-shift-name brain_modality_matrix_axt2_to_axt1pre \
  --test-noise-snr-db 10 \
  --test-noise-seed 9008 \
  --eval-seed 8 \
  --resume
```

无额外测试噪声时使用：

```sh
--test-noise-snr-db clean
```

计划中的三个 seed 对为：

- `--eval-seed 7 --test-noise-seed 9007`
- `--eval-seed 8 --test-noise-seed 9008`
- `--eval-seed 9 --test-noise-seed 9009`

## 8. 汇总主实验

```sh
conda run -n uncertainty_tta python scripts/summarize_modality_matrix.py \
  --test-noise-snr-db 20 \
  --test-noise-seed 9007 \
  --eval-seed 7 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 20260702 \
  --strict-completeness
```

主要输出位于 `outputs/tta/shifts/modality_matrix/analysis/`：

- `volume_metrics.csv`
- `cell_summary.csv`
- `paired_cell_comparison.csv`
- `macro_bootstrap.csv`
- `completeness.json`
- `robustness_directions.json`
- Brain/Knee before、after、delta NMSE heatmaps

## 9. 日志、失败恢复和安全检查

训练和 TTA 日志位于各自 output directory 的 `run.log`。实时查看示例：

```sh
tail -f outputs/shifts/modality_matrix/main/brain/axt2/ensure_r4_w1_unroll12_seed7_train_noise_snr15_25_val20_seed7007/run.log
```

失败后先查看日志；确认不是数据、CUDA 或显存错误后，原命令加 `--resume` 重跑。TTA 只有在 `summary.json` 显示 `num_failed=0` 且存在 metrics 时才会被视为完成。

正式结果开始前务必确认：

- 日志中的 device 是预期的 `cuda:N`。
- checkpoint、manifest、SNR 和 seed 与任务名一致。
- `metrics.csv` 没有 failed rows。
- `recons/` 和 `curves/` 实际存在。
