我按你补充的口径，只看 `tta_loss="ensure"` 的 TRUE-ENSURE TTA。结论先说：这不是 ENSURE 模型本身全面失败，尤其 modality shift 里 before TTA 反而是 ENSURE 更强；真正的问题集中在 TRUE-ENSURE TTA 的优化目标、收敛速度、DC/validation/final mask 切换，以及 runtime 实现上。

**核心差异**
传统 TTA 在 [tta_shift_supervised_baseline.py](/home/hulabdl/Deng_proj/cardiac_ensure/scripts/tta_shift_supervised_baseline.py:461) 里做的是：

`train_mask/val_mask` 拆分后，用 `model(train_zf, train_mask)`，再最小化 `A_train(pred) vs y_train` 的 normalized complex L1。它非常直接，优化目标和 measured k-space 一致。

ENSURE TTA 在 [tta.py](/home/hulabdl/Deng_proj/cardiac_ensure/tta.py:327) 走的是 TRUE-ENSURE loss：先解 `rho_ls`，再算 projected error，再算 measurement divergence。对应实现在 [ensure_loss.py](/home/hulabdl/Deng_proj/cardiac_ensure/losses/ensure_loss.py:156)。这本质上更贵，也更不等价于“把 measured k-space 拟合好”。

所以 modality shift 的现象可以解释成：ENSURE prior 很好，但 TTA 更新没有传统 L1 那么会利用 target measured k-space。表里 modality before：ENSURE PSNR/SSIM 是 `30.76 / 0.789`，传统是 `29.77 / 0.706`；after 才被传统拉到 `39.56 / 0.972`，ENSURE 停在 `38.05 / 0.938`。

**我看到的主要问题**
1. **modality 的 TRUE-ENSURE TTA 明显没收敛。**  
   100 个 modality 样本里 82 个打满 250 steps，best step 平均 236，最后 20 steps 的 self-val 还在下降。这不是过拟合，是步子太慢。当前 `lr=1e-5`、`grad_clip=1.0` 是拿传统 L1 的设置硬套 ENSURE loss。

2. **ENSURE loss 尺度很大且未归一化。**  
   曲线里 modality 第一步 `data_term` 平均约 5821，最后也还有约 34；`div_contribution` 最后约 19。用同样 `lr=1e-5` 很可能保守了。建议先试 `lr=3e-5/5e-5/1e-4`，或把整个 ENSURE loss 乘一个常数缩放，不改变最优点但改善优化速度。

3. **初始模型没有进入 best-state 竞争。**  
   [tta.py](/home/hulabdl/Deng_proj/cardiac_ensure/tta.py:317) 里 `best_state` 是初始模型，但 `best_val_loss=inf`，所以第 1 步一定覆盖初始状态。acceleration 里 3 个 NMSE 灾难点大概率和这个有关。应该先计算 step 0 的 self-val，把初始模型作为合法 best。这样可以减少负迁移。

4. **runtime 有明显可省的 CG。**  
   `rho_ls` 只依赖 `train_kspace/maps/train_mask`，但现在每一步都在 [ensure_loss.py](/home/hulabdl/Deng_proj/cardiac_ensure/losses/ensure_loss.py:157) 重新解。TTA 中应预计算一次。  
   另外 projected divergence 当前对 base 和 perturbed 各做一次 projection；projection 是线性的，可以改成只对差值做一次 projection。这个不改 ENSURE 本质，但每 step 可少掉很多 CG。

5. **DC 权重几乎不动。**  
   DC block 在 [temporal_normunet.py](/home/hulabdl/Deng_proj/cardiac_ensure/models/temporal_normunet.py:154)，sigmoid 后初始化接近 0.99。曲线里 TTA 前后 DC weight 只动到 `1e-5` 量级。如果你怀疑 DC 项不适配 modality，当前 TTA 基本修不了 DC。可以试 final DC polish 或单独重参数化 DC scalar。

6. **TTA train mask 与 final full mask 有切换。**  
   训练/validation 用的是 `train_zf + train_mask`，最终 after 用 `zf + full mask`，见 [tta.py](/home/hulabdl/Deng_proj/cardiac_ensure/tta.py:302) 和 [tta.py](/home/hulabdl/Deng_proj/cardiac_ensure/tta.py:427)。传统 L1 对这个切换更鲁棒；ENSURE 的 `rho_ls/projection` 对 mask 更敏感。可以加一个 final full-mask DC polishing，用 held-out self-val 选择强度。

7. **manifest 的 eval mask 当前没用上。**  
   shift manifest 有 `eval_mask_path/eval_mask_actual_acceleration`，但 dataset 在 [shift_manifest_dataset.py](/home/hulabdl/Deng_proj/cardiac_ensure/datasets/shift_manifest_dataset.py:500) 重新采样 Bernoulli-Gaussian mask。acceleration shift 的 target 标的是 R=2，但 checkpoint config 仍用 R=4。若实验定义要求固定 eval mask，这里需要修；虽然两种方法都受影响，但会影响结论的可信度。

**我建议的优先方向**
第一优先级：修 runtime 和负迁移，不改方法本质。  
预计算 `rho_ls`；projected divergence 用差值投影；step 0 加入 best-state；关闭 per-step GT/SSIM 诊断，只保留最终 SSIM。这个组合很可能同时降 runtime、减少 outlier。

第二优先级：为 TRUE-ENSURE 单独调 TTA 优化。  
modality 先跑小网格：`lr={3e-5,5e-5,1e-4}`，`grad_clip={1,5,None}`，`steps={150,250}`。目标是让 250 step 内 self-val 不再明显下降。

第三优先级：提高 modality after SSIM。  
加一个很小的 measured-kspace L1 辅助项或两阶段策略：前半 TRUE-ENSURE，后半少量 k-space L1 polish。ENSURE 仍是主损失，但让结果更贴近传统 TTA 擅长的 measured consistency。

第四优先级：final DC polish。  
在 best ENSURE 模型输出后，对 full measured k-space 做 1-3 次轻量 DC correction，或在 `model_output` 和 `rho_ls/full-DC-corrected output` 间用 self-val 选 alpha。这对 SSIM/PSNR 可能比继续盲目加 ENSURE steps 更划算。

最值得先做的小实验：`step0 best fallback + lr=5e-5 + rho_ls cache`。这三个不动 ENSURE 本质，但最可能同时改善 modality SSIM 和 runtime/slice。