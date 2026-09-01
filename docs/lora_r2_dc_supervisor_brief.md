# TRUE-ENSURE + Conv-LoRA + DC：导师汇报说明

![TRUE-ENSURE + Conv-LoRA + DC 方法图](../outputs/figures/lora_tta_method/ensure_lora_dc_method.svg)

## 一句话概括

源域仍然使用原来的 TRUE-ENSURE 无监督训练；目标域仍然使用原来的 measured-kspace 自监督 L1 做 TTA，但测试时不再更新整个重建网络，而是冻结源域 CNN，只更新插入中间卷积层的低秩 LoRA 分支和 12 个 data-consistency 标量。

因此，改进的核心不是“换 loss”，而是“增加可适配旁路并删除全参数 TTA 的大部分自由度”。

## 方法如何与现有 ENSURE + TTA 结合

### 1. 源域阶段完全不变

- 输入欠采样 k-space、sampling mask 和 sensitivity maps。
- 经过 12-stage unrolled reconstruction；每一级由 CNN prior 和 soft data consistency 组成。
- 使用 TRUE-ENSURE loss 训练原模型参数，仍然不需要 fully-sampled ground truth。
- 得到源域 checkpoint \(W_0\)。

LoRA 没有参与这一轮已有 checkpoint 的源域训练；当前实现是在已经训练好的 checkpoint 上注入 zero-output adapter。

### 2. 测试时给 CNN 增加低秩旁路

原来一个中间卷积层为

\[
h' = \operatorname{Conv}(W_0,h).
\]

加入 Conv-LoRA 后为

\[
h' = \operatorname{Conv}(W_0,h)
+ \frac{\alpha}{r}\operatorname{Conv}_{1\times1}
\left(B,\operatorname{Conv}_{3\times3}(A,h)\right).
\]

其中：

- \(W_0\) 是 ENSURE 学到的源域卷积核，测试时冻结；
- \(A\) 是 `3×3, 64→r` 的降维卷积；
- \(B\) 是 `1×1, r→64` 的升维卷积；
- 只有 \(A,B\) 参与 TTA；
- \(B\) 初始化为 0，所以插入 LoRA 后的 step 0 输出与源 checkpoint 完全相同。

LoRA 插在每个 denoiser 的三个中间 `64→64` 卷积层，不改输入层、输出层和复数通道接口。

### 3. 同时适配 data consistency

每个 unroll 的 soft DC 为

\[
x_{k+1}=x_{k+\frac12}
-\lambda_k\left(A^HAx_{k+\frac12}-x_0\right).
\]

当前选定的 `R2 + DC` 方案除 LoRA 外，还允许 12 个 \(\lambda_k\) 在测试时更新。含义是：

- LoRA 负责调整目标域图像先验；
- DC 标量负责调整每个重建阶段中“网络先验”和“实测数据”的平衡。

当前 DC 改进仍只是 12 个标量，并不是 spatial/frequency-dependent DC controller；后者可以作为下一阶段的结构创新。

### 4. 测试 loss 与安全选择不变

- 将目标样本已采集的 k-space lines 划分为约 95% TTA train 和 5% self-validation。
- 使用 measured-kspace normalized complex L1 更新 LoRA 和 DC。
- 使用 held-out 5% k-space 选择最佳 step，并允许选择 step 0。
- 最后恢复完整 sampling mask 进行重建。

整个过程没有使用目标图像 ground truth。

## LoRA rank `r=1,2,4` 到底是什么

这里的 `r` 是低秩更新的维度，不是 MRI 加速倍数。

一个 `64→64` 的 `3×3` 卷积展平后为

\[
W_0\in\mathbb{R}^{64\times576}.
\]

普通全参数微调允许任意 \(\Delta W\)；LoRA 将其限制为

\[
\Delta W=BA,\qquad \operatorname{rank}(\Delta W)\le r.
\]

- `r=1`：只能沿一个低秩方向修正，容量最小、约束最强。
- `r=2`：允许两个低秩方向，是容量和稳定性的折中。
- `r=4`：表达能力更强，但参数更多，也更容易用单样本 TTA 拟合偶然噪声或 mask 特征。

每个中间卷积层的 LoRA 参数量为

\[
r(64\times3\times3)+64r=640r.
\]

每个 denoiser 插入三个中间层，因此为 \(1920r\)。

| LoRA rank | 共享一个 CNN：LoRA-only | 加 12 DC | 12 个独立 CNN：LoRA-only | 加 12 DC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,920 | 1,932 | 23,040 | 23,052 |
| 2 | 3,840 | **3,852** | 46,080 | **46,092** |
| 4 | 7,680 | 7,692 | 92,160 | 92,172 |

第一阶段 rank 消融中的 `R1/R2/R4` 是 LoRA-only；之后选择的方案是 `R2 + DC`。

> 汇报时建议统一写成小写 \(r_{\mathrm{LoRA}}\)，把 MRI 加速写成 \(R_{\mathrm{acc}}\)。本轮所有实验固定 \(R_{\mathrm{acc}}=4\)，变化的是 \(r_{\mathrm{LoRA}}=1,2,4\)。

## 为什么最终选择 `r=2 + DC`

主 modality shift、共享 CNN checkpoint 的消融结果：

| 方法 | 可训练参数 | PSNR | SSIM | mean NMSE | 负适配率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-parameter TTA | 112,908 | 36.8407 | 0.9339 | 0.004493 | 4% |
| LoRA r=1 | 1,920 | 36.4320 | 0.9299 | 0.004702 | 1% |
| LoRA r=2 | 3,840 | 36.5805 | 0.9307 | 0.007343 | 1% |
| LoRA r=4 | 7,680 | 36.6425 | 0.9303 | 0.007466 | 2% |
| **LoRA r=2 + DC** | **3,852** | **36.7987** | **0.9360** | **0.003588** | **1%** |

选择依据不是“r 越大越好”：

1. `r=4` 相比 `r=2` 只带来很小 PSNR 增益，没有稳定改善 NMSE/SSIM。
2. 给 `r=2` 增加 12 个 DC 标量后，PSNR 接近 full TTA，SSIM 和 mean NMSE 更好。
3. 负适配率由 full TTA 的 4% 降到 1%。
4. 说明跨域适配不仅需要改变 CNN prior，也需要改变 unrolled network 中 prior/DC 的平衡。

需要谨慎表述：`r=2 + DC` 的逐样本 NMSE 胜率并没有超过 50%，mean NMSE 改善部分来自减少少数严重异常样本。更稳妥的结论是“接近 full-TTA 平均质量，同时改善尾部稳定性和负适配率”。

## 其他跨模态实验

AXT1 源域 checkpoint 使用 12 个独立 denoiser，因此 `r=2 + DC` 更新 46,092 个参数，约占全模型的 3.4%。所有测试均为 clean、\(R_{\mathrm{acc}}=4\)。

| Target shift | NMSE | PSNR | SSIM | 适配时间/片 | 负适配率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 → FLAIR | 0.003062 | 37.6830 | 0.9543 | 49.59 s | 0% |
| T1 → T2 | 0.005457 | 36.6681 | 0.9547 | 45.84 s | 4% |
| T1 → POST | 0.004002 | 37.6683 | 0.9506 | 49.37 s | 2% |

## 建议的 60 秒口头讲法

> 我原来的方法是在源域用 TRUE-ENSURE 无监督训练一个 12-stage unrolled reconstruction network，到目标域后用 measured-kspace L1 对整个网络做 TTA。问题是单个测试样本更新整个网络自由度太高，容易过拟合，而且每个样本都要维护全参数梯度。
>
> 这次我没有改 ENSURE loss，也没有更换 TTA loss，而是在每个 denoiser 的三个中间卷积层加入 Conv-LoRA。源域卷积核全部冻结，目标域只通过 rank-r 的低秩旁路修正图像先验。同时我保留了 12 个可训练 DC 标量，让每个 unroll 可以重新调整数据一致性和网络先验的权重。
>
> rank 不是 MRI 加速倍数，它表示权重更新子空间的维度。r=1 约束最强，r=4 容量更大。消融发现并不是 rank 越大越好；r=2 加 DC 在只更新约 3.4% 参数的情况下，PSNR 基本接近 full TTA，SSIM更高，负适配率从 4% 降到 1%，所以最终选择 r=2+DC。

## 导师可能追问的问题

### “这是不是只把 NLP 的 LoRA 搬过来？”

LoRA 本身不是新贡献，不能这样包装。当前可强调的是：

- 将低秩适配嵌入 physics-unrolled MRI reconstruction 的 image-prior 模块；
- 与可适配 DC 结合，形成 prior-side 和 physics-side 的双重适配；
- 源域 TRUE-ENSURE、目标域无 GT TTA、held-out k-space 安全选择组成完整无监督跨域流程。

如果要加强论文结构创新，下一步应把静态 DC 标量升级为 residual-conditioned DC controller，或者让不同 cascade 使用共享低秩基但具有 cascade-specific gates。

### “参数少了，为什么共享模型上没有大幅提速？”

冻结参数减少了权重梯度和优化器状态，但仍需要通过 12 个 unroll 做前向和激活反传。共享模型本身只有约 11 万参数，主要耗时不是参数更新，因此速度提升有限。对于含 12 套独立 CNN 的模型，冻结约 135 万主干参数后，实测适配时间下降更明显。

### “为什么不直接只更新 DC？”

DC 只能改变对实测数据的信任强度，无法表达 T1、T2、FLAIR、POST 之间的图像先验差异；LoRA 负责 prior shift，DC 负责 physics/prior balance，两者作用互补。正式论文应补充 DC-only 消融来进一步证明这一点。
