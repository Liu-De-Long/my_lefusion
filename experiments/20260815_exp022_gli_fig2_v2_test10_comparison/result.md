# 20260815 exp022 结果

## 状态与范围

实验已完成。没有训练或微调任何模型；没有修改 `paper3` 的 Fig. 2、正文或 200 例定量表。本实验的 10 例指标仅用于配对重建和图件质量审计，不计算 FID、FSD、KID 或 Rad-MMD。

## 冻结样本

- 数据：当前 LeFusion_v2 `64×64×32` test split，共 1038 个 patch、74 位患者。
- 选择：10 个 patch、10 位不同患者；只使用 GT anchor label、boundary/interior role、GT union 体积和路径哈希，不使用生成质量或 filtered 覆盖率。
- 配额：NETC/SNFH/ET/RC=`2/4/2/2`，boundary/interior=`5/5`。
- GT union 体积百分位：`0.0501, 0.1504, 0.2498, 0.3500, 0.4503, 0.5497, 0.6490, 0.7512, 0.8515, 0.9499`。
- selection manifest SHA-256：`4c6c38dfa8d97c482245423906769828a8d385df8c4271c6e84d9cec279b432d`。

## 条件、模型与审计

exp016 best classifier checkpoint SHA-256 为 `3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617`，filter threshold 固定为 `0.964`。direct/filtered 各 10 个 overlay，filtered fallback 为 0。五方法共享同一个 filtered-retained union；Ours 额外接收 NETC/SNFH/ET/RC 四通道 mask 与重算 histogram。

旧四基线按原协议运行：RePaint EMA step 4000/NFE 300、Med-DDPM model-20/NFE 250、Pix2Pix step 20000/NFE 1、Latent RFlow step 20000 与 VAE step 20000/NFE 50。Filtered (v2) 使用 step 46000 best/EMA、FP32、NFE 300、CFG 2.0。五个 checkpoint 和九个旧推理源码/配置全部通过固定 SHA-256、权重键、shape、有限性与输出域审计；没有静默替换模型。

单例 preflight 与正式运行均通过。每方法恰好 10 个结果；统一适配后每方法另有 10 个 v2 `[-1,1]` NPZ。shared hole 在五方法间逐 voxel 相同，显式 masked input 的 hole 内只有固定填充值，所有输出有限且无需 range clipping；区域外恢复原 T1c 后最大绝对误差为 `0`。

## 十例 paired 指标

均值 ± 样本标准差如下。主指标均在 shared filtered-retained union 内计算；full-patch 指标只作辅助审计。

| 方法 | PSNR ↑ | 三视图局部 SSIM ↑ | MAE ↓ | Hist-W1 ↓ | Full PSNR ↑ | Full SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| RePaint-3D | 18.8591 ± 4.9683 | 0.5446 ± 0.1802 | 0.20665 ± 0.15507 | 0.14467 ± 0.16818 | 26.1783 ± 6.6058 | 0.7896 ± 0.1483 |
| Med-DDPM-T1c | 14.7468 ± 3.3952 | 0.3851 ± 0.1420 | 0.33111 ± 0.20821 | 0.21927 ± 0.25628 | 22.0661 ± 4.8472 | 0.7088 ± 0.1471 |
| Pix2Pix-3D | 13.3612 ± 3.0244 | 0.4167 ± 0.1382 | 0.35945 ± 0.15401 | 0.28086 ± 0.18081 | 20.6805 ± 3.7024 | 0.7333 ± 0.1264 |
| Latent RFlow-3D | 11.9103 ± 1.9252 | 0.4140 ± 0.1378 | 0.41834 ± 0.10565 | 0.32962 ± 0.07878 | 19.2296 ± 3.2882 | 0.7092 ± 0.1520 |
| Filtered (v2) | **23.1414 ± 4.5379** | **0.7408 ± 0.1155** | **0.11079 ± 0.06615** | **0.07269 ± 0.07259** | **30.4607 ± 6.3511** | **0.8921 ± 0.0759** |

在这 10 个冻结病例上，Filtered (v2) 的四项 shared-region 均值均优于四个外部基线。由于样本是为 Fig. 2 定性展示而设计的 10 例分层集合，该结果不能替代 200 例分布指标或论文统计结论。

## 图件与产物

图件保留 8 行：Ground-truth T1c、Shared filtered mask、Masked input、四个外部基线、Filtered (v2)。列按 GT union 体积升序排列，每列取 shared mask 轴向面积最大的切片。标注版/清洁版 PNG 与 PDF、figure protocol、comparison manifest、checkpoint contracts、逐例/汇总指标和输出审计均位于 `outputs/comparison/`。

- 标注 PNG SHA-256：`3aa17a4865aadac16d123a4a48887e012697eba9388b19a75225e01a6869b1fe`
- 清洁 PNG SHA-256：`3f2e2f9cd19b44264b19204e797b9d81d67b5e17cf74e4084bb0bc1eb2f51a9f`
- comparison manifest SHA-256：`b4f54dee3524c932fff612c220e260b311c57837345f18e7cd85ca0063e09b11`
- checkpoint contracts SHA-256：`1f7bbaf7ba6e6f9de2357ded6eaa8e6c7f52c2752cccaeaa76652745e73abf85`

视觉 QA 已检查标注图：10 列病例顺序、8 行方法顺序、shared mask、slice 和背景均对齐，无明显错列或错层。图件仅保存在 LeFusion_v2 实验目录，等待确认后再决定是否更新论文。
