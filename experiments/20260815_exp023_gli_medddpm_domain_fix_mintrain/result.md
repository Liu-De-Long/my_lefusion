# exp023：Med-DDPM 数据域最小修复

## 目标

在两小时硬预算内修复旧 Med-DDPM 训练数据未归一化的问题，保持 mask-only 条件和原网络不变，从头进行最多 5000 step 训练，并在 exp022 冻结的 10 例测试集上复评。

## 已确认根因与最小修复

- 旧训练 T1c 的审计范围约为 `0–13698`，均值约 `990`，与 Fig. 2 适配层及当前 v2 数据的 `[-1,1]` 域不一致。
- 新 loader 直接读取当前 p64 NPZ，将 XYZ 转为 `1×32×64×64`，严格检查 T1c 位于 `[-1,1]`；训练条件仍为 GT 单通道 union，未加入 masked T1c。
- 保持 Med-DDPM 的 64 channels、2 residual blocks、250 diffusion steps、L1 noise loss、Adam `1e-5`、EMA `0.995`、effective batch 16、FP32 和双 A100；从头初始化模型、EMA 与 optimizer。
- 数据审计通过：train/val/test 为 `7772/1032/1038` patches、`584/73/74` patients，无患者级 split 泄漏；manifest SHA-256 为 `42f71687b3edfe1f54819a01490224e3bd9b6a56be25125aed0d311b99e683ad`，split SHA-256 为 `7fd0ea31dbbc7d68142852df49abb63b3cc5bdedeceeb1baef53adfdc894e8b1`。

## 训练与 validation

| step | EMA validation loss | checkpoint SHA-256 | 门禁 |
|---:|---:|---|---|
| 0 | 0.7978831 | — | 随机初始化基线 |
| 500 | 0.7516929 | `16369f88e5fd30ae1b904e5196f73c57233a2cf549d62c16df2c466bd7aa3263` | loss/梯度/权重有限；较初始化下降；250-step 样本非空、非平坦；reload 通过 |
| 3000 | 0.2489628 | `72cf6f9f67eb9a1c9a1dddfaeb8facced51a6a6c2a1b68154ad3724bfd5d18ab` | 通过；继续至硬上限以争取去除颗粒噪声 |
| 5000 | **0.1365685** | `e69be574a526c5cc46932386cbe16fd418c232cd746e7dd2e38847cf1de59bc1` | validation 最优并冻结；独立单卡 reload 与 250-step 采样通过 |

实际停止于 step `5000`。5000-step 独立采样范围为 `[-0.9849,1.0]`、标准差 `0.1609`，数值有限且非平坦。从 3000 续训至 5000 的运行时间为 `993.8 s`。

## 固定 test10 结果

复用 exp022 冻结的 10 位患者、病例顺序、shared filtered-retained union、seed `20260806`、NFE `250` 和轴向切片。共生成 10 个有限 NPZ；每例 shared union 外最大误差均为 `0`。模型选择只依据完整 validation loss，未根据 test 反向选模。

| 模型 | PSNR ↑ | 三视图局部 SSIM ↑ | MAE ↓ | Hist-W1 ↓ |
|---|---:|---:|---:|---:|
| 旧 Med-DDPM（exp022） | 14.7468 | 0.3851 | 0.33111 | 0.21927 |
| domain-fixed step 3000 | 11.0434 | 0.2829 | 0.47824 | 0.37966 |
| domain-fixed step 5000 | 13.4748 | 0.3363 | 0.39412 | 0.33929 |

5000-step 相比 3000-step 四项均改善，但相对旧 Med-DDPM 四项均未改善，未达到“至少三项严格改善”的论文替换门槛。视觉 QA 同时发现 Med-DDPM 行仍普遍存在高频颗粒/盐粒纹理，尤其 S01、S03–S10，仍属于明显短预算欠训练。

## 结论

数据域错误已被最小修复，训练与 checkpoint 合约均成功；但 5000 step 的短预算不足以把 mask-only Med-DDPM 训练到可用于 Fig. 2 的质量。根据预先冻结的失败处理规则，本次候选图**不写入论文**，`paper3` 的现有 Fig. 2、PDF 和 FSD/KID/Rad-MMD/定量表均保持不变。

候选图的 figure protocol 已标注 `med_ddpm_protocol=domain-fixed short-budget` 与 `paper3_modified=false`。候选 PNG SHA-256 为 `de1935b321b32aa769d5a11f57c4b1e7f8c41ee1a6db7d45c8e888c4b0c3afd1`。

## 下一步

若后续仍需将 Med-DDPM 纳入论文图，应单独授权更长训练预算，并继续按 validation-only 选模；不应复用本次 test10 指标调参。当前两小时任务到 5000 step 即停止，不自动扩展到 10k/20k。

## 输出路径

- 远端 checkpoint、validation、test NPZ：`experiments/20260815_exp023_gli_medddpm_domain_fix_mintrain/outputs/med_ddpm/`
- 远端指标、审计、标注版/清洁版 PNG/PDF：`experiments/20260815_exp023_gli_medddpm_domain_fix_mintrain/outputs/comparison/`
- 本地视觉 QA 候选图：`experiments/20260815_exp023_gli_medddpm_domain_fix_mintrain/outputs/comparison/fig2_v2_test10_labeled.png`

大权重、日志、NPZ 与生成图件不进入 Git。
