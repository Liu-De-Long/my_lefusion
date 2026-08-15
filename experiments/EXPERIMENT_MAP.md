# 实验地图

| 实验 ID | 类型 | 目的 | 状态 |
|---|---|---|---|
| `20260804_exp000_brats_gli_data_preprocess_audit` | 数据审计 | 整理 BraTS2024 GLI 数据审计与预处理统计产物 | 已完成 |
| `20260804_exp001_t1c_local_patch_dataset` | 方法实验 | 构建 BraTS 2024 GLI 的 T1c 四类局部病灶 patch 数据集 | 已完成 |
| `20260805_exp002_gli_loader_t1c_multilesion` | 方法实验 | 实现并注册 GLI T1c scalar label 与四通道 lesion mask loader | 已完成 |
| `20260805_exp003_gli_lesion_aware_training` | 方法实验 | 接入 GLI 四通道 lesion-aware loss、训练传递与矩形空间 shape | 已完成 smoke 验收 |
| `20260805_exp004_gli_inference_closed_loop` | 方法实验 | 完成 GLI checkpoint、train-only cluster、RePaint、四通道合成与保存闭环 | 已完成 smoke 验收 |
| `20260805_exp005_gli_formal_training_baseline` | 方法实验 | 实现分层 sampler、正式 validation、checkpoint/resume 与 W&B online 门禁 | 训练契约不适用于伪病灶生成；p64/p80 仅保留失败审计 |
| `20260806_exp006_gli_official_repaint_alignment` | 方法实验 | 删除 GLI 额外 post-denoiser hard clamp，按原始 LeFusion 语义重跑冻结 val/test 子集 | 已完成，但因训练契约错误而失效 |
| `20260806_exp007_gli_masked_input_qa` | 方法实验 | 仅用 8 例冻结 val QA 比较显式挖空输入的多标签与 anchor-union 单标签生成 | 已停止；QA 暴露旧 checkpoint 未学习条件式修复 |
| `20260806_exp008_gli_conditional_inpainting_training` | 方法实验 | 以挖空 T1c、四通道 mask 与 hist 作为条件，重新训练 p64 病灶修复扩散模型 | 已完成 50,000 step 与两组 8 例 QA；生成退化到挖空值，仅保留失败审计 |
| `20260806_exp009_gli_p64_weak_supervision_classifier` | 方法实验 | 用 1000 个有标签 p64 patch 训练 T1c 与总 mask 的四类逐体素分类器 | 独立分支运行中 |
| `20260807_exp010_gli_single_state_noise` | 方法实验 | 用单一完整 T1c 扩散状态消除跨通道捷径并预测噪声 | 长程训练在 step 28k 因 NaN 退出；step 14k best/EMA 已完成 519 例 test 50%，paired 修复有效但 hist/四类语义未证实 |
| `20260807_exp011_gli_lesion_only_noise` | 方法实验 | 用 lesion-only 单通道状态短程预测噪声 | 已完成 5k 与固定 8 例 QA；union histogram CF 5/8，未入选 |
| `20260807_exp012_gli_lesion_only_x0_hist` | 方法实验 | 用 lesion-only 状态直接预测 x0 并增加 soft-histogram 约束 | 已完成 5k 与固定 8 例 QA；histogram 响应最强，类别视觉/语义一致性未证实 |
| `20260807_exp013_gli_short_method_comparison` | 比较实验 | 统一比较 exp010–exp012 的固定 8 例五变体 QA、定量门槛与横向图 | 已完成；exp012 为 hist 响应候选，exp010 为 paired 重建最佳方法 |
| `20260808_exp014_gli_four_class_counterfactual_qa` | QA 方法扩展 | 对同一冻结病例、union mask 和采样随机序列分别请求四类，检查 histogram 排名与无亮度泄漏纹理可分性 | 已完成 32 个结果；只证实亮暗/统计控制，四类视觉与医学语义控制未通过 |
| `20260808_exp016_gli_exp010_exp012_four_class_comparison` | 配对 QA 比较 | 为 exp010 补充与 exp012 完全相同的冻结 8 例 × 4 类生成，并逐例比较 histogram、GLCM、margin 与背景保持 | 已完成；exp012 的 hist/GLCM 为 21/32、26/32，优于 exp010 的 18/32、21/32，保留语义未验证边界 |
| `20260811_exp018_gli_exp016_pseudomask_lefusion` | 方法与对照实验 | 用 exp016 direct/置信度过滤四通道 mask 分别执行 exp010 FP32 全量训练 | 已完成两路线 50k、best checkpoint gate、完整 real/overlay val 与固定 QA；test 未访问 |
| `20260811_exp019_gli_exp010_pseudomask_test200_comparison` | 配对 test 比较 | 在冻结 200 例上比较原始 exp010、direct 和 filtered 的真实-mask 300-step 重建及分布指标 | 已完成；filtered 的 PSNR/SSIM/MAE/FID/FSD 最佳，direct 的 Hist-W1 最佳；端到端 3:24:36 |
| `20260813_exp020_gli_exp010_pseudomask_conditioned_test200` | 配对 test 重评 | 在同一 200 例上比较 GT、direct 与 filtered 各自推理条件下的重建表现 | 已完成；共同 retained 区域 filtered 最佳，伪 mask 覆盖与 copied region 已分开审计 |
| `20260815_exp022_gli_fig2_v2_test10_comparison` | Fig. 2 配对质控比较 | 用当前 v2 test 数据、共享 filtered union 和既有五方法 checkpoint 重跑 10 位患者的定性/配对重建比较 | 已完成；五方法各 10 例，自动审计与视觉 QA 通过；标注版已写入 paper3 |
| `20260815_exp023_gli_medddpm_domain_fix_mintrain` | 外部基线最小修复 | 修复 Med-DDPM T1c 数据域并在两小时内从头训练至最多 5k，再复评 Fig. 2 固定 test10 | 已完成至 step 5000；val 明显下降但 test 四项均未优于旧模型且仍有颗粒噪声，论文门禁失败，未替换 Fig. 2 |

实验 ID 只描述研究目的。两种 patch 大小分别记录在同一实验目录的配置文件中，使用同一份裁剪代码和患者划分。

补充：`20260813_exp020_gli_exp010_pseudomask_conditioned_test200` 已完成。该配对 test 重评中 exp010 使用 GT 条件，direct/filtered 使用各自 test 伪 mask/hist；共同 retained 区域 pooled PSNR/SSIM 为 `19.1322/0.4988`、`20.7710/0.5977`、`21.4968/0.6636`，filtered coverage 为 80.9007%。exp019 明确归档为统一 GT 推理条件的历史对照。
