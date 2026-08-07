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
| `20260807_exp010_gli_single_state_noise` | 方法实验 | 用单一完整 T1c 扩散状态消除跨通道捷径并短程预测噪声 | 已完成 5k 与固定 8 例 QA；四项门槛通过，paired 重建最佳 |
| `20260807_exp011_gli_lesion_only_noise` | 方法实验 | 用 lesion-only 单通道状态短程预测噪声 | 已完成 5k 与固定 8 例 QA；union 门槛 5/8，未入选 |
| `20260807_exp012_gli_lesion_only_x0_hist` | 方法实验 | 用 lesion-only 状态直接预测 x0 并增加 soft-histogram 约束 | 已完成 5k 与固定 8 例 QA；四项门槛全通过，当前最佳可控候选 |

实验 ID 只描述研究目的。两种 patch 大小分别记录在同一实验目录的配置文件中，使用同一份裁剪代码和患者划分。
