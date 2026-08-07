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
| `20260806_exp008_gli_conditional_inpainting_training` | 方法实验 | 以挖空 T1c、四通道 mask 与 hist 作为条件，重新训练 p64 病灶修复扩散模型 | 实施与验证中 |
| `20260806_exp009_gli_p64_weak_supervision_classifier` | 方法实验 | 用 1000 个有标签 p64 patch 训练 T1c 与总 mask 的四类逐体素分类器 | 已完成 val；C0 focus mIoU 0.5436，未过 0.85 门禁，test 封存 |
| `20260807_exp013_gli_p64_multiscale_semisupervised_classifier` | 方法实验 | 在同一 1000 patch 上训练多尺度残差 3D U-Net，并用剩余 train patch 做 mean-teacher 半监督 | 已完成 val；监督 0.5730 优于半监督 0.5727，未过 0.85，test 封存 |
| `20260807_exp014_gli_p64_geometry_boundary_classifier` | 方法实验 | 在同一 1000 patch 上验证无泄漏几何通道、Lovász 与类间边界监督 | 15/15 tests 与 CPU/GPU0 零步 preflight 通过；待正式训练，test 封存 |

实验 ID 只描述研究目的。两种 patch 大小分别记录在同一实验目录的配置文件中，使用同一份裁剪代码和患者划分。
