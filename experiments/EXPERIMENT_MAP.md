# 实验地图

| 实验 ID | 类型 | 目的 | 状态 |
|---|---|---|---|
| `20260804_exp000_brats_gli_data_preprocess_audit` | 数据审计 | 整理 BraTS2024 GLI 数据审计与预处理统计产物 | 已完成 |
| `20260804_exp001_t1c_local_patch_dataset` | 方法实验 | 构建 BraTS 2024 GLI 的 T1c 四类局部病灶 patch 数据集 | 已完成 |
| `20260805_exp002_gli_loader_t1c_multilesion` | 方法实验 | 实现并注册 GLI T1c scalar label 与四通道 lesion mask loader | 已完成 |
| `20260805_exp003_gli_lesion_aware_training` | 方法实验 | 接入 GLI 四通道 lesion-aware loss、训练传递与矩形空间 shape | 已完成 smoke 验收 |
| `20260805_exp004_gli_inference_closed_loop` | 方法实验 | 完成 GLI checkpoint、train-only cluster、RePaint、四通道合成与保存闭环 | 已完成 smoke 验收 |

实验 ID 只描述研究目的。两种 patch 大小分别记录在同一实验目录的配置文件中，使用同一份裁剪代码和患者划分。
