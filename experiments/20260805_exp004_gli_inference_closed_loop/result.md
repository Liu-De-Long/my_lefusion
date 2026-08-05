# 实验结果

## 目标

在不启动正式训练的前提下，完成 GLI patch 级 validation checkpoint、train-only histogram cluster、逐步 RePaint、四通道病灶合成和单通道 NPZ/NIfTI 保存闭环，并审计当前 normalization 所依赖的原始 T1c 非零区域。

## 变更

- 将原 147 个 holdout subject 确定性拆分为 73 个 val 和 74 个 test，保留原 584 个 train subject。
- 新增 GLI inference dataset adapter、显式多模态 brain support、train-only 分标签 histogram clustering 和可视化工具。
- 按 scalar segmentation 确定性展开四通道 lesion mask；RePaint 每个反向转换均使用共享的前向扩散 T1c 背景状态。
- sampler 内部将四通道终态折叠为单通道；不执行采样结束后的原始健康区硬覆盖。
- 新增严格 checkpoint schema、矩形 shape、DHW/XYZ、NIfTI 回读、分区指标和 provenance 验证。

## 配置

- `config_64x64x32.yaml`：训练 smoke 20 step，inference batch size 1，缩减 RePaint schedule `t_T=5`。
- `config_80x96x80.yaml`：训练 smoke 1 step，inference batch size 1，缩减 RePaint schedule `t_T=5`。
- histogram condition：每个 label 独立使用 train patch 聚类，空 label 保持零 block，存在 label 选择与 source histogram 最近的 train center。
- 显式 brain support：四模态至少两个非零，保留最大 3D 连通域并填洞，最后并入 `segmentation > 0`；仅用于 QA 和合法性检查。

## 结果

待远端 smoke 验收后填写。本轮不会启动长期或正式训练。

## 结论

待两种 patch 的 checkpoint、cluster、RePaint、保存回读和 normalization 审计完成后填写。

## 下一步

1. 完成远端全量测试与两份 Hydra 配置解析。
2. 生成 versioned split、cluster 图和 64 subject normalization audit。
3. 分别运行两种 patch 的 validation checkpoint 与 test inference smoke。
4. 仅在完整闭环验收并确认标签医学语义后，另行决定是否启动正式训练。

## 输出路径

- 远端实验根目录：`/workspace/LeFusion_v2/my_experiment/experiments/20260805_exp004_gli_inference_closed_loop`
- checkpoint、W&B 缓存和生成体积位于该目录的 `outputs/` 下，不进入 Git。
- versioned split、配置、结果记录和代码版本进入 Git。
