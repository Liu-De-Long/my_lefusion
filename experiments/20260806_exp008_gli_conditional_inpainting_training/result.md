# exp008 条件式病灶修复训练

## 目标

把 GLI 训练从“完整病灶图加噪、mask 仅参与 loss”改为与伪病灶生成任务一致的条件式修复：模型必须根据挖空后的 T1c 背景、四通道病灶 mask 和 histogram 生成病灶区域。

## 变更

- 扩散目标 `x0` 保留原始病灶 T1c，并按 NETC/SNFH/ET/RC 复制为四通道。
- 在所有真实病灶的 union 内把空间上下文置为 `0.0`，形成单通道 `masked_context`。
- denoiser 每步接收四通道 `x_t`、单通道 `masked_context` 和四通道 lesion mask，共 9 个空间输入通道。
- histogram 仍为四组 16-bin block，共 64 维全局条件。
- denoiser 输出四通道噪声预测；loss 仍只在对应病灶通道内计算，并对非空 `(sample, lesion channel)` 等权平均。
- 训练与 RePaint 推理复用同一空间条件契约。

## 配置

- patch：`64x64x32`
- seed：`20260805`
- 最大步数：`50000`
- effective batch：`4`
- W&B run：`exp008-p64-s20260805`
- checkpoint：`outputs/patch_64x64x32/seed_20260805/checkpoints/`

## 结果

待完成远端测试、preflight 和正式训练启动后补充。

## 结论

待实验运行。

## 下一步

通过真实 patch 回归与 GPU preflight 后，启动且只启动本实验的 p64 seed `20260805` 正式训练。

## 输出路径

`experiments/20260806_exp008_gli_conditional_inpainting_training/outputs/`
