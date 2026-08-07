# exp010 单一完整 T1c 扩散状态

## 目标

删除 exp008 的四份重复 T1c 状态，验证单一完整 T1c 扩散状态能否在 5000 step 内学习挖空区域的病灶生成，并响应四组 histogram 条件。

## 变更

- 扩散状态和输出均改为单通道 T1c。
- 空间条件固定为挖空 T1c 加四通道 lesion mask。
- 预测目标为噪声，loss 按非空病灶类别区域逐体素归一化。
- histogram condition dropout 为 0.1，推理使用 CFG。

## 配置

- patch：`64×64×32`
- seed：`20260805`
- 最大 optimizer step：`5000`
- W&B run：`exp010-p64-s20260805`
- QA：固定 8 个 validation patch，五种条件变体；不运行 test split。

## 结果

- 方法实现 commit：`d188ce170040037ec477d5ff398e1f2cf141608a`。
- 待完成远端回归、训练与 8 例 QA 后补充数值结果。

## 结论

待定。

## 下一步

完成 tensor/GPU/W&B preflight 后，在不占用现有训练 GPU 的前提下运行到 step 5000。

## 输出路径

`experiments/20260807_exp010_gli_single_state_noise/outputs/`
