# exp012 lesion-only x0 与 soft histogram

## 目标

验证直接预测病灶 x0，并显式约束四类 soft histogram，能否在 5000 step 内同时获得病灶纹理和 histogram 控制能力。

## 变更

- 扩散状态沿用 exp011 的 lesion-only 状态。
- denoiser 直接预测 x0，基础损失为病灶区域归一化 L1。
- 增加四类可微 soft-histogram L1，权重在前 500 step 从 0 线性升至 0.1。
- 推理根据预测 x0 计算 posterior，最终只替换 union mask 内区域。

## 配置

- patch：`64×64×32`
- seed：`20260805`
- 最大 optimizer step：`5000`
- W&B run：`exp012-p64-s20260805`
- QA：固定 8 个 validation patch，五种条件变体；不运行 test split。

## 结果

- 方法实现 commit：`d188ce170040037ec477d5ff398e1f2cf141608a`。
- 待完成远端回归、训练与 8 例 QA 后补充数值结果。

## 结论

待定。

## 下一步

与 exp010/exp011 使用同一数据、seed、step 和 QA 合同完成公平比较。

## 输出路径

`experiments/20260807_exp012_gli_lesion_only_x0_hist/outputs/`
