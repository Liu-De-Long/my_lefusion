# exp011 lesion-only 噪声预测

## 目标

验证只扩散病灶信号、洞外状态恒为零噪声轨迹，是否比完整 T1c 状态更快学会伪病灶生成。

## 变更

- 训练状态为 `y0=x0×union_mask`，洞外为零。
- 预测目标为加入 `y0` 的噪声。
- masked context、四类 mask、histogram 条件与 exp010 完全相同。
- 推理时洞外状态使用零值的前向噪声轨迹，最终只把生成病灶粘回真实背景。

## 配置

- patch：`64×64×32`
- seed：`20260805`
- 最大 optimizer step：`5000`
- W&B run：`exp011-p64-s20260805`
- QA：固定 8 个 validation patch，五种条件变体；不运行 test split。

## 结果

- 方法实现 commit：`d188ce170040037ec477d5ff398e1f2cf141608a`。
- 待完成远端回归、训练与 8 例 QA 后补充数值结果。

## 结论

待定。

## 下一步

与 exp010 使用同一数据、seed、step 和 QA 合同完成公平比较。

## 输出路径

`experiments/20260807_exp011_gli_lesion_only_noise/outputs/`
