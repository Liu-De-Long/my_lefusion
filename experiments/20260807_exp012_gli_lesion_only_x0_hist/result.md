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

- 正式运行代码 commit：`d0c6aaa056b38e34e4cb929f85106cc90af05426`；远端完整回归测试 `39/39` 通过。
- BF16 正式预检（零 optimizer update）通过：总 loss `0.773416`，基础 L1 `0.653153`，soft-hist loss `1.202632`，梯度范数 `20.736454`，反向峰值显存 `23324.62 MiB`；checkpoint 恢复成功。
- 预检 validation：总 loss `0.704994`、基础 L1 `0.584811`、soft-hist loss `1.201826`、hist 权重 `0.1`；覆盖 `1032` patches、`73` subjects、`3207` 个有效类单元及全部 4 类。
- W&B 预检 run：`exp012-p64-preflight-s20260805`。正式 5k 训练在串行队列中等待 exp010、exp011 完成，未提前或重复启动。

## 结论

待定。

## 下一步

等待现有串行队列自动进入 exp012；完成后只运行固定 8 例五种 QA 变体并执行三方法统一比较。

## 输出路径

`experiments/20260807_exp012_gli_lesion_only_x0_hist/outputs/`
