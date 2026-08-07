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
- 正式训练仅使用 GPU1、seed `20260805`，完成 `5000` optimizer step；W&B run
  [`exp012-p64-s20260805`](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp012-p64-s20260805)
  已 finished，未提前或重复启动。
- `milestone-500/1000/2500/5000.pt` 与 `latest.pt` 齐全。step 5000 的 EMA validation total/base/hist loss
  为 `0.144356 / 0.111711 / 0.326442`，hist 权重为 `0.1`，覆盖口径与另外两种方法一致。
- 五种冻结 8 例 QA 均完成 `8/8`，NPZ、metrics、contract 与 contact sheet 齐全；没有运行 test split、519 例或全量 test。
- `qa_original_real` 平均 lesion MAE 为 `0.163218`，相对零填洞基线 `0.323883` 平均改善
  `47.5%`；8/8 病例通过 lesion 生成门槛。
- histogram counterfactual 为 `8/8`，union histogram counterfactual 为 `8/8`，mask 外最大绝对误差为 `0`；histogram 配对一致性为三方法最佳。
- 该 counterfactual 只比较 mask 内 16-bin 强度 histogram 与请求/交换请求的 L1 距离，不衡量空间纹理、形态或医学类别语义。
- 目检显示 first/last cluster 会产生清楚的亮暗分布变化，但 union mask 内没有呈现明确的跨病例同类外观；个别 5k 样本仍出现偏黑或偏亮团块。

## 结论

exp012 是本轮当前最强 histogram 条件响应候选，但现有 union 8/8 不能证明它能生成视觉统一、类别明确的同一种病灶。该结论仅限固定 8 例 validation QA，不代表医学有效性、其他 seed 或全量 test 表现。

## 下一步

先用相同病例、相同 union mask 和相同噪声分别生成四个目标类别，验证类内一致性与类间可分性；确认后再决定是否以 exp012 为后续方法起点。exp010 继续作为 paired 重建对照。任何新增训练、seed 或扩大测试范围均需另行授权。

## 输出路径

- 实验输出：`experiments/20260807_exp012_gli_lesion_only_x0_hist/outputs/`
- 统一比较：`experiments/20260807_exp013_gli_short_method_comparison/outputs/`
