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

- 正式运行代码 commit：`d0c6aaa056b38e34e4cb929f85106cc90af05426`；远端完整回归测试 `39/39` 通过。
- BF16 正式预检（零 optimizer update）通过：loss `1.065113`，梯度范数 `17.252542`，反向峰值显存 `23324.62 MiB`；checkpoint 恢复成功。
- 预检 validation：`val/ema/total_loss=0.999704`，覆盖 `1032` patches、`73` subjects、`3207` 个有效类单元及全部 4 类。
- 正式训练仅使用 GPU1、seed `20260805`，完成 `5000` optimizer step；W&B run
  [`exp011-p64-s20260805`](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp011-p64-s20260805)
  已 finished，未提前或重复启动。
- `milestone-500/1000/2500/5000.pt` 与 `latest.pt` 齐全。step 5000 的 EMA validation total loss
  为 `0.136704`，覆盖口径与 exp010 一致。
- 五种冻结 8 例 QA 均完成 `8/8`，NPZ、metrics、contract 与 contact sheet 齐全；没有运行 test split、519 例或全量 test。
- `qa_original_real` 平均 lesion MAE 为 `0.164752`，相对零填洞基线 `0.323883` 平均改善
  `45.6%`；lesion 生成门槛通过 `7/8`。
- histogram counterfactual 为 `7/8`，但 union-as-single counterfactual 仅 `5/8`，未达到 `6/8`；mask 外最大绝对误差为 `0`。
- 目检可见非零、与 mask 对齐的病灶结构，但 union first/last 的纹理响应较弱且不稳定。

## 结论

exp011 能学习病灶生成并保持洞外背景，但未通过 union-as-single 门槛，因此不作为当前优胜方法。单纯把扩散状态改为 lesion-only 噪声预测没有稳定优于完整 T1c 状态。

## 下一步

保留为失败门槛对照，不增加训练步数、不扩大测试范围；若复用其 lesion-only 状态，应采用 exp012 的直接 x0 与显式 histogram 约束方向。

## 输出路径

- 实验输出：`experiments/20260807_exp011_gli_lesion_only_noise/outputs/`
- 统一比较：`results/20260807_exp010_exp012_short_comparison/`
