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
- W&B 预检 run：`exp011-p64-preflight-s20260805`。正式 5k 训练在串行队列中等待 exp010 成功生成 `milestone-5000.pt`，未提前或重复启动。

## 结论

待定。

## 下一步

等待现有串行队列自动进入 exp011；完成后只运行固定 8 例五种 QA 变体。

## 输出路径

`experiments/20260807_exp011_gli_lesion_only_noise/outputs/`
