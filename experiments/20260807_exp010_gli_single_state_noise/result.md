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

- 正式运行代码 commit：`d0c6aaa056b38e34e4cb929f85106cc90af05426`；远端完整回归测试 `39/39` 通过。
- BF16 正式预检（零 optimizer update）通过：loss `1.069728`，梯度范数 `16.731850`，反向峰值显存 `23324.62 MiB`；checkpoint 恢复成功。
- 预检 validation：`val/ema/total_loss=0.996760`，覆盖 `1032` patches、`73` subjects、`3207` 个有效类单元及全部 4 类。
- 正式训练使用 GPU1，W&B：`exp010-p64-s20260805`。首次后台启动因登录 shell 丢失 conda PATH，在导入 `einops` 时以 step 0 退出；保留失败日志后改用已验证的 conda Python 绝对路径启动，未产生重复训练或 checkpoint。
- step 500：`latest.pt` 与 `milestone-500.pt` 已落盘；`val/ema/total_loss=0.178548`，覆盖口径与预检一致。训练随后正常恢复，最近审计到 step 571，有限 loss 为 `0.113238 / 0.358248 / 0.121547`。
- 训练与固定 8 例 QA 尚在进行，最终门槛结果待补充。

## 结论

待定。

## 下一步

继续现有唯一串行队列到 step 5000；不得重复启动。完成后只运行固定 8 例五种 QA 变体。

## 输出路径

`experiments/20260807_exp010_gli_single_state_noise/outputs/`
