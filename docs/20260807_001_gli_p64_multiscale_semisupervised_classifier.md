# p64 多尺度残差与半监督亚区分类方案

## 1. 决策背景

exp009 的 M0/M1/M2/C0 最高保存 checkpoint 的患者等权 ET/RC focus mIoU 为 0.5436，训练
历史绝对最大值为 0.5482，均远低于 0.85。C0 显著优于 MLP，说明完整三维上下文必要；主要
失败集中在 ET/RC 小区域、ET 与 SNFH 混淆、RC 与 SNFH/NETC 混淆。

## 2. 数据与泄漏边界

- 有标签数据固定复用 exp009 的 1000 个 train p64 patch，不重新抽样。
- val/test 患者隔离与 `splits_v2.json` 保持不变。
- 半监督池为 train pool 排除这 1000 patch 后的所有剩余 patch。
- 未标签 loader 只能返回 T1c、总 mask、case/subject/path，不加载四分类 target，不使用
  `anchor_label`、per-class hist、类别体素量或任何真实亚区派生特征。
- 模型选择始终只依据完整 p64 重建后的患者等权 val；test 继续 fail-closed。

## 3. 监督阶段

输入 `[B,2,32,64,64]`，两通道为归一化 T1c 与总病灶 mask。网络使用两级残差 3D U-Net，
bottleneck 并行 dilation 1/2/4 多尺度上下文，GroupNorm 支持小 batch，输出
`[B,4,32,64,64]`。loss 为 class-balanced CE、focal 和 soft Dice 的组合，并对 ET/RC
赋予 1.25 倍 focus 权重。训练加入一致的三轴翻转与 T1c 强度扰动。

## 4. 半监督阶段

监督阶段最佳 checkpoint 同时初始化 student 与 EMA teacher。每一步分别进行：

1. 冻结 1000 patch 的监督 loss；
2. 未标签 patch 的 teacher 弱扰动预测；
3. student 强扰动预测；
4. 仅在总 mask 内选择置信度至少 0.9 的伪标签；
5. 每类最多选择相同上限的最高置信体素，避免 SNFH 主导；
6. 反向传播伪标签 CE 与概率一致性，更新 student，再以 0.99 EMA 更新 teacher。

每 epoch 确定性选取 1000 个未标签 patch，循环偏移保证数个 epoch 内覆盖完整剩余池。验证使用
teacher 权重，不使用未标签数据中实际存在的四分类标签。

## 5. 训练与门禁

- 两阶段使用不同固定 W&B run ID，online 初始化失败时在模型训练前退出。
- best checkpoint 对任何绝对 val 新高都保存；`min_delta` 只控制 early-stopping patience。
- 验证门禁：focus mIoU ≥0.85、ET IoU ≥0.80、RC IoU ≥0.80、mask 外非零率为 0、
  union Dice 为 1。
- 任一门禁失败时不得运行 test；两阶段全部完成后只选择 val 最优方法。
- 只有最终冻结的 config、checkpoint、subset 和门禁记录一致时，才运行一次冻结 test。

## 6. 停止条件

- 监督 U-Net 若未超过 C0 至少 0.03，优先检查容量、增强与优化，不直接启动低置信伪标签。
- 半监督阶段若伪标签选中率长期接近 0，或 val 连续 8 次无 0.005 显著改善，则停止。
- 若半监督相对监督最佳无正增益，保留监督模型，不以 test 调参。
- 即使验证未达到 0.85，也必须如实停止在 val 并报告单模态信息与小区域上限，不放宽门禁。
