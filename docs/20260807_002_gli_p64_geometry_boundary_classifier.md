# p64 几何通道与边界 IoU 优化方案

## 1. 决策依据

exp013 监督多尺度残差 3D U-Net 的患者等权 ET/RC focus mIoU 为 `0.573000`；mean-teacher
为 `0.572729`。半监督阶段伪标签覆盖约 `19%–20%`、平均置信度约 `0.993–0.995`，但没有
验证增益，说明相同教师的高置信伪标签主要复制已有偏差。下一步不再重复相同一致性配方，转而
显式提供总 mask 几何，并用更贴近 IoU 和类间边界的监督目标。

## 2. 输入与泄漏边界

每个 patch 的 18 个输入通道为：T1c、patch XYZ、相对总 mask bbox XYZ、相对总 mask 质心
XYZ、总 mask 内归一化距离、3×3×3 与 5×5×5 的 T1c 局部均值/标准差及 mask occupancy，
再加显式总 mask。所有通道仅由 T1c 与总 mask 计算。

真实四分类 target 只能用于监督 loss、边界权重和验证指标；禁止把真实 per-class hist、类别
体素量、anchor label、类边界图或任何四分类标签派生张量送入模型。

## 3. 模型与 warm-start

网络保持 exp013 的 base-32 两级多尺度残差 3D U-Net。除首个 residual block 的主分支与
skip 分支输入卷积外，所有参数从 exp013 监督验证最优 checkpoint 原样加载。输入卷积中：

- exp013 T1c 权重复制到 exp014 T1c 通道；
- exp013 总 mask 权重复制到 exp014 最后一个总 mask 通道；
- 其余 16 个新增通道权重置 0。

因此 warm-start 时函数起点与 exp013 基线一致，后续增益才可归因于新增特征与监督目标。

## 4. loss

- CE 保留全体 mask 内体素监督；
- Focal 关注难分类体素；
- soft Dice 保持四类重叠优化；
- Lovász-Softmax 直接近似优化四类 Jaccard/IoU；
- 类间边界由真实 target 的 3×3×3 邻域多类共存计算，只对 loss 加权，不进入输入。

边界检测忽略总 mask 外背景，只强调病灶亚区之间的界面。ET/RC 继续使用 `1.25` focus
multiplier，但不改变验证指标定义。

## 5. 训练与门禁

固定复用 SHA-256 为
`aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a` 的 1000 patch；
train/val/test 患者隔离保持不变。正式训练必须 W&B online fail-closed，只依据完整 p64 重建后
的 73 患者等权 val 选择。

验证门禁保持：focus mIoU ≥0.85、ET IoU ≥0.80、RC IoU ≥0.80、mask 外非零率为 0、
union Dice 为 1。五项全部通过前不得运行冻结 test；通过后也只允许一次 test。

## 6. Preflight 结果

- 远端 focused tests 15/15 通过。
- CPU 零步 preflight 覆盖八个 `anchor×role` 层，真实 exp013 checkpoint warm-start 成功。
- GPU0 完整 p64、batch 8 零步 preflight 输入/输出为
  `[8,18,32,64,64] -> [8,4,32,64,64]`；loss 与梯度有限，峰值 allocated/reserved 为
  `2829/3958 MiB`。
- 两次 preflight 的 optimizer steps 均为 0，subset 与 initial checkpoint SHA 完全一致。
