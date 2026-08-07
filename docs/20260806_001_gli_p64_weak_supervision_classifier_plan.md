# GLI p64 少标签逐体素亚区分类方案

## 1. 目标与范围

本方案面向现有 BraTS2024 GLI `64×64×32` T1c 局部 patch，在已知总病灶位置的前提下，
训练一个少标签、逐体素四分类器：

- 输入：归一化 T1c 和同尺寸总病灶二值 mask；
- 输出：总 mask 内的 NETC、SNFH、ET、RC 四分类；
- 最终标量标签：`0=mask 外背景、1=NETC、2=SNFH、3=ET、4=RC`；
- 不使用真实四分类 mask、per-class histogram、manifest 类别体素数或其他标签派生特征作为模型输入；
- 本轮只冻结方案，不实现代码、不创建实验卡、不启动训练。

项目内沿用“弱监督分类器”的名称，但从监督形式看，首阶段更准确地属于“约 10% patch
具有完整体素标签的少标签监督”。其余 patch 只有 T1c 和总 mask，可在后续独立实验中用于
半监督一致性或伪标签训练，首个 baseline 不自动使用。

## 2. 数量口径

现有 p64 数据共有 `9842` 个 patch、`1621` 个原始病例目录和 `731` 个患者组。因此：

- `9842` 个 patch 的 10% 约为 `984`；
- `1621` 个原始病例的 10% 约为 `162`；
- “10% 且约 1000 个 Case”只有在 `Case` 指 p64 patch 样本时才成立。

本方案默认把 `Case` 解释为 p64 patch，并采用便于四类精确平衡的固定规模：

```text
有标签训练 patch：1000
NETC anchor：250
SNFH anchor：250
ET anchor：250
RC anchor：250
```

这相当于全部 9842 个 p64 patch 的 `10.16%`，但因为只能从冻结的 train pool 选择，
相当于 7772 个 train patch 的 `12.87%`。若必须严格执行 10%，可改为 `984` 个 patch，
每类 `246` 个；该差异属于实施前必须确认的配置选择，不需要重新生成 NPZ。

## 3. 数据划分与有标签子集

### 3.1 冻结外层划分

继续使用：

```text
experiments/20260805_exp004_gli_inference_closed_loop/splits_v2.json
```

当前划分为：

| split | 患者组 | 原始病例 | p64 patch |
|---|---:|---:|---:|
| train | 584 | 1275 | 7772 |
| val | 73 | 172 | 1032 |
| test | 74 | 174 | 1038 |

train/val/test 患者组交集为 0。有标签训练子集只能从 7772 个 train patch 中抽取；
val/test 的四分类标签只用于评价，不能参与子集选择、特征计算、归一化拟合或阈值调整。

患者 ID 当前来自病例名回退规则，而不是已成功读取的原始 metadata。实施前应先验证该规则
与官方患者映射一致；若不一致，只重建 split/manifest 元数据，不必重算 T1c/seg patch。

### 3.2 确定性抽样

默认使用 seed `20260806`，在 train pool 内按以下顺序抽取：

1. 按 `anchor_label × sample_role` 建立 8 个层；
2. 每个类别选 125 个 interior 和 125 个 boundary，共 250 个；
3. 层内先按患者轮转，再在患者内按稳定路径哈希排序，避免 patch 多的患者占主导；
4. 优先保证同一 `(case_id, origin_xyz)` 只选一次，避免重复坐标 patch；
5. 冻结相对路径列表、患者数、病例数、各类实际体素暴露量和 SHA-256；
6. val/test 保持完整自然分布，不做重采样。

四类各 250 指的是 anchor patch 数平衡，不代表四类体素数平衡。现有训练 patch 内 SNFH
体素占比约 60%，NETC 约 4%，因此训练时仍需体素级类别平衡。

### 3.3 有标签与无标签边界

有标签子集允许读取 scalar segmentation 作为 target，但模型输入只能由以下内容组成：

- T1c；
- `total_mask = scalar_seg > 0`；
- 从 T1c 或 total mask 确定性推导的无泄漏特征。

剩余 train patch 在首阶段不进入优化器。若后续启用半监督阶段，无标签 dataset 返回值中
不应包含 scalar segmentation、四通道 lesion mask、histogram 或 anchor label，防止训练代码
意外访问隐藏标签。

## 4. 数据与空间契约

NPZ 以 `[X,Y,Z]=[64,64,32]` 保存，loader 转换为：

```text
T1c:        [B,1,D,H,W] = [B,1,32,64,64]
total_mask: [B,1,D,H,W] = [B,1,32,64,64]
target:     [B,1,D,H,W] = [B,1,32,64,64]
```

模型输出：

```text
logits: [B,4,32,64,64]
```

mask 内将标签 `1..4` 映射为 CE target `0..3`。推理结果按下式恢复：

```python
pred_inside = logits.argmax(dim=1) + 1
pred = torch.where(total_mask[:, 0], pred_inside, 0)
```

因此 mask 外始终为背景 0，预测类别 union 与输入总 mask 必须精确一致。

## 5. 模型方案

### 5.1 判断

任务参数量可以很小，但并不等同于仅靠单体素强度就容易分类。既有 train-only 强度审计中，
NETC 与 SNFH 的归一化 T1c 直方图重叠约 `0.85`，说明强度 MLP 只能作为下限。
ET 与 RC 在 T1c 上相对更显著，当前将其作为重点性能类别，但完整输出仍必须保留四类。

### 5.2 M0：单体素强度 MLP

```text
输入：中心体素 T1c，1 维
结构：1 → 64 → 64 → 4
参数量：约 4.5K
```

用途是验证单体素可辨识性，不作为满足 85% mIoU 的默认候选。

### 5.3 M1：无泄漏空间特征 MLP（推荐首个模型）

每个 mask 内体素使用约 17 维特征：

- 中心 T1c：1；
- patch 内归一化 XYZ：3；
- 相对总 mask bbox 坐标：3；
- 相对总 mask 质心坐标：3；
- 到总 mask 边界的归一化距离：1；
- `3×3×3`、`5×5×5` T1c 局部均值和标准差：4；
- 两个尺度的总 mask occupancy：2。

网络：

```text
17 → 128 → 64 → 4
Linear + LayerNorm + GELU + Dropout(0.1)
参数量：约 11K
```

所有空间特征只由 T1c 和 total mask 计算。禁止读取 patch 的全局 per-class histogram、
四分类 lesion mask 或 label voxel count。

### 5.4 M2：局部邻域 MLP

将中心体素周围 `3×3×3` 的 T1c 和 total mask 展平为 54 维，再输入：

```text
54 → 128 → 64 → 4
参数量：约 15K
```

如果 M1 已显著优于 M0，再用 M2 判断原始局部纹理是否比手工局部统计有效。
只有 `3×3×3` 有明确收益时才扩展到 `5×5×5`，避免不必要的 unfold 内存和 I/O。

### 5.5 C0：轻量 3D CNN 对照

若 MLP 无法达到目标，使用一个不下采样的轻量 fully convolutional 3D CNN：

- 输入 `[B,2,32,64,64]`，即 T1c 与 total mask；
- 16/32 个通道的残差卷积；
- dilation `1,2,4,2,1` 扩大三维感受野；
- `1×1×1` 输出四通道；
- 参数量控制在约 0.1–0.3M。

该模型仍属于轻量逐体素分类器，但能提供 MLP 缺少的共享三维上下文。它是达到较高 IoU
更现实的兜底方案，不使用完整 3D U-Net 作为首个实现。

## 6. 训练策略

### 6.1 体素采样

对 MLP 不把所有病灶体素一次性展开保存。每个有标签 patch 在线抽样：

- 每类最多 512 或 1024 个体素；
- 类内均匀随机，seed 可恢复；
- 极小类别保留全部体素，其他类别下采样；
- 每个 batch 的四类有效体素数尽量相等；
- 同一患者的 patch 数受上层 sampler 限制。

标签只用于选择监督体素和计算 loss，不拼入特征。

### 6.2 loss

MLP 首版使用四类 class-balanced CrossEntropy。类别平衡已经由体素采样完成，不再叠加
完整 inverse-frequency 权重。可在验证后增加轻度平方根逆频率权重，但应避免对 NETC
进行双重过采样。

轻量 3D CNN 使用完整 patch 前向和 masked loss：

```text
0.7 × class-balanced CrossEntropy + 0.3 × macro soft Dice
```

首版不使用 Focal Loss，不做连通域或形态学后处理。

### 6.3 少标签阶段与可选半监督阶段

阶段 1 只使用冻结的 1000 个有标签 patch，依次运行 M0、M1、M2，必要时运行 C0。

只有阶段 1 未达到目标且用户另行确认时，才新增半监督方法实验：

- EMA teacher；
- T1c 强度增强下的一致性 loss；
- 按类别设置置信度阈值；
- 伪标签只在 total mask 内生成；
- 阈值只用 val 调整，test 始终封存。

半监督逻辑属于新的方法变化，不应悄悄并入首个 MLP baseline。

## 7. 性能目标与评价

“TC 模态”在本方案中暂按现有单模态输入 `T1c` 理解。重点类别暂定为 ET 和 RC。

核心指标：

```text
focus_mIoU = (IoU_ET + IoU_RC) / 2
```

建议成功门禁：

- 患者等权 val `focus_mIoU ≥ 0.85`；
- 模型与阈值冻结后，患者等权 test `focus_mIoU ≥ 0.85`；
- ET、RC 单类 IoU 均不低于 0.80，避免均值掩盖单类失败；
- mask 外非零率为 0，预测 union 与输入总 mask 的 Dice 为 1。

85% 是目标门禁，不是根据现有强度分布即可保证的结果。尤其逐体素 MLP 缺少长程三维上下文，
若 M1/M2 明显低于门禁，应优先验证 C0，而不是继续扩大纯 MLP。

所有指标必须先恢复完整 `[32,64,64]` patch，再按患者聚合；禁止把所有患者体素混成单个
全局 confusion matrix。至少报告：

- NETC/SNFH/ET/RC 各类 IoU、Dice、precision、recall；
- 四类 macro IoU、macro Dice；
- ET/RC focus mIoU；
- balanced accuracy；
- 患者归一化 confusion matrix；
- mask 内错误率；
- `1–100`、`101–1000`、`>1000` 体素的小区域分层指标；
- 患者 bootstrap 95% 置信区间。

## 8. 最小实验矩阵与停止条件

| 组别 | 目的 | 停止条件 |
|---|---|---|
| M0 | 单强度可辨识性下限 | balanced accuracy 接近随机或重点类 recall 塌缩时保留下限结果，不调大网络 |
| M1 | 验证总 mask 几何和局部统计是否足够 | patient val focus mIoU 连续 8 次验证无 `0.005` 改善则停止 |
| M2 | 验证原始 `3×3×3` 邻域贡献 | 相对 M1 focus mIoU 提升不足 `0.01` 时不扩展 `5×5×5` |
| C0 | 验证共享三维上下文是否必要 | 若稳定优于 MLP 则作为正式候选；仍达不到 0.85 时再讨论半监督或更强 CNN |

模型选择只看 val。test 只在配置、checkpoint、后处理和重点类别定义全部冻结后运行一次。

## 9. 标签泄漏与数据泄漏门禁

以下字段可以用于有标签子集的离线分层或监督 target，但不得进入模型输入：

- NPZ `hist`；
- 四通道 `lesion_mask`；
- manifest `anchor_label`、`label_1_voxels` 至 `label_4_voxels`；
- per-class bbox、质心、距离或强度统计；
- val/test 的标签分布。

实现时应建立专用 leak-safe dataset，模型输入字段采用白名单，并在测试中断言：

```text
输入只能包含 T1c、total_mask 及其无标签派生特征。
```

患者划分先于 10% 子集抽取；同一患者的任何 patch 不得跨 train/val/test。重复 origin patch
不得同时增加有标签样本权重。

## 10. 实施建议

建议后续实验 ID：

```text
20260806_exp009_gli_p64_weak_supervision_classifier
```

建议分支：

```text
feature/20260806-exp009-gli-p64-weak-supervision
```

M0、M1、M2、C0 属于同一研究问题的消融配置，应放入同一实验目录。方法实现应新建独立的
分类器 dataset、model、metrics 和 training 模块，不修改现有 diffusion loss 或 exp008 配置。

当前 exp008 p64 正在远端双 GPU 运行。在其结束或用户明确安排独立资源前，不切换远端分支、
不运行分类器 preflight、不占用 GPU。

## 11. 实施前确认项

1. `Case` 是否确认指 p64 patch，而不是原始病例目录；
2. 选择固定 1000 个 patch（每类 250）还是严格 984 个（每类 246）；
3. “TC”是否确认指 T1c；
4. 两个重点亚型是否确认是 ET 和 RC；
5. 85% 门禁是否要求 ET/RC 两类平均 mIoU，且单类 IoU 均不低于 80%；
6. 首阶段是否只使用有标签子集，未标签 train patch 暂不加入优化。

## 12. 2026-08-07 实施结果与长期决策

用户已确认按 1000 个 p64 patch、T1c、ET/RC focus mIoU 门禁实施，并授权在独立 exp009
worktree 严格串行训练 M0/M1/M2/C0。冻结子集 SHA-256 为
`aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a`，覆盖 480 个 train
subject；val 为 1032 patch、73 subject，患者级隔离保持成立。

验证集结果如下：

| 模型 | focus mIoU | ET IoU | RC IoU | 判断 |
|---|---:|---:|---:|---|
| M0 | 0.3385 | 0.3084 | 0.3686 | 单强度只适合作为下限 |
| M1 | 0.3938 | 0.4029 | 0.3847 | 几何与局部统计有有限增益 |
| M2 | 0.3815 | 0.4060 | 0.3569 | 不优于 M1，不扩展 5×5×5 |
| C0 | **0.5436** | 0.4937 | 0.5935 | 三维上下文必要，但远未达到门禁 |

C0 的 focus mIoU 患者 bootstrap 95% CI 为 `[0.5000, 0.5862]`。四种模型均未通过
`focus mIoU >= 0.85` 且 ET/RC 单类 IoU >=0.80 的验证门禁，因此 test 保持封存，从未运行。

该结果把长期路线从“优先尝试 MLP”更新为：MLP 只保留下限与消融价值，后续若继续追求
0.85，应采用更强的多尺度 3D 网络，或在另一个方法实验中使用剩余 train patch 做半监督；
不得继续把当前轻量 MLP/CNN 的小幅调参视为足以达到目标的方案。新方法仍必须维持 T1c+总
mask 输入白名单、患者隔离、完整 p64 后患者等权评价和一次性 test 门禁。

本次 classifier 训练没有 W&B online 记录，违反项目正式训练日志要求；结果可作为验证集方法
审计，但 checkpoint 不能标记为 W&B 完整的正式最佳模型。后续训练必须先补 W&B fail-closed。
