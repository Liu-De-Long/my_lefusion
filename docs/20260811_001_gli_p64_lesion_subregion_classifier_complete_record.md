# GLI p64 病灶亚区分类完整记录

## 1. 文档目的与当前结论

本文汇总 `20260806_exp009` 至 `20260808_exp016` 的 p64 病灶亚区分类数据契约、模型演进、
验证结果、失败证据、泄漏边界和后续路线。它是跨实验的长期入口；各实验的实现细节和哈希仍以
对应实验卡、原始输出及既有专题文档为准。

截至 2026-08-11，结论如下：

- 当前任务在项目内沿用“弱监督分类器”名称，但冻结的 1000 个训练 patch 具有完整逐体素四分类
  target，因此首阶段准确地说是“少标签、全体素监督”；剩余 train patch 才属于无标签池。
- 现有 p64 数据不需要重新生成标签。scalar segmentation 足以同时构造总病灶 mask 和四分类 target；
  真实四分类 target 只允许进入 loss 和离线审计，禁止作为模型输入或输入特征来源。
- 单体素 T1c MLP 的患者等权 ET/RC focus mIoU 仅 `0.3385`，只值得作为可辨识性下限。完整 p64
  三维上下文、总 mask 几何和多模态 MRI 都比继续扩大 MLP 更合理。
- 当前正式最佳为 exp016 四模态 3D U-Net：患者等权 focus mIoU `0.664752`，较最佳 T1c 几何模型
  exp014 提升 `0.072541`，但距 `0.85` 门禁仍差 `0.185248`。
- exp016 的 pooled ET/RC IoU 已为 `0.858449/0.840951`，但 `1–100` 体素 ET/RC Dice 仅
  `0.131446/0.216415`。瓶颈是小区域和患者等权泛化，不能用总体大体素表现替代。
- 截至本文提交，所有分类实验均未通过五项验证门禁，冻结 test 从未运行；不产生 test 结论。

## 2. 任务定义与不可变输入契约

### 2.1 标签语义

scalar segmentation 的语义固定为：

| scalar 值 | 语义 |
|---:|---|
| 0 | 总病灶 mask 外背景 |
| 1 | NETC |
| 2 | SNFH |
| 3 | ET |
| 4 | RC |

总病灶 mask 定义为 `seg > 0`。网络在 mask 内输出四类 logits，类别索引 `0..3` 分别对应
`NETC/SNFH/ET/RC`；重建时 `argmax + 1`，mask 外强制写为 scalar `0`。

### 2.2 shape 与轴顺序

- NPZ 存储空间轴为 XYZ=`[64,64,32]`。
- loader 转换为 DHW=`[32,64,64]`。
- T1c 单模态方案的完整 patch 输入为 `[B,2,32,64,64]`：T1c 与总 mask。
- exp014 的输入为 `[B,18,32,64,64]`：仅由 T1c 和总 mask 派生的无泄漏几何通道。
- exp016 的输入为 `[B,5,32,64,64]`：`T1c/T1n/T2f/T2w/total_mask`。
- 所有空间网络输出均为 `[B,4,32,64,64]` logits。
- 逐体素 MLP 可只抽取 mask 内体素训练；完整 patch CNN/U-Net 必须整块前向后做 masked loss。

### 2.3 输入白名单与标签泄漏红线

允许进入 forward 的信息：

- 已归一化的 MRI 模态；
- 总病灶二值 mask；
- 仅由 patch 网格、MRI 和总 mask 推导的 XYZ、bbox/质心相对坐标、distance transform、总 mask
  组件几何、局部均值/方差、局部邻域和无标签自监督表征。

禁止进入 forward、伪装为先验或参与 val/test 调参的信息：

- 真实四分类 mask、one-hot 四分类 mask；
- manifest 的 `anchor_label`、`label_*_voxels`、per-class histogram、类别存在性或类别组件；
- 从真实亚区标签推导的 bbox、质心、距离、局部统计或采样角色；
- val/test 标签分布、test 指标或 test 驱动的阈值和后处理。

训练 sampler 和 loss 可以读取训练 target 做 class balance、component weighting 和离线审计，但这些量不得
进入网络 forward；无标签 dataset 必须在返回前丢弃 scalar 四分类值，只保留 MRI 与由其生成的 total mask。

## 3. p64 数据契约审计

### 3.1 总数据与正式患者划分

T1c p64 数据位于服务器：

`/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches/patch_64x64x32`

原始 manifest 共 `9842` 个 patch、`1621` 个病例目录、`731` 个患者组。原始 manifest 只有
`train/val` 两池：train 为 `7772` patch、584 患者；原始 val 池为 `2070` patch、147 患者。
`splits_v2.json` 再把原始 val 池按患者拆为正式 val/test：

| 正式 split | 患者组 | 原始病例 | p64 patch | padding patch |
|---|---:|---:|---:|---:|
| train | 584 | 1275 | 7772 | 170 |
| val | 73 | 172 | 1032 | 21 |
| test | 74 | 174 | 1038 | 10 |

train/val/test 患者交集均为 `0`。同一患者的多个 patch 可以同时存在，但只能留在同一个 split；
因此它们不会造成跨 split 泄漏。训练时仍需限制单患者 patch 贡献，评价时必须患者等权，避免多 patch
患者主导优化或指标。

`splits_v2.json` 的 manifest SHA-256 为
`42f71687b3edfe1f54819a01490224e3bd9b6a56be25125aed0d311b99e683ad`，split 文件 SHA-256 为
`7fd0ea31dbbc7d68142852df49abb63b3cc5bdedeceeb1baef53adfdc894e8b1`。

### 3.2 类别体素与 patch 共存

正式 split 的自然分布为：

| split | NETC 体素 | SNFH 体素 | ET 体素 | RC 体素 |
|---|---:|---:|---:|---:|
| train | 9,933,234 | 159,955,833 | 46,467,399 | 51,440,460 |
| val | 1,200,098 | 19,934,864 | 6,750,480 | 6,800,579 |
| test | 1,255,255 | 20,108,616 | 6,047,164 | 8,241,826 |

train 病灶体素比例约为 NETC `3.71%`、SNFH `59.73%`、ET `17.35%`、RC `19.21%`。
因此 NETC 与 SNFH 相差约 16 倍，逐体素监督高度不平衡。

全体 9842 patch 中，按实际正体素包含的亚区数统计：仅 1 类 `248`、2 类 `1869`、3 类 `4287`、
4 类 `3438`；至少两类共存 `9594`，占 `97.48%`。四分类并不是“一个 patch 一个亚型”的 patch
分类问题，而是在同一病灶内划分共存亚区的三维 segmentation 问题。

### 3.3 冻结 1000 个有标签训练 patch

冻结子集：

`experiments/20260806_exp009_gli_p64_weak_supervision_classifier/labeled_subset_1000.json`

- 共 `1000` 个唯一 train patch，占全体 p64 的 `10.16%`，占 train pool 的 `12.87%`；
- 四个 anchor 各 `250`，每类 interior/boundary 各 `125`；
- 覆盖 `480` 个 train 患者，单患者最多 `6` 个 patch；
- SHA-256：`aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a`；
- loss-only 类别体素为 NETC `1,459,601`、SNFH `20,784,984`、ET `6,753,399`、RC
  `6,308,505`，约占 `4.13%/58.87%/19.13%/17.87%`。

“四类各 250”仅表示 anchor patch 平衡，不代表 patch 内实际类别存在性或逐体素数量平衡。真实标签统计
可用于训练 loss 权重与审计，但不能转化为输入特征。

排除冻结子集后的剩余 train pool 为 `6772` patch，SHA-256 为
`8248585a47003f33cf0f318b4df7170cee457b2f2ed40663d5112496b643057b`。它只能用于严格 train-only
的无标签学习；正式 val/test 均不得进入预训练、模型选择或校准。

### 3.4 normalization、padding 与 boundary 风险

- T1c 使用病例非零体素的 `p0.5/p99.5` 归一化契约；四模态数据重放保持原 T1c、seg、affine、
  origin 和 padding 精确一致，并逐病例对每个 MRI 模态独立归一化。
- 9842 patch 中有 `201` 个涉及 padding。padding 与 boundary patch 本身不是标签泄漏，但网络可能把
  人工边缘当成位置先验；必须保留 padding 分层指标，增强时不得把填充值与某一亚区绑定。
- interior/boundary 是基于 anchor 采样生成的 manifest 元数据，只能用于冻结子集分层与审计，禁止送入模型。
- 空类别在单 patch 中普遍存在；Dice/Lovász、precision/recall 必须明确空真值/空预测规则，不能用 NaN
  丢弃困难 patch。小区域至少按 `1–100`、`101–1000`、`>1000` 体素分层报告。

### 3.5 是否需要重新生成 p64

T1c 方案无需重新生成：现有 scalar segmentation 可直接得到 total mask 与四分类 target。exp016 因增加
T1n/T2f/T2w，已按同一 manifest 窗口重放四模态 train+val 数据；已物化 `8804` patch（train 7772、
val 1032），四模态退化病例为 0。正式 test 的 1038 条 manifest 仍保留，但多模态 test NPZ 未物化，
门禁通过前不得生成或运行。

## 4. 模型可辨识性判断

### 4.1 独立体素 MLP

只输入单体素 T1c 时，同一强度可对应坏死、术后腔、液化区、浸润/水肿样组织和增强边界。特别是 NETC
与 SNFH、ET 边缘与其他实性组织、RC 与低强度坏死/液体之间可能高度重叠。归一化还会消除病例间
绝对强度标尺。该模型没有纹理、边界、形状、病灶内相对位置和邻域一致性，无法稳定划分共存亚区。

exp009 M0 的 focus mIoU `0.3385` 已验证它只适合作为下限、数据管线 smoke 和强度可辨识性审计，
不应作为 85% 目标的最终模型。

### 4.2 空间特征与邻域 MLP

T1c+XYZ+distance/局部统计的 M1 提升至 `0.3938`；`3×3×3` 邻域 M2 为 `0.3815`，未优于 M1。
这些特征能补充局部位置，但仍逐点决策，无法高效建模长程三维结构、多尺度边界和同一区域预测一致性。
M2 没有证据支持继续盲目扩展到 `5×5×5` 或更大向量。

### 4.3 完整 p64 3D 网络

C0 的 focus mIoU `0.5436`，显著高于所有 MLP；exp013–exp016 继续提升到最高 `0.664752`。
因此完整 p64 应采用整块三维前向、mask 内 loss，并在输出端硬约束 mask 外背景。对当前任务，小型残差
3D U-Net 比逐体素 MLP 更符合数据结构；参数“轻量”不等于任务在强度域容易。

## 5. 实验演进与验证结果

所有结果均只按 val 选择，指标先恢复完整 `[32,64,64]` patch，再按患者聚合。pooled 指标只作解释，
不得用于 best 选择或 test 门禁。

| 实验/模型 | 主要变量 | 参数量 | focus mIoU | ET IoU | RC IoU | 结论 |
|---|---|---:|---:|---:|---:|---|
| exp009 M0 | 单体素 T1c MLP | 约 4.5K | 0.3385 | 0.3084 | 0.3686 | 强度下限 |
| exp009 M1 | T1c+XYZ+mask 几何/局部统计 MLP | 约 11K | 0.3938 | 0.4029 | 0.3847 | 有限几何增益 |
| exp009 M2 | `3×3×3` 邻域 MLP | 约 15K | 0.3815 | 0.4060 | 0.3569 | 不扩展邻域 MLP |
| exp009 C0 | 完整 p64 轻量 3D CNN | 约 0.1–0.3M | 0.5436 | 0.4937 | 0.5935 | 三维上下文必要 |
| exp013 supervised | 多尺度残差 3D U-Net | 3.06M | 0.573000 | 0.510031 | 0.635969 | 更强空间基线 |
| exp013 mean-teacher | 同网络+6772 无标签 patch | 3.06M | 0.572729 | 0.523194 | 0.622264 | 无正增益 |
| exp014 | 18 通道几何+Lovász+边界监督 | 约 3.06M | 0.592211 | 0.522322 | 0.662099 | 小幅增益 |
| exp015 | 患者/组件等权+Tversky/presence | 约 3.06M | 0.555747 | 0.503885 | 0.607609 | 重分配错误，反而下降 |
| exp016 | 四模态 MRI+total mask | 3,065,600 | **0.664752** | 0.589268 | 0.740236 | 当前最佳，仍未过门禁 |

### 5.1 exp009：MLP 与轻量 CNN 下限

- C0 保存 best 的 bootstrap 95% CI 为 `[0.5000,0.5862]`。
- C0 四类 IoU 为 NETC `0.2492`、SNFH `0.8679`、ET `0.4937`、RC `0.5935`；四类 macro
  Dice `0.6230`，balanced accuracy `0.7755`。
- C0 的 `1–100/101–1000/>1000` 体素 Dice：ET `0.0981/0.5258/0.8587`，RC
  `0.1031/0.4542/0.7797`，小区域问题已出现。
- 旧实现把 `min_delta=0.005` 同时用于 patience 和 best 保存，导致 history 的绝对最高 `0.5482`
  没有覆盖 epoch 22 的 `0.5436` checkpoint。后续实现已分离“绝对 best”和“显著改善 patience”。
- exp009 没有 W&B online 正式记录，因此只作方法审计，不是日志完备的正式最佳模型。

### 5.2 exp013：多尺度残差 U-Net 与 mean-teacher

- 监督模型 focus mIoU `0.573000`，95% CI `[0.529404,0.618380]`。
- mean-teacher focus mIoU `0.572729`，95% CI `[0.526628,0.617570]`；伪标签覆盖约
  `19%–20%`、平均置信度 `0.993–0.995`，仍无正增益。
- 高置信不代表正确。该配方主要复制教师对小区域和患者级错误的偏差，不应原样重复。
- W&B：
  [supervised](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp013-unet-sup-s20260807)，
  [mean-teacher](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp013-unet-mt-s20260807)。

### 5.3 exp014：几何、Lovász 与边界监督

- focus mIoU `0.592211`，95% CI `[0.547210,0.638076]`；pooled ET/RC IoU
  `0.813983/0.775111`。
- `1–100` 体素 ET/RC Dice `0.124810/0.196575`；`>1000` 体素为
  `0.877092/0.808596`。
- 18 通道全部只由 T1c、total mask 和 patch 网格产生，无标签泄漏。它与 Lovász/边界监督有小幅贡献，
  但不是从约 0.57 通向 0.85 的主路径。
- W&B：
  [exp014-geometry-boundary-s20260807](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp014-geometry-boundary-s20260807)。

### 5.4 exp015：患者与组件等权

- best focus mIoU `0.555747`，95% CI `[0.510204,0.600693]`，相对 exp014 下降
  `0.036464`。
- `1–100` 体素 ET/RC Dice `0.167198/0.232198` 偶有提高，但没有转化为患者等权总体收益。
- 患者轮换、组件等权、偏召回 Tversky 和 patch presence 主要重新分配 false positive/false negative，
  没有增加 T1c+mask 的可分信息；不再沿此方向继续叠加 loss 权重。
- W&B：
  [exp015-patient-component-s20260807](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp015-patient-component-s20260807)。

### 5.5 exp016：四模态 MRI

- 模型沿用 base-32 两级多尺度残差 3D U-Net，参数量 `3,065,600`，输入
  `[B,5,32,64,64]`，输出 `[B,4,32,64,64]`。
- 从 exp014 best warm-start：共享 trunk/decoder 复用，输入 stem 的 T1c 与 total mask 映射到新通道，
  T1n/T2f/T2w 初始权重置零；loss 保持 CE/Focal/Dice/Lovász/边界组合。
- GPU0 batch 8 零步 preflight 峰值 allocated/reserved 显存为 `2777.291/3920 MiB`。
- 唯一正式 run 在 epoch 21 后暂停，从同一原子 `latest.pt` 恢复 epoch 22，epoch 29 以
  `bad_epochs=10` 自然 early stop；best 为 epoch 24。
- 患者等权 focus mIoU `0.6647520799`，95% CI `[0.6192260550,0.7065009737]`；ET/RC IoU
  `0.5892676951/0.7402364647`，macro IoU `0.6418862227`，macro Dice `0.7025038481`，
  balanced accuracy `0.8095769745`，mask 内错误率 `0.0513752989`。
- pooled ET/RC IoU `0.8584494977/0.8409506198`；ET 小/中/大 Dice
  `0.131446/0.581826/0.905619`，RC `0.216415/0.582456/0.853121`。
- 相对 exp014 提升 `0.072541`，说明多模态是目前最有效的单一变量；但患者等权小区域泛化仍不足。
- 只读权重范数审计显示首层新模态权重明显小于 T1c：body 的 T1c/T1n/T2f/T2w 约
  `0.4632/0.0439/0.0400/0.0324`，skip 约 `0.4080/0.0196/0.0169/0.0174`。这是新模态可能
  未被充分利用的线索，不等同于因果结论，需用 val-only occlusion 验证。
- W&B：
  [exp016-multimodal-s20260808](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp016-multimodal-s20260808)。
  在线 history 为 `0–20,22–29`，暂停前 epoch 21 未刷新；服务器 history 完整为 `0–29`，无训练缺口。
- 正式代码提交 `ff288093578b4ade4c34a0318021574254225691`；最终审计提交
  `c02da58f2e0e4afa7068b91ff0310da2685a3a36`。
- best checkpoint SHA-256：`3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617`；
  final val metrics SHA-256：`6069fe97e7ebb21f810d0bda89b9b04dd26f1e2cbfb649ccb5ca84c911333c5c`。

## 6. 推荐模型与训练原则

### 6.1 已由实验更新的 baseline 判断

如果从零复现本问题，首个有实际价值的 baseline 应是完整 p64 的轻量残差 3D U-Net，而不是独立体素
MLP。当前最合理的复现基线是 exp016：

- 输入 `[B,5,32,64,64]`，输出 `[B,4,32,64,64]`；
- base-32 两级残差 3D U-Net，约 3.07M 参数；
- 整块前向，mask 内 masked loss；mask 外 logits 不参与四分类 loss；
- loss 使用 class-balanced CE + soft Dice 为基本盘，Focal/Lovász/边界项只作为已验证过的辅助，不再把
  继续调权当作主要突破方向；
- batch 8 在当前 GPU0 峰值 reserved 约 3.92 GiB；新结构必须重新做零 optimizer-step preflight；
- 预测先形成 `[32,64,64]` 四类索引，再在 total mask 内 `+1`，mask 外硬写 0；
- 默认不做连通域删除或形态学后处理。小组件可能是真实小病灶，任何后处理必须 val-only、按患者配对
  审计，并且不能读取真实四分类组件。

### 6.2 class balance

推荐组合训练层面的患者/patch 贡献控制与适度 class weighting，但避免 exp015 已失败的激进组件重加权。
不能只按总体体素反频率无限放大 NETC，也不能把 anchor 250/类误当成像素平衡。所有权重只用 train
统计冻结；val 只用于模型选择，test 不参与。

## 7. 评价协议与 85% 门禁

每个模型必须先重建完整 p64，再按患者聚合，禁止把所有患者体素混成一个全局 confusion matrix 后只报告
pooled 指标。至少报告：

- NETC/SNFH/ET/RC 每类 IoU、Dice、precision、recall；
- macro IoU、macro Dice、balanced accuracy；
- 患者归一化 confusion matrix；
- mask 内四分类错误率；
- `1–100`、`101–1000`、`>1000` 体素区域/patch 分层；
- 患者等权均值与患者 bootstrap 95% CI；
- pooled 指标，仅作大体素总体可分性的补充解释；
- mask 外非零率和预测类别 union 与 total mask 的 Dice。

focus 指标固定为：

```text
focus_mIoU = (patient_equal_IoU_ET + patient_equal_IoU_RC) / 2
```

运行冻结 test 前必须同时满足：

1. 患者等权 val focus mIoU `>= 0.85`；
2. 患者等权 val ET IoU `>= 0.80`；
3. 患者等权 val RC IoU `>= 0.80`；
4. mask 外非零率 `= 0`；
5. 预测四类 union 与输入 total mask 的 Dice `= 1`。

五项全部通过后，才可冻结 config、checkpoint、subset、阈值和后处理，并在单独授权下运行一次 test。
截至本文提交，前三项均未通过，后两项通过，`test_metrics.json` 不存在。

## 8. 风险登记

### 8.1 标签泄漏

最大风险是把 manifest 的 per-class 字段、anchor、真实类别组件或四分类 mask 派生几何混入输入。代码评审
必须同时检查 dataset 返回值、feature builder、sampler、augmentation、checkpoint config 和评估脚本；仅在
文档中声明“无泄漏”不足以构成证据。

### 8.2 患者泄漏与重复 patch

患者划分必须先于 1000 patch 抽取。同患者多病例、多 patch、重复/相邻 origin 不得跨 split。训练可轮换
患者内 patch，验证必须患者等权；不能随机按 patch 重新拆 train/val。

### 8.3 类别不平衡与空类

NETC 稀少、SNFH 占多数，ET/RC 的小区域又决定 focus 门禁。总体 accuracy 和 pooled IoU 会被 SNFH 与大
病灶主导；空类规则、small-region 分层和患者 bootstrap 必须固定，不能在看到 test 后修改。

### 8.4 指标错配

exp014/016 已显示 pooled ET/RC 可超过 0.8，而患者等权 focus 仍只有约 0.59/0.66。任何把 pooled 指标
替换为门禁、按体素混合患者或只报告大区域结果的做法都会高估临床可迁移性。

### 8.5 伪标签确认偏差

exp013 表明高置信 mean-teacher 仍可无收益。再次使用无标签池前，应先改变表示学习目标、融合结构或教师
校准；不能仅降低阈值或增加一致性权重。

## 9. 建议复用的代码与资产

- 主训练/评估入口：`scripts/gli_p64_classifier.py`
- T1c p64 loader：`scripts/gli_p64_classifier_data.py`
- 四模态重放与 loader：`scripts/gli_p64_multimodal_data.py`
- 患者级 split：`experiments/20260805_exp004_gli_inference_closed_loop/splits_v2.json`
- 冻结有标签子集：
  `experiments/20260806_exp009_gli_p64_weak_supervision_classifier/labeled_subset_1000.json`
- exp013 多尺度残差 U-Net/半监督配置：
  `experiments/20260807_exp013_gli_p64_multiscale_semisupervised_classifier/`
- exp014 几何/Lovász/边界监督配置：
  `experiments/20260807_exp014_gli_p64_geometry_boundary_classifier/`
- exp016 四模态配置与最终审计：
  `experiments/20260808_exp016_gli_p64_multimodal_classifier/`

复用原则是继承数据、患者、评价和 test seal 契约，而不是无条件复制旧模型的 stem 初始化或失败 loss 配方。

## 10. 下一步 exp017 建议（尚未实施）

建议实验 ID：

```text
20260811_exp017_gli_p64_multimodal_representation_classifier
```

建议分支：

```text
feature/20260811-exp017-gli-p64-multimodal-representation
```

### 10.1 V0：val-only 逐模态诊断

先用 exp016 epoch 24 best 复现 focus `0.664752`，随后运行：完整输入、`-T1c/-T1n/-T2f/-T2w`、
只保留 T1c+mask、去掉 T1c 只保留辅助模态+mask、mask 不进 feature 但仍用于输出 clamp。每个条件报告
患者配对差值、bootstrap CI 和 small-region 指标。V0 不训练、不运行 test。

它回答新增模态是否被实际利用，以及 exp016 的增益来自哪个模态；若完整复现误差超出实现规定的数值容差，
先修复评估复现，不进入新训练。

### 10.2 B0：独立模态 stem 监督对照

每个 MRI 使用独立 `1→8→8` 的 Conv3d+GroupNorm+GELU stem，mask 使用独立 `1→8` stem，拼接
40 通道后 `1×1×1` 融合到 32 通道，再接现有 U-Net trunk；加入约 `0.15` modality dropout。新 stem
等能量初始化，避免 T1c warm-start 先验压制 T1n/T2f/T2w。参数量预计仍约 3.1M。

B0 使用同一冻结 1000 patch，作为结构变量对照。若最终 focus 低于 exp016 超过 `0.02` 或训练不稳定，
判定融合结构失败；不能靠 test 决定是否保留。

### 10.3 B1：train-only 多模态自监督预训练

在全部 7772 个 train p64 上做缺失模态/块重建预训练，内部再按患者划分 95/5 pretrain train/holdout；
正式 val/test 完全排除。可使用 lesion-centered block masking、随机整模态 dropout 和 modality/patch 等权
SmoothL1。scalar seg 只转 total mask 后立即丢弃类别值，不生成 per-class 输入。

丢弃重建 head 后，在同一冻结 1000 patch 上 fine-tune。预训练第 5 epoch 若 heldout 重建损失没有至少
5% 相对下降则停止。B1 只有在患者等权 focus 至少达到 `0.684752`（相对 exp016 `+0.02`）、患者配对
bootstrap 差值下界大于 0、且 ET/RC 任一不下降超过 `0.01` 时，才值得进入更复杂的 B2。

### 10.4 B2：可选的小区域 component refiner

仅在 B1 显著胜出后加入全分辨率浅残差 refiner。允许输入独立 stem 高分辨率特征、detached coarse
logits/entropy，以及只由 total mask 得到的 distance 和组件几何；禁止输入真实类别组件、体素直方图或
类别存在性。先冻结 coarse 训练 refiner，再小学习率联合微调。

若 B2 到预设检查点仍无 `+0.01` focus 增益，或 ET/RC 任一下降超过 `0.02`，方法级停止。无论 B0/B1/B2
结果如何，原五项 test 门禁不变。

### 10.5 预计文件改动范围

实施前预计只涉及：

- `scripts/gli_p64_classifier.py`：独立 stem、occlusion/evaluate 与自监督训练入口；
- `scripts/gli_p64_multimodal_data.py`：严格 train-only 预训练视图和 modality dropout；
- `tests/`：输入白名单、患者隔离、test seal、occlusion、预训练不返回标签、resume/W&B fail-closed；
- 新的 `experiments/20260811_exp017_gli_p64_multimodal_representation_classifier/`；
- 对应长期方案、README、STATUS 和 CHANGELOG。

不得修改或覆盖 exp009–exp016 的冻结 config、checkpoint、history、metrics 和 W&B run。

## 11. 实施前需用户确认

1. exp017 是否仍以患者等权 ET/RC focus mIoU `0.85`、ET/RC 单类 IoU `0.80` 为不可放宽门禁；
2. 是否先只授权 V0 val-only 诊断，再根据结果分步授权 B0/B1，而不是一次启动全部训练；
3. 是否接受 B1 使用全部 7772 个 train patch 做严格无标签多模态自监督，但仍只用冻结 1000 patch
   做亚区监督 fine-tune；
4. 是否接受独立模态 stem 从等能量初始化，而不是继续将 T1c warm-start 权重作为共享 stem 主导；
5. B2 是否必须等 B1 达到 `+0.02` 且配对 bootstrap 下界大于 0 后才实施；
6. 所有阶段是否继续只用 GPU0、唯一 W&B online、val-only 选模，五项门禁前不物化或运行 test。

## 12. 追溯入口

- 初始少标签方案与 exp009：`docs/20260806_001_gli_p64_weak_supervision_classifier_plan.md`
- exp013 多尺度/半监督：`docs/20260807_001_gli_p64_multiscale_semisupervised_classifier.md`
- exp014 几何/边界：`docs/20260807_002_gli_p64_geometry_boundary_classifier.md`
- exp015 患者/组件等权：`docs/20260807_003_gli_p64_patient_component_balanced_classifier.md`
- exp016 四模态：`docs/20260808_001_gli_p64_multimodal_classifier.md`

本文只记录和审计既有工作及下一步设计；本次没有创建 exp017、没有修改训练代码、没有运行 test、
没有启动训练，也没有占用 GPU。
