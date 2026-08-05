# GLI patch 级 inference 闭环：详细分析与实施方案

## 1. 文档定位

本文是跨实验复用的 GLI patch 级推理接口和验收方案，服务于
`20260805_exp004_gli_inference_closed_loop` 及后续正式训练、checkpoint 验证和
生成质量评估。本文回答“如何把 GLI 训练 checkpoint 变成可复现的 patch inference
产物”，不把一次 smoke 的数值误认为医学有效性结论。

本方案覆盖：

- test-only inference 数据读取和 subject 划分；
- T1c、segmentation、四通道 lesion mask、histogram condition 的契约；
- train-only histogram cluster 构建和推理选择；
- 显式 brain support、padding 和 healthy/outside 区域约束；
- 逐反向步 RePaint 状态融合；
- 四通道输出合成、NPZ/NIfTI 保存和 provenance；
- 64×64×32 与 80×96×80 两种矩形 patch 的 shape 验收。

不在本文范围内：全脑 sliding-window 拼接、医学标签语义的最终裁定、正式模型
质量结论、正式训练授权以及尚未定义的全脑临床推理协议。

## 2. 当前问题与目标

GLI 训练已经支持四通道输入 `[B,4,D,H,W]`、四通道 `lesion_mask` 和 64 维
histogram condition，但训练闭环并不自动产生可验证的推理接口。若直接复用旧的
LIDC/EMIDEC inference，会出现以下问题：

1. inference 入口只接受旧数据集类型；
2. 没有 GLI test loader 和固定 subject split；
3. 没有四个 label 独立的 histogram condition 来源；
4. `GaussianDiffusion_Nolatent` 的采样 shape 不能依赖正方形 `image_size`；
5. RePaint 若只在最后一步覆盖原图，会掩盖中间状态错误；
6. 四个生成 channel 不能直接平均或直接当作 segmentation；
7. normalized T1c 的数值不能可靠反推出 brain mask；
8. checkpoint、cluster、输入 patch 和输出体积可能被不同实验混用。

目标是形成如下可追溯链路：

```text
训练 checkpoint
  -> train-only split / histogram cluster / provenance
  -> test-only GLI inference loader
  -> 显式 brain support + padding + lesion mask
  -> 共享背景的逐步 RePaint
  -> 四通道 lesion-aware 输出合成
  -> NPZ/NIfTI + metrics.json + manifest + shape/provenance 校验
```

## 3. 数据、标签和空间契约

### 3.1 segmentation 语义

当前 loader 按数据集约定解释 scalar segmentation：

| 值 | 名称 | 推理用途 |
|---:|---|---|
| 0 | background | 非病灶，参与 outside/healthy 约束 |
| 1 | NETC | lesion channel 0 |
| 2 | SNFH | lesion channel 1 |
| 3 | ET | lesion channel 2 |
| 4 | RC | lesion channel 3 |

标签语义已通过 BraTS 官方评测说明确认；patch、四通道 mask、histogram、loss、
cluster 和 inference 均固定使用 `NETC/SNFH/ET/RC` 顺序。该确认只解决标签契约，
不代表当前 smoke 输出已经具备医学有效性。

### 3.2 输入输出 shape

NPZ/NIfTI 存储顺序为 XYZ，模型内部顺序为 CDHW：

| patch_size_xyz | model spatial_shape_dhw | model input | NIfTI/NPZ 输出 |
|---|---|---|---|
| `[64,64,32]` | `[32,64,64]` | `[B,4,32,64,64]` | `[64,64,32]` |
| `[80,96,80]` | `[80,80,96]` | `[B,4,80,80,96]` | `[80,96,80]` |

必须在 loader、diffusion、sampler、保存和回读五个位置分别校验 shape；禁止用
单一 `image_size` 推导矩形 patch 的全部空间维度。

### 3.3 split 与泄漏门禁

使用版本化的 `splits_v2.json`，当前规划为 train/val/test subject 数量
`584/73/74`，三组 subject 零交集。规则如下：

- checkpoint 和 cluster 只能读取 train；
- validation 只能用于接口调试和选择 smoke checkpoint；
- inference smoke 和正式评估只能读取 test；
- patch 级样本不能替代 subject 级去重；
- 输出 manifest 必须记录 `subject_id`、`case_id`、split、patch path 和 Git SHA。

## 4. inference loader 设计

新增或复用 `GLIInferenceDataset`，不能直接把训练 dataset 的 `split="train"`
逻辑复制到推理脚本。每个样本至少返回：

```text
GT / source_t1c       [1,D,H,W]  原始归一化 T1c
conditioning_seg      [1,D,H,W]  scalar 0..4
lesion_mask           [4,D,H,W]  seg == 1..4
hist                  [64]       四个 16-bin block
explicit_brain_support_mask [1,D,H,W]
healthy_brain_mask    [1,D,H,W]
outside_mask          [1,D,H,W]
pad_mask              [1,D,H,W]
affine_xyz             [4,4]
patch_size_xyz, origin_xyz, pad_before_xyz, pad_after_xyz
subject_id, case_id, relative_path, split
```

loader 必须拒绝：shape 不匹配、非法 seg 值、非有限数、manifest patch size 不一致、
路径越界、test split 之外的样本，以及 metadata 与实际数组不一致的样本。

## 5. normalization 与显式 brain support

### 5.1 当前 normalization 的限制

预处理使用原始 `T1c != 0` 计算每个病例的 0.5/99.5 percentile，再映射到
`[-1,1]`。因此原始精确零背景不参与 percentile，但所有原始非零体素都可能参与；
归一化后合法脑体素也可能恰好变成 0。结论是：

- 不能用 `normalized_t1c != 0` 反推 brain mask；
- 不能把 patch padding 的 0 和真实脑内 0 混为一类；
- inference 必须显式构造并保存 support/mask；
- 是否重新做 brain-mask normalization 另立预处理实验，不在本闭环中静默更换。

### 5.2 support 构造策略

`explicit_brain_support_mask` 只用于推理合法性、QA 和背景融合，不作为训练标签。
规划的确定性构造为：

1. 从原始四模态体数据构造 `nonzero_any`；
2. 要求至少两个模态非零，降低单模态伪影影响；
3. 保留最大 3D 连通区域并填洞；
4. 与 `segmentation > 0` 合并，避免病灶被 support 排除；
5. 与 `pad_mask` 求交，禁止把 patch 外填充值当成脑区；
6. 派生 `healthy_brain_mask = support & (seg == 0)`；
7. 派生 `outside_mask = ~support`，并单独记录 boundary outer shell。

该 mask 是工程支持区域，不是人工标注的脑组织真值。每个样本应保存 support 的
体素数量、连通域数量、padding fraction 和与 segmentation 的覆盖率，供 QA 复核。

## 6. histogram condition 与 cluster 计划

### 6.1 构建规则

每个 label 使用独立的 16-bin histogram block。训练 patch 中缺失的 label 保持
零 block，不参与该 label 的拟合。为避免 patch 数量多的 subject 主导聚类，
每个 `(subject,label)` 的 patch 权重和固定为 1。

cluster 资产必须包含：

- schema/version；
- patch size XYZ 与 DHW；
- label 顺序和 histogram bins；
- 每个 label 的 `k`、centers、membership、cluster size；
- k-sweep、距离指标、PCA 或等价可视化摘要；
- 数据 split、输入 manifest hash、生成时间和 Git SHA。

### 6.2 推理选择规则

推理时：

- 当前样本存在的 label 选择对应 train cluster 中距离最近的 center；
- 不存在的 label 使用零 block；
- 四个 block 按 label 1→4 拼接成 64-D condition；
- 禁止读取 val/test histogram 重新拟合 cluster；
- 禁止用随机 histogram 替代正式 cluster，除非明确标记为独立 ablation。

必须验证 condition shape `[B,64]`、有限性、block 边界和 cluster provenance。

## 7. checkpoint schema 与加载

GLI checkpoint 不能只保存裸 `state_dict`。至少应记录：

```text
experiment_id
git_sha
data_type=gli
patch_size_xyz
spatial_shape_dhw
channels=4
base_dim / model architecture
cond_dim=64
timesteps
loss_type
normalization contract
split_file / manifest hash
cluster asset path and hash
training step / seed
```

加载时先校验 schema、空间尺寸、通道数、condition 维度、模型结构和 cluster 版本；
不一致应在采样前报错，不能“尽量加载”后继续生成。

## 8. RePaint 状态融合语义

### 8.1 背景轨迹

使用原始单通道 T1c 构造共享背景的前向扩散轨迹。每一个反向 timestep 都使用与
当前 timestep 对应的背景状态，而不是只在采样结束后覆盖原图。

### 8.2 四通道生成

模型内部保持四通道状态 `[B,4,D,H,W]`。每个 channel 只在自身 lesion mask 内保留
生成状态；非 lesion 区域与同一 timestep 的共享背景状态融合。这样可避免四通道
分别产生不一致的健康脑背景。

必须验证：

- lesion mask 与 scalar segmentation 一致；
- 四个 mask 互斥且都在 support 内；
- 每个反向步的 state shape 不变；
- 没有采样后原图硬覆盖掩盖 sampler 错误；
- `p_sample_repaint` 的 deterministic/fixed noise smoke 可复现。

### 8.3 采样 schedule

`t_T=5` 仅用于接口 smoke，不能代表正式生成质量。正式生成需要单独配置完整
RePaint schedule、随机种子、采样数量和 W&B 记录。所有输出必须记录 schedule，
避免把缩减 schedule 的结果混入正式结果目录。

## 9. 四通道输出合成与保存

采样完成后先保留内部四通道结果，再按项目定义合成为单通道 patch：

1. 对每个 voxel 根据 source scalar label 选择对应生成 channel；
2. background voxel 取共享背景状态；
3. 不对四个 channel 做平均；
4. 不把生成强度直接写成 segmentation；
5. 保存四通道 NPZ 以便调试，同时保存单通道合成结果；
6. 逆变换 DHW→XYZ，使用原始 patch affine 写 NIfTI；
7. 回读 NIfTI 并验证 shape、affine、有限性和 round-trip 误差。

每个输出目录必须隔离 patch 尺寸、checkpoint、cluster、schedule 和 split，并生成
`manifest.json` 或 `metrics.json`，记录输入、输出、seed、耗时、显存、cluster id、
Git SHA 和 checkpoint hash。

## 10. 代码和配置改造清单

### 10.1 代码

- `LeFusion/inference/inference.py`：增加 GLI 分支、配置校验、输出 manifest 和
  shape/provenance 门禁；
- `LeFusion/dataset/gli_hist_in.py`：实现 test-only inference adapter、support、
  padding 和 metadata 返回；
- `LeFusion/inference/gli_utils.py`：cluster schema、nearest-center condition、
  XYZ/DHW 转换；
- `LeFusion/ddpm/diffusion.py`：实现并校验 GLI `p_sample_repaint` 状态融合；
- `scripts/gli_build_inference_assets.py`：生成 versioned split、cluster 和审核摘要；
- `scripts/gli_brain_support_audit.py`：输出 support 与 normalization 差异 QA。

### 10.2 配置

为两个 patch 尺寸各提供独立 inference 配置，至少包含：

```yaml
data_type: gli
patch_size_xyz: [64, 64, 32]  # 或 [80, 96, 80]
spatial_shape_dhw: [32, 64, 64]  # 或 [80, 80, 96]
channels: 4
cond_dim: 64
split: test
conditioning.source: cluster
conditioning.selection: nearest
repaint.t_T: 5  # smoke；正式运行必须显式改为完整 schedule
output_root: experiments/20260805_exp004_gli_inference_closed_loop/outputs/...
```

checkpoint、cluster、inference 输出和临时 Hydra 目录不能跨 patch 尺寸复用或覆盖。

## 11. 分阶段实施与测试计划

### 阶段 A：静态契约

- 两份 Hydra 配置 `--cfg job` 无 `???`；
- loader 返回字段、dtype、shape 和 XYZ/DHW 轴顺序正确；
- checkpoint schema 与 cluster provenance 可读；
- 非法 split、shape、label、cluster 和 affine 均能明确失败。

### 阶段 B：纯函数与单元测试

- scalar seg 到四通道 mask 的确定性展开；
- support、healthy、outside、padding 互斥/覆盖关系；
- histogram nearest-center 选择、缺失 label 零 block、subject 权重；
- `mix_gli_repaint_state` 每一步保持背景和 lesion 区域约束；
- 四通道合成不平均、不误写 segmentation；
- DHW↔XYZ 和 affine round-trip。

### 阶段 C：64×64×32 smoke

- test sample 输入 `[1,4,32,64,64]`；
- 固定 checkpoint、seed、condition、noise，使用 `t_T=5`；
- 验证逐步模型调用、最终 `[1,1,32,64,64]`、NPZ/NIfTI 回读；
- 记录耗时、峰值显存、healthy-brain MAE、outside residual 和 boundary 指标。

### 阶段 D：80×96×80 smoke

- test sample 输入 `[1,4,80,80,96]`；
- 精确验证 `[80,80,96]` 的采样链和 XYZ `[80,96,80]` 保存；
- batch size 1，不静默替换 attention；
- 若 OOM，只按已有门禁依次评估 AMP、gradient accumulation 或 checkpointing，
  并在新实验记录中说明。

### 阶段 E：正式 inference 门禁

- 使用正式训练 checkpoint，而不是 validation smoke checkpoint；
- 使用在线 W&B run URL 和完整 provenance；
- 先完成接口/数据一致性验收，再讨论生成质量或医学指标；
- 任何标签语义变化、normalization 重做或全脑拼接均创建新的方法实验。

## 12. 当前验收标准与已知限制

当前闭环至少应达到：

- 22/22 代码与真实数据测试通过；
- 两种 patch 的输入、内部四通道、最终单通道和 NIfTI XYZ shape 一致；
- train/val/test subject 无交集，cluster/checkpoint 不读取 test；
- healthy-brain、outside 和 boundary outer-shell 不被生成状态污染；
- 输出可由 manifest 追溯到 checkpoint、cluster、输入 patch 和 Git SHA；
- smoke 产物与正式训练产物目录隔离。

已知限制：

- `t_T=5` smoke 不能证明正式采样质量；
- 显式 brain support 是工程 QA mask，不是医学真值；
- 当前 normalization 仍基于原始非零区域，不能从 normalized patch 反推 brain mask；
- 四通道生成结果的医学语义仍需官方标签说明确认；
- 尚未实现全脑 sliding-window 合并和下游临床评价。

## 13. 实验管理与输出

本方案对应实验 `20260805_exp004_gli_inference_closed_loop`。代码、配置、split、
结果记录和 `code_version.txt` 进入实验目录；checkpoint、cluster、W&B cache、
生成体积等大型或环境相关产物只保留在隔离的 `outputs/`，不进入 Git。

正式实施必须：

1. 在本地 feature branch 完成代码和测试；
2. 创建 commit 并写入完整 SHA；
3. 推送后核对远端 branch、commit、status；
4. 远端只切换到同名 branch 验证，不形成独立提交；
5. 将 smoke 结果、失败原因、输出路径和下一步写入对应 `result.md`、`STATUS.md`、
   `CHANGELOG.md`。

## 14. exp005 正式 checkpoint 的 validation、闭环 QA 与半量 test 方案

### 14.1 授权与固定模型入口

2026-08-06，用户确认对 `20260805_exp005_gli_formal_training_baseline` 执行正式
checkpoint 评估，并固定使用：

- checkpoint：`outputs/patch_64x64x32/seed_20260805/checkpoints/best.pt`；
- 权重：`ema`，不得在 QA 或 test 阶段改用 raw `model`；
- patch：仅 `64×64×32`；
- 不启动其他 seed，不启动 `80×96×80`；
- validation、零更新 resume 和完整 val inference QA 全部通过后，自动开始冻结的半量
  test，不再临时选择 checkpoint 或调整采样参数。

`best.pt` 为 schema 2 正式训练 checkpoint，step 为 `46000`。它记录的 EMA validation
best 为 `0.0965079195`。50k 最终 validation 为 `0.0961709834`，虽然数值略低，但相对
改善约 `0.35%`，没有达到 early-stopping 配置的 `0.5%` relative min-delta，因此
`best.pt` 没有更新。本次遵循用户决定，仍冻结 step 46000 的 `best.pt/ema`。

### 14.2 正式运行前的兼容修改

现有 exp004 inference 配置不能原样用于正式 checkpoint：它只接受 schema 1、指向
validation smoke checkpoint、使用 raw `model`、读取 `test` 且仅运行 `t_T=5`。实施时
必须：

1. 让 inference checkpoint loader 直接接受 schema 2，不创建转换版 checkpoint；
2. 从 `resolved_config.model` 校验网络结构，并校验 experiment、patch shape、split、
   manifest、Git SHA 和 W&B run ID；
3. 规范化 DataParallel 的 `module.` 前缀后 strict-load `ema`；
4. provenance 同时支持正式字段 `git_sha`，并记录 checkpoint step、schema、SHA-256；
5. 创建 exp005 独立的 val QA 与半量 test 配置和输出目录，禁止覆盖 exp004 smoke。

这些修改属于评估基础设施兼容，不改变模型结构、loss、normalization、cluster 或 RePaint
数学语义；继续归入 exp005，但必须形成 Git commit 并通过回归测试。

### 14.3 validation 与 resume 门禁

test 前必须依次通过：

1. 用 `best.pt/ema` 重跑固定 val，覆盖 1032 patch、73 subject、四个 anchor label 和
   3207 个有效单元；指标应与 step 46000 的记录一致，允许的纯浮点误差上限为 `1e-6`；
2. 对实际 `latest.pt` 执行零 optimizer update 的 resume preflight，恢复 model、EMA、
   optimizer、AMP scaler、sampler/batch offset 和 Python/NumPy/Torch/CUDA RNG；
3. resume 只校验恢复和下一 batch 指纹，不继续训练、不登录或创建新的 W&B run；
4. metadata、config hash、split hash、manifest hash、Git SHA 或 W&B run ID 任一不一致时
   fail closed，不进入 inference QA。

### 14.4 val 完整闭环 QA

QA 只使用 val，不提前读取 test 结果。确定性选择至少 8 个 patch，覆盖：

- NETC/SNFH/ET/RC 四个 anchor label；
- interior 与 boundary；
- 尽可能多的多标签 patch。

`t_T=5` 只作为 wiring smoke；同一组样本必须继续完成 `t_T=300`、`n_sample=1`、
`jump_length=1`、`jump_n_sample=1` 的正式 QA，最终门禁只由完整 schedule 决定。

硬性验收项：

- channel 0/1/2/3 严格对应 `NETC/SNFH/ET/RC` 和 label 1/2/3/4；
- scalar segmentation 与四通道 lesion mask 逐 voxel 一致、互斥；
- explicit brain support 为 bool、shape/affine 正确、非空，并覆盖全部 lesion；
- healthy brain、support 外区域及 lesion 外边界相对输入的变化均不超过 `1e-6`；
- 分开记录“推理新增背景变化”和“输入继承的 support 外非零”，不得把两者混为背景污染；
- 每个样本完整 schedule 的 model calls 为 300，输出有限且 shape 正确；
- NPZ/NIfTI、affine、DHW/XYZ 和保存回读一致；
- 至少连续 3 个完整 batch 后显存不持续增长，无 OOM、NaN 或残留进程；
- 生成随机、分层和最差样本 montage，人工复核是否存在明显颅外或边界伪影。

任一硬门禁失败时停止，不启动 test。

### 14.5 test 的确定性半量子集

test 不做全量，只评估 test patch 的精确 50%。子集必须在看到生成结果前冻结：

1. 读取 test manifest 并确认总 patch、subject、anchor label、sample role 分布；
2. 按 `anchor_label × sample_role` 分层，以 largest-remainder 配额分配到总数的
   `floor(N/2)`；若当前 N 为 1038，则选择 519 个 patch；
3. 层内按固定 seed `20260806` 与稳定 `relative_path` 的 SHA-256 排序选择，不依赖文件
   枚举顺序；
4. 写出 `subset_manifest.json`，记录选择规则、seed、split hash、完整入选路径、各层配额、
   subject 覆盖和 complement 统计；
5. subset manifest、checkpoint SHA-256、cluster hash、inference config hash 和 Git SHA
   一经冻结，不得根据 test 输出修改。

半量 test 使用两张 GPU 独立分片，而不是 batch size 1 的 DataParallel。将冻结 manifest
按稳定顺序交错分成两个互斥 shard；预期为 260/259 个 patch。两个进程使用相同模型、
EMA、cluster、schedule 和随机种子契约，输出到独立 shard 目录。合并前必须验证：

- 两个 shard 无重叠；
- 合集与 subset manifest 完全一致；
- 每个 patch 只有一份结果；
- 失败 shard 可按 manifest 原位 resume，不重复已完成样本；
- 合并结果明确标记为“test 50% 确定性子集”，不得表述为全量 test。

### 14.6 输出目录与预计成本

```text
experiments/20260805_exp005_gli_formal_training_baseline/outputs/
  patch_64x64x32/seed_20260805/
    validation_best_ema/
    resume_preflight/
    inference_qa_val/
    test_subset_50/
      subset_manifest.json
      shard_0/
      shard_1/
      merged/
```

按已有 5-call smoke 线性估算，519 个 patch 的完整 `t_T=300` 双 GPU test 约需
`10–13` 小时，预计输出 `2–4 GiB`。实际时间和峰值显存以 val 完整 QA 实测为准。

### 14.7 2026-08-06 正式门禁结果与半量 test 启动记录

正式 checkpoint、validation、resume 与 val inference QA 已按本节冻结契约完成：

- 固定 checkpoint 为 `best.pt`，SHA-256 为
  `2d554fd10ce2671ffe6b62231c180e91e5809ce83c8a590f374475c563a37d69`；固定读取
  schema 2 的 `ema` 权重，step 为 `46000`，未回退到 raw `model`。
- best EMA 固定 validation 覆盖 `1032` patch、`73` subject、4 个 anchor label 和
  `3207` 个有效单元。复算 total loss 为 `0.0965080350`，与 checkpoint 记录
  `0.0965079195` 的绝对误差为 `1.16e-7`，小于 `1e-6` 门限；NETC/SNFH/ET/RC
  loss 分别为 `0.108614461/0.073248001/0.126505197/0.087730055`。
- `latest.pt` 被独立恢复两次，均执行 `0` 次 optimizer update；两次恢复后的下一 batch
  指纹均为 `0eee8b5...`，证明 sampler/batch offset 与 RNG 恢复一致。该检查没有继续训练，
  也没有创建 W&B run。
- val QA manifest 冻结为 8 个 patch，覆盖 label 1/2/3/4 与 interior/boundary；7 个 subject，
  且每个入选 patch 均包含四个 lesion label。manifest SHA-256 为
  `51a67b56...`。
- `t_T=5` wiring smoke 与同一批样本的正式 `t_T=300` QA 均通过。正式 QA 共执行
  `8 × 300` 次 model call，耗时 `155.945 s`，推理峰值显存 `1087.46 MiB`；最后三个
  batch 的 allocated/reserved span 分别为 `1/0 MiB`，判定显存稳定。
- inference 内部通道 shape 恒为 `[4,32,64,64]`，顺序严格为
  `1=NETC, 2=SNFH, 3=ET, 4=RC`；输出 DHW/XYZ 分别为 `[32,64,64]` 与
  `[64,64,32]`。全部输出有限，lesion support 外体素数为 0。
- 8/8 样本的 healthy-brain 与 explicit-support 外相对输入变化均精确为 0，boundary
  outer-shell 最大变化也为 0。一个 boundary patch 的输入本身在 support 外有
  `0.7448%` 非零体素；输出逐体素原样继承且未扩大。代表性 montage 目检未见推理新增的
  颅外或边界伪影，因此该输入继承现象不构成 RePaint 背景污染 blocker。

全部 val 门禁通过后，于 2026-08-06 03:42 CST 自动启动冻结的 test 50% 子集：

- test 总数 `1038`，确定性选择 `519`；覆盖 `73` 个 subject。anchor label 计数为
  `1:74, 2:174, 3:129, 4:142`，sample role 为 `boundary:260, interior:259`。
- subset manifest SHA-256 为
  `295b20a01327dcd0071058efd8fe854135688a843c1888ef33242a908b6c3698`。
- GPU 0/1 分片分别为 `260/259`，交集为 0，并集严格等于 519 个冻结样本；两个进程均固定
  使用相同 `best.pt/ema`、cluster、`t_T=300` 与逐样本 seed 契约。
- 启动检查时两张 GPU 均约占用 `1753 MiB`、利用率约 `91%`，两分片均已完成并落盘首个
  样本。该运行没有登录或创建 W&B run，也不会扩展到剩余 519 个 test patch、其他 seed
  或 `80×96×80`。

按完整 val QA 的实测吞吐估算，双 GPU 半量 test 约需 `1.4–1.8` 小时；最终耗时、磁盘、
完整性与汇总指标必须等待两个 shard 完成并执行合并审计后记录，不得把当前“已启动”表述为
“test 已完成”。
