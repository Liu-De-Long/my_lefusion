# 实验变更记录

## 实验 ID

20260804_init_docs_v1

## 日期

2026-08-04

## 目标

初始化 BraTS2024 GLI 迁移实验工作区的项目管理文档。

## 方法

- 检查复制后的 `my_experiment` 仓库结构。
- 检查已有 README。
- 确认当前训练和推理入口。
- 检查数据集工厂支持情况。
- 检查 EMIDEC 训练、推理、数据集和配置文件，将其作为迁移参照。
- 创建 `README.md`、`STATUS.md` 和 `CHANGELOG.md`。

## 结果

已创建项目管理入口文档。本次实验未运行模型训练，也未执行 GLI 迁移代码。

## 结论

当前工作区已经具备可追踪实验管理入口。下一项技术任务是实现并注册 GLI 数据集，同时定义模态、标签、通道和直方图约定。

---

## 实验 ID

20260804_gli_data_audit_v1

## 日期

2026-08-04

## 目标

检查 BraTS2024 GLI 数据集是否已经存在、目录结构是否可用于迁移、数据规模和样本格式是否适配 LeFusion 后续数据集实现。

## 方法

- 使用 `ssh-remote-python-workspace` 在远端只读检查 `/workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI`。
- 统计顶层结构、数据大小、病例数量、NIfTI 文件数量和各模态数量。
- 抽样读取 NIfTI header，确认尺寸、spacing 和数据类型。
- 抽样读取训练分割标签唯一取值。
- 使用一次性只读检查脚本和远端 shell 命令完成检查，并将详细结果记录到 `docs/20260804_001_brats2024_gli_data_audit.md`。
- 检查结论记录完成后，删除不会复用的临时脚本，避免干扰项目结构。

## 结果

- 数据集总大小约 `182G`，总文件数 `8859`，其中 NIfTI 文件 `8857`。
- `train` 有 `1621` 个病例目录，每例包含 `seg/t1c/t1n/t2f/t2w`。
- `val` 有 `188` 个病例目录，每例包含 `t1c/t1n/t2f/t2w`，未发现 `seg`。
- 抽样尺寸为 `182 x 218 x 182`，spacing 为 `1.0 x 1.0 x 1.0`。
- 抽样标签取值包含 `0,1,2,3,4`，单个病例不一定包含所有病灶类别。
- 未发现 `.npy`、`.npz`、`.pkl`、`.pt`、`.pth`、`.h5`、`.hdf5` 等 LeFusion 预处理缓存格式。

## 结论

该数据目录是 BraTS 发布格式的 NIfTI 数据，不是 LeFusion 已处理缓存。后续迁移应基于 `train` split 实现 GLI dataloader，并在配置中固定模态顺序为 `t1c, t1n, t2f, t2w`。在实现前还需要确认标签 `1,2,3,4` 的官方语义，并决定 GLI 标签到 LeFusion 多通道病灶表示的映射方式。

## 下一步

确认 GLI 标签语义和通道映射，然后实现 `LeFusion/dataset` 下的 GLI 数据集类与 `get_dataset.py` 注册逻辑。

---

## 实验 ID

20260804_gli_patch_stats_preprocess_v1

## 日期

2026-08-04

## 目标

统计 BraTS2024 GLI/PTG 训练集病灶区域大小和模态病灶强度分布，为 LeFusion 数据接入选择 patch 尺寸、裁剪方式、缩放方式和 histogram 条件设计。

## 方法

- 新增正式可复用脚本 `scripts/brats_gli_lesion_patch_stats.py`。
- 对 `train` split 的 `1621` 个病例做全量 mask/bbox 统计，输出到 `experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/region_patch_stats_full`。
- 对前 `100` 个病例做四模态病灶强度抽样统计，输出到 `experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/modality_intensity_stats_100cases`。
- 按 BraTS post-treatment glioma 语义暂定标签：`1=NETC`、`2=SNFH`、`3=ET`、`4=RC`。
- 将 LeFusion 论文中的 lesion-focused loss、histogram texture control 和 multi-channel decomposition 约束整理为预处理方案。

## 结果

- `WT=1+2+3` 的 p90 bbox 为 `95 x 120 x 91`，p95 bbox 为 `103 x 129 x 98`。
- `TC=1+3` 的 p90 bbox 为 `59 x 74 x 64`，p95 bbox 为 `68.6 x 86 x 73.6`。
- `ET=3` 的 p90 bbox 为 `59 x 74 x 64`，p95 bbox 为 `68 x 86 x 73`。
- 包含 `RC` 的整体异常区域 p95 patch 建议为 `112 x 144 x 104`。
- 100 例模态强度抽样显示四个 MRI 模态强度尺度差异明显，不能直接共享一个全局 intensity range。
- 详细方案记录到 `docs/20260804_002_brats2024_gli_patch_preprocessing_plan.md`。

## 结论

GLI 第一版接入建议使用 3D mask-centered crop，不做空间重采样；主配置采用 `patch_size=[112,144,104]` 覆盖 `WT` p95。强度预处理应按病例、按模态对非零脑区做 clip 和归一化。LeFusion v1 建议先使用 `WT` 单通道和四模态 histogram 条件，`cond_dim=64`，通过小规模 overfit 验证流程后再扩展到 `SNFH/TC/RC` 或 `NETC/SNFH/ET/RC` 多通道分解。

## 下一步

复核官方标签语义后，实现 GLI dataset v1，并在 dataloader 中记录 patch 截断比例和 histogram 条件维度。

---

## 实验 ID

20260804_exp001_t1c_local_patch_dataset

## 日期

2026-08-04

## 目标

构建 BraTS 2024 GLI 的 T1c 单模态局部病灶 patch 数据集，为 NETC、SNFH、ET、RC 四类病灶分别提供内部纹理和边界纹理样本。

## 方法

- 新增可复用的确定性 patch 裁剪器 `scripts/brats_gli_crop_local_patches.py`。
- 使用 `train` split 的 1621 例有标签病例，按患者组做 80/20 划分。
- 固定输入模态为 `t1c`，保留标量 segmentation，后续 loader 再展开为 NETC/SNFH/ET/RC 四通道 lesion mask。
- 生成 `64×64×32` 与 `80×96×80` 两种 patch，每种尺寸共享同一份患者划分。
- 每个存在的 anchor label 生成 interior 和 boundary 两类样本。
- 按非零 foreground 做 `p0.5-p99.5` clip 并归一化到 `[-1,1]`。
- 每个标签计算 16-bin histogram，保存为四标签 histogram 条件。
- 使用 staging 目录完成构建、QA 和逐 NPZ 完整性验证后原子发布。

## 结果

- 输出目录：`/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches`。
- 输入病例 1621 例，归组为 731 个患者；train 584，val 147，无患者泄漏。
- `64×64×32`：9842 个 patch，padding patch 201（2.04%）。
- `80×96×80`：9842 个 patch，padding patch 1545（15.70%）。
- 两种尺寸合计 19684 个 NPZ；NPZ 逻辑压缩容量约 16.0 GiB，JuiceFS `du` 分配量约 65 GiB。
- 每种尺寸 anchor 数一致：NETC 1412、SNFH 3236、ET 2446、RC 2748。
- 每种尺寸 interior/boundary 各 4921 个样本。
- 失败病例 0，归一化退化病例 0。
- 每种尺寸生成 16 张 QA 图，最终逐 NPZ 验证通过，无临时文件或 staging 残留。

## 结论

数据集已完成并通过发布门禁。当前有效路线从旧的 `WT` 单通道四模态方案调整为 T1c 单模态图像加 NETC/SNFH/ET/RC 四通道 lesion mask 的局部 patch 方案；两种 patch 尺寸可直接用于后续 patch size 对照和 GLI loader 接入。

## 下一步

实现 GLI loader：读取单份 T1c 与标量 segmentation，在加载阶段展开 NETC/SNFH/ET/RC 四个 lesion channel，并验证 `[X,Y,Z]` 到 `[C,D,H,W]` 的轴转换与 histogram 条件拼接。

---

## 实验 ID

20260805_exp002_gli_loader_t1c_multilesion

## 日期

2026-08-05

## 目标

实现并注册 BraTS2024 GLI T1c 局部 patch loader，参考原始 EMIDEC 的 scalar label 语义，同时暴露四通道 lesion mask 和 64 维 histogram 条件。

## 方法

- 在 `LeFusion/dataset/gli_hist.py` 实现可复用 loader。
- 保留 `label` 为 scalar segmentation，新增 `lesion_mask` 为 NETC/SNFH/ET/RC 四通道二值 mask。
- 将 NPZ 的 `[X,Y,Z]` 按 `[channel,z,x,y]` 转换为 LeFusion 的 `[C,D,H,W]`。
- 在 `get_dataset.py` 注册 `gli`，新增 focused tests 和实验配置。
- 本轮不启动训练。

## 结果

代码已提交到 `feature/20260805-exp002-gli-loader`，commit 为 `fc7d7e1`。本地语法检查通过；远端 focused tests 4/4 通过，包含真实发布数据两种 patch 尺寸迭代；全量测试 10 项通过、1 项因本地未设置真实数据路径跳过。

## 结论

GLI loader 的 scalar label 与四通道 lesion mask 语义已分离，避免将原始 EMIDEC 的前景选择逻辑错误套用到 GLI。当前训练 loss 和推理入口仍未改造。

## 下一步

后续将 GLI loss、Trainer 传递和矩形 patch 支持作为独立方法实验；本次未启动训练。

---

## 文档整理

## 日期

2026-08-05

## 目标

将已实施的 GLI loader 设计、scalar label 兼容语义、四通道 lesion mask、轴转换、padding metadata 和验证标准整理为可复用方案文档。

## 结果

新增：`docs/20260805_001_gli_loader_implementation_plan.md`。

## 结论

方案文档已与 `20260805_exp002_gli_loader_t1c_multilesion` 实验记录关联，后续 GLI loss、训练和推理接入可按该文档继续拆分实验。
