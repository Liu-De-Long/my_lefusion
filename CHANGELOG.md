# 实验变更记录

## 2026-08-06 — 记录 p64 少标签逐体素亚区分类方案

- 新增长期方案 `docs/20260806_001_gli_p64_weak_supervision_classifier_plan.md`。
- 将“10% 且约 1000 Case”解释为从冻结 train pool 中确定性选择 1000 个 p64 patch，
  每个 anchor 类 250，并在类别内平衡 interior/boundary 和患者贡献；同时保留严格 984 个
  patch 作为实施前可选口径。
- 首个推荐模型改为无标签泄漏的空间特征 MLP，并保留单强度 MLP、局部邻域 MLP 和轻量
  3D CNN 对照；不因模型较小而假设单体素 T1c 足以区分四类。
- 暂定 ET 与 RC 为 T1c 下两个重点亚型，成功门禁为患者等权 focus mIoU 不低于 85%，
  同时继续报告四类完整指标和小区域表现。
- 明确 NPZ histogram、四通道 lesion mask、anchor label 和 per-class voxel count 不得进入
  模型输入；剩余未标签 train patch 首阶段不参与优化。
- 本次只修改中文方案与状态文档，没有修改代码、创建实验卡、登录 W&B、启动训练或占用 GPU。

## 2026-08-06 — 停止错误训练契约并启动 exp008 条件式修复方法

- QA 确认 exp005 的 denoiser 没有接收挖空 T1c 和四通道 lesion mask 空间条件；旧 p64/p80
  checkpoint 不满足伪病灶生成目标，exp006/exp007 结果同步降级为失败审计。
- 按用户要求停止 p80 主进程与两个数据加载子进程；最后完整 checkpoint 为 step `11500`，
  GPU 0/1 已释放至 `1 MiB`、`0%`，未删除任何 checkpoint。
- 新建方法实验 `20260806_exp008_gli_conditional_inpainting_training` 和分支
  `feature/20260806-exp008-gli-conditional-inpainting`。
- 数据 loader 新增病灶 union 内置零、洞外保持原值的单通道 `masked_context`；denoiser 空间输入
  改为四通道 `x_t`、一通道 masked T1c 与四通道 lesion mask，输出仍为四通道噪声预测。
- 训练、validation、preflight 与 RePaint 共用该空间条件，hist 保持四组 16-bin、loss 保持
  仅在对应病灶 mask 内计算。
- 实现 commit：`c117bc74519c4329ff721817d31b3e2171be90b2`。
- 兼容与 preflight 门禁修复后的运行版本：`6e29f355294c762cfb8928f73abcb350200c2e70`。
- 远端真实 patch 全套回归 `34/34` 通过；online preflight 完成 0 次 optimizer update 的真实
  batch 前/反向、完整 validation 和 checkpoint resume，反向峰值显存 `23344.58 MiB`。
- preflight W&B run：`exp008-p64-preflight-s20260805`；临时 checkpoint 已自动删除。
- 首次 preflight 被旧 `exp005-` ID 硬编码在计算前拦截，已泛化门禁并补回归测试。
- `2026-08-06 08:28:25 UTC` 启动且只启动 exp008 p64 seed `20260805` 正式训练；W&B run
  `exp008-p64-s20260805` 已 online，同步流已记录约 164 个 optimizer step 和有限 loss。
- 启动核验时两张 GPU 均参与计算，显存约 `13383/13161 MiB`；未启动 p80、其他 seed 或 test。
- 首个约 `580.4 MB` 的 `latest.pt` 已按 500-step 规则写入，写入后双 GPU 训练继续正常。
- 清理旧 exp007 隔离 worktree 中仅存的 `.hydra/config.yaml`、`hydra.yaml`、`overrides.yaml`
  临时解析产物，并注销 `/workspace/LeFusion_v2/.exp007_qa_worktree`；该 worktree 无源码改动，
  旧实验输出与 checkpoint 均未删除。
- 已删除本地与远端一次性 `.tmp_inspect_checkpoint.py`；它仅用于只读记录停止点，不是实验资产。

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

---

## 文档整理

## 日期

2026-08-05

## 目标

将 GLI lesion-aware 训练接入、四通道 loss、DHW shape 约定和验收门禁整理为跨实验可复用的长期方案。

## 结果

新增：`docs/20260805_002_gli_lesion_aware_training_integration_plan.md`。

文档覆盖训练入口、Trainer 数据传递、非空 `(sample, lesion channel)` 等权 loss、64×64×32 与 80×96×80 shape 约束、配置隔离和 smoke 验收标准。

## 结论

该方案文档作为后续 GLI 训练和矩形 patch 实验的共同接口说明；单次运行结果仍保留在各实验目录的 `result.md` 中。

## 下一步

后续 GLI inference loader、RePaint keep-mask 和输出合并继续使用独立方法实验管理。

---

## 实验 ID

20260805_exp003_gli_lesion_aware_training

## 日期

2026-08-05

## 目标

实现 GLI 四通道 lesion-aware loss、Trainer 的 `lesion_mask` 传递和完整三维空间 shape 支持，并验证 `64×64×32` 与 `80×96×80` 两种 patch 的训练接入可行性。

## 方法

- 训练入口接受 `gli`，解耦 `base_dim` 与 `spatial_shape_dhw`。
- GLI loss 对所有非空 `(sample, lesion channel)` 单元等权平均，每个单元先按自身病灶 voxel 数归一化。
- histogram 使用当前训练 device，不再写死 `.cuda()`。
- 参数化 temporal relative-position `max_distance`，GLI 配置统一使用 `128`。
- 增加两份 Hydra 配置、可复用 smoke 工具和 loss/shape/Hydra 集成测试。
- 推理入口改用 diffusion 的完整 sample shape，但没有开放 GLI RePaint 推理。

## 结果

- 代码 commit：`dbeabd5e1d9f6f35fc0348031c800043e554585f`。
- 新增集成测试 5/5 通过；项目全量测试 16 项通过；真实发布数据 loader 4/4 通过。
- 两份 Hydra 配置均通过 `--cfg job --resolve`，无未解析字段。
- `64×64×32`：真实输入 `[1,4,32,64,64]`，固定 batch 20 步首 5 步 loss 均值 `0.785975`、末 5 步 `0.360158`，峰值显存 `6309.063 MiB`。
- `80×96×80`：真实输入 `[1,4,80,80,96]`，单步 loss `0.852194`，前向/反向通过，峰值显存 `33944.751 MiB`，未 OOM。
- W&B 64 patch run：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/23hogegm>
- W&B 矩形 patch run：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/228g4h4j>
- 初次在线启动因远端没有已配置凭据而在模型计算前失败；随后以 offline 模式完成 smoke，并在用户提供临时环境凭据后同步成功。该凭据未写入项目文件、配置或 Git。
- 同步完成后删除 `experiments/20260805_exp003_gli_lesion_aware_training/outputs/` 下的 W&B offline 临时缓存；未生成 checkpoint，因此整个临时 outputs 目录可安全删除。

## 结论

`64×64×32` 已通过固定 batch overfit smoke；`80×96×80` 已通过真实单 batch 前向和反向，当前 A100 80GB 上 batch size 1 的峰值显存约 `33.15 GiB`。两种尺寸的训练接口均可行，但正式训练仍未启动，当前结果不属于模型性能结论。

## 下一步

先复核 GLI 标签医学语义，再确认正式训练 batch size、gradient accumulation、验证策略和在线 W&B 环境；GLI RePaint inference 作为后续独立方法实验实施。

---

## 实验 ID

20260805_exp004_gli_inference_closed_loop

## 日期

2026-08-05

## 目标

在不启动正式训练的前提下，完成 GLI patch 级 validation checkpoint、train-only histogram cluster、逐 timestep RePaint、四通道病灶合成、单通道保存和 normalization QA 闭环。

## 方法

- 保留原 584 个 train subject，将原 147 个 holdout subject 确定性拆分为 73 val 和 74 test。
- scalar segmentation 作为事实源，确定性展开四通道 lesion mask；逐 batch 校验标签映射和互斥性。
- 四个 lesion channel 在每个反向转换中共享同一条固定 noise 的前向扩散 T1c 背景轨迹；终态背景直接取 sampler 的共享状态，不执行采样结束后的原图硬覆盖。
- 在 train patch 上按 label 独立聚类 16-bin histogram，并按 `(subject,label)` 归一权重；inference 选择最近 cluster center。
- 从四模态原始体积构建显式 support，仅用于 QA、分区指标和 lesion 合法性，不参与 RePaint。
- 分别生成 64 patch 20-step 和矩形 patch 1-step validation checkpoint，并在 test split 上执行缩减 `t_T=5` inference smoke。

## 结果

- 实现 commit：`bcf9097a6ccbd20db6ab6d992ae72872c5ea65dd`；分支：`feature/20260805-exp004-gli-inference`。
- 全量测试 22/22 通过；四份 Hydra 配置解析通过且无未解析字段。
- split 为 `584/73/74` 且零交集；两种 patch 的 NETC/SNFH/ET/RC cluster 均选择 `k=[5,3,2,2]`。
- 64-subject normalization 审计四项门禁全部通过：median Dice `0.999997`、p05 `0.990682`、median extra `0`、p95 extra `0.2444%`。最差和随机 montage 的视觉检查未见系统性颅外伪影，保留当前 normalization。
- 64 validation checkpoint：20-step overfit 首 5 步 loss `0.785975`、末 5 步 `0.360158`，峰值显存 `6309.063 MiB`。
- 矩形 validation checkpoint：1-step loss `0.852194`，峰值显存 `33944.751 MiB`。
- 64 inference：5 次模型调用、`2.6067 s`、`1083.212 MiB`；矩形 inference：5 次模型调用、`4.6900 s`、`6518.637 MiB`。两者四通道/单通道 shape、XYZ/DHW、affine 和 NIfTI round-trip 均通过。
- 首次 online W&B 初始化因远端没有环境凭据而在模型计算前失败；随后以 offline 模式完成，run ID 为 `30rwobgb` 和 `30wigck6`，暂无线上的 run URL。

## 清理

- 删除 Hydra 配置解析在远端仓库根生成的临时 `.hydra/config.yaml`、`.hydra/hydra.yaml` 和 `.hydra/overrides.yaml`；这些文件仅是解析副产物，不是实验资产。
- 未删除 validation checkpoint、W&B offline run、cluster、normalization QA 和 inference 输出；它们仍是当前闭环的有效远端资产，并由 `.gitignore` 排除。

## 结论

GLI patch 级训练 checkpoint→cluster condition→RePaint→四通道合成→单通道保存闭环已打通，64 与矩形 patch 均可行。当前 checkpoint 和缩减 schedule 只证明接口正确，不能作为正式模型质量结论；正式训练仍需官方标签语义与 W&B 在线记录。

## 下一步

复核标签医学语义并配置新的远端 W&B 凭据；只有在用户另行确认正式训练配置后，才启动长期训练和完整 RePaint 质量评估。

---

## 文档整理

## 日期

2026-08-05

## 目标

记录 BraTS2024 GLI 正式训练前的标签确认、normalization、类别平衡、训练参数、validation/test 隔离、checkpoint/resume、W&B 安全接入和项目管理决策。

## 方法

- 使用 BraTS 官方评测说明确认 `0=background、1=NETC、2=SNFH、3=ET、4=RC`。
- 核对 patch、四通道 mask、histogram、loss、cluster 和 inference channel 顺序。
- 复核 exp004 normalization support audit，决定保留当前 normalization。
- 对两种 patch 的 train/val/test 做 anchor、lesion、subject、role 和多标签分布统计。
- 比较自然采样、anchor 平衡、`(subject,label)` 平衡和分层 sampler，推荐分层 sampler baseline。
- 设计两阶段正式训练参数、validation、checkpoint、resume、early stopping、三 seed 复现和 W&B online 安全流程。

## 结果

- 新增长期方案：`docs/20260805_004_gli_formal_training_plan.md`。
- 标签语义证据充分，不再是 blocker。
- 保留当前 `T1c != 0 -> p0.5/p99.5 -> clip -> [-1,1]` normalization。
- 推荐先训练 `64×64×32`，再将 `80×96×80` 作为第二阶段对照。
- 推荐新实验 ID：`20260805_exp005_gli_formal_training_baseline`。
- 推荐分层 sampler；保持现有 loss 不变，但该训练流程变化需要新 branch 和 Git commit。
- 用户已说明在 F 盘准备新的 W&B key；本次未读取、未显示、未写入项目，也未登录或创建 run。

## 结论

当前仍不能直接启动正式训练。正式训练前必须先实现 sampler、validation、完整 checkpoint/resume、W&B online fail-closed 和 preflight，并统一 normalization audit provenance。

## 下一步

等待用户确认方案文档中列出的 experiment ID、branch、sampler、训练参数、复现次数和 W&B entity；确认后先实施代码与测试，不自动启动训练。

---

## 文档整理

## 日期

2026-08-05

## 目标

保留并同步扩写后的 GLI patch 级 inference 闭环长期接口文档。

## 结果

- 扩写 `docs/20260805_003_gli_inference_closed_loop.md`，补全 test-only loader、显式 brain support、histogram cluster provenance、逐 timestep RePaint、四通道输出合成、NPZ/NIfTI 保存回读和分阶段验收方案。
- 将文档中旧的标签待确认描述更新为已经官方确认的 `0=background、1=NETC、2=SNFH、3=ET、4=RC` 契约。
- 本次只整理并同步文档，不修改 inference 代码、不登录 W&B、不创建 run、不启动训练。

## 结论

该文档作为 exp005 正式 checkpoint 验证及后续生成质量评估的长期接口依据继续保留。

---

## 实验 ID

`20260805_exp005_gli_formal_training_baseline`

## 日期

2026-08-05

## 目标

在不登录 W&B、不创建 run、不启动训练的前提下，实现 GLI 正式训练所需的采样、验证、恢复和在线日志门禁。

## 方法

- 实现按 `anchor_label × sample_role` 分层、层内按 subject 平衡的确定性 sampler，保持现有 lesion-aware loss 不变。
- 新增固定 timestep/noise 的监督 val，记录 EMA 总 loss、NETC/SNFH/ET/RC loss、各通道和总有效单元数、patch/subject/anchor label 覆盖。
- 实现 early stopping、原子 `latest.pt`、`best.pt`、最近三个 milestone 和完整 checkpoint/resume。
- W&B 使用固定 entity/project/run ID 与 `resume=never/must`，online 初始化失败时 fail closed，不读取或保存 key。
- 新增两种 patch 的正式 Hydra 配置；参数保持可配置，50,000 step 作为上限而非必须跑满。

## 结果

- 实现代码版本：`7a288dc2f59947db2c8bf00200a59f19f6e0bc7a`。
- focused tests 首轮通过；全量测试曾发现旧 `SimpleNamespace` factory 兼容问题，已在 `cd2affb` 修复。
- 直接训练脚本配置解析曾发现 `train.tracking` 包解析问题，已在 `4ee7ebf` 修复。
- 最终远端全量非训练测试 28/28 通过；两份正式配置解析通过且无 `???`。
- exp004 normalization audit stale 字段已改为人工 QA 完成并保留当前 normalization，JSON 回读通过。
- 本轮没有登录 W&B、创建 run、执行 GPU preflight 或启动训练。

## 结论

exp005 已具备进入独立 preflight 的代码基础，但仍不允许正式全量训练。micro-batch 可根据显存调整并用 accumulation 保持 effective batch；三个 seed 是本项目复现候选，不是原始 LeFusion 的强制规则；训练可在 validation 收敛时提前停止。

## 下一步

用户安全配置远端 W&B key 后，另行授权 W&B online、显存、validation 和 resume preflight；preflight 通过后仍需再次确认才能启动首个正式 run。

---

## 实验 ID

`20260805_exp005_gli_formal_training_baseline`

## 日期

2026-08-05

## 目标

在不启动正式训练的前提下，执行已授权的 W&B online、64 patch 显存、完整 validation 和 checkpoint resume preflight。

## 方法

- 新增可复用 `scripts/gli_formal_training_preflight.py`，使用独立 preflight W&B run ID。
- 对 `64×64×32` 的 `batch=4/accum=1` 仅执行一次 fixed-batch forward/backward，不调用 optimizer/scaler step；随后运行完整固定 validation、保存/重载临时 checkpoint 并验证 resume。
- 成功后自动删除临时 `resume_preflight.pt`，保留轻量 metrics 和日志；不生成正式 checkpoint。

## 结果

- 新 W&B key 已在远端安全可用，成功 run：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-preflight-s20260805-r3>。
- 0 次 optimizer update；batch `[4,4,32,64,64]`；反向峰值显存 `23330.56 MiB`；完整 validation 峰值 `5482.45 MiB`。
- validation：1032 patch、73 subject、4 label、3207 有效单元；EMA total loss `0.959372`；NETC/SNFH/ET/RC loss `0.893387/0.904111/1.140945/0.887598`。
- checkpoint resume 恢复成功；临时 checkpoint 已删除。
- 首次 preflight 因本地 SSH 会话传输中断而留下未完成 online run；第二次 retry 暴露合法负 `origin_xyz` 被 loader 拒绝。已修复该 bug（`b34d867f944343f9f6ff6e4edde5abe3a0b0805b`），修复后真实 patch 根目录下远端全量测试 28/28 通过。

## 结论

64 patch 的所有技术门禁已通过。该 preflight 不构成正式训练；正式 50,000-step seed `20260805` 仍需用户单独授权。

## 下一步

等待用户决定是否启动首个正式 `64×64×32` run；不自动启动另外两个 seed 或 80 patch 对照。

---

## 正式训练授权

## 日期

2026-08-05

## 结果

- 用户授权仅启动 `64×64×32` 的正式 seed `20260805`，明确禁止自动扩展至其他 seed 或 `80×96×80` 对照。
- 远端确认 GPU 0、1 均为空闲 A100 80GB；正式配置改为 DataParallel，保持全局 batch 4、每卡 micro-batch 2、accumulation 1、effective batch 4。
- 运行代码版本固定为 `75b187ce6664d4fa44ddad32dd328112998ff5c4`。
## 2026-08-05 — 启动 exp005 首个正式双 GPU 训练

- 经用户授权，已启动且仅启动 `20260805_exp005_gli_formal_training_baseline` 的 `64×64×32`、seed `20260805` 正式训练；未启动其他 seed 或 `80×96×80` 对照。
- 运行使用 GPU 0、1 的 `DataParallel`，全局 batch 为 4（每卡 2），运行时 Git HEAD 为 `ea88464f3bf51350f5bd7d33f1bfcc4d7f80b6c1`，实现/配置版本为 `75b187ce6664d4fa44ddad32dd328112998ff5c4`。
- 已确认 W&B online run：`exp005-p64-s20260805`，链接为 https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-s20260805 。
- 正式输出目录为 `experiments/20260805_exp005_gli_formal_training_baseline/outputs/patch_64x64x32/seed_20260805/`；运行日志为其中的 `train.run.log`。

## 2026-08-06 — 修订 exp005 正式闭环 QA 与半量 test 方案

- 用户固定使用正式 `best.pt` 的 `ema` 权重，并授权 schema 2 checkpoint inference 兼容、QA 指标、双 GPU test 分片和 Git 提交。
- 将正式门禁写入 `docs/20260805_003_gli_inference_closed_loop.md`：先复核 best EMA validation 和 actual-checkpoint 零更新 resume，再在 val 上执行完整 `t_T=300` 闭环 QA。
- 只有全部 val 门禁通过才自动开始 test；test 范围由全量改为确定性 50% patch 子集。
- 半量子集按 `anchor_label × sample_role` 分层，以固定 seed `20260806` 和稳定路径 SHA-256 排序选择精确 `floor(N/2)`，冻结 subset manifest 后由 GPU 0、1 独立分片运行。
- test 输出必须标记为“test 50% 确定性子集”，不得外推或表述为全量 test。
- 实现提交：`9662ba1e3695191c0c368f55f3f62a7bca1a080a`；该提交只扩展正式 checkpoint 评估基础设施，不改变已完成训练的模型和 loss。

## 2026-08-06 — exp005 正式 val QA 通过并启动 test 50% 子集

- 固定使用 schema 2、step 46000 的 `best.pt/ema`；固定 val 复算 loss 为
  `0.0965080350`，与 checkpoint 记录误差 `1.16e-7`，完整覆盖与 resume 门禁通过。
- 8 个分层 val patch 的正式 `t_T=300` QA 通过：四通道顺序、shape、explicit support、
  RePaint 闭环、有限值、背景不变性和显存稳定性均满足硬门禁；代表性 montage 目检无推理
  新增明显背景污染。
- 冻结 test 50% manifest：1038 中选择 519，SHA-256 为
  `295b20a01327dcd0071058efd8fe854135688a843c1888ef33242a908b6c3698`；两分片
  260/259、交集为 0、并集等于冻结子集。
- 已于 2026-08-06 03:42 CST 启动 GPU 0/1 两个独立 shard；未启动全量 test、其他 seed
  或 80 patch，也未为 inference 创建 W&B run。

## 2026-08-06 — 启动 exp006 原始 LeFusion RePaint 语义对齐

- 经用户确认，将 GLI inference 中额外的 post-denoiser `target_background` hard clamp 删除；
  保留 denoiser 前的真实背景前向加噪注入。
- 对齐原始 LeFusion 分支：背景噪声改为每次反向调用重新采样，单次调用内四个 lesion channel
  共享，不再缓存整条轨迹的固定 noise。
- 不重新训练，固定复用 exp005 step 46000 `best.pt/ema`、train-only cluster、8 例 val
  manifest 和 519 例 test 50% manifest。
- 新增 exp006 独立配置与输出目录；exp005 的 519 例 hard-clamp 输出保留作错误版本对照，
  不删除、不覆盖。
- QA 从“病灶外必须严格零变化”改为记录 healthy/support 外/outer-shell 的 MAE、p95、最大值、
  变化比例和 lesion/healthy 边界 jump；exact 字段只保留用于新旧对照。
- 新增可复用 `scripts/gli_audit_inference_subset.py`，用于审计两个 shard 的无重叠无遗漏、
  channel/role 分组、support 约束、异常样本和 exp005/exp006 同路径配对差异。

## 2026-08-06 — exp006 完整 val 与 test 50% 子集验收通过

- 远端真实发布 patch 环境全量回归 `31/31` 通过；8 例完整 `t_T=300` val QA 的 healthy、
  support 外和 outer-shell MAE 均约 0.009，边界 jump p95 最大增量 `0.004737`，显存稳定。
- 同一冻结 manifest 的 519 例 test 在 GPU 0/1 完成 260/259 两个 shard；交集 0、并集严格
  等于 manifest，每例 300 calls，checkpoint/EMA/cluster/channel/shape provenance 一致。
- test healthy/support 外/outer-shell 的变化大于 0.1 比例均为 0；边界 jump p95 增量
  mean/p95/max 为 `-0.001774/0.004392/0.014357`，最差样本图未见明显背景污染或接缝。
- NETC/SNFH/ET/RC 的 lesion change MAE 为
  `0.010038/0.010379/0.010303/0.010995`；histogram L1 为
  `0.476755/0.510642/0.439694/0.688289`。
- exp005 旧版 519 例（5.4 GiB）未删除；exp006 新版约 6.6 GiB。step 46000
  `best.pt/ema` 与 exp006 inference 语义冻结为 64 patch 工程 baseline，不外推为医学有效性
  或全量 test 结论。

## 2026-08-06 — 启动 exp007 显式挖空输入 8 例 QA

- 用户授权只复用冻结的 8 例 validation QA，禁止运行 519 例 test 子集或全量 test。
- 新建方法实验 `20260806_exp007_gli_masked_input_qa`：两个变体均将真实病灶 union 区域在送入 LeFusion 前显式置零。
- `masked_multilabel` 保留原四通道 mask 与逐标签最近 train-only cluster；`masked_anchor_union` 将完整病灶 union 赋给 patch 的 `anchor_label` 通道，其他 mask 通道和 condition block 置零。
- QA 固定输出原始输入、挖空输入、生成输出、绝对 difference 和 conditioning mask 五联图。
- 新增可复用 `scripts/gli_build_qa_contact_sheet.py`，将每个变体的 8 张五联图整理为 2×4 总览图。
- 实现 commit：`000e3735ea1604981a316cfee3a97eb3f17c6e5a`。
- 检查时 p80 正在占用 GPU 0/1；在其结束前只实施与测试代码，不启动本实验 GPU 推理，不切换远端训练工作树。

## 2026-08-07 — exp009 完成 p64 少标签逐体素分类验证

- 在分支 `feature/20260806-exp009-gli-p64-weak-supervision` 实现 leak-safe dataset、M0/M1/M2/C0、
  患者等权指标、原子 checkpoint、验证门禁与一次性 test 封存入口。
- 冻结 1000 个 train p64 patch：四个 anchor 各 250、interior/boundary 各 500、覆盖 480 个
  subject；子集 SHA-256 为 `aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a`。
- 修复 NumPy 指标 JSON 序列化和 CUDA checkpoint RNG 恢复；增加只缓存 T1c/总 mask 派生
  mask 内特征行的运行时优化。最终代码 commit 为 `cf0b3789305ad7cd6c18644c22f57545824e83f7`。
- GPU0 严格串行完成 M0/M1/M2/C0；患者等权 val focus mIoU 分别为
  `0.3385/0.3938/0.3815/0.5436`。C0 最佳 epoch 22，ET/RC IoU 为 `0.4937/0.5935`，
  focus mIoU 95% CI 为 `[0.5000,0.5862]`。
- 四种模型均未通过 0.85 focus mIoU 和 ET/RC 单类 0.80 门禁；未创建 test 结果，冻结 test
  从未运行。结论是单体素/局部 MLP 只适合作为下限，完整三维上下文必要但当前轻量 C0 仍不足。
- classifier 训练未接入 W&B online，已记录为正式日志缺口；本次 checkpoint 仅作验证审计，
  后续新训练必须先补 W&B fail-closed。
- 汇总正式远端 `outputs/` 后删除本地独立 worktree 中约 5.1 MiB 的 `.tmp_exp009_results/`
  临时结果拷贝；远端 checkpoint、history、验证指标和日志均未删除。
- 追加只读训练历史审计：C0 epoch 29 实际 focus mIoU 为 `0.5482`，高于已保存 epoch 22
  checkpoint 的 `0.5436`；确认根因是 `early_stopping_min_delta` 被错误复用于 best 保存判断。
  由于该值仍远低于 0.85，冻结 test 继续封存；本地 `.tmp_c0_history.jsonl` 临时副本在汇总结论
  后删除，远端正式 history 未删除。

## 2026-08-07 — 启动 exp013 多尺度残差与半监督分类方法

- 用户授权保持 exp009 同一冻结 1000 patch，升级多尺度残差 3D U-Net，并允许剩余 train
  patch 进入半监督一致性/伪标签训练。
- exp010–exp012 已被另一个隔离任务占用，因此创建
  `20260807_exp013_gli_p64_multiscale_semisupervised_classifier` 与独立分支/worktree，避免冲突。
- 修复“`early_stopping_min_delta` 同时控制 best 保存”的缺陷：任何绝对 val 新高都保存，
  min delta 只控制 patience 重置。
- 新增两级残差 3D U-Net、dilation 1/2/4 多尺度 bottleneck、focus-weighted CE/focal/Dice、
  T1c/空间增强、W&B online fail-closed 和 mean-teacher 训练框架。
- mean-teacher 未标签 loader 不返回 target，也不使用 anchor/per-class hist/类别体素量；高置信
  伪标签在总 mask 内按类限额，teacher 以 EMA 更新。
- 当前只进入实现与测试阶段；尚未启动 GPU 训练，冻结 test 未运行。
- GPU0 初始完整 p64 零 optimizer-step preflight 通过：base 24、batch 2 峰值 allocated/reserved
  显存为 `609/804 MiB`；据此将正式监督配置调整为 base 32、batch 8，半监督为 labeled 与
  unlabeled batch 各 4，调整后需重新 preflight。
- 调整后监督/半监督完整 p64 GPU0 preflight 均通过：参数量 3,062,912；监督 batch 8 峰值
  `2769/3914 MiB`，半监督双 batch 4 峰值 `2687/3292 MiB`，均为零 optimizer step。
- 半监督真实数据 preflight 冻结 6772 个剩余 train patch，pool SHA-256 为
  `8248585a47003f33cf0f318b4df7170cee457b2f2ed40663d5112496b643057b`，确认未加载 target。
- 初查发现 `WANDB_API_KEY` 环境变量为空且 `wandb status` 未展示认证；后续只检查标准位置，
  确认 `/root/.netrc` 存在 W&B 主机条目，`wandb login --verify` 在线验证当前 entity 成功。
  全程未读取或输出 key，正式训练继续使用 online fail-closed，不回退 offline。
