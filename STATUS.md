# 当前状态

## 当前版本

v0.4.0-gli-inference-closed-loop-smoke

## 当前最佳实验

`20260805_exp004_gli_inference_closed_loop`

## 当前最佳结果

已发布的数据资产仍为两种尺寸各 9842 个 patch；当前最新方法结果是 patch 级 GLI 训练 checkpoint、train-only cluster、逐步 RePaint、四通道合成和单通道保存闭环。该结果是接口 smoke，不是正式模型性能结果。

## 当前流程

当前实验工作区包含一份 LeFusion 代码副本：

```text
/workspace/LeFusion_v2/my_experiment
```

已存在的代码路径：

- 已有 LIDC 训练和推理脚本。
- 已有 EMIDEC 训练和推理脚本。
- 训练入口：`LeFusion/train/train.py`
- 推理入口：`LeFusion/inference/inference.py`
- 数据集工厂：`LeFusion/get_dataset/get_dataset.py`
- 核心扩散模型实现：`LeFusion/ddpm/diffusion.py`

当前迁移目标：

- 将 EMIDEC 风格的多通道、直方图条件 LeFusion 流程迁移到 BraTS2024 GLI。
- 当前优先路线为 T1c 单模态图像输入，并在 loader 阶段将标量 segmentation 展开为 NETC/SNFH/ET/RC 四通道 lesion mask。

## 已完成

- 已创建项目管理入口文档。
- 已确认 `my_experiment` 是一份 LeFusion 实验工作区副本。
- 已确认当前训练和推理路径支持 `lidc` 与 `emidec`。
- 已确认 EMIDEC 使用双通道重复图像/标签张量，并将两组 16-bin 病灶直方图拼接为 `cond_dim=32`。
- 已检查 BraTS2024 GLI 数据目录：`train` 有 1621 例且包含 `seg/t1c/t1n/t2f/t2w`，`val` 有 188 例且仅包含 `t1c/t1n/t2f/t2w`。
- 已确认抽样 NIfTI 尺寸为 `182 x 218 x 182`、spacing 为 `1.0 x 1.0 x 1.0`。
- 已新增数据检查记录：`docs/20260804_001_brats2024_gli_data_audit.md`。
- 已新增可复用统计脚本：`scripts/brats_gli_lesion_patch_stats.py`。
- 已完成 GLI/PTG 全量病灶 bbox 统计，结果目录为 `experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/region_patch_stats_full`。
- 已完成 100 例模态病灶强度抽样统计，结果目录为 `experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/modality_intensity_stats_100cases`。
- 已新增预处理方案文档：`docs/20260804_002_brats2024_gli_patch_preprocessing_plan.md`。
- 已新增确定性局部裁剪器：`scripts/brats_gli_crop_local_patches.py`。
- 已完成 `20260804_exp001_t1c_local_patch_dataset`：1621 例、731 个患者组、两种尺寸合计 19684 个 NPZ，失败病例为 0，归一化退化病例为 0。
- 已发布数据集到 `/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches`，并保留 manifest、患者划分、QA 图和完整性汇总。
- 已完成 `20260805_exp002_gli_loader_t1c_multilesion`：loader 保留 scalar `label`，新增 NETC/SNFH/ET/RC 四通道 `lesion_mask`，固定 `hist` 条件维度为 64；远端 focused tests 4/4 通过。
- 已记录 GLI loader 实施方案：`docs/20260805_001_gli_loader_implementation_plan.md`。
- 已整理 GLI lesion-aware 训练接入长期方案：`docs/20260805_002_gli_lesion_aware_training_integration_plan.md`。
- 已完成 `20260805_exp003_gli_lesion_aware_training`：训练入口接受 GLI，Trainer 传递四通道 `lesion_mask`，histogram 使用当前 device，diffusion 支持完整 `[D,H,W]` shape。
- 已实现 GLI loss：所有非空 `(sample, lesion channel)` 单元等权平均，不按病灶大小加权；空单元跳过，全空 batch 报错。
- 已完成两份 Hydra 配置解析和全量测试：16 项通过；真实发布数据 loader 4/4 通过。
- `64×64×32` 真实固定 batch 20 步 overfit smoke 通过，首 5 步 loss 均值 `0.785975`，末 5 步 `0.360158`，峰值显存 `6309.063 MiB`。
- `80×96×80` 真实单 batch 前向/反向通过，输入 `[1,4,80,80,96]`，峰值显存 `33944.751 MiB`，未 OOM。
- 两次 smoke 已同步到 W&B，run ID 为 `23hogegm` 和 `228g4h4j`；同步后已删除远端临时 offline 缓存，未生成 checkpoint。
- 已完成 `20260805_exp004_gli_inference_closed_loop`，当前实现分支为 `feature/20260805-exp004-gli-inference`，验证代码 commit 为 `bcf9097a6ccbd20db6ab6d992ae72872c5ea65dd`。
- 已生成 versioned `splits_v2.json`：train/val/test 为 `584/73/74`，subject 零交集；两种 patch 共用。
- 已按 label 对 train-only 16-bin histogram 聚类，两种 patch 的 NETC/SNFH/ET/RC 均选择 `k=[5,3,2,2]`，并保留 k-sweep、PCA、中心分位带、cluster 规模和代表 patch 图。
- 已完成 64 个 train subject 的 normalization 审计：median Dice `0.999997`、p05 Dice `0.990682`、median extra `0`、p95 extra `0.2444%`，四项自动门禁通过；最差和随机 montage 未见系统性颅外伪影，因此保留当前 normalization，同时保留显式 support 仅用于 QA。
- 已生成两份 validation-only checkpoint：64 patch 20-step overfit 通过，矩形 patch 1-step 前向/反向通过；均绑定 commit `bcf9097`。
- 真实 test inference smoke 均通过：64 patch 为 5 次模型调用、`2.61 s`、`1083 MiB`；矩形 patch 为 5 次模型调用、`4.69 s`、`6519 MiB`。两者 DHW/XYZ、affine、NIfTI round-trip、内部四通道与最终单通道均正确。
- 全量测试在真实 patch 根目录下 22/22 通过；四份 Hydra 配置完整解析且无 `???`。
- 已通过 BraTS 官方评测说明确认标签语义为 `0=background、1=NETC、2=SNFH、3=ET、4=RC`；代码中的 patch、四通道 mask、histogram、loss、cluster 和 inference channel 顺序一致，标签语义不再是正式训练 blocker。
- 已完成正式训练可行性复核并保留当前 normalization；长期方案记录于 `docs/20260805_004_gli_formal_training_plan.md`。
- 已确定推荐 baseline 为按 `anchor_label × sample_role` 分层并在层内按 subject 平衡的 sampler；该方案保持现有 loss 不变，但属于训练流程方法变化。

## 失败尝试

- exp004 首次在线 W&B checkpoint smoke 因远端无环境凭据而在模型计算前失败；随后改为 W&B offline 完成。offline run 尚未同步，不能满足正式训练的在线日志门禁。

## 已知问题

- 当前闭环实现位于 `feature/20260805-exp004-gli-inference`，验证代码 commit 为 `bcf9097a6ccbd20db6ab6d992ae72872c5ea65dd`；尚未合并到 `main`。
- GLI 正式训练尚未启动，当前 smoke 结果不能作为最佳模型或正式指标。
- 当前两个 checkpoint 仅由 20-step/1-step smoke 产生，不能作为正式训练 checkpoint 或医学质量模型。
- exp004 W&B run 仅保存在远端 offline 目录，尚无在线 run URL。用户已说明在 F 盘准备新的 W&B key，但该凭据尚未以安全方式加载到远端环境并验证 online；key 禁止写入项目文件或 Git。
- `val` split 未发现 `seg` 标签，不能直接作为监督验证集。
- 当前 RePaint smoke 使用 `t_T=5`，生成结果呈随机纹理，只证明闭环、shape 和 mask 语义正确，不证明病灶生成质量。
- 当前 Trainer 尚无正式 validation、per-channel metrics、early stopping、完整 checkpoint/resume 和分层 sampler，不能直接启动全量训练。
- normalization audit 的 `summary.json` 仍保留 `manual_qa_status=pending`，与已完成视觉检查的项目文档不一致，正式训练前需统一 provenance。

## 下一步

1. 等待用户确认 `docs/20260805_004_gli_formal_training_plan.md` 中列出的 experiment ID、branch、sampler、两阶段 patch 和训练参数。
2. 确认后创建 exp005 feature branch，实施 sampler、validation、checkpoint/resume、W&B online fail-closed 和测试；先不启动正式训练。
3. 由用户将新的 W&B key 安全加载到远端环境变量或凭据缓存；不得读取、输出或写入 Git。
4. 另行授权后执行 online、显存、validation 和 resume preflight；通过后先训练 `64×64×32` seed `20260805`。
5. 正式 checkpoint 可用并冻结模型选择后，再决定 `80×96×80` 对照和完整 RePaint/test 医学 QA。
