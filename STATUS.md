# 当前状态

## 当前版本

v0.7.1-gli-conditional-inpainting-qa

## 当前最佳实验

当前没有满足“挖空背景条件下生成伪病灶”目标的有效最佳 checkpoint。exp008 已按正确的
空间条件契约完成训练，但 8 例冻结 val QA 显示病灶区输出几乎退化为挖空值 `0`，因此也只
保留为失败审计。exp005 的 p64/p80 与 exp006/exp007 同样不再作为当前最佳模型或医学有效性依据。

## 当前最佳结果

数据资产仍为两种尺寸各 9842 个 patch。旧 exp005 p64 的低 validation noise-prediction loss
不能证明伪病灶生成有效，因为 denoiser 未接收挖空 T1c 与 lesion mask 空间条件。p80 已在
step `11500` 由用户要求停止；`latest.pt`、`best.pt` 和 milestone 仅保留用于失败审计。

exp008 p64 seed `20260805` 已完成 50,000 step，最终 EMA validation total loss 为 `0.091035`；
`best.pt` 为 step `48000`、EMA，SHA-256 为
`660e23533064eaf9f32310b42500a8067c476dbd2bddbfd88a88dbc0f2c64857`。仅对冻结的 8 个
val patch 运行了 `masked_multilabel` 和 `masked_anchor_union` 两组完整 `t_T=300` QA，
每组 2,400 次模型调用；没有运行 519 例、test split 或任何全量 test。

两组 QA 的病灶区生成绝对强度均值仅为 `0.00834/0.00949`，而原始病灶为 `0.32388`；
histogram L1 约为 `1.49–1.98`，目检显示输出基本保留挖空后的均匀灰值，没有恢复病灶纹理。
因此低 noise-prediction validation loss 不能证明条件式生成成功，exp008 不属于有效最佳结果。

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
- 已保留并扩写 `docs/20260805_003_gli_inference_closed_loop.md`，补全 test-only loader、显式 support、cluster provenance、逐步 RePaint、四通道合成、保存回读和分阶段验收契约，作为 exp005 及后续正式 checkpoint 评估的长期接口文档。
- 已生成 versioned `splits_v2.json`：train/val/test 为 `584/73/74`，subject 零交集；两种 patch 共用。
- 已按 label 对 train-only 16-bin histogram 聚类，两种 patch 的 NETC/SNFH/ET/RC 均选择 `k=[5,3,2,2]`，并保留 k-sweep、PCA、中心分位带、cluster 规模和代表 patch 图。
- 已完成 64 个 train subject 的 normalization 审计：median Dice `0.999997`、p05 Dice `0.990682`、median extra `0`、p95 extra `0.2444%`，四项自动门禁通过；最差和随机 montage 未见系统性颅外伪影，因此保留当前 normalization，同时保留显式 support 仅用于 QA。
- 已生成两份 validation-only checkpoint：64 patch 20-step overfit 通过，矩形 patch 1-step 前向/反向通过；均绑定 commit `bcf9097`。
- 真实 test inference smoke 均通过：64 patch 为 5 次模型调用、`2.61 s`、`1083 MiB`；矩形 patch 为 5 次模型调用、`4.69 s`、`6519 MiB`。两者 DHW/XYZ、affine、NIfTI round-trip、内部四通道与最终单通道均正确。
- 全量测试在真实 patch 根目录下 22/22 通过；四份 Hydra 配置完整解析且无 `???`。
- 已通过 BraTS 官方评测说明确认标签语义为 `0=background、1=NETC、2=SNFH、3=ET、4=RC`；代码中的 patch、四通道 mask、histogram、loss、cluster 和 inference channel 顺序一致，标签语义不再是正式训练 blocker。
- 已完成正式训练可行性复核并保留当前 normalization；长期方案记录于 `docs/20260805_004_gli_formal_training_plan.md`。
- 已确定推荐 baseline 为按 `anchor_label × sample_role` 分层并在层内按 subject 平衡的 sampler；该方案保持现有 loss 不变，但属于训练流程方法变化。
- 已创建 `feature/20260805-exp005-gli-formal-training` 和 exp005 实验卡，完成分层 sampler、固定 val timestep/noise、总 loss/各 lesion channel loss/有效单元/覆盖指标、early stopping、best/latest/milestone 和完整 checkpoint/resume。
- 正式 checkpoint 已覆盖 optimizer、AMP scaler、sampler/epoch/batch offset、Python/NumPy/Torch/CUDA RNG、resolved config/hash、split/manifest hash、Git SHA 和 W&B run ID，并对不一致状态 fail closed。
- W&B 正式配置已固定 entity `jinyuanbao719-xi-an-jiaotong-university-`、run ID 和 `resume=never/must` 语义；online 初始化失败会在训练计算前退出，不回退 offline。
- 两份正式 Hydra 配置解析通过；远端全量非训练测试 28/28 通过，包含真实发布 patch 的两种尺寸 loader 和精确数据序列 resume。
- normalization audit 输出已将 stale 状态更新为 `completed_no_systematic_background_pollution` 和 `keep_t1c_nonzero_percentile_normalization`，JSON 回读通过。
- 用户已安全配置远端 W&B key；64 patch online preflight 使用独立 run `exp005-p64-preflight-s20260805-r3`，完成零 optimizer update 的显存/梯度检查、完整 val、checkpoint reload 和 resume；W&B URL：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-preflight-s20260805-r3>。
- 64 preflight 的反向峰值显存为 `23330.56 MiB`，完整 validation 峰值为 `5482.45 MiB`；validation 为 1032 patch、73 subject、3207 有效单元，四个 label 全覆盖。临时 resume checkpoint 已自动删除，仅保留远端轻量 `metrics.json` 和日志。
- 修复真实边界 patch 的负 `origin_xyz` loader 契约，代码 commit 为 `b34d867f944343f9f6ff6e4edde5abe3a0b0805b`；修复后远端全量测试 28/28 通过。
- 已新增 `docs/20260806_001_gli_p64_weak_supervision_classifier_plan.md`，冻结 p64 少标签逐体素
  四分类的数据子集、MLP/轻量 CNN、标签泄漏门禁和 ET/RC focus mIoU 评价方案；本次未实现或训练。

## 失败尝试

- exp004 首次在线 W&B checkpoint smoke 因远端无环境凭据而在模型计算前失败；随后改为 W&B offline 完成。offline run 尚未同步，不能满足正式训练的在线日志门禁。

## 已知问题

- 当前条件式修复实现位于 `feature/20260806-exp008-gli-conditional-inpainting`，实现 commit 为
  `6e29f355294c762cfb8928f73abcb350200c2e70`；尚未合并到 `main`。
- exp008 的训练 loss 与 validation loss 正常下降，但完整反向采样在病灶区退化到接近 `0`；
  下一步必须先区分 teacher-forced 噪声预测、逐步 `x0` 重建和自由采样之间的失配，不能直接扩展数据规模。
- exp005 p64 虽完成 50,000 step，但其训练契约缺少挖空 T1c 与 mask 空间条件，已失效；
  p80 在 step 11500 停止。两者 checkpoint 均不得作为新的正式结果入口。
- exp004 的 20-step/1-step checkpoint 仍只是 smoke，不能与 exp005 正式 checkpoint 混用。
- exp004 W&B run 仍仅是历史 offline smoke；正式 exp005 训练已有 online run。W&B key 始终
  不进入项目文件或 Git。
- `val` split 未发现 `seg` 标签，不能直接作为监督验证集。
- `t_T=5` 仍只证明接线；本次正式结论来自 `t_T=300`。工程 QA 不能替代医学有效性、下游
  分割增益或全量 test 结论。
- exp008 只完成了已授权的 `64×64×32` seed `20260805` 与两组各 8 例 val QA；不得自动启动
  其他 seed、p80、519 例或全量 test。
- baseline 的 batch 是 preflight 初值而非硬编码：64 可从 `4/1` 调为 `2/2` 或 `1/4`；`50,000` optimizer steps 是上限，可由 early stopping 提前结束。
- 原始 LeFusion 入口没有强制三 seed；`20260806/20260807` 是在首个 seed 通过后再决定的正式复现候选。
- 少标签分类方案中的“10% 且约 1000 Case”存在口径差异：默认按 1000 个 p64 patch
  解释，并暂定 ET/RC 为重点类别、TC 为 T1c；实施前仍需用户确认。

## 下一步

1. 先做小规模诊断：按 timestep 审计 epsilon 与 `x0` 重建误差，并比较 teacher-forced 单步重建和从随机噪声自由采样。
2. 检查 mask-only noise loss 是否需要增加 `x0`/histogram/边界重建约束；在方案确认前不重新训练。
3. 不自动启动其他 seed、p80、519 例或全量 test；任何扩展均需新的用户授权。
4. exp009 少标签分类工作保持独立，不用其分支或 GPU 资产改写 exp008 的失败结论。
