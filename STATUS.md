# 当前状态

## 当前版本

v0.7.0-gli-conditional-inpainting-training

## 当前最佳实验

当前没有满足“挖空背景条件下生成伪病灶”目标的有效最佳 checkpoint。正在实施的新方法实验为
`20260806_exp008_gli_conditional_inpainting_training`。exp005 的 p64/p80 与 exp006/exp007
推理仅保留为错误训练契约的审计资产，不再作为当前最佳模型或医学有效性依据。

## 当前最佳结果

数据资产仍为两种尺寸各 9842 个 patch。旧 exp005 p64 的低 validation noise-prediction loss
不能证明伪病灶生成有效，因为 denoiser 未接收挖空 T1c 与 lesion mask 空间条件。p80 已在
step `11500` 由用户要求停止；`latest.pt`、`best.pt` 和 milestone 仅保留用于失败审计。

exp008 的冻结契约为：四通道扩散状态 `x_t`，单通道挖空 T1c 加四通道 lesion mask 的
五通道空间条件，64 维 histogram 全局条件，监督目标为加入 `x0` 的噪声，loss 仅在对应病灶
mask 内计算。远端回归 `34/34` 与 online GPU preflight 已通过；exp008 p64 seed `20260805`
已启动并完成首批 optimizer step，W&B 与双 GPU 状态正常。
首个 `latest.pt` 已按 500-step 规则写入，训练继续运行。

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
- 已完成 `20260806_exp009_gli_p64_weak_supervision_classifier`：冻结 1000 个 train p64 patch，
  依次训练 M0/M1/M2/C0；验证最佳为 C0，ET/RC focus mIoU `0.5436`，95% CI
  `[0.5000,0.5862]`，未过 0.85 门禁，冻结 test 未运行。长期方案与结论记录于
  `docs/20260806_001_gli_p64_weak_supervision_classifier_plan.md`。

## 失败尝试

- exp004 首次在线 W&B checkpoint smoke 因远端无环境凭据而在模型计算前失败；随后改为 W&B offline 完成。offline run 尚未同步，不能满足正式训练的在线日志门禁。

## 已知问题

- 当前条件式修复实现位于 `feature/20260806-exp008-gli-conditional-inpainting`，实现 commit 为
  `6e29f355294c762cfb8928f73abcb350200c2e70`；尚未合并到 `main`。
- exp005 p64 虽完成 50,000 step，但其训练契约缺少挖空 T1c 与 mask 空间条件，已失效；
  p80 在 step 11500 停止。两者 checkpoint 均不得作为新的正式结果入口。
- exp004 的 20-step/1-step checkpoint 仍只是 smoke，不能与 exp005 正式 checkpoint 混用。
- exp004 W&B run 仍仅是历史 offline smoke；正式 exp005 训练已有 online run。W&B key 始终
  不进入项目文件或 Git。
- `val` split 未发现 `seg` 标签，不能直接作为监督验证集。
- `t_T=5` 仍只证明接线；本次正式结论来自 `t_T=300`。工程 QA 不能替代医学有效性、下游
  分割增益或全量 test 结论。
- 用户已授权按 exp008 新契约重新启动且只启动 `64×64×32` seed `20260805`；不得自动启动
  其他 seed、p80 或全量 test。
- baseline 的 batch 是 preflight 初值而非硬编码：64 可从 `4/1` 调为 `2/2` 或 `1/4`；`50,000` optimizer steps 是上限，可由 early stopping 提前结束。
- 原始 LeFusion 入口没有强制三 seed；`20260806/20260807` 是在首个 seed 通过后再决定的正式复现候选。
- 少标签分类方案中的“10% 且约 1000 Case”存在口径差异：默认按 1000 个 p64 patch
  解释；用户已确认本次按 1000 patch、T1c、ET/RC 门禁实施。exp009 最佳仅 0.5436，
  且训练未接入 W&B online，因此 checkpoint 只作为验证审计，不是正式最佳模型。
- exp009 的 C0 `history.jsonl` 实际最高 focus mIoU 为 epoch 29 的 `0.5482`，但旧实现把
  `early_stopping_min_delta` 复用于 best 保存，导致 `best.pt` 停在 epoch 22 的 `0.5436`；
  该问题不改变门禁失败，但必须在下一次训练前修复。

## 下一步

1. 监控 exp008 p64 seed `20260805` 的 W&B、latest/best/milestone 与 validation 收敛。
2. 新 checkpoint 可用后仅先做少量冻结 QA，不运行 519 例或全量 test。
3. 不自动启动其他 seed、p80 或额外训练；任何扩展均需新的用户授权。
4. 若继续追求分类 focus mIoU 0.85，先补 W&B fail-closed，再由用户确认采用更强多尺度 3D
   网络，还是允许剩余 train patch 进入半监督一致性/伪标签实验；验证门禁通过前不运行 test。
