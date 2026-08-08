# 当前状态

## 当前版本

v0.11.0-gli-paired-four-class-comparison-complete

## 当前最佳实验

当前没有已证实能够生成视觉统一、类别明确病灶的最终最佳模型。严格配对 QA 后，条件生成
主干优先候选为 `20260807_exp012_gli_lesion_only_x0_hist` 的 p64、seed `20260805`、step `5000`
checkpoint。它使用 lesion-only 扩散状态、直接 x0 预测和 soft-histogram loss，在冻结 8 例
五种 QA 中达到 lesion `8/8`、hist counterfactual `8/8`、union histogram counterfactual
`8/8`、mask 外最大误差 `0`。这里的 union 指标只比较 mask 内 16-bin 强度 histogram 与
请求/交换请求的距离，不能证明病灶亚型的视觉或语义一致性。exp016 的同病例四类别比较显示
exp012 的 histogram/GLCM Top-1 为 `21/32、26/32`，高于 exp010 的 `18/32、21/32`；因此在
exp010/exp012 之间选择 exp012 作为后续条件控制主干，但仍不能称为医学语义最终模型。

exp010 同样通过全部门槛，并以平均 lesion MAE `0.138066` 获得三方法最佳 paired 重建；
exp011 因 union histogram counterfactual 仅 `5/8` 未入选。exp008、exp005 p64/p80 与 exp006/exp007 继续
仅保留为失败审计。

后续研究执行顺序已由用户调整：优先对 exp010 做 50k 双 GPU 全量训练，exp012 仅作为之后的
同规模对照。该决策优先利用 exp010 的 paired 重建优势，不改写 exp016 中 exp012 的 5k
histogram/GLCM 条件响应更强这一历史统计。

## 当前最佳结果

数据资产仍为两种尺寸各 9842 个 patch。本轮 exp010–exp012 均只运行同一 seed 的 5000 step，
W&B 已 finished，`milestone-500/1000/2500/5000.pt`、`latest.pt` 与最终 validation 齐全。
step 5000 EMA validation total loss 分别为 `0.135519 / 0.136704 / 0.144356`；exp012 的
base/hist loss 为 `0.111711 / 0.326442`。

统一冻结 8 例 QA 结果：exp010 为 lesion/hist-CF/union-hist-CF `8/7/7`，exp011 为 `7/7/5`，exp012
为 `8/8/8`；三者 mask 外最大绝对误差均为 `0`。原始真实 hist 模式的平均 lesion MAE 分别为
`0.138066 / 0.164752 / 0.163218`，零填洞基线为 `0.323883`。统一指标、摘要和横向图保存在
`experiments/20260807_exp013_gli_short_method_comparison/outputs/`。完整比较设定、结论和 QA
目录说明见 `docs/20260807_001_gli_short_method_comparison.md`。

目检确认三者均生成非零、非平坦且与 mask 对齐的结构；exp012 对 first/last cluster 的亮暗
分布响应最明显，但 union-as-single 没有呈现明确的跨病例同类外观，个别样本仍有偏黑/偏亮
团块。为解除类别与病例绑定的混杂，后续已执行 exp014 同病例四类别 counterfactual QA。没有
运行 p80、其他 seed、519 例、test split 或全量 test。

exp014 已完成上述同病例四类别 QA：固定同一 8 例、挖空输入、union mask、checkpoint、EMA、
`t_T=300`、每例 seed 和采样随机序列，共生成 `8×4=32` 个结果。合同逐数组核对通过，mask 外
最大误差为 `0`。四类 histogram 目标 Top-1 为 `21/32`，均值 rank `1.34375`；NETC/SNFH/ET/RC
分别为 `3/8、2/8、8/8、8/8`。强度归一化 3D GLCM texture 的 leave-one-case-out 正确数为
`26/32`，但 RC 仅 `4/8`，且该指标只描述生成样本统计可分性，不能证明医学语义。

目检显示 ET 在全部病例中稳定显著高亮，NETC/RC 多为较暗结构、SNFH 居中；NETC、SNFH、RC
之间仍大量重叠，局部形态继续由病例背景和 union 形状主导。由此确认 exp012 学会了明显的
亮暗/histogram 控制，但四类视觉统一性和医学亚型控制仍未通过；当前继续没有最终最佳模型。

exp016 已为 exp010 补齐完全相同的 8 例 × 4 类冻结 QA，并逐数组确认两模型的原图、挖空输入、
union/目标 mask、条件 hist 和 sample seed 一致。exp010/exp012 的平均 histogram margin 为
`0.100785/0.329680`，exp012 在 8/8 病例的四类平均 margin 上均胜；GLCM texture margin 的
病例级胜负为 4:4，但 Top-1 为 `21/32` 对 `26/32`。两者背景最大误差均为 `0`。这支持优先继续
exp012，但不能把差异单独归因于 `λhist`，因为两模型的扩散状态与预测目标也不同。

## 当前流程

exp010 双 GPU 长程训练没有正常达到 50k：step `27838` 起训练 loss/grad 变为 NaN，step
`28000` EMA validation 也为 NaN，有限性门禁抛错退出。W&B run
`exp010-p64-full50k-2gpu-s20260805` 已结束，但该状态不能改写本地异常证据。step 28000
`latest.pt` 的 model/EMA 各有 291 个含非有限值 tensor，已禁止使用；step `14000` 的
`best.pt/EMA` 参数全有限，validation loss `0.126450`，是本次唯一冻结评估入口。exp012 未启动。

当前按用户授权准备在 GPU0/1 对上述 step 14k best/EMA 运行 test 的确定性 50% 子集：固定
manifest SHA-256 `295b20...c3698`，精确 `519/1038` patch，两个互斥 shard 为 `260/259`。
输入继续使用挖空 T1c 加原始四通道 mask，条件为 nearest train-cluster hist，`t_T=300`、CFG
scale `2.0`；每张 GPU 只运行一个推理进程和 4 个 DataLoader worker，不运行全量 test。

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
- exp010、exp011、exp012 均在 GPU1 串行完成 seed `20260805` 的 5000 step，W&B finished，
  checkpoint 与最终 validation 审计通过。
- 固定 8 例五种 QA 全部完成；经用户授权，剩余 QA 从单 worker 切换为 GPU1 上两个输出互斥
  worker（父 PID `31788/31790`），未启动第三个 worker或重复变体。
- 已仅运行一次 `scripts/gli_compare_short_generation_experiments.py`，生成统一指标表、摘要与
  三方法横向 contact sheet，并完成原图、挖空输入、生成、difference、mask 及 union first/last 目检。
- 已完成 exp014 同病例四类别 counterfactual QA：仅复用 exp012 step 5000 EMA 和冻结 8 例，
  使用 GPU1 四个输出互斥 worker 生成 32 个结果；未运行训练、p80、其他 seed、519 例或全量 test。
- exp014 保留 `summary.json`、`case_metrics.csv`、`texture_features.csv`、8 张逐病例横向图和
  `four_class_contact_sheet.png`；四类输入/mask/seed 合同、32 个 NPZ 和背景严格保持均已审计。
- 已完成 exp016：只为 exp010 新增与 exp014 完全相同的冻结 32 个四类别结果，并复用 exp012
  输出做 32 对比较；保留两模型 confusion matrix、逐对/逐病例 margin 与双模型横向图。

## 失败尝试

- exp004 首次在线 W&B checkpoint smoke 因远端无环境凭据而在模型计算前失败；随后改为 W&B offline 完成。offline run 尚未同步，不能满足正式训练的在线日志门禁。

## 已知问题

- exp010–exp012 实现位于 `feature/20260807-exp010-012-gli-short-compare`，正式运行 commit
  已冻结为 `d0c6aaa056b38e34e4cb929f85106cc90af05426`，尚未合并到 `main`。
- exp012 虽在固定 8 例的 histogram 与 union 控制上最好，但个别生成含偏黑/偏亮团块；
  当前只可作为下一轮方法候选，不能直接用于医学结论或数据增强发布。
- exp014 解除类别与病例绑定后，NETC/SNFH 的四类 histogram Top-1 仅 `3/8、2/8`；ET 的高亮
  响应最明显，但 NETC/SNFH/RC 未形成稳定可辨外观。生成样本 GLCM-LOCO `26/32` 不能替代
  真实 held-out T1c 上的无标签泄漏类别可分性上限验证。
- exp016 表明 exp012 在配对条件下优于 exp010，但不是只改变 `λhist` 的单因素消融；若要声明
  soft-histogram loss 的独立增益，仍需保持 lesion-only x0 契约不变并补做 `λhist=0` 对照。
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

1. 在调整 `λhist` 或继续训练前，先验证真实 held-out T1c 病灶的无标签泄漏四类可分性上限；
   分类输入不得包含 mask 形状、hist block、anchor、病例 ID 或路径信息，并需报告 histogram-only 对照。
2. 若真实 T1c 四类可分，再针对 NETC/SNFH/RC 的重叠设计空间纹理或类别表征约束；exp010 继续
   作为 paired 重建质量对照，exp012 只作为 histogram 响应起点。
3. 在任何新增训练前建立新实验 ID、配置和 Git commit，并重新执行零更新 preflight；未经用户
   另行授权，不启动其他 seed、p80、519 例或全量 test，exp009 保持独立。
