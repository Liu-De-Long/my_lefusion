# LeFusion 迁移到 BraTS2024 GLI 实验

> 最新配对评估：exp020 已改为让 direct/filtered 在 test 时真正使用各自的 exp016 伪四通道 mask 与重算 histogram。冻结 200 例上，direct union 精确等于 GT total，filtered coverage 为 80.9007%；共同 filtered-retained 生成区域的 pooled PSNR/SSIM 分别为 exp010 19.1322/0.4988、direct 20.7710/0.5977、filtered 21.4968/0.6636。exp019 仅保留为统一 GT 推理条件的历史对照。

本目录是将 LeFusion 方法迁移到 BraTS2024 GLI 数据集的实验工作区。

## 项目目标

目标是在 `my_experiment` 中保留一份可控、可追踪的 LeFusion 实验副本，并以原项目在 EMIDEC 数据集上的流程为主要参照，将 lesion-focused diffusion 方法迁移到 BraTS2024 GLI。

当前迁移方向：

- 使用 BraTS2024 GLI 的多模态 MRI 作为多通道输入。
- 参考 EMIDEC 的多类别病灶处理方式，采用多通道分解。
- 参考 EMIDEC 的病灶直方图条件，扩展为适配 GLI 病灶区域的多组直方图条件。
- 生成带病灶控制的 3D 图像和对应标签，用于后续分割模型增强实验。

LeFusion 论文入口：<https://arxiv.org/abs/2403.14066>

## 技术流程

当前代码是 LeFusion 原始流程的实验副本，已保留 LIDC 和 EMIDEC 路径，并完成 BraTS2024 GLI loader、四通道 lesion-aware loss、训练 batch 传递、矩形空间 shape 和 patch 级 RePaint inference 闭环。

当前可复用主流程：

1. 数据集读取：`LeFusion/dataset/*_hist.py`
2. 数据工厂：`LeFusion/get_dataset/get_dataset.py`
3. 训练入口：`LeFusion/train/train.py`
4. 扩散模型：`LeFusion/ddpm/diffusion.py`
5. 推理入口：`LeFusion/inference/inference.py`
6. histogram 条件：`LeFusion/inference/hist_clusters/*.json`
7. shell 参数入口：`emidec_train.sh`、`emidec_inference.sh`

BraTS2024 GLI 流程状态：

1. 已建立 T1c 局部 patch GLI dataset，并定义 NETC/SNFH/ET/RC 四通道 lesion mask。
2. 已计算四组 16-bin histogram，并在训练 dataset factory 中注册 GLI。
3. 已增加 GLI lesion-aware loss、`lesion_mask` 训练传递和三维矩形 shape 支持。
4. 已完成 `64×64×32` 固定 batch overfit smoke 和 `80×96×80` 单 batch 前向/反向 smoke。
5. exp005 `64×64×32` seed `20260805` 已完成，但因缺少挖空 T1c 与 mask 空间条件而失效；
   p80 已在 step 11500 停止，两者只保留失败审计。
6. 已完成 GLI inference loader、train-only histogram cluster、原始 LeFusion 对齐的
   pre-denoiser 背景 RePaint、四通道病灶合成和 NPZ/NIfTI 保存回读。
7. 已通过 BraTS 官方评测说明确认标签为 `0=background、1=NETC、2=SNFH、3=ET、4=RC`。
8. exp005 已完成分层 sampler、正式 validation、完整 checkpoint/resume、early stopping、
   W&B online 和正式训练。
9. exp008 将训练契约改为 `x_t + masked T1c + four-channel mask + hist -> noise`，
   p64 已完成 50,000 step；随后仅在冻结的 8 例 val patch 上做完整 `t_T=300` QA。
   工程闭环通过，但生成病灶区几乎退化为挖空值 `0`，因此 exp008 也只保留为失败审计，
   不作为有效伪病灶模型或全量 test 结论。
10. exp010–exp012 将扩散状态缩减为单通道，分别比较完整 T1c 噪声预测、lesion-only 噪声
    预测和 lesion-only x0/soft-histogram 目标；三者均已完成同一 seed 的 5,000 step 和固定
    8 例五种 validation QA。exp012 的 lesion/hist-CF/union-hist-CF 为 `8/8/8`、背景误差为
    `0`，当前仅作为最强 histogram 条件响应候选；union 指标不证明同类病灶视觉/语义一致性，
    结论不外推到医学有效性、其他 seed 或全量 test。
11. exp014 已对 exp012 执行同一 8 例、同一 union mask 和同一采样随机序列的四类别
    counterfactual QA，共 32 个结果。四类 histogram Top-1 为 `21/32`，其中 NETC/SNFH 仅
    `3/8`、`2/8`；目检仅稳定看到 ET 高亮及整体亮暗变化，没有证据证明四类视觉或医学语义
    控制成立，因此暂不调整 `λhist`，也不选定最终生成模型。
12. exp016 已为 exp010 补齐完全相同的 8×4 四类别 QA，并与 exp012 做 32 对逐数组配对比较。
    exp010/exp012 的 histogram Top-1 为 `18/32、21/32`，GLCM-LOCO 为 `21/32、26/32`；
    后续条件生成优先使用 exp012，exp010 保留为 paired 重建对照。该结果仍不证明医学亚型语义，
    也不能将差异单独归因于 soft-histogram loss。
13. exp010 的完整 train split、双 GPU 正式训练在 step 27838 起出现 NaN，并在 step 28000
    validation 有限性门禁处退出，未正常达到 50k。step 14000 的 best/EMA 参数全有限、validation
    loss 为 `0.126450`，已冻结为 test 50% 评估入口；污染的 latest 禁止使用。exp012 仍未启动。
14. exp010 step 14k best/EMA 已完成冻结 test 50%：精确 `519/1038` patch，双分片 `260/259`
    无重叠。平均 lesion MAE `0.157524`，`493/519` 相对零填洞改善至少 20%，519/519 非零、
    非平坦且 mask 外严格不变；仍有 17 例负改善，nearest-cluster histogram 控制和四类医学
    语义均未证实。本结论不等同于全量 test。

## 项目结构

```text
my_experiment/
|-- README.md
|-- STATUS.md
|-- CHANGELOG.md
|-- requirements.txt
|-- emidec_train.sh
|-- emidec_inference.sh
|-- lidc_train.sh
|-- lidc_inference.sh
|-- LeFusion_architecture_notes.md
`-- LeFusion/
    |-- dataset/
    |   |-- emidec_hist.py
    |   |-- emidec_hist_in.py
    |   |-- lidc_hist.py
    |   `-- lidc_hist_in.py
    |-- ddpm/
    |   |-- diffusion.py
    |   |-- scheduler.py
    |   |-- time_embedding.py
    |   `-- unet.py
    |-- get_dataset/
    |   `-- get_dataset.py
    |-- inference/
    |   |-- inference.py
    |   |-- confs/infer.yaml
    |   `-- hist_clusters/
    `-- train/
        |-- train.py
        `-- config/
```

## 运行方法

远端实验目录：

```bash
cd /workspace/LeFusion_v2/my_experiment
```

安装依赖：

```bash
pip install -r requirements.txt
```

当前原始 EMIDEC 训练入口：

```bash
bash emidec_train.sh
```

当前原始 EMIDEC 推理入口：

```bash
bash emidec_inference.sh
```

注意：上述 EMIDEC 脚本仍使用原始数据布局和参数，不能替代 GLI 配置。GLI 的可复用 smoke 入口为：

```bash
python scripts/gli_training_smoke.py +experiment=gli_64x64x32
python scripts/gli_training_smoke.py +experiment=gli_80x96x80
```

GLI smoke 和后续训练使用 W&B；凭据只能通过服务器环境变量 `WANDB_API_KEY` 或服务器本机登录缓存提供。

exp005 正式配置的只读解析入口为：

```bash
python LeFusion/train/train.py +experiment=gli_formal_64x64x32 --cfg job --resolve
python LeFusion/train/train.py +experiment=gli_formal_80x96x80 --cfg job --resolve
```

以上命令只解析配置，不登录 W&B、不创建 run、不启动训练。正式训练入口必须在 W&B、显存、validation 和 resume preflight 另行授权并通过后才能执行。

64 patch 的可复用 preflight 入口为：

```bash
python scripts/gli_formal_training_preflight.py +experiment=gli_formal_64x64x32 preflight.run_id=<unused-preflight-run-id> preflight.run_name=<unused-preflight-run-name>
```

该命令会创建一个独立 W&B preflight run，执行零 optimizer update 的显存/梯度检查、完整 validation 和 checkpoint reload；它不执行正式训练。成功后会删除临时 preflight checkpoint。

GLI inference 闭环的可复用入口为：

```bash
python scripts/gli_build_inference_assets.py --dataset-root /workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches --output-root experiments/20260805_exp004_gli_inference_closed_loop/outputs --split-output experiments/20260805_exp004_gli_inference_closed_loop/splits_v2.json
python LeFusion/inference/inference.py --config-name gli_64x64x32
python LeFusion/inference/inference.py --config-name gli_80x96x80
```

上述 inference 配置默认使用缩减的 `t_T=5` schedule，只验收接口和保存闭环，不代表正式采样质量。

## 实验管理方式

本项目用三个根文档管理实验：

- `README.md`：唯一项目入口，只记录目标、流程、结构、运行方式和管理规则。
- `STATUS.md`：只记录当前有效状态，包括当前版本、最佳实验、最佳结果、已完成事项、已知问题和下一步。
- `CHANGELOG.md`：记录所有成功和失败实验历史，每个实验需要包含实验 ID、日期、目标、方法、结果、结论。

实验 ID 建议格式：

```text
YYYYMMDD_short_goal_vN
```

示例：

```text
20260804_init_docs_v1
20260805_gli_dataset_loader_v1
20260806_gli_multihist_train_v1
```

每次实验结束后：

1. 在 `CHANGELOG.md` 追加实验记录。
2. 如果实验改变了当前最佳结果或当前流程，更新 `STATUS.md`。
3. 不把临时路径、一次性 debug 输出、未验证猜测写入 `README.md`。

## 当前有效结果入口

当前已发布 BraTS2024 GLI T1c 局部病灶 patch 数据集：

```text
/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches
```

该数据集使用 T1c 单模态图像和标量 segmentation，后续 loader 将展开为 NETC/SNFH/ET/RC 四通道 lesion mask。两种 patch 尺寸分别为 `64×64×32` 和 `80×96×80`，每种尺寸 9842 个 patch。

GLI loader 的当前契约为：`label` 保留 scalar segmentation，`lesion_mask` 提供 NETC/SNFH/ET/RC
四通道二值 mask；`target_t1c` 提供新的单通道扩散目标，旧四通道 `data` 仅为历史 checkpoint
兼容保留；`masked_context` 在四类病灶 union 内置 0、洞外保持原值。exp010–exp012 denoiser
每步接收单通道状态、单通道 `masked_context` 与四通道 lesion mask，共 6 个空间通道；histogram
条件按四个 16-bin block 拼接为 `cond_dim=64`。

GLI loss 对 batch 中所有非空 `(sample, lesion channel)` 单元等权平均；每个单元先按自身病灶体素数归一化，空单元跳过，整个 batch 全空时直接报错。两种训练空间分别为 `[D,H,W]=[32,64,64]` 和 `[80,80,96]`。

请查看：

- 当前状态：`STATUS.md`
- 实验历史：`CHANGELOG.md`
- GLI 数据统计和预处理方案：`docs/20260804_002_brats2024_gli_patch_preprocessing_plan.md`
- GLI loader 实施方案：`docs/20260805_001_gli_loader_implementation_plan.md`
- GLI lesion-aware 训练接入方案：`docs/20260805_002_gli_lesion_aware_training_integration_plan.md`
- GLI patch 级 inference 闭环接口：`docs/20260805_003_gli_inference_closed_loop.md`
- GLI 正式训练方案与可行性判断：`docs/20260805_004_gli_formal_training_plan.md`
- GLI p64 少标签逐体素亚区分类方案：`docs/20260806_001_gli_p64_weak_supervision_classifier_plan.md`
- exp010–exp012 单通道短程对比设定与结论：`docs/20260807_001_gli_short_method_comparison.md`
- exp012 同病例四类别 QA：`experiments/20260808_exp014_gli_four_class_counterfactual_qa/result.md`
- 可复用 GLI 病灶统计脚本：`scripts/brats_gli_lesion_patch_stats.py`
- T1c 局部 patch 实验：`experiments/20260804_exp001_t1c_local_patch_dataset/result.md`
- GLI lesion-aware 训练接入：`experiments/20260805_exp003_gli_lesion_aware_training/result.md`
- GLI inference 闭环：`experiments/20260805_exp004_gli_inference_closed_loop/result.md`
- exp010–exp012 统一比较：`experiments/20260807_exp013_gli_short_method_comparison/result.md`
- exp016 伪 mask 驱动 exp010 的 direct/filtered 全量训练：
  `experiments/20260811_exp018_gli_exp016_pseudomask_lefusion/result.md`
- exp010/direct/filtered 冻结 200 例配对 test 对比：
  `experiments/20260811_exp019_gli_exp010_pseudomask_test200_comparison/result.md`

exp018 已完成 direct/filtered 两次 FP32 50k 正式训练；best step 分别为 44000/46000，真实-mask
val loss 为 `0.1126854883/0.1122985579`。checkpoint/resume gate、完整 real/overlay val 和固定
8-patch paired QA 均通过，全程未访问 test。

exp019 已在同一组 200 个真实-mask test patch 上完成三模型 FP32、300-step 配对比较。filtered
取得最佳 lesion PSNR/SSIM/MAE（`21.8071/0.7277/0.12087`）和 FID/SwAV-FSD
（`47.0280/2.4970`）；direct 的 Hist-W1 最低（`0.06749`）。三模型各只访问冻结的 200 例，
端到端用时 `3:24:36`。
