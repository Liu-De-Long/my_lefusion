# LeFusion 迁移到 BraTS2024 GLI 实验

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
   loss 仍只在对应病灶区域计算；当前重新训练 p64，旧 QA 不再作为有效方法结论。

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
四通道二值 mask；原始 T1c 按四个 lesion channel 复制为扩散目标 `data`，同时生成单通道
`masked_context`，在四类病灶 union 内置 0、洞外保持原值。denoiser 每步接收四通道 `x_t`、
单通道 `masked_context` 与四通道 lesion mask，共 9 个空间通道；histogram 条件按四个 16-bin
block 拼接为 `cond_dim=64`。

GLI loss 对 batch 中所有非空 `(sample, lesion channel)` 单元等权平均；每个单元先按自身病灶体素数归一化，空单元跳过，整个 batch 全空时直接报错。两种训练空间分别为 `[D,H,W]=[32,64,64]` 和 `[80,80,96]`。

请查看：

- 当前状态：`STATUS.md`
- 实验历史：`CHANGELOG.md`
- GLI 数据统计和预处理方案：`docs/20260804_002_brats2024_gli_patch_preprocessing_plan.md`
- GLI loader 实施方案：`docs/20260805_001_gli_loader_implementation_plan.md`
- GLI lesion-aware 训练接入方案：`docs/20260805_002_gli_lesion_aware_training_integration_plan.md`
- GLI patch 级 inference 闭环接口：`docs/20260805_003_gli_inference_closed_loop.md`
- GLI 正式训练方案与可行性判断：`docs/20260805_004_gli_formal_training_plan.md`
- 可复用 GLI 病灶统计脚本：`scripts/brats_gli_lesion_patch_stats.py`
- T1c 局部 patch 实验：`experiments/20260804_exp001_t1c_local_patch_dataset/result.md`
- GLI lesion-aware 训练接入：`experiments/20260805_exp003_gli_lesion_aware_training/result.md`
- GLI inference 闭环：`experiments/20260805_exp004_gli_inference_closed_loop/result.md`
