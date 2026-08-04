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

当前代码仍是 LeFusion 原始流程的精简副本，已实现 LIDC 和 EMIDEC 两条路径。BraTS2024 GLI 迁移尚未接入数据工厂。

当前可复用主流程：

1. 数据集读取：`LeFusion/dataset/*_hist.py`
2. 数据工厂：`LeFusion/get_dataset/get_dataset.py`
3. 训练入口：`LeFusion/train/train.py`
4. 扩散模型：`LeFusion/ddpm/diffusion.py`
5. 推理入口：`LeFusion/inference/inference.py`
6. histogram 条件：`LeFusion/inference/hist_clusters/*.json`
7. shell 参数入口：`emidec_train.sh`、`emidec_inference.sh`

计划中的 BraTS2024 GLI 流程：

1. 建立 GLI 数据集类，读取多模态 MRI 和分割标签。
2. 定义 GLI 病灶标签到多通道表示的映射。
3. 为每个目标病灶区域计算直方图条件。
4. 在 `get_dataset.py` 中注册 GLI train/inference dataloader。
5. 添加 GLI 训练和推理配置。
6. 训练 LeFusion GLI 模型。
7. 用 RePaint-style sampling 生成 GLI 合成图像和标签。
8. 用下游分割实验评估合成数据收益。

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

注意：上述 EMIDEC 脚本仍使用原始数据布局和参数。BraTS2024 GLI 的训练/推理脚本尚未创建，不能直接用 EMIDEC 脚本替代。

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

## 当前最佳结果入口

当前已发布 BraTS2024 GLI T1c 局部病灶 patch 数据集：

```text
/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches
```

该数据集使用 T1c 单模态图像和标量 segmentation，后续 loader 将展开为 NETC/SNFH/ET/RC 四通道 lesion mask。两种 patch 尺寸分别为 `64×64×32` 和 `80×96×80`，每种尺寸 9842 个 patch。

请查看：

- 当前状态：`STATUS.md`
- 实验历史：`CHANGELOG.md`
- GLI 数据统计和预处理方案：`docs/20260804_002_brats2024_gli_patch_preprocessing_plan.md`
- 可复用 GLI 病灶统计脚本：`scripts/brats_gli_lesion_patch_stats.py`
- T1c 局部 patch 实验：`experiments/20260804_exp001_t1c_local_patch_dataset/result.md`
