# 当前状态

## 当前版本

v0.2.0-brats-gli-t1c-local-patches

## 当前最佳实验

`20260804_exp001_t1c_local_patch_dataset`

## 当前最佳结果

已发布 BraTS2024 GLI T1c 局部病灶 patch 数据集：两种尺寸各 9842 个 patch，合计 19684 个 NPZ，完整性验证通过。

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

## 失败尝试

暂无记录。

## 已知问题

- 尚未实现 BraTS2024 GLI 数据集类。
- `LeFusion/get_dataset/get_dataset.py` 尚未注册 GLI 数据集。
- `LeFusion/train/train.py` 和 `LeFusion/inference/inference.py` 目前只接受 `lidc` 与 `emidec`。
- GLI loader 的四通道展开和模型接入尚未实现；裁剪数据已固定为 T1c、四标签、16-bin histogram、两种 patch 尺寸和患者级划分。
- 尚未创建 GLI 训练/推理脚本或 Hydra 配置。
- 尚未创建 GLI 直方图聚类中心 JSON。
- `val` split 未发现 `seg` 标签，不能直接作为监督验证集。
- GLI 标签 `1,2,3,4` 的医学语义尚未从官方说明中确认。
- 数据检查中使用的一次性临时脚本已删除；后续如需稳定数据审计工具，应重新设计为可复用脚本并补充文档说明。
- 当前统计按 BraTS post-treatment glioma 语义命名：`1=NETC`、`2=SNFH`、`3=ET`、`4=RC`。后续实现前仍需用官方说明或 metadata 复核一次。

## 下一步

1. 用官方说明或 metadata 复核 GLI/PTG 标签语义，尤其 `label_4` 是否为 `RC`。
2. 实现 GLI dataset v1：T1c 单模态输入，将标量 segmentation 展开为 NETC/SNFH/ET/RC 四通道 lesion mask。
3. 接入 `64×64×32` 和 `80×96×80` 两份已发布 patch 配置，保持 `[X,Y,Z]` 到 `[C,D,H,W]` 的轴转换一致。
4. 在 dataloader 中暴露 patch padding/截断相关 metadata，并验证 histogram 条件维度为 `4×16=64`。
5. 在 `LeFusion/get_dataset/get_dataset.py` 中注册 GLI。
6. 添加 GLI 训练和推理脚本或 Hydra 配置。
7. 运行第一次 loader shape/hist 冒烟测试，再决定是否启动 5-10 例小规模过拟合实验。
