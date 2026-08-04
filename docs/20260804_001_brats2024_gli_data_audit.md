# BraTS2024 GLI 数据检查记录

## 检查时间

2026-08-04

## 数据路径

远端数据路径：

```text
/workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI
```

当前实验路径：

```text
/workspace/LeFusion_v2/my_experiment
```

## 数据集结构

顶层结构：

```text
BraTS-GLI/
|-- BraTS-PTG supplementary demographic information and metadata.xlsx
|-- CITATIONS.bib
|-- train/
`-- val/
```

病例目录示例：

```text
train/BraTS-GLI-00008-100/
|-- BraTS-GLI-00008-100-seg.nii.gz
|-- BraTS-GLI-00008-100-t1c.nii.gz
|-- BraTS-GLI-00008-100-t1n.nii.gz
|-- BraTS-GLI-00008-100-t2f.nii.gz
`-- BraTS-GLI-00008-100-t2w.nii.gz

val/BraTS-GLI-02073-100/
|-- BraTS-GLI-02073-100-t1c.nii.gz
|-- BraTS-GLI-02073-100-t1n.nii.gz
|-- BraTS-GLI-02073-100-t2f.nii.gz
`-- BraTS-GLI-02073-100-t2w.nii.gz
```

## 大小和数量

- 总大小：约 `182G`
- 总文件数：`8859`
- NIfTI 文件数：`8857`
- 额外文件：`metadata.xlsx` 和 `CITATIONS.bib`
- `train` 病例数：`1621`
- `val` 病例数：`188`

按模态统计：

| split | seg | t1c | t1n | t2f | t2w |
|---|---:|---:|---:|---:|---:|
| train | 1621 | 1621 | 1621 | 1621 | 1621 |
| val | 0 | 188 | 188 | 188 | 188 |
| total | 1621 | 1809 | 1809 | 1809 | 1809 |

## 尺寸和数据类型

抽样读取 NIfTI header 后确认：

- 训练样本 `BraTS-GLI-00008-100` 的 `seg/t1c/t1n/t2f/t2w` 尺寸均为 `182 x 218 x 182`。
- 验证样本 `BraTS-GLI-02073-100-t1c` 尺寸为 `182 x 218 x 182`。
- spacing 均为 `1.0 x 1.0 x 1.0`。
- 训练样本 header 数据类型为 NIfTI datatype `16`，即 `float32`。
- 验证样本 `t1c` header 数据类型为 NIfTI datatype `64`，即 `float64`。

注意：本次为抽样读取 header，不是全量读取所有体数据。后续实现 dataloader 时仍建议在首次构建索引时增加 shape 一致性断言。

## 标签取值

抽样读取训练分割标签：

- `BraTS-GLI-00008-100-seg.nii.gz`：`0, 2, 3, 4`
- `BraTS-GLI-00009-100-seg.nii.gz`：`0, 1, 2, 3, 4`

结论：

- 标签为多类别体素标签，背景为 `0`。
- 单个病例不一定包含所有病灶类别。
- 后续需要结合 BraTS2024 GLI 官方说明或 `CITATIONS.bib`/metadata 明确 `1,2,3,4` 的医学语义，再决定 LeFusion 的通道分解方式。

## 原始数据还是处理后数据

本目录更接近 BraTS 发布格式的 NIfTI 数据，而不是 LeFusion 已处理缓存：

- 文件为 `.nii.gz`，按病例目录组织。
- 包含四个 MRI 模态：`t1c`、`t1n`、`t2f`、`t2w`。
- 训练集包含 `seg` 标签，验证集不包含 `seg` 标签。
- 未发现 `.npy`、`.npz`、`.pkl`、`.pt`、`.pth`、`.h5`、`.hdf5` 等常见预处理缓存文件。
- 数据已经是 BraTS 风格的配准、统一 spacing 的医学影像发布格式；但不是当前 `my_experiment`/LeFusion 所需的切片、归一化、直方图条件或多通道病灶表示缓存。

## 对迁移的影响

- 训练 loader 应优先使用 `train` split，因为只有 `train` 有 `seg`。
- `val` 可用于无标签生成或推理检查，不能直接作为监督验证集，除非另有标签来源。
- 输入模态顺序建议先固定为 `t1c, t1n, t2f, t2w`，并在配置中显式记录。
- 现有 LeFusion EMIDEC 流程使用两通道图像/标签和 `cond_dim=32` 的直方图条件；GLI 需要重新定义标签到通道的映射和直方图条件维度。
- 由于体数据尺寸为 `182 x 218 x 182`，后续需要决定是沿某个轴做 2D slice 训练，还是改造为 3D patch/volume 训练。当前 LeFusion 代码更接近 2D slice 流程，需要谨慎迁移。

## 检查脚本处理

本次检查过程中曾创建只读检查脚本用于读取目录统计、NIfTI header 和标签唯一取值。检查结果已经记录到本文档、`STATUS.md` 和 `CHANGELOG.md`，这些脚本不会作为后续稳定流程复用，因此已按项目管理规则删除，避免干扰项目结构。
