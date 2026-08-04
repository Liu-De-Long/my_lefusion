# BraTS2024 GLI T1c 四通道 Loader 实施方案

## 文档目的

记录 `20260805_exp002_gli_loader_t1c_multilesion` 的 loader 设计、数据接口、验证标准和后续 LeFusion 接入边界。本方案不包含训练启动，不生成 checkpoint，也不修改 GLI diffusion loss。

## 标签与通道约定

GLI 的 NPZ 保留标量 `seg`。参考原始 LeFusion EMIDEC 数据处理，loader 保留 scalar label 语义，同时提供独立的四通道 lesion mask：

```text
modality = t1c
image_modality_channels = 1
lesion_channel_order = NETC, SNFH, ET, RC
label_values = 1, 2, 3, 4
hist_bins = 16
cond_dim = 4 × 16 = 64
```

输出字段：

```text
data:        [4,D,H,W]  # T1c 复制到四个 lesion 对齐通道
label:       [1,D,H,W]  # scalar segmentation
lesion_mask: [4,D,H,W]  # NETC/SNFH/ET/RC 二值 mask
hist:        [64]       # 四个 16-bin block 按标签顺序拼接
```

展开规则：

```python
lesion_mask[0] = label == 1  # NETC
lesion_mask[1] = label == 2  # SNFH
lesion_mask[2] = label == 3  # ET
lesion_mask[3] = label == 4  # RC
```

## 轴转换

NPZ 的空间数组使用 `[X,Y,Z]`。loader 固定转换为 LeFusion 使用的 `[C,D,H,W]`：

```text
t1c[x,y,z] -> data[:,z,x,y]
seg[x,y,z] -> label[:,z,x,y]
```

对应两种 patch 的 shape 为：

```text
64×64×32 -> data [4,32,64,64]
80×96×80 -> data [4,80,80,96]
```

## Loader 实现

正式运行时代码放在 `LeFusion/dataset/gli_hist.py`，因为它需要被 dataset factory 和后续训练/推理入口导入。`scripts/` 仅保留裁剪、审计、转换等命令行工具，不放运行时 loader。

loader 负责：

1. 根据 patch size 读取对应目录的 `manifest.csv`。
2. 按 `train/val` 过滤样本。
3. 读取 NPZ 中的 `t1c`、`seg`、`hist`、`affine`。
4. 校验键、dtype、shape、有限值和标签范围。
5. 展开四通道 `lesion_mask`，复制 T1c 到四个 `data` 通道。
6. 按 `NETC/SNFH/ET/RC` 顺序将 `[4,16]` histogram 展平为 `[64]`。
7. 返回 manifest 中的 case、anchor、origin、padding 和归一化 metadata。

## Padding metadata

当前 manifest 没有完整病灶在原始影像中的总 voxel 数，因此不计算 lesion coverage 截断比例。loader 暴露的是空间边界 padding 比例：

```text
valid_xyz = patch_size_xyz - pad_before_xyz - pad_after_xyz
valid_volume_fraction = prod(valid_xyz) / prod(patch_size_xyz)
padding_fraction = 1 - valid_volume_fraction
```

同时返回：

```text
is_padded
pad_before_xyz
pad_after_xyz
origin_xyz
valid_volume_fraction
padding_fraction
```

## Dataset 注册

已在以下位置完成注册和配置：

```text
LeFusion/dataset/__init__.py
LeFusion/get_dataset/get_dataset.py
LeFusion/train/config/dataset/gli.yaml
```

本轮只注册 dataset factory，不放开 `train.py`。原因是当前 diffusion loss 仍只有 `lidc/emidec` 分支，单独放开 `gli` 会造成无法计算 loss 的不完整接口。

## 验证标准

正式测试位于 `tests/test_gli_loader.py`，覆盖：

- `[X,Y,Z] -> [C,D,H,W]` 轴映射。
- scalar `label` shape 和标签值保持。
- 四通道 lesion mask 精确展开。
- histogram 四个 block 的拼接顺序。
- padding metadata 计算。
- 两种 patch size 的单样本和 batch shape。
- `gli` dataset factory 注册。
- 真实发布数据的两种 patch 尺寸迭代。

当前验证结果：

- 远端 focused tests：4/4 通过。
- 真实发布数据：`64×64×32` 和 `80×96×80` 均可迭代。
- 既有测试回归：10 项通过，1 项因未设置真实数据环境变量跳过。

## 后续接入边界

后续训练实验仍需独立完成：

- GLI 四通道 lesion-mask loss。
- Trainer 将 `lesion_mask` 传入 GLI loss。
- `diffusion_num_channels=4`、`cond_dim=64` 的训练配置。
- 对矩形 `80×96×80` patch 的 diffusion shape 支持。

后续推理实验还需实现 GLI inference loader、四通道 RePaint keep-mask、histogram cluster 文件和输出合并逻辑。
