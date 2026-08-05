# BraTS2024 GLI lesion-aware 训练接入方案

## 文档目的

本文档整理 GLI 四通道训练接入的长期方案，供后续训练实验、矩形 patch 配置和代码维护复用。本文档描述接口、数据流、loss 约定、shape 约束和验收门禁；单次实验结果仍记录在对应实验目录的 `result.md`。

## 当前数据接口

GLI loader 输出：

```text
data:        [4,D,H,W]       # T1c 复制到四个 lesion channel
label:       [1,D,H,W]       # scalar segmentation，保留兼容语义
lesion_mask: [4,D,H,W]       # NETC/SNFH/ET/RC 二值 mask
hist:        [64]            # 4 个 16-bin block
```

固定通道顺序：

```text
channel 0: NETC, label 1
channel 1: SNFH, label 2
channel 2: ET,   label 3
channel 3: RC,   label 4
```

上述标签医学语义在正式训练前仍需用官方说明或 metadata 最终复核。

## 训练接口改造

### 训练入口

`LeFusion/train/train.py` 应：

- 接受 `dataset.data_type=gli`；
- 将 `base_dim` 与输入空间尺寸分开配置；
- 根据 `spatial_shape_dhw` 构建 diffusion；
- 启动前校验 GLI 的 4 通道输入、64 维条件和 XYZ→DHW 映射；
- 正式训练启用 W&B，记录实验 ID、代码版本、配置、指标和 checkpoint 路径。

### Trainer 数据传递

- LIDC/EMIDEC 继续使用 `label`；
- GLI 使用 `lesion_mask`；
- histogram 必须通过 `.to(device)` 移动，不允许写死 `.cuda()`；
- `data`、`lesion_mask` 和 histogram 必须位于同一 device。

## GLI lesion-aware loss

设预测噪声为 `prediction`，目标噪声为 `target`，mask 为 `M[b,c]`。对每个非空 `(sample, lesion channel)` 单元独立计算：

```text
unit_loss[b,c] = sum(error[b,c] * M[b,c]) / sum(M[b,c])
```

其中 `error` 为逐 voxel L1 或 L2 误差。

最终 loss 为 batch 中所有非空 `(sample, lesion channel)` 单元的简单平均：

```text
loss = mean(unit_loss[b,c] for all sum(M[b,c]) > 0)
```

约定：

- 不按病灶体素总数加权；
- 不先做通道级平均；
- 空病灶单元跳过；
- 如果整个 batch 没有任何病灶 voxel，直接报错；
- 正式训练的类别平衡由采样策略或显式固定通道权重控制，不隐含在 loss 归一化中。

## 空间 shape 约定

loader 使用 XYZ，模型使用 DHW：

```text
patch_size_xyz = [64,64,32] -> spatial_shape_dhw = [32,64,64]
patch_size_xyz = [80,96,80] -> spatial_shape_dhw = [80,80,96]
```

当前 UNet 的 `[1,2,4,8]` 配置实际执行三次 H/W stride-2 下采样，因此 H/W 必须被 8 整除；depth 不下采样。

### 第一阶段

```text
输入: [B,4,32,64,64]
下采样: [32,64,64] -> [32,32,32] -> [32,16,16] -> [32,8,8]
```

### 第二阶段

```text
输入: [B,4,80,80,96]
下采样: [80,80,96] -> [80,40,48] -> [80,20,24] -> [80,10,12]
```

两种尺寸都可以被精确上采样恢复。第二阶段的主要风险是 depth=80 时 temporal attention 的显存，而不是几何 shape。

## 配置约定

每个 GLI patch 配置至少包含：

```yaml
dataset:
  data_type: gli
  root_dir: /workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches
  patch_size_xyz: [64, 64, 32]
  split: train

model:
  base_dim: 64
  spatial_shape_dhw: [32, 64, 64]
  diffusion_num_channels: 4
  cond_dim: 64
  temporal_max_distance: 128
```

不同 patch 尺寸应分别指定 `results_folder` 和 W&B run name，禁止共享 checkpoint 或输出目录。

## 验收门禁

### 静态和单测

- Hydra 配置可完整解析，不含 `???`；
- loss 数值验证覆盖非空单元等权平均、空单元跳过和全空 batch 报错；
- mask、histogram、输入张量 device 一致；
- LIDC/EMIDEC 回归测试通过。

### 真实数据 smoke

- `64×64×32`：真实 batch `[B,4,32,64,64]`，完成前向、反向和固定 batch overfit；
- `80×96×80`：真实 batch `[B,4,80,80,96]`，至少完成 batch size 1 前向和反向；
- 记录峰值显存；
- smoke 不生成正式 checkpoint，也不作为正式模型性能结论。

## 范围边界

本方案覆盖 GLI 训练接入和矩形 shape 基础设施，不包含：

- GLI inference loader；
- 四通道 RePaint keep-mask；
- GLI histogram cluster 文件；
- 四通道输出合并和下游分割评估。

这些内容应作为后续独立方法实验管理。

