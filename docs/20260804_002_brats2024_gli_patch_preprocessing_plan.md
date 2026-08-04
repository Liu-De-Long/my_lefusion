# BraTS2024 GLI 病灶统计与 LeFusion 预处理方案

## 日期

2026-08-04

## 目标

为将 LeFusion 接入 BraTS2024 GLI 数据集做正式预处理设计，重点回答病灶区域大小如何分布、选择多大的 patch 更合理、是否应该缩放或裁剪、以及如何结合 LeFusion 的 lesion-focused loss、histogram texture control、multi-channel decomposition 和 lesion mask diffusion 设计 GLI 数据流。

## 参考依据

- LeFusion 论文与 ICLR 2025 页面说明：LeFusion 只重建病灶区域，背景由 forward-diffused context 保留；对复杂病灶使用 histogram-based texture control 和 multi-channel decomposition；mask diffusion 用于控制病灶大小、位置和边界。
- BraTS2024 post-treatment glioma 相关说明：四个 MRI 模态为 `t1c/t1n/t2f/t2w`，标签语义通常为 `1=NETC`、`2=SNFH`、`3=ET`、`4=RC`。组合区域为 `TC=NETC+ET`，`WT=NETC+SNFH+ET`，`RC` 是 resection cavity，不应默认合入肿瘤核心。

注意：本数据目录顶层 metadata 文件名包含 `PTG`，且 `val` 数量为 `188`，与 BraTS post-treatment glioma 验证集规模一致。因此本轮统计按 PTG 标签语义命名。后续仍建议用官方数据说明或 metadata 再做一次确认。

## 统计脚本

已新增正式可复用脚本：

```text
scripts/brats_gli_lesion_patch_stats.py
```

用途：

- 统计每个病例、每个病灶区域的体素量、物理体积和 3D bbox。
- 统计每个病例、每个病灶区域、每个 MRI 模态的病灶内强度分布。
- 输出 patch 建议和机器可读汇总，供 GLI dataloader、训练配置和 histogram 条件设计复用。

全量 mask/bbox 统计命令：

```bash
cd /workspace/LeFusion_v2/my_experiment
python scripts/brats_gli_lesion_patch_stats.py \
  --data-root /workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI \
  --split train \
  --out-dir experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/region_patch_stats_full \
  --skip-intensity
```

100 例模态强度抽样命令：

```bash
cd /workspace/LeFusion_v2/my_experiment
python scripts/brats_gli_lesion_patch_stats.py \
  --data-root /workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI \
  --split train \
  --out-dir experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/modality_intensity_stats_100cases \
  --max-cases 100 \
  --intensity-sample-limit 100000
```

## 输出目录

全量病灶 bbox 统计：

```text
/workspace/LeFusion_v2/my_experiment/experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/region_patch_stats_full
```

100 例模态强度抽样：

```text
/workspace/LeFusion_v2/my_experiment/experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/modality_intensity_stats_100cases
```

主要输出文件：

- `case_region_stats.csv`：每个病例、每个病灶区域的体素量、体积和 bbox。
- `region_summary.csv`：病灶区域体积和 bbox 分位数。
- `modality_intensity_stats.csv`：每个病例、区域、模态的病灶强度统计。
- `modality_intensity_summary.csv`：每个区域、模态的强度聚合统计。
- `summary.json`：机器可读 patch 建议。
- `patch_recommendations.md`：人读 patch 建议。

## 全量病灶大小统计

本次全量统计范围为 `train` split，病例数为 `1621`。

| 区域 | 非空病例数 | p90 bbox | p95 bbox | p90 patch | p95 patch |
|---|---:|---|---|---|---|
| `abnormality_including_rc` = 1+2+3+4 | 1621 | `95 x 120 x 92` | `103 x 129 x 98` | `96 x 128 x 96` | `112 x 144 x 104` |
| `wt_without_rc` = 1+2+3 | 1619 | `95 x 120 x 91` | `103 x 129 x 98` | `96 x 128 x 96` | `112 x 144 x 104` |
| `tc_without_rc` = 1+3 | 1229 | `59 x 74 x 64` | `68.6 x 86 x 73.6` | `64 x 80 x 64` | `80 x 96 x 80` |
| `label_1_netc` | 706 | `47.5 x 60 x 49.5` | `55 x 68.75 x 58` | `48 x 64 x 56` | `64 x 80 x 64` |
| `label_2_snfh` | 1618 | `94 x 120 x 90` | `101.15 x 129 x 98` | `96 x 128 x 96` | `112 x 144 x 104` |
| `label_3_et` | 1223 | `59 x 74 x 64` | `68 x 86 x 73` | `64 x 80 x 64` | `80 x 96 x 80` |
| `label_4_rc` | 1374 | `49 x 62 x 58` | `54 x 75 x 68` | `64 x 64 x 64` | `64 x 80 x 72` |

结论：

- 如果目标是覆盖完整 `WT` 或包含 `RC` 的整体异常区域，建议优先评估 `112 x 144 x 104`。
- 如果目标是先做 LeFusion 的肿瘤核心或增强区域生成，建议优先评估 `80 x 96 x 80`。
- 如果显存压力较大，可以从 `96 x 128 x 96` 的 `WT` p90 覆盖版本开始，但需要记录会截断约 10% 大 bbox 病例。

## 模态强度观察

100 例抽样显示，各模态病灶强度分布差异明显：

- `t1c` 中 `label_3_et` 的病灶强度最高，符合增强组织预期。
- `t2f` 对 `label_2_snfh` 和 `WT` 更敏感，但强度尺度低于 `t1c/t1n`。
- `t2w` 中 `label_4_rc` 强度较高，提示 resection cavity 的纹理与肿瘤组织应分开建模。
- 原始强度范围不是 `[-1, 1]`，不能直接套用 EMIDEC 的 `RescaleIntensity(out_min_max=(-1, 1))` 后混合四模态计算 histogram，必须先做每模态归一化。

## 推荐预处理方案

### 1. 数据索引

只使用 `train` split 做监督训练，因为 `val` 没有 `seg`。每个病例固定读取：

```text
t1c, t1n, t2f, t2w, seg
```

模态顺序必须写入配置，建议固定为：

```text
["t1c", "t1n", "t2f", "t2w"]
```

### 2. 标签映射

建议先按 PTG 语义建模：

```text
1 = NETC
2 = SNFH
3 = ET
4 = RC
TC = 1 + 3
WT = 1 + 2 + 3
abnormality = 1 + 2 + 3 + 4
```

LeFusion 多通道分解建议有两个可选阶段：

- 阶段 A：`WT` 单通道或 `WT + RC` 双通道，先验证背景保真和 patch 流程。
- 阶段 B：`SNFH / TC / RC` 或 `NETC / SNFH / ET / RC` 多通道，正式利用 multi-channel decomposition。

不建议一开始直接把 `1/2/3/4` 全部作为独立生成通道训练大模型，因为类别长尾明显，`label_1_netc` 非空病例只有 `706/1621`。

### 3. 空间处理

原始数据已经是 `182 x 218 x 182`、`1mm` spacing，现阶段不建议重采样缩放。

推荐做 mask-centered crop：

- 初始可行配置：`80 x 96 x 80`，用于 `TC/ET` 或小规模冒烟。
- 推荐主配置：`112 x 144 x 104`，用于覆盖 `WT` p95。
- 显存保守配置：`96 x 128 x 96`，覆盖 `WT` p90。

裁剪策略：

1. 根据目标区域 bbox 计算中心。
2. 训练时围绕中心加随机平移，增强位置鲁棒性。
3. patch 超出边界时使用 padding。
4. 对超大病例允许截断，但必须记录截断比例和被截断病例数。

### 4. 强度归一化

建议每个病例、每个模态单独归一化：

1. 用 brain foreground 或非零区域估计强度分布。
2. clip 到该模态非零区域的 `p0.5-p99.5` 或 `p1-p99`。
3. z-score 或 min-max 到 `[-1, 1]`。
4. histogram 条件只在归一化后的病灶区域内计算。

不建议直接跨模态共享一个强度归一化范围。

### 5. Histogram 条件

结合 LeFusion 现有 EMIDEC 代码，histogram 条件应从病灶 mask 内的归一化强度计算。

GLI 建议：

- 每个目标通道、每个模态计算 16-bin histogram。
- 若使用 `WT` 单通道和 4 模态，则 `cond_dim = 4 x 16 = 64`。
- 若使用 `SNFH/TC/RC` 三通道和 4 模态，则 `cond_dim = 3 x 4 x 16 = 192`。
- 若使用 `NETC/SNFH/ET/RC` 四通道和 4 模态，则 `cond_dim = 4 x 4 x 16 = 256`。

为降低第一版风险，建议先做：

```text
target_channels = ["wt_without_rc"]
modalities = ["t1c", "t1n", "t2f", "t2w"]
hist_bins = 16
cond_dim = 64
patch_size = [112, 144, 104]
```

### 6. LeFusion 接入方式

当前 LeFusion 代码更像 3D patch 输入，而不是整脑 volume 输入。GLI dataloader 应返回：

```text
{
  "data": image_patch_channels,
  "label": mask_patch_channels,
  "hist": histogram_condition
}
```

需要注意：

- `data` 不应简单 repeat 单模态图像，应明确支持 4 MRI 模态和 n 个 lesion channels。
- 如果沿用 LeFusion 的 multi-channel decomposition，可能需要把每个 lesion channel 对应的图像 patch 构造成同形张量，并调整 `diffusion_num_channels`。
- loss mask 应只关注目标病灶区域，背景通过 LeFusion 的 forward-diffused context 保留。
- 推理阶段需要先有目标 lesion mask；可先使用真实 mask 做 reconstruction/sanity check，再做 DiffMask 或 mask perturbation。

## 推荐下一步

1. 确认官方标签语义，尤其 `label_4` 是否为 `RC`。
2. 实现 GLI dataset v1：`WT` 单通道、4 模态输入、`patch_size=[112,144,104]`。
3. 添加截断比例统计，验证 p95 patch 是否足够。
4. 先做 5-10 例 overfit 冒烟实验，不直接全量训练。
5. 再扩展到 `SNFH/TC/RC` 多通道和更高维 histogram 条件。
