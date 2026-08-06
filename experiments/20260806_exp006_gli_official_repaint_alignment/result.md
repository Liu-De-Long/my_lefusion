# 20260806_exp006_gli_official_repaint_alignment

## 目标

将 GLI inference 修正为原始 LeFusion 的 RePaint 调用语义：每次反向调用前向病灶外注入当前
timestep 的真实背景前向加噪状态，反向去噪输出直接进入下一步；删除 exp005 评估实现中额外
加入的反向输出后 `target_background` hard clamp。

## 冻结项

- 不重新训练；固定复用 exp005 的 step 46000 `best.pt/ema`。
- 不改训练 loss、normalization、四通道标签顺序或 train-only histogram cluster。
- 固定复用 exp005 的 8 例 val QA manifest 和 519 例 test 50% subset manifest。
- test 仍按原有 260/259 两个互斥 shard 使用 GPU 0/1。
- exp005 的 hard-clamp val/test 输出全部保留，不删除、不覆盖，作为错误版本对照。

## 方法修正

1. 保留 denoiser 前的前向加噪背景注入。
2. 每次反向调用重新采样一份 `[B,1,D,H,W]` 背景噪声，并在四个 lesion channel 间共享；
   不再跨 timestep 缓存同一份 `background_noise`。
3. 删除 denoiser 输出后的 `target_background` 二次硬覆盖。
4. 终态 lesion voxel 按 scalar label 选择 NETC/SNFH/ET/RC 对应 channel；病灶外固定选择
   channel 0，不平均四个通道，也不覆盖原始图像。
5. QA 不再把 healthy brain、outer shell 或 support 外变化严格等于 0 作为门禁；改为记录
   MAE、p95、最大变化、变化比例，以及 lesion/healthy 边界 6 邻域强度跳变相对输入的变化。
6. 使用 `scripts/gli_audit_inference_subset.py` 对两个 shard 执行冻结子集并集/交集、provenance、
   四通道、interior/boundary、support、异常样本以及 exp005/exp006 配对差异审计。

## 结果

### 代码与回归

- inference 修正实现提交：`d94f8b92fa5eb3b1f1a07273484f74cd81bc0c2b`；正式配对审计脚本
  提交：`1ea165d`。
- 新增 regression 覆盖“pre-denoiser 四通道共享背景”“每次反向调用 fresh noise”以及
  “post-denoiser 不 hard clamp”；远端真实发布 patch 环境全量测试 `31/31` 通过。
- 三份 exp006 Hydra 配置完整解析，固定 `best.pt/ema`、`t_T=300` 与独立输出目录。

### 完整 val QA

- 8 个冻结样本全部完成，每例 300 model calls，总耗时 `158.996 s`；峰值显存
  `1088.71 MiB`，末三例 allocated/reserved span 为 `1/0 MiB`，判定稳定。
- healthy-brain MAE 均值/最差为 `0.009058/0.009296`；support 外变化 MAE 为
  `0.008722/0.011476`；lesion outer-shell MAE 为 `0.008919/0.009461`。
- lesion/healthy 边界 jump p95 相对输入的最大增量为 `0.004737`；8 张 QA 图目检未见明显
  颅外污染、亮带或硬接缝。完整 val QA 通过。

### test 50% 冻结子集

- 继续复用冻结 manifest SHA-256
  `295b20a01327dcd0071058efd8fe854135688a843c1888ef33242a908b6c3698`；覆盖 519 patch、
  73 subject，anchor label 为 `1:74, 2:174, 3:129, 4:142`，role 为
  `interior:259, boundary:260`。
- GPU 0/1 shard 为 `260/259`，交集 0，并集严格等于冻结子集；每个 shard 的 NPZ/QA 数分别
  为 `260/260` 与 `259/259`，NIfTI 数为 `520/518`。每例 300 model calls。
- 两 shard 耗时 `5174.26/5145.91 s`，峰值显存均为 `1090.71 MiB`，显存稳定；结束后 GPU
  均释放。修正版 test 目录约 `6.6 GiB`。
- 多标签分布：仅 1/2/3/4 个 lesion label 的 patch 为 `22/111/216/170`；所有 lesion 均在
  explicit support 内，`lesion_outside_support_voxels=0`。

### 背景、边界与四通道指标

- healthy-brain MAE mean/p95/max：`0.009122/0.009696/0.010513`。
- support 外 change MAE mean/p95/max（494 个存在 outside voxel 的 patch）：
  `0.008953/0.010520/0.022897`。
- lesion outer-shell MAE mean/p95/max：`0.008973/0.009716/0.010746`。
- healthy、support 外和 outer-shell 的变化大于 `0.1` 的比例在全部有效 patch 中均为 `0`。
- 边界 jump p95 增量 mean/p95/max：`-0.001774/0.004392/0.014357`；interior 与 boundary 的
  outer-shell MAE 分别为 `0.008953/0.008992`，未见 role 相关退化。
- NETC/SNFH/ET/RC 的 present patch 为 `243/519/402/408`，lesion change MAE 均值为
  `0.010038/0.010379/0.010303/0.010995`；histogram L1 为
  `0.476755/0.510642/0.439694/0.688289`，Wasserstein 为
  `0.081254/0.083504/0.131476/0.139706`。
- 与 exp005 同路径配对后，修正版 lesion change MAE 平均差为 `-0.000631`；四通道 histogram
  距离整体与旧版接近。三张数值最差 QA 图目检仍未见明显背景污染或边界断裂。

## 结论

exp006 的原始 LeFusion RePaint 对齐实现、完整 val QA 和同一 519 例 test 子集均通过工程
门禁。step 46000 `best.pt/ema` 可作为 `64×64×32` 的正式冻结工程 baseline，并固定搭配
exp006 inference 语义；exp005 hard-clamp 输出仅保留作错误版本对照。该结论不等于医学有效性
或全量 test 结论，后续仍需独立下游/临床指标验证。

## 输出隔离

- 旧版：`experiments/20260805_exp005_gli_formal_training_baseline/outputs/...`
- 修正版：`experiments/20260806_exp006_gli_official_repaint_alignment/outputs/...`
- 正式审计：`.../test_subset_50/paired_audit/audit.json` 与 `anomaly_samples.csv`
