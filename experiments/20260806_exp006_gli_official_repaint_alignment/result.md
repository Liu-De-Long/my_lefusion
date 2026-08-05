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

## 结果

待远端回归测试、完整 val QA 和同一 519 例 test 子集完成后填写。

## 输出隔离

- 旧版：`experiments/20260805_exp005_gli_formal_training_baseline/outputs/...`
- 修正版：`experiments/20260806_exp006_gli_official_repaint_alignment/outputs/...`
