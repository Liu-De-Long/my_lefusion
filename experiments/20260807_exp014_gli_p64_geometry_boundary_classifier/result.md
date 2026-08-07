# exp014 p64 几何与边界监督分类

## 目标

在 exp009/exp013 同一冻结 1000 个有标签 p64 patch、同一患者划分和同一验证门禁下，验证仅由
T1c 与总病灶 mask 推导的显式几何/局部统计通道，以及 Lovász 与类间边界监督，能否把
ET/RC 患者等权 focus mIoU 从 exp013 的 `0.573000` 提升至至少 `0.85`。

## 变更

- 新增 `geometry_unet3d`：输入为 M1 的 17 个 leak-safe dense feature channel 加总 mask，
  共 18 通道；不读取真实四分类 hist、类别体素量或其他标签派生输入。
- 从 exp013 监督 `best.pt` warm-start：共享层完全复用；两个输入 stem 仅复制 T1c 与总 mask
  权重到新通道位置，其余 16 个几何通道权重初始化为 0。
- 新增 mask 内 Lovász-Softmax 与真实类间边界加权；边界只用于训练 loss，不进入模型输入。
- 保持完整 p64 前向、mask 内 loss、患者等权 val 和五项 test fail-closed 门禁。

## 配置

- 模型：base channels 32，约 307 万参数，两级多尺度残差 3D U-Net。
- 输入/输出：`[B,18,32,64,64] -> [B,4,32,64,64]`。
- loss：CE/Focal/Dice/Lovász=`0.25/0.10/0.25/0.40`，ET/RC multiplier `1.25`，
  类间边界 multiplier `1.5`。
- batch 8，初始学习率 `2e-4`，最多 50 epoch，patience 12。
- W&B online fail-closed run：`exp014-geometry-boundary-s20260807`。

## 结果

- 本地静态编译通过；远端 focused tests 15/15 通过。
- CPU 零 optimizer-step preflight 通过：输入/输出为 `[1,18,8,16,16] -> [1,4,8,16,16]`，
  loss `0.463277`，gradient norm `11.4095`，八个 `anchor×role` 层全部覆盖。
- GPU0 完整 p64 零 optimizer-step preflight 通过：batch 8，输入/输出为
  `[8,18,32,64,64] -> [8,4,32,64,64]`，loss `0.258514`，gradient norm `1.84317`，
  峰值 allocated/reserved 显存为 `2829/3958 MiB`。
- 两次 preflight 的初始 checkpoint SHA-256 均为
  `de3636a2b169373c1b2e0bfb678dec07c7f7136a0cb1465c69ac6c2f6646b37a`，冻结 subset SHA
  均为 `aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a`。
- 正式训练在 GPU0、W&B online run `exp014-geometry-boundary-s20260807` 上运行。服务器曾在
  epoch 6 期间重启；恢复前核验 `latest.pt` 的 epoch、config/subset/warm-start SHA、optimizer、
  scheduler 与 RNG 状态完整，随后从 epoch 5 原子 checkpoint 安全 resume。
- epoch 6 后患者等权 focus mIoU 长期在约 `0.56–0.59` 波动；学习率从 `2e-4` 先后降到
  `1e-4` 与 `5e-5`，未形成通向 `0.85` 的趋势。用户在 epoch 19 后触发方法级 early stop，
  最终保留 20 条 epoch 历史。
- 验证集绝对最佳为 epoch 19：患者等权 ET/RC focus mIoU `0.592211`，95% bootstrap CI
  `[0.547210,0.638076]`；ET/RC IoU 为 `0.522322/0.662099`，ET/RC Dice 为
  `0.589892/0.746911`，macro IoU/Dice 为 `0.595358/0.668240`，balanced accuracy
  `0.774047`，mask 内错误率 `0.080047`。
- 仅作参考的 pooled ET/RC IoU 为 `0.813983/0.775111`，明显高于患者等权指标，证明总体
  大病灶体素性能不能代表小病灶患者性能。
- 小区域是主瓶颈：`1–100` 体素 ET/RC patch Dice 仅 `0.124810/0.196575`，而 `>1000`
  体素 ET/RC patch Dice 已达 `0.877092/0.808596`。
- 空间门禁通过：mask 外非零率 `0`、union Dice `1`；性能门禁全部失败：focus mIoU 未达
  `0.85`，ET/RC IoU 均未达 `0.80`。冻结 test 未运行，也未生成 test 文件。
- best checkpoint SHA-256 为
  `b293e5d35b078f6a290a24695f0081e0ce0599154513bf6bcbf7372a007898c2`；最终冻结 val
  指标文件 SHA-256 为
  `166562c2823cc2e4562534f4d35ae49eab5220ababb62d7db1ed543123295e3e`。
- 终止信号使 W&B 一度将 run 标为 crashed 且只同步到 epoch 18；随后仅恢复同一个 run，补记
  epoch 19 最终 val、bootstrap、门禁和 `user_triggered_method_early_stop` 后正常 finish，未训练、
  未创建新 run。

## 结论

几何通道、Lovász 与边界监督相对 exp013 的 `0.573000` 有小幅增益，但不是达到患者等权
`0.85` 的主路径。模型对大区域 ET/RC 已有较强总体可分性，当前主要受小区域漏检/错分以及
患者等权聚合拖累；继续增加同一路线 epoch 的预期收益不足，因此已停止。

## 下一步

下一方法应直接改变小区域优化目标，而不是继续同配方训练：优先考虑 patch/患者等权的小区域
重采样或 loss、component-aware/hierarchical head，并先在 val 做不读取 test 的连通域/小组件
后处理审计。任何新训练应建立新实验 ID；仍复用患者隔离 split 和泄漏白名单，仍须五项门禁
全部通过后才允许一次冻结 test。

## 输出路径

`experiments/20260807_exp014_gli_p64_geometry_boundary_classifier/outputs/geometry_boundary/`

关键产物：`best.pt`、`latest.pt`、`history.jsonl`、`best_val_metrics.json`、`run_metadata.json`。

W&B：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp014-geometry-boundary-s20260807>
