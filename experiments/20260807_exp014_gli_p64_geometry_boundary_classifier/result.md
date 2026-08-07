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
- 正式 CPU/GPU preflight 与训练尚未运行。
- 冻结 test 未运行。

## 结论

等待 preflight 与验证集证据；不得从设计意图推断性能提升。

## 下一步

先执行 CPU 和 GPU0 零 optimizer-step preflight，验证真实 exp013 checkpoint warm-start、完整
p64 shape、loss/梯度、显存与输入泄漏边界。全部通过并同步文档后，才启动 GPU0 正式训练。

## 输出路径

`experiments/20260807_exp014_gli_p64_geometry_boundary_classifier/outputs/geometry_boundary/`
