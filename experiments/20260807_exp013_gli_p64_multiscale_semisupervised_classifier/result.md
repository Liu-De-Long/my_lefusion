# exp013 p64 多尺度与半监督逐体素亚区分类

## 目标

在 exp009 冻结的同一 1000 个有标签 p64 patch 上，将 ET/RC 患者等权 focus mIoU 从轻量
C0 的 0.5436 提升至至少 0.85；监督阶段使用多尺度残差 3D U-Net，随后允许剩余 train
patch 以无标签方式进入 EMA teacher 一致性与高置信伪标签训练。

## 变更

- 新增约 306 万参数的两级多尺度残差 3D U-Net，bottleneck 使用 dilation 1/2/4。
- 拆分绝对最佳 checkpoint 与 early-stopping min delta，任何验证新高都会保存。
- 新增 focus-weighted CE/focal/Dice、三轴翻转与 T1c 强度扰动。
- 新增 EMA teacher、强弱扰动一致性、mask 内高置信按类限额伪标签。
- 新增 W&B online fail-closed、监督/半监督独立 run ID 和完整 p64 GPU 零步 preflight。

## 配置

- `config_supervised.yaml`：冻结 1000 patch 的监督多尺度残差 3D U-Net。
- `config_mean_teacher.yaml`：从监督最佳 checkpoint 初始化，使用剩余 train patch 的
  mean-teacher 半监督阶段。
- 输入白名单仍为 T1c、总病灶 mask 及其无标签派生；未标签 patch 不加载四分类 target。
- 所有正式训练必须 W&B online fail-closed。

## 结果

- 最终远端 classifier 单测 12/12 通过；新增批量空间验证与单 patch 验证等价性覆盖。
- 监督 CPU preflight 覆盖八个 `anchor × role` 层，半监督 CPU preflight 确认剩余 train pool
  为 6772 patch，SHA-256 为 `8248585a47003f33cf0f318b4df7170cee457b2f2ed40663d5112496b643057b`，
  且 `unlabeled_target_present=false`。
- GPU0 完整 p64 监督 preflight：batch 8，参数量 3,062,912，峰值 allocated/reserved 显存
  `2769/3914 MiB`，零 optimizer step，loss 与梯度有限。
- GPU0 完整 p64 半监督 preflight：labeled/unlabeled batch 各 4，峰值 allocated/reserved
  `2687/3292 MiB`，零 optimizer step，伪标签路径与梯度正常。
- 进一步只读检查发现 `/root/.netrc` 存在 W&B 主机认证；`wandb login --verify` 已在线确认
  当前登录 entity 与配置一致，未读取或输出 key。
- 监督阶段运行 40 epoch，患者等权最佳 focus mIoU 为 `0.573000`，95% CI
  `[0.529404,0.618380]`；ET/RC IoU 为 `0.510031/0.635969`，macro IoU 为 `0.586163`，
  mask 外非零率为 `0`，union Dice 为 `1`。checkpoint SHA-256 为
  `de3636a2b169373c1b2e0bfb678dec07c7f7136a0cb1465c69ac6c2f6646b37a`。
- mean-teacher 使用全部 6772 个剩余 train patch 的无标签池，在 epoch 8 因连续八次无显著
  改善早停；最佳出现在 epoch 0，focus mIoU 为 `0.572729`，95% CI
  `[0.526628,0.617570]`，ET/RC IoU 为 `0.523194/0.622264`。伪标签覆盖约 `19%–20%`，
  平均置信度约 `0.993–0.995`，但没有带来验证增益。
- 监督 W&B：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp013-unet-sup-s20260807>；
  mean-teacher W&B：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp013-unet-mt-s20260807>。
- 两阶段均未通过 focus mIoU、ET IoU、RC IoU 三项性能门禁；冻结 test 从未运行。

## 结论

多尺度残差 U-Net 相对 exp009 C0 的历史绝对最佳 `0.5482` 提升约 `0.0248`，但仍远低于
`0.85`。mean-teacher 的高置信伪标签主要复制监督教师偏差，未产生正增益，因此 exp013
最终保留监督模型为验证最优候选，但不晋级 test。

## 下一步

下一方法实验保持同一冻结 1000 patch 和患者划分，优先加入仅由总 mask 推导的距离、bbox、
质心与 patch 坐标空间通道，并改善 ET/RC 边界监督；不再重复当前 mean-teacher 配方。
验证五项门禁全部通过前，冻结 test 继续封存。

## 输出路径

`experiments/20260807_exp013_gli_p64_multiscale_semisupervised_classifier/outputs/`
