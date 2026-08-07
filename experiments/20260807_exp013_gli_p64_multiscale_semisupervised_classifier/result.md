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

- 远端 classifier 单测 11/11 通过。
- 监督 CPU preflight 覆盖八个 `anchor × role` 层，半监督 CPU preflight 确认剩余 train pool
  为 6772 patch，SHA-256 为 `8248585a47003f33cf0f318b4df7170cee457b2f2ed40663d5112496b643057b`，
  且 `unlabeled_target_present=false`。
- GPU0 完整 p64 监督 preflight：batch 8，参数量 3,062,912，峰值 allocated/reserved 显存
  `2769/3914 MiB`，零 optimizer step，loss 与梯度有限。
- GPU0 完整 p64 半监督 preflight：labeled/unlabeled batch 各 4，峰值 allocated/reserved
  `2687/3292 MiB`，零 optimizer step，伪标签路径与梯度正常。
- 进一步只读检查发现 `/root/.netrc` 存在 W&B 主机认证；`wandb login --verify` 已在线确认
  当前登录 entity 与配置一致，未读取或输出 key。正式训练的 W&B 门禁已解除；当前尚无监督或
  半监督验证指标，冻结 test 未运行。

## 结论

代码、数据泄漏、资源与 W&B 认证 preflight 已通过，但 0.85 目标尚未验证。

## 下一步

先运行监督阶段，再从其绝对最佳 val checkpoint 顺序运行半监督阶段。
只有 focus mIoU、ET IoU、RC IoU、mask 外背景和 union 一致性五项验证门禁全部通过，才允许
对冻结 test 运行一次。

## 输出路径

`experiments/20260807_exp013_gli_p64_multiscale_semisupervised_classifier/outputs/`
