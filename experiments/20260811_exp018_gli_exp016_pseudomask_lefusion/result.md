# exp018：exp016 伪四通道 mask 驱动 exp010 LeFusion

## 目标与固定契约

- 用 exp016 epoch 24 best（SHA-256
  `3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617`）在正式 p64
  train/val 内划分 NETC/SNFH/ET/RC，不运行 test。
- direct 在真实 total lesion mask 内 argmax；filtered 对 26 连通组件按平均 max-softmax 过滤。
- val 上患者×类别等权 CRR–ERR 交点冻结为 `0.964`：CRR `0.589991`、ERR `0.590110`。
- 两路线沿用 exp010、真实 `anchor_label × sample_role` sampler、seed `20260805`、双 A100、
  global batch 4、LR `1e-4`、FP32 和 50k step；真实 mask val 选模，overlay val 只审计。

## 伪 mask 统计

- train/val sidecar 为 `7772/1032`，两套各 `8804` 文件；direct union 精确等于真实 total mask。
- direct 全 train 的患者等权 macro IoU/Dice 为 `0.666245/0.722457`，ET/RC focus mIoU
  `0.671966`（bootstrap 95% CI `0.655603–0.689023`）。教师见过的 1000 patch focus
  mIoU 为 `0.734637`，未见的 6772 patch 为 `0.668695`。
- filtered 全 train 保留覆盖率 `0.842106`、拒绝率 `0.157894`；拒绝视为错误时 macro
  IoU/Dice 为 `0.473924/0.518069`，ET/RC focus mIoU `0.433933`（95% CI
  `0.411107–0.457330`）。过滤显著提高保留预测的 precision，但牺牲 recall，不能表述为提高完整-mask IoU。
- filtered 共保留 `20579`、删除 `112694` 个组件，`334` 个 patch 触发最高置信组件兜底。
- direct/filtered files manifest SHA-256 分别为
  `ac906b0e759ab09e9212928a1654fe4289407c216a885f006cbfe1d9b24177be` 和
  `dd91c3963b52563255b41661afb1d53aebccfef8bd395a895eaadc5b7ad936c1`；审计均为
  `test_accessed=false`。

## 正式训练结果

| 路线 | 终止 step | best step | best checkpoint SHA-256 | 真实-mask val loss |
|---|---:|---:|---|---:|
| direct | 50000 | 44000 | `3e3c30585ef27c3540a8f3e3bf0bc3f4b48e1c1fb84ac7b85c06adfac3f70e0a` | `0.1126854883` |
| filtered | 50000 | 46000 | `5ddd8f112e0d63fc671f641351010cd2fe0fe11a5009f25f3df4e31cf617ff6c` | `0.1122985579` |

- 两次运行都正常到达 50k，没有 NaN/Inf。checkpoint gate 复算误差分别为
  `7.56e-09/1.34e-08`，均低于 `1e-6`；latest checkpoint 两次恢复后的下一 batch SHA-256
  一致，且 gate 为零 optimizer update。
- direct best 在 direct overlay val 上 total loss `0.1100249112`；filtered best 在 filtered
  overlay val 上为 `0.0909269676`。filtered overlay 只有 `1756` 个有效 lesion unit，不能与
  direct 的 `3367` 个有效单元直接作生成质量优劣比较。
- 固定 8-patch QA 的 lesion MAE（真实/对应 overlay）为 direct
  `0.158310/0.149746`、filtered `0.160406/0.124033`；histogram L1 为 direct
  `0.427756/0.408914`、filtered `0.511716/0.274987`。
- 四组固定 QA 的非零率、非平坦率和背景精确不变率均为 `1.0`，背景最大绝对变化为 `0`。
  filtered overlay 中只有 10 个非空 class-unit，空类按设计不纳入 histogram L1。

## 审计 gate 修复与验收

- 旧 gate 错把条件模型的 `spatial_condition_channels` 默认为 0，导致正确的 5 通道 checkpoint
  被拒绝。提交 `3ffc9bae421ec9747741507f446e4cacf0ec43bb` 改为同时验证
  `spatial_condition_channels=5`、objective、GLI state mode 和 overlay contract hash。
- 修复后 direct/filtered gate、全量 train mask 审计、完整 1032-patch real/overlay val、固定
  8-patch paired QA 和最终聚合全部通过；审计链在 `2026-08-11T15:11:11Z` 以
  `phase=complete` 结束。
- 远端相关单元测试 `14/14` 通过；训练运行提交为
  `5b15bf9356a6c5b348d6f2b66302f3166cff6ff9`，审计代码提交为
  `3ffc9bae421ec9747741507f446e4cacf0ec43bb`。

## 产物

- sidecar：`/workspace/LeFusion_v2/dataset/brats2024_gli_exp016_pseudomasks/`
- 汇总：`outputs/best_validation/summary.json`
- 固定 QA 总览：`outputs/best_validation/qa_montage.png`
- direct/filtered gate：`outputs/{direct,filtered}_mask_fp32_50k/checkpoint_gate.json`
- direct/filtered 完整 val：`outputs/{direct,filtered}_mask_fp32_50k/best_validation/metrics.json`
- W&B 正式运行：[direct](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp018-direct-mask-fp32-50k-s20260805)、
  [filtered](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp018-filtered-mask-fp32-50k-s20260805)。

## 结论

两条伪 mask 路线均已完成同契约 50k 全量训练和成对审计。filtered 的真实-mask best val loss
仅比 direct 低 `0.000387`，而过滤会大量移除 NETC/ET/RC 有效单元；当前证据不足以宣称 filtered
优于 direct。实验不访问 test，结论限定于伪 mask 质量、真实 val loss 和固定 8-patch 工程审计。
