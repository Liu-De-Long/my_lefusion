# exp008 条件式病灶修复训练

## 目标

把 GLI 训练从“完整病灶图加噪、mask 仅参与 loss”改为与伪病灶生成任务一致的条件式修复：模型必须根据挖空后的 T1c 背景、四通道病灶 mask 和 histogram 生成病灶区域。

## 变更

- 扩散目标 `x0` 保留原始病灶 T1c，并按 NETC/SNFH/ET/RC 复制为四通道。
- 在所有真实病灶的 union 内把空间上下文置为 `0.0`，形成单通道 `masked_context`。
- denoiser 每步接收四通道 `x_t`、单通道 `masked_context` 和四通道 lesion mask，共 9 个空间输入通道。
- histogram 仍为四组 16-bin block，共 64 维全局条件。
- denoiser 输出四通道噪声预测；loss 仍只在对应病灶通道内计算，并对非空 `(sample, lesion channel)` 等权平均。
- 训练与 RePaint 推理复用同一空间条件契约。

## 配置

- patch：`64x64x32`
- seed：`20260805`
- 最大步数：`50000`
- effective batch：`4`
- W&B run：`exp008-p64-s20260805`
- checkpoint：`outputs/patch_64x64x32/seed_20260805/checkpoints/`

## 结果

- 真实发布 patch 环境的远端全套回归 `34/34` 通过。
- Hydra 冻结配置确认空间条件为 5 通道，正式与 preflight W&B run ID、checkpoint 输出目录相互独立。
- online preflight 严格执行 `0` 次 optimizer update；真实 batch shape 为
  `[4,4,32,64,64]`，loss `1.153605`，梯度范数 `13.804416`。
- 反向峰值显存 `23344.58 MiB`，完整 validation 峰值 `5499.75 MiB`。
- validation 覆盖 1032 patch、73 subject、4 个 anchor label、3207 个有效单元；EMA total
  loss `1.195817`，NETC/SNFH/ET/RC loss 为
  `0.913812/0.859972/0.876844/2.098268`。这些是未训练随机初始化模型的接线基准，不能作质量比较。
- checkpoint 保存/重载恢复成功，临时 `resume_preflight.pt` 已自动删除。
- W&B preflight：
  [exp008-p64-preflight-s20260805](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp008-p64-preflight-s20260805)。

### 预检失败记录

- 首次调用在模型计算和 W&B 初始化前被旧脚本的 `exp005-` run ID 硬编码拦截，没有占用 GPU、
  没有 optimizer update、没有有效训练产物。
- 已将门禁泛化为“run ID 必须包含 `preflight` 且不得等于正式 run ID”，并补充回归测试。

## 结论

条件式训练契约、真实数据、显存、validation、W&B 与 resume 门禁均通过；但最终是否有效必须
由完整反向采样 QA 判定，不能由 noise-prediction loss 单独判定。

## 下一步

先完成 p64 seed `20260805`，随后只在冻结的少量 val patch 上做完整采样 QA，不运行 test split。

## 输出路径

`experiments/20260806_exp008_gli_conditional_inpainting_training/outputs/`

## 正式训练启动记录

- 服务器启动时间：`2026-08-06 08:28:25 UTC`（北京时间 `16:28:25`）。
- 仅启动 p64 seed `20260805`；未启动 p80、其他 seed 或任何 test。
- 运行时 Git HEAD：`67a598d366b691c832125fee0f1a4f48c5df2471`；方法实现版本：
  `6e29f355294c762cfb8928f73abcb350200c2e70`。
- W&B 正式 run：
  [exp008-p64-s20260805](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp008-p64-s20260805)。
- 启动核验时 GPU 0/1 显存约 `13383/13161 MiB`，两卡均有计算利用率。
- W&B 本地流已写到约 optimizer step `164`；最近抽查的 `train/total_loss` 为
  `0.376631/0.341066/0.232471/0.402093`，均为有限值，训练未在初始化或首批次失败。
- 正式日志：`outputs/patch_64x64x32/seed_20260805/train.run.log`；checkpoint 按每 500 step
  latest、每 5000 step milestone、每 2000 step validation/best 的冻结规则生成。
- 首个 `latest.pt` 已于服务器时间 `08:31:55 UTC` 写入，文件约 `580.4 MB`；写入后训练进程
  与两张 GPU 均继续正常运行。

## 正式训练完成记录

- 训练在 step `50000` 正常完成，未 early stop；W&B 已完整同步。
- 最后一次记录的 train loss 为 `0.056426`，最终 EMA validation total loss 为 `0.091035`；
  NETC/SNFH/ET/RC validation loss 为 `0.103730/0.069262/0.116849/0.084226`。
- 正式 `best.pt` 为 step `48000`、`ema` 权重，checkpoint SHA-256 为
  `660e23533064eaf9f32310b42500a8067c476dbd2bddbfd88a88dbc0f2c64857`。
- 最终保留 `milestone-40000.pt`、`milestone-45000.pt`、`milestone-50000.pt`、`latest.pt`
  与 `best.pt`；训练结束后未发现残留训练进程。

## 8 例完整采样 QA

### 范围与配置

- QA 配置 commit：`16d0a4e`；focused Hydra/checkpoint 配置测试 `1/1` 通过。
- 只复用冻结的 8 个 val patch，selection manifest SHA-256 为
  `51a67b56b0fc9a4d256994a880ca6676a34a8c9528ba1a026a9b8558bf7f4b79`。
- `masked_multilabel`：保留原四通道 lesion mask，各标签使用对应的最近 train-only cluster hist。
- `masked_anchor_union`：完整病灶 union 只写入 anchor label 通道，其他 mask 通道和 hist block 置零。
- 两组均使用 step `48000` `best.pt/ema`、完整 `t_T=300`、每例 300 次模型调用；各完成
  `8/8`、共 `2400` 次模型调用。未运行 519 例、test split 或任何全量 test。

### 工程门禁

- 两组均为 8 个唯一输入，生成 NPZ `8`、NIfTI `16`、五联 QA PNG `8`；所有数组和指标有限。
- 病灶区挖空最大绝对值严格为 `0`；病灶均位于 explicit support 内。
- 多标签 mask 与源 mask 逐体素一致；anchor-union 仅目标通道非空，其余 mask/hist block 为零。
- 两组显存稳定，峰值约 `1095.80/1098.68 MiB`，耗时约 `166.15/166.31 s`。

### 生成质量

- `masked_multilabel` 病灶区生成绝对强度均值为 `0.00834`，`masked_anchor_union` 为
  `0.00949`；对应原始病灶均值为 `0.32388`。生成值几乎等于挖空填充值 `0`。
- 病灶区相对原图 MAE 为 `0.32359/0.32052`；两种 conditioning 的病灶区配对输出 MAE
  仅 `0.00588`，说明 hist/mask 模式改变对结果影响很小。
- 各标签生成 histogram 对目标 cluster 的 L1 约为 `1.49–1.98`，接近明显失配范围。
- 病灶边界 jump p95 增量均值为 `0.24496/0.24166`，最大约 `0.51567/0.51564`。
- 五联图目检一致显示：原始病灶被正确挖空，但生成输出仍是接近零的均匀灰色区域，没有恢复
  可辨认的病灶强度或纹理。两种 QA 的图像几乎相同。

### QA 输出

- `outputs/patch_64x64x32/seed_20260805/val_qa_masked_multilabel_8cases/`
- `outputs/patch_64x64x32/seed_20260805/val_qa_masked_anchor_union_8cases/`
- 每个目录保留 `metrics.json`、`manifest.csv`、`run_contract.json`、逐例 NPZ/NIfTI、五联图和
  `qa_contact_sheet.png`。

## 最终结论

exp008 的训练、checkpoint、条件输入和完整采样工程闭环均正确运行，但生成质量失败。低
noise-prediction validation loss 没有转化为从随机噪声恢复病灶的能力；当前 checkpoint 在病灶区
退化到接近挖空值 `0`，不能作为有效伪病灶生成模型，也不能扩展到 519 例或全量 test。

## 后续建议

先用少量 patch 做 timestep 分层诊断，比较 teacher-forced epsilon 预测、由预测 epsilon 反推的
`x0` 误差和完整自由采样，定位训练目标与反向生成之间的失配；再决定是否增加 `x0`、histogram
或边界重建约束。未经新方案确认，不启动新的训练、seed、p80 或扩大测试规模。
