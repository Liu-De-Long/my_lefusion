# exp018：exp016 伪四通道 mask 驱动 exp010 LeFusion

## 目标

在正式 p64 train 上用 exp016 epoch 24 best 划分 NETC/SNFH/ET/RC 四通道 mask，比较直接 mask
与按置信度过滤组件的 mask，并以相同 exp010 方法、采样序列和 FP32 训练配置完成两次全量训练。

## 方法与固定契约

- exp016 checkpoint SHA-256：`3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617`。
- direct 在真实 total lesion mask 内 argmax，只划分亚区，不预测病灶 union。
- filtered 使用 26 连通组件平均 max-softmax；val 的患者×类别等权 CRR–ERR 交点为 `0.964`。
- train/val sidecar 分别为 `7772/1032`，没有访问 test；原 MRI 不复制。
- 两次 LeFusion 保留真实 `anchor_label × sample_role` sampler；真实 val mask 用于选模，伪 val mask只作最终审计。

## 当前结果

- direct mask contract：`8804` 文件，files manifest SHA-256
  `ac906b0e759ab09e9212928a1654fe4289407c216a885f006cbfe1d9b24177be`。
- filtered mask contract：`8804` 文件；保留组件 `20579`、删除组件 `112694`，`334` 个 patch
  触发最高置信组件兜底；files manifest SHA-256
  `dd91c3963b52563255b41661afb1d53aebccfef8bd395a895eaadc5b7ad936c1`。
- 阈值交点：CRR `0.589991`、ERR `0.590110`、绝对差 `0.000119`。
- direct 全 train 患者等权 focus mIoU `0.671966`、macro IoU `0.666245`；未见 6772 patch
  focus `0.668695`，教师见过 1000 patch focus `0.734637`。
- filtered 全 train 覆盖率 `0.842106`；拒绝视为错误时 focus mIoU `0.433933`、macro IoU
  `0.473924`。该过滤是置信度拒绝路线，不应表述为提高完整-mask IoU。
- direct/filtered 双卡 FP32 preflight 均通过，真实 val 均为 `1032` patch、`73` subject、
  `3207` 有效单元、total loss `0.996759`，checkpoint 保存与恢复成功。
- direct preflight loss/grad norm 为 `0.914796/12.402021`；filtered 为
  `1.071336/21.619751`。两路线 GPU0/1 反向峰值显存均约 `15230.53/15070.36 MiB`。

## 状态

mask 生成、阈值冻结、初版审计、测试和两路线 preflight 已完成。正式训练按 direct 后 filtered
顺序执行；本文件将在两次 run 到达 50k、early stop 或 fail-closed 终态后补充最终结果。

## 输出路径

- sidecar：`/workspace/LeFusion_v2/dataset/brats2024_gli_exp016_pseudomasks/`
- mask 审计与阈值：`experiments/20260811_exp018_gli_exp016_pseudomask_lefusion/outputs/`
- W&B preflight：
  [direct](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp018-direct-mask-fp32-preflight-s20260805)、
  [filtered](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp018-filtered-mask-fp32-preflight-s20260805)。

## 结论与下一步

当前只可得出伪 mask 数据与训练接口已闭环、两种 FP32 配置可运行。正式 LeFusion 结果尚未完成；
不得启动 test，也不得从 mask 离线 IoU 推断下游生成效果。
