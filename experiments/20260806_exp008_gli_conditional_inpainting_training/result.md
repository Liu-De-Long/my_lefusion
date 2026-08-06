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

条件式训练契约、真实数据、显存、validation、W&B 与 resume 门禁均通过，可以启动且只启动
p64 seed `20260805` 正式训练。

## 下一步

启动且只启动本实验的 p64 seed `20260805` 正式训练，并核验首批 loss、W&B 与 GPU 状态。

## 输出路径

`experiments/20260806_exp008_gli_conditional_inpainting_training/outputs/`
