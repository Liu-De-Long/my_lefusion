# 20260805_exp005_gli_formal_training_baseline

## 目标

实现正式 GLI 训练所需的分层 sampler、监督 validation、完整 checkpoint/resume、early stopping 和 W&B online fail-closed 门禁；本阶段只实施代码和测试，不启动训练。

## 变更

- 分层 baseline 固定为 `anchor_label × sample_role` 平衡，并在层内按 subject 平衡。
- 保持现有 lesion-aware loss 规则不变。
- 训练参数全部配置化；micro-batch 可由显存 preflight 调整，gradient accumulation 同步补偿以保持 effective batch。
- `50,000` optimizer steps 是最大预算，不是必须跑满；以固定 validation 的 EMA 总 loss 执行 early stopping。
- 首轮只授权 seed `20260805` 的实施与 preflight；`20260806/20260807` 是正式复现候选，不是原始 LeFusion 代码的强制要求。

## 配置

- 第一阶段：`64×64×32`，初始 `batch=4/accumulation=1`，OOM 时允许在 preflight 后改为 `2/2` 或 `1/4`。
- 第二阶段对照：`80×96×80`，初始 `batch=1/accumulation=4`。
- effective batch 均为 4，LR `1e-4`，AMP 开启，gradient clipping `1.0`。
- validation 每 2,000 optimizer steps；latest 每 500 step；milestone 每 5,000 step并保留最近 3 个。
- W&B entity：`jinyuanbao719-xi-an-jiaotong-university-`；凭据由用户在远端安全配置，不进入项目或 Git。

## 结果

实施与测试进行中。尚未登录 W&B、创建 run 或启动训练。

## 结论

待代码、配置和测试全部通过后再判断是否具备 preflight 条件；preflight 与正式训练仍需另行授权。

## 下一步

完成实现、远端非训练测试和 Git 同步，然后报告剩余 blocker。

## 输出路径

正式运行预留路径：`experiments/20260805_exp005_gli_formal_training_baseline/outputs/`。当前未创建训练输出。
