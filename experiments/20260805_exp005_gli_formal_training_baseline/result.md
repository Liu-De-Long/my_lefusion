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

- 已完成分层 sampler、正式 validation、per-channel loss/有效单元/覆盖指标、early stopping、best/latest/milestone、完整 checkpoint/resume 和 W&B fail-closed 接入。
- checkpoint 保存并校验 model、EMA、optimizer、scaler、optimizer step、sampler/epoch/batch offset、Python/NumPy/Torch/CUDA RNG、early-stopping 状态、resolved config/hash、split/manifest hash、Git SHA 和 W&B run ID。
- 两份正式 Hydra 配置均可直接解析且无 `???`；64 配置允许通过 override 改为 `batch=2/accumulation=2`，保持 effective batch 为 4。
- 远端全量 `unittest` 28/28 通过，包含真实发布 patch 的两种尺寸 loader；新增测试覆盖 sampler 平衡、validation 聚合、early stopping、checkpoint 元数据和精确数据序列 resume。
- exp004 normalization audit 的 stale 状态已更新为 `completed_no_systematic_background_pollution` 和 `keep_t1c_nonzero_percentile_normalization`，JSON 回读通过。
- 实现代码版本：`7a288dc2f59947db2c8bf00200a59f19f6e0bc7a`。
- 尚未读取或使用 W&B key，未登录 W&B、未创建 run、未做 GPU preflight、未启动训练。

## 结论

代码和非训练测试已通过，具备进入独立 preflight 的代码条件。当前仍不能启动正式训练：用户需先把新 key 安全配置到远端，且 W&B online、64 patch 显存、validation 和 resume preflight 均需另行授权并通过。

## 下一步

由用户安全配置远端 W&B 凭据；另行确认后执行不超过门禁范围的 online、显存、validation 和 resume preflight，不自动继续正式训练。

## 输出路径

正式运行预留路径：`experiments/20260805_exp005_gli_formal_training_baseline/outputs/`。当前未创建训练输出。
