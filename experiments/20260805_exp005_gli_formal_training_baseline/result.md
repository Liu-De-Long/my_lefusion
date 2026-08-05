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
- 正式首个 run 使用两张空闲 A100 80GB 的 `DataParallel`（GPU 0、1），全局 batch 4 均分为每卡 2；effective batch、LR 和训练步数不变。
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
- 代码实施阶段未读取或使用 W&B key，也未登录、创建 run 或启动训练；后续获得用户单独授权后才执行本节记录的 online preflight。

## Preflight 结果

- 用户已在远端安全配置新 W&B key；本项目未读取、打印、复制或写入该 key。
- 在线 preflight 成功 run：[exp005-p64-preflight-s20260805-r3](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-preflight-s20260805-r3)。
- 预检严格执行 `0` 次 optimizer update；仅进行一次固定 batch 的 forward/backward 来测显存和梯度有效性，随后做完整 validation 与 checkpoint reload，不构成正式训练。
- 输入 shape 为 `[4,4,32,64,64]`；preflight loss `1.132638`；gradient norm（clip 前）`9.194938`；反向峰值显存 `23330.56 MiB`；完整 validation 峰值显存 `5482.45 MiB`。
- 完整 validation 覆盖 `1032` patch、`73` subject、4 个 anchor label、`3207` 个有效 `(sample,channel)` 单元；EMA total loss `0.959372`，NETC/SNFH/ET/RC loss 分别为 `0.893387/0.904111/1.140945/0.887598`。
- checkpoint 重载恢复 step、epoch、batch offset、sampler/RNG 等状态并能继续取得下一 batch；临时 `resume_preflight.pt` 已自动删除，未留下可被误用为正式模型的 checkpoint。
- 预检轻量输出保留在远端 `outputs/patch_64x64x32/preflight/metrics.json` 与日志中。

## 失败记录与修复

- 首次 online preflight [run](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-preflight-s20260805) 因本地 SSH 会话在 30 秒传输限制中断，未形成 metrics；未启动训练。
- 第二次 retry [run](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-preflight-s20260805-r2) 在 validation 发现真实边界 patch 的合法负 `origin_xyz` 被 loader 错拒绝。
- 已在 `b34d867f944343f9f6e4edde5abe3a0b0805b` 修复 origin 契约并补充回归测试；修复后真实 patch 根目录下远端全量测试 `28/28` 通过。

## 结论

W&B online、64 patch 显存、validation 与 resume preflight 均已通过。当前不启动正式训练仅因为用户尚未作出新的正式训练授权；技术门禁已不再阻塞首个 `64×64×32` seed。

## 下一步

等待用户单独确认是否启动 `64×64×32` 的正式 seed `20260805`；得到授权后仍只运行该一个 seed，不自动扩展到另外两个 seed 或 80 patch 对照。

## 输出路径

正式运行输出路径：`experiments/20260805_exp005_gli_formal_training_baseline/outputs/patch_64x64x32/seed_20260805/`。训练正在运行，checkpoint 将按既定规则生成。

## 正式训练启动记录

- 启动时间：2026-08-05；仅启动 `64×64×32`、seed `20260805`，未启动其他 seed 或 `80×96×80`。
- 运行方式：GPU 0、1 的 `DataParallel`；全局 batch 为 4（每卡 2），gradient accumulation 为 1。
- 运行时 Git HEAD：`ea88464f3bf51350f5bd7d33f1bfcc4d7f80b6c1`；实现和配置提交：`75b187ce6664d4fa44ddad32dd328112998ff5c4`。
- W&B online run：[exp005-p64-s20260805](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-s20260805)。
- 输出目录：`outputs/patch_64x64x32/seed_20260805/`；运行日志：`train.run.log`；checkpoint 将由既定的 latest/best/milestone 规则写入该目录。

## 正式闭环 QA 与半量 test 授权

- 2026-08-06，用户固定使用 `best.pt` 的 `ema` 权重，不在 QA/test 阶段改选 checkpoint。
- 先完成 schema 2 inference 兼容、best EMA 固定 val、actual-checkpoint 零更新 resume 和完整 `t_T=300` val inference QA；任一门禁失败即停止。
- val 全部门禁通过后，自动运行 test 的确定性 50% patch 子集，不运行全量 test。
- test 子集按 `anchor_label × sample_role` 分层，以固定 seed `20260806` 和稳定路径 SHA-256 顺序选择精确 `floor(N/2)`，冻结 subset manifest 后使用 GPU 0、1 独立分片。
- 详细接口、验收阈值、输出目录和合并契约记录于 `docs/20260805_003_gli_inference_closed_loop.md` 第 14 节。
