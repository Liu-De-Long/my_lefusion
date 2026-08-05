# BraTS2024 GLI 正式训练方案与可行性判断

## 1. 文档目的

本文记录 GLI 正式全量训练前的标签、normalization、类别平衡、训练参数、validation、checkpoint、W&B 和项目管理决策，作为 `20260805_exp005_gli_formal_training_baseline` 及后续 patch-size 对照的长期实施依据。

本文只定义方案和启动门禁，不构成启动训练、创建 W&B run 或使用 test split 的授权。

## 2. 当前结论

- 当前不允许直接启动正式全量训练。
- 官方标签语义已经确认，不再是 blocker。
- 保留当前 `T1c != 0 -> p0.5/p99.5 -> clip -> [-1,1]` normalization。
- 类别平衡 baseline 使用分层 sampler，不改变现有 lesion-aware loss。
- 第一阶段先训练 `64×64×32`，第二阶段再训练 `80×96×80` 对照。
- 下一个实验 ID 为 `20260805_exp005_gli_formal_training_baseline`。
- 正式训练前仍需实现 sampler、validation、checkpoint/resume 和 W&B online fail-closed 接入，并完成显存与恢复 preflight。

## 3. 标签语义与代码通道顺序

### 3.1 官方语义

BraTS 2024 官方组织者的评测说明明确规定：

```text
0 = background
1 = NETC
2 = SNFH
3 = ET
4 = RC
```

官方说明同时定义 `WT=(1,2,3)`、`TC=(1,3)`。参考：

- BraTS 官方 Synapse 评测说明：<https://www.synapse.org/Synapse%3Asyn53708249/discussion/threadId%3D11350>
- BraTS 2024 GLI challenge paper：<https://arxiv.org/abs/2405.18368>
- TCIA MU-Glioma-Post：<https://www.cancerimagingarchive.net/collection/mu-glioma-post/>

当前数据 segmentation 只包含 `{0,1,2,3,4}`，全部病灶类别已对应 `1–4`，因此 `0` 为 background。证据充分，标签语义不再阻塞正式训练。

### 3.2 代码一致性

| 环节 | 当前顺序与约定 |
|---|---|
| patch 裁剪 | `{1:netc, 2:snfh, 3:et, 4:rc}` |
| NPZ histogram | 第 0–3 行依次对应 label 1–4 |
| loader | `LABEL_VALUES=(1,2,3,4)`，输出 NETC/SNFH/ET/RC 四通道 mask |
| 64-D condition | 四个 16-bin block 按 label 1–4 拼接 |
| loss | 直接使用上述四通道 `lesion_mask`，不重新映射 |
| cluster | 按 label 1–4 独立拟合和装载 |
| RePaint | 从 scalar segmentation 按 `(1,2,3,4)` 展开，并与传入 mask 严格比对 |
| 最终合成 | scalar label 选择对应生成 channel，背景取共享 sampler 状态 |

未发现 channel swap、ET/RC 旧编码混用或 histogram block 错位。

## 4. Normalization 决策

### 4.1 审计结果

exp004 对 64 个分层 train subject 的 normalization support 审计结果：

- median Dice：`0.999997`
- p05 Dice：`0.990682`
- median extra fraction：`0`
- p95 extra fraction：`0.2444%`
- 四项自动门禁全部通过
- 最差 16 例和随机 16 例 montage 未见系统性颅外伪影

个别病例存在局部原始非零区与多模态 explicit support 差异，但没有形成系统性背景污染。

### 4.2 正式决策

保留当前 normalization：

```text
原始 T1c 非零区域
-> 病例级 p0.5 / p99.5
-> clip
-> 映射到 [-1,1]
-> 原始零区域保持 0
```

不改用 explicit-support percentile。当前收益不足以抵消重建全部数据、histogram、cluster 和 checkpoint 的成本。

### 4.3 Explicit brain support 用途

- 训练：只用于输入 QA、病灶脑内合法性检查和分区统计；不作为输入、不改变 normalization、不参与 loss。
- RePaint：不参与逐 timestep 背景混合，继续使用共享的 normalized T1c 背景轨迹。
- Inference QA：定义 healthy-brain、outside、boundary 区域，计算 lesion 合法性和分区指标。
- 禁止使用 `normalized_t1c != 0` 反推 brain support。

### 4.4 如果未来更换 normalization

若改为 explicit-support percentile，必须重建：

- 两种尺寸的全部 NPZ patch；
- manifest 中的 normalization 字段与 hash；
- 四组 16-bin histogram；
- 两种尺寸的 train-only cluster、membership 和 provenance；
- normalization audit 与 QA；
- 所有 smoke 和正式 checkpoint；
- 所有依赖旧强度尺度的 inference 输出。

subject split 和几何裁剪中心可以保持，但所有下游 manifest hash 和 provenance 必须更新。

## 5. 数据规模与类别平衡

### 5.1 Train/val/test 分布

`splits_v2.json` 使用 `584/73/74` 个 train/val/test subject，三组零交集。

| Patch / split | Patch / subject / case | Anchor NETC/SNFH/ET/RC | 含该 lesion 的 patch NETC/SNFH/ET/RC | Interior/Boundary | 每 patch 含 1/2/3/4 类 |
|---|---:|---:|---:|---:|---:|
| 64 train | 7772 / 584 / 1275 | 1118/2550/1934/2170 | 3843/7769/6342/6290 | 3886/3886 | 179/1451/3405/2737 |
| 64 val | 1032 / 73 / 172 | 148/338/254/292 | 517/1024/830/836 | 516/516 | 24/208/433/367 |
| 64 test | 1038 / 74 / 174 | 146/348/258/286 | 479/1038/817/814 | 519/519 | 45/210/449/334 |
| 80 train | 7772 / 584 / 1275 | 1118/2550/1934/2170 | 4124/7772/6526/6689 | 3886/3886 | 73/1278/3202/3219 |
| 80 val | 1032 / 73 / 172 | 148/338/254/292 | 547/1024/860/886 | 516/516 | 12/188/399/433 |
| 80 test | 1038 / 74 / 174 | 146/348/258/286 | 527/1038/862/873 | 519/519 | 18/170/458/392 |

Train subject 的 label 覆盖为 NETC/SNFH/ET/RC `293/584/461/511`。每个 train subject 有 `2–60` 个 patch，p50 为 `12`、p95 为 `32`；每个 `(subject,anchor label)` 有 `2–20` 个 patch。自然采样会同时偏向 SNFH、多标签 patch 和多时间点 subject。

### 5.2 保持不变的 loss

继续使用当前规则：

- 所有非空 `(sample,channel)` 等权；
- 每个单元按自身 lesion voxel 数归一化；
- 空单元跳过；
- 整个 batch 全空时报错；
- 不增加固定 channel weight，不改变 loss 公式。

batch size 1 下，自然采样的长期期望 loss 贡献为：

| Patch | NETC | SNFH | ET | RC |
|---|---:|---:|---:|---:|
| 64 natural | 13.58% | 35.01% | 24.74% | 26.68% |
| 80 natural | 14.26% | 33.25% | 24.79% | 27.70% |

### 5.3 推荐 baseline sampler

只推荐分层 sampler：

1. 对 `anchor_label × sample_role` 八个 strata 等概率采样；
2. 每个 stratum 内对 subject 等概率循环；
3. subject 内再对 case/patch 循环；
4. epoch 长度固定为 7,772 个样本；
5. sampler RNG、游标和循环状态进入 checkpoint。

该策略控制 anchor label、interior/boundary 和多时间点 subject，但不会使多标签共现下的最终 channel 贡献严格相等。它改变训练采样流程，属于方法变化，需要新 experiment、feature branch 和 Git commit。

## 6. 正式训练阶段与参数

### 6.1 Patch 顺序

先训练 `64×64×32`，再把 `80×96×80` 作为第二阶段对照。

依据：

- batch 1 实测峰值显存约 `6.3 GiB` 对 `33.15 GiB`；
- padding patch 比例为 `2.04%` 对 `15.70%`；
- 64 patch 更适合先验证 sampler、validation、resume 和 W&B；
- 80 patch 提供更大上下文和更多四标签共现，但训练成本显著更高。

### 6.2 推荐参数

| 参数 | 64×64×32 | 80×96×80 |
|---|---:|---:|
| 单 run GPU | 1×A100 80GB | 1×A100 80GB |
| Micro batch | 4 | 1 |
| Gradient accumulation | 1 | 4 |
| Effective batch | 4 | 4 |
| Optimizer | Adam | Adam |
| LR | `1e-4` constant | `1e-4` constant |
| Betas / eps / weight decay | `0.9,0.999 / 1e-8 / 0` | 相同 |
| AMP | 开启 | 开启 |
| Gradient clipping | global norm `1.0` | `1.0` |
| EMA | `0.995`，每 10 step 更新 | 相同 |
| Max optimizer steps | 最多 50,000 | 最多 50,000 |
| Effective epochs | `50000×4/7772=25.73` | 相同 |
| Validation | 每 2,000 optimizer steps | 相同 |
| Latest checkpoint | 每 500 step 原子更新 | 相同 |
| Milestone | 每 5,000 step，保留最近 3 个 | 相同 |
| Best | 最低 `val/ema/total_loss` | 相同 |
| Early stopping | 10k step 后启用；patience 8；relative min delta 0.5% | 相同 |
| Seeds | 首轮 `20260805`；候选复现 `20260806/20260807` | 相同 |

这些值是可配置的正式 baseline，不是不可更改的硬编码。64 patch 的 batch 4 尚未真实测量，必须先做显存 preflight；若 OOM，可在报告后改为 `batch=2, accumulation=2`，仍不足时改为 `batch=1, accumulation=4`，保持 effective batch 为 4。`50,000` optimizer steps 是训练预算上限，不要求必须跑满；10k step 后由固定 validation 的 `val/ema/total_loss` 判断是否收敛并 early stop。任何 batch 或训练上限调整都记录为配置变化，不静默修改。

### 6.3 Validation 与 test 隔离

- Trainer 只创建 train 和 val loader，不创建 test loader。
- val 使用 73 个 subject、1,032 个 patch，`shuffle=False`，不使用训练 sampler。
- 固定 validation seed，并为每个 patch 固定 timestep/noise，使 checkpoint 间可比较。
- 使用 EMA 权重计算 early-stopping metric。
- 至少记录：总 loss、NETC/SNFH/ET/RC loss、各 channel 有效单元数、总有效单元数、patch/subject/anchor label 覆盖、空单元数。
- 64 val 预期有效单元总数为 `3207`，80 val 为 `3317`。
- `t_T=5` smoke 图、随机纹理或 test inference 不能作为 early stopping 指标。
- test 只在模型选择冻结后使用一次，不能参与调参或 checkpoint 选择。

### 6.4 Checkpoint 与 resume

正式 checkpoint 必须保存：

- model、EMA、optimizer、AMP scaler；
- optimizer step、effective epoch、best metric；
- sampler 状态；
- Python、NumPy、Torch、CUDA RNG；
- resolved config 和 config hash；
- split/manifest hash；
- Git SHA；
- W&B run ID；
- checkpoint schema/version。

`latest.pt` 原子覆盖；`best.pt` 只在 validation 改善时更新；另保留最近三个 5,000-step milestone。resume 前必须校验 config、patch shape、split、sampler、Git SHA 和 W&B run ID，不一致则 fail closed。

### 6.5 复现次数

- 原始 LeFusion 训练入口没有“三个 seed 自动重复”或必须三 seed 的规则；三 seed 是本项目为降低单次随机性、形成更可靠正式结论而提出的复现设计。
- 第一阶段先运行 seed `20260805`。
- 单 run 通过完整 validation、resume 和 W&B 门禁后，再运行 `20260806/20260807`。
- 64 patch 正式结论使用三次独立 seed。
- 80 patch 若作为正式对照，也使用相同三次 seed；可以在第一 seed 验证后再决定是否完成另外两次高成本重复。

### 6.6 资源估计

- 64 batch 4：预计 `20–28 GiB`，约 `20–36 小时/run`。
- 80 batch 1、accumulation 4：预计 `34–40 GiB`，约 `6–12 天/run`。
- 当前估时基于既有 smoke，实施前需短 timing preflight 校准。
- 每个 run 的 best/latest/milestone/W&B 缓冲预计 `3–5 GiB`。
- 六个正式 run 预计 `18–30 GiB`；远端当前可用空间约 `837 GiB`，磁盘不是 blocker。

## 7. W&B online 安全设计

### 7.1 命名

```text
project: lefusion-brats2024-gli
group:   20260805_exp005_gli_formal_training_baseline
run:     exp005-p64-s20260805
run:     exp005-p80-s20260805
```

### 7.2 凭据安全

- 曾暴露的旧 key 禁止复用。
- 用户已说明在 F 盘准备新的 W&B key；本项目不读取、不打印、不复制该 key，也不记录其明文或具体私有路径。
- 实施时由用户通过远端环境变量 `WANDB_API_KEY`、远端凭据缓存或不进入 Git 的私有环境文件安全提供。
- key 禁止写入 config、代码、shell、Markdown、日志或 Git。
- 当前文档记录不登录 W&B、不创建 run。

### 7.3 Run 与断网恢复

- 首次运行使用固定 run ID 和 `resume=never`，防止误接旧 run。
- checkpoint resume 使用同一 ID 和 `resume=must`，防止意外创建新 run。
- online 初始化失败则在模型计算前退出，不静默切换 offline。
- 运行中断网时保留本地 W&B spool 与 checkpoint；恢复网络后使用同一 run ID 续传。
- W&B history、summary 和 URL 验证完整前，不把 run 标记为正式最佳实验。
- 记录 experiment ID、Git SHA、split/hash、sampler、配置、LR、grad norm、显存、train/val 各通道指标、有效单元数及 checkpoint 路径/校验和。

## 8. Experiment 与 Git 管理

### 8.1 新 experiment

```text
20260805_exp005_gli_formal_training_baseline
```

类型为方法实验。两种 patch 和三个 seed 放在同一个 experiment 下作为配置变体。

### 8.2 分支与 commit

建议分支：

```text
feature/20260805-exp005-gli-formal-training
```

以当前已验证的 exp004 HEAD 为基础建立。由于 sampler、validation、checkpoint/resume 和 W&B 流程会发生变化，必须创建 Git commit，并将完整 SHA 写入 exp005 的 `code_version.txt`。

### 8.3 可复用资产

可复用：

- 当前两种 patch 数据与 manifest；
- `splits_v2.json`；
- scalar label、四通道 loader 和 64-D histogram；
- 当前 lesion-aware loss；
- exp004 两种 train-only cluster；
- explicit support 与 RePaint/inference 代码；
- 当前 shape、checkpoint metadata 和 inference 测试基础设施。

不能作为正式训练起点或最佳模型复用：

- exp004 的 20-step/1-step validation checkpoint；
- `t_T=5` inference 输出；
- exp004 offline W&B run；
- smoke loss 数值。

## 9. 预计文件修改

### 9.1 Config

- `LeFusion/train/config/experiment/gli_formal_64x64x32.yaml`
- `LeFusion/train/config/experiment/gli_formal_80x96x80.yaml`
- exp005 下每个 patch/seed 对应的运行 config

### 9.2 Dataset 与 sampler

- 新增 `LeFusion/dataset/gli_sampler.py`
- 修改 `LeFusion/get_dataset/get_dataset.py`
- 必要时只为 metadata 暴露修改 `LeFusion/dataset/gli_hist.py`

### 9.3 Trainer、validation、checkpoint、W&B

- `LeFusion/ddpm/diffusion.py`
- `LeFusion/train/train.py`
- 新增 `LeFusion/train/validation.py`
- `LeFusion/checkpointing.py`
- 可新增 `LeFusion/train/tracking.py`

### 9.4 测试与项目文档

- 新增 `tests/test_gli_sampler.py`
- 新增 `tests/test_gli_formal_training.py`
- 扩充 checkpoint/resume、val/test 隔离和指标聚合测试
- 新增 exp005 的 `result.md`、`code_version.txt` 和配置
- 更新 `README.md`、`STATUS.md`、`CHANGELOG.md`、`experiments/EXPERIMENT_MAP.md`
- 修正 normalization audit 中 stale 的人工 QA 状态

本方案不修改模型结构，不修改现有 loss 权重，不重建 patch/histogram/cluster。

## 10. 正式训练 blockers

已完成的正式训练门禁：

1. 用户已安全配置远端新 W&B key，online preflight 已连接指定 entity/project/run。
2. 64 patch `batch=4/accum=1` 通过零 optimizer update 的 forward/backward 显存检查，反向峰值 `23330.56 MiB`。
3. 完整固定 validation 通过：1032 patch、73 subject、4 label 覆盖、3207 有效单元；checkpoint reload/resume 通过，临时 checkpoint 已删除。
4. 真实边界 patch 的负 `origin_xyz` loader bug 已修复；远端全量测试 28/28 通过。

Trainer validation/per-channel metrics/early stopping、完整 checkpoint/resume、分层 sampler、normalization audit provenance、W&B online 和 64 patch preflight 均不再是技术 blocker。剩余条件是用户对正式 50,000-step 训练的单独授权。

标签语义和 normalization 不再是 blocker。

## 11. 用户需要确认或决定的事项

### 已确认

1. experiment ID：`20260805_exp005_gli_formal_training_baseline`。
2. branch：`feature/20260805-exp005-gli-formal-training`，并授权实施代码、测试和 Git commit。
3. 分层 sampler 作为唯一 baseline；属于方法变化，但保持 loss 不变。
4. 第一阶段先实施 `64×64×32`，`80×96×80` 作为第二阶段对照。
5. baseline 初值为 64 `batch=4/accum=1`、80 `batch=1/accum=4`、effective batch 4、LR `1e-4`、最多 50,000 optimizer steps；允许根据 preflight 显存调整 micro-batch/accumulation，并允许按 validation 收敛 early stop。
6. W&B entity：`jinyuanbao719-xi-an-jiaotong-university-`；key 由用户自行安全配置到远端。

### W&B 仍需用户完成

7. 确认 F 盘新增的是全新的、未暴露的 W&B key，旧 key 已撤销或不再使用。
8. 提供该 key 的安全使用方式：由用户自行加载到远端 `WANDB_API_KEY` 或远端凭据缓存；不需要把 key 内容发给 Codex。
9. entity 已确认；project 固定为 `lefusion-brats2024-gli`。仍需在不向 Codex 暴露 key 内容的前提下完成远端配置。

### 可以分阶段决定

10. 第一 seed 完成后，是否立即运行另外两个 64 seed。
11. 64 patch 三 seed 结论完成后，是否启动成本更高的 80 patch 对照。
12. 正式模型选择冻结后，是否另行授权完整 RePaint/test 医学 QA。

## 12. 确认后的实施顺序

1. 核对本地/远端工作树、branch 和 commit 一致。
2. 从已验证 exp004 HEAD 创建 exp005 feature branch。
3. 创建 exp005 实验卡和正式配置，不启动训练。
4. 实现分层 sampler、validation、checkpoint/resume 和 W&B fail-closed 接入。
5. 补齐单测、val/test 隔离测试和 resume 一致性测试。
6. 创建 Git commit，更新 `code_version.txt` 和中文项目文档。
7. 同步远端并只读核对 Git、配置、数据 hash 和全量测试。
8. 用户安全配置新的远端 W&B 凭据。
9. 在另行授权后进行 online 连通性、64 batch 4、validation 和 resume preflight。
10. 汇报 preflight；确认通过后先运行 64 patch seed `20260805`。
11. 第一 run 通过后完成其余 64 seed，再决定 80 patch 对照。
12. 模型选择冻结后，另行决定 test 和完整 RePaint 评估。
