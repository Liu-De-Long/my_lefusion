# exp009 p64 少标签逐体素亚区分类

## 目标

仅使用 T1c p64 与同尺寸总病灶二值 mask，在冻结的 1000 个 train patch 有标签子集上学习
NETC、SNFH、ET、RC 四分类；以完整 p64 重建后患者等权的 ET/RC `focus_mIoU >= 0.85`
作为成功门槛，同时要求 ET、RC 单类 IoU 均不低于 0.80。

## 变更

- 新增 leak-safe classifier dataset，模型输入白名单仅包含 T1c、总 mask 及其无标签派生特征。
- 新增确定性的 `anchor_label × sample_role` 均衡、层内 subject round-robin 的 1000 patch
  子集冻结器；anchor 和真实类别体素量只用于离线选择与 loss，不进入模型。
- 新增 M0 单强度 MLP、M1 空间特征 MLP、M2 `3×3×3` 邻域 MLP 和 C0 轻量 3D CNN。
- 新增完整 p64 恢复、mask 外强制背景、患者等权指标、bootstrap 置信区间和小区域分层评估。
- 新增原子 `latest/best` checkpoint、严格 config/subset hash resume 和 early stopping。
- 正式训练 fail-closed：要求 Git 工作树干净，并在 metadata/checkpoint 中冻结 branch、HEAD、
  Python、Torch 与 CUDA 版本。
- test 入口 fail-closed：只有 `best_val_metrics.json` 的五项门禁全部通过，且 checkpoint、config、
  subset SHA-256 与验证记录一致时，才允许读取冻结的 `best.pt` 运行 test。
- 新增真实 p64 CPU preflight：覆盖八个 `anchor × role` 分层，执行一次前反向但固定
  `optimizer_steps=0`；C0 使用真实病灶中心小裁块控制 CPU 成本，正式完整 p64 仍须等待 GPU。
- 修复验证指标中 NumPy 标量无法 JSON 序列化的问题，以及 CUDA checkpoint 恢复时 RNG
  状态被错误搬到 GPU 的问题。
- 为 M0/M1/M2 增加进程内 mask 内特征行缓存；缓存特征只由 T1c 与总 mask 推导，target
  作为独立张量仅参与监督采样，不进入模型输入。该优化不改变数值、采样与评价口径。

## 配置

- 数据：p64 `[X,Y,Z]=[64,64,32]`，模型使用 `[D,H,W]=[32,64,64]`。
- 有标签训练集：冻结 train pool 中 1000 patch，每个 anchor 类 250，每类
  interior/boundary 各 125。
- 验证与测试：复用 `splits_v2.json`，按 subject 隔离；模型选择只看 val，test 只在最终冻结后运行。
- 配置文件：`config_m0.yaml`、`config_m1.yaml`、`config_m2.yaml`、`config_c0.yaml`。
- 冻结子集：`labeled_subset_1000.json`，作为小型 provenance 资产纳入 Git，不放入忽略的
  `outputs/`。

## 结果

### 数据与运行审计

- 冻结子集为 1000 个唯一 train patch，四个 anchor 各 250，interior/boundary 各 500；
  覆盖 480 个 subject，单 subject 最多 6 个 patch。
- 子集 SHA-256：`aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a`。
- val 为 1032 patch、73 subject；所有指标均先恢复完整 p64，再按 subject 等权聚合。
- 远端真实 p64 CPU preflight 覆盖八个 `anchor × role` 层，四个模型均完成有限 loss 与非零梯度，
  且 `optimizer_steps=0`；远端 classifier 单测最终为 7/7 通过。
- M0、M1、M2、C0 在 GPU0 严格串行运行；GPU1 上后续出现的 exp010 属于另一个隔离 worktree，
  未被本实验使用或干扰。

### 验证集模型比较

| 模型 | epoch 数 | 最佳 epoch | ET IoU | RC IoU | ET/RC focus mIoU | 95% CI | macro IoU | mask 内错误率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 10 | 0 | 0.3084 | 0.3686 | 0.3385 | [0.3027, 0.3716] | 0.3091 | 0.3668 |
| M1 | 12 | 3 | 0.4029 | 0.3847 | 0.3938 | [0.3561, 0.4298] | 0.3933 | 0.2479 |
| M2 | 12 | 3 | 0.4060 | 0.3569 | 0.3815 | [0.3458, 0.4196] | 0.3920 | 0.2237 |
| C0 | 31 | 22 | 0.4937 | 0.5935 | **0.5436** | [0.5000, 0.5862] | 0.5510 | 0.1032 |

四个模型的患者等权逐类指标如下；`n` 为该类可评价患者数：

| 模型 | 类别 | IoU | Dice | precision | recall | n |
|---|---|---:|---:|---:|---:|---:|
| M0 | NETC | 0.0226 | 0.0410 | 0.0433 | 0.1184 | 73 |
| M0 | SNFH | 0.5370 | 0.6703 | 0.7630 | 0.6372 | 73 |
| M0 | ET | 0.3084 | 0.4003 | 0.4032 | 0.7076 | 73 |
| M0 | RC | 0.3686 | 0.4781 | 0.5763 | 0.5903 | 73 |
| M1 | NETC | 0.0831 | 0.1319 | 0.1042 | 0.6765 | 73 |
| M1 | SNFH | 0.7024 | 0.8089 | 0.8699 | 0.7950 | 73 |
| M1 | ET | 0.4029 | 0.4935 | 0.5021 | 0.7529 | 73 |
| M1 | RC | 0.3847 | 0.4929 | 0.5487 | 0.6147 | 73 |
| M2 | NETC | 0.0702 | 0.1167 | 0.1026 | 0.4984 | 72 |
| M2 | SNFH | 0.7351 | 0.8312 | 0.8057 | 0.8966 | 73 |
| M2 | ET | 0.4060 | 0.4997 | 0.5475 | 0.6700 | 73 |
| M2 | RC | 0.3569 | 0.4681 | 0.6379 | 0.5069 | 73 |
| C0 | NETC | 0.2492 | 0.3249 | 0.3664 | 0.5929 | 55 |
| C0 | SNFH | 0.8679 | 0.9168 | 0.9275 | 0.9327 | 73 |
| C0 | ET | 0.4937 | 0.5657 | 0.5529 | 0.7718 | 73 |
| C0 | RC | 0.5935 | 0.6844 | 0.7072 | 0.8046 | 73 |

C0 为验证集最佳模型，其四类结果为：

| 类别 | IoU | Dice | precision | recall |
|---|---:|---:|---:|---:|
| NETC | 0.2492 | 0.3249 | 0.3664 | 0.5929 |
| SNFH | 0.8679 | 0.9168 | 0.9275 | 0.9327 |
| ET | 0.4937 | 0.5657 | 0.5529 | 0.7718 |
| RC | 0.5935 | 0.6844 | 0.7072 | 0.8046 |

C0 的 balanced accuracy 为 0.7755，macro Dice 为 0.6230；mask 外非零率为 0，预测类别
union 与输入总 mask 的 Dice 为 1。患者等权归一化 confusion matrix（行是真值，列是预测，
顺序为 NETC/SNFH/ET/RC）为：

```text
0.5929  0.0804  0.1619  0.1648
0.0050  0.9327  0.0218  0.0405
0.0130  0.1211  0.7718  0.0941
0.0522  0.1035  0.0397  0.8046
```

C0 的小区域 patch Dice 显示明显体积依赖：ET 的 `1–100/101–1000/>1000` 体素分层均值为
`0.0981/0.5258/0.8587`，RC 为 `0.1031/0.4542/0.7797`，说明总体结果不能掩盖极小区域失败。

训练后复核 `history.jsonl` 发现：epoch 29 的实际验证 focus mIoU 为 `0.5482`，高于保存的
epoch 22 `best.pt` 的 `0.5436`。原因是实现错误地把 `early_stopping_min_delta=0.005` 同时用于
early-stopping patience 和 best-checkpoint 保存，导致小于 0.005 的真实新高没有保存。
epoch 29 权重已被 epoch 30 的 `latest.pt` 覆盖，无法事后恢复。该审计不改变门禁结论：即使按
验证历史最大值计算，`0.5482` 仍远低于 `0.85`；后续训练必须把“绝对验证新高保存”和
“达到 min_delta 后重置 patience”拆开。

所有模型的 `focus_mIoU >= 0.85`、ET IoU >= 0.80、RC IoU >= 0.80 三项门禁均失败；
因此没有创建 `test_metrics.json`，冻结 test 从未运行。C0 最佳 checkpoint SHA-256 为
`a605f0a6ad45505e2ea41b519dac284a40680abdd29fc66d8d80419cecf48eb1`。

训练入口未接入项目规则要求的 W&B online 记录。本次文件级 provenance、history、checkpoint
与验证结果完整，但 W&B 日志不完整；因此不得把本次 checkpoint 表述为满足项目管理规则的
正式最佳模型。后续正式复现实验必须先补 W&B fail-closed 接入。

## 结论

- 单体素强度 MLP 只适合作为可辨识性下限；M1 的几何/局部统计有增益，但远低于目标。
- `3×3×3` 邻域 MLP 没有优于 M1，不值得在当前方案内继续扩展 `5×5×5`。
- 完整 p64 轻量 3D CNN 明显优于三种 MLP，证明共享三维上下文必要；但最佳 focus mIoU
  checkpoint 只有 0.5436；训练历史绝对最大值也只有 0.5482，不能支持“达到 85% 可分性”的结论。
- 当前 1000 patch、T1c+总 mask、轻量 C0 的监督基线已经证伪“任务相对简单、MLP 即可达到
  85%”这一假设。继续追求 0.85 需要新的方法实验，而不是对当前 MLP 做小幅调参。

## 下一步

1. 在任何新训练前补齐 W&B online fail-closed 记录与复现门禁。
2. 由用户确认下一阶段是在同一 1000 patch 监督口径下升级更强 3D U-Net/残差网络，还是允许
   使用剩余 train patch 做半监督一致性/伪标签；两者都属于新的方法实验。
3. 新方法仍只使用 T1c 与总 mask 作为输入，模型选择只看 val；只有全部验证门禁通过后才允许
   对冻结 test 运行一次。

## 输出路径

`experiments/20260806_exp009_gli_p64_weak_supervision_classifier/outputs/`
