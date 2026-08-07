# exp010–exp012 单通道病灶生成短程对比

## 1. 研究目的

exp008 在固定 8 例 validation QA 中退化为接近零填洞，说明仅加入挖空上下文、四通道 mask
和 histogram 条件仍不足以保证模型学习有效病灶。exp010–exp012 因此在相同数据划分、seed、
训练步数和 QA 样本上比较三种单通道扩散契约，目标是在不运行 519 例或全量 test 的前提下，
判断模型能否在 5000 optimizer step 内学习非零病灶，并响应 histogram 与 union-as-single 条件。

本轮只回答短程工程可行性和条件控制能力，不回答医学真实性、跨 seed 稳定性或全量泛化能力。

## 2. 共同训练契约

三种方法共同使用以下条件：

```text
真实目标图 x0       [B,1,D,H,W]
masked context      [B,1,D,H,W]，union lesion mask 内严格置 0
lesion masks        [B,4,D,H,W]，顺序为 NETC/SNFH/ET/RC
hist                [B,64]，每类 16 bins，顺序与 mask 一致
denoiser 空间输入    单通道 diffusion state + masked context + four masks，共 6 通道
loss 区域            union lesion mask 内；按非空病灶类别区域归一化
hist condition dropout 0.1，用于 classifier-free guidance
```

公平性约束：

- patch 固定为 `64×64×32`，seed 固定为 `20260805`。
- 使用相同 train/validation 划分、采样器和冻结的 8 例 validation QA manifest。
- 每种方法最多训练 `5000` optimizer step，统一使用 BF16；均保存
  `milestone-500/1000/2500/5000.pt` 和 `latest.pt`。
- 三种方法均完成 5000 step 和最终 validation；未运行 p80、其他 seed、519 例、test split
  或任何全量 test。
- 训练与 QA 代码固定为 commit
  `d0c6aaa056b38e34e4cb929f85106cc90af05426`。

## 3. 三种方法

| 方法 | Diffusion state | 预测目标与 loss | 推理合成 |
|---|---|---|---|
| exp010 | `q(x0,t)`，完整单通道真实 T1c 加噪 | 预测噪声 `ε`；mask-normalized ε loss | 生成后只替换 union mask 内区域 |
| exp011 | `q(y0,t)`，其中 `y0=x0×union_mask` | 预测噪声 `ε`；mask-normalized ε loss | 将生成的 lesion-only 区域粘回真实背景 |
| exp012 | `q(y0,t)`，与 exp011 相同 | 直接预测 `x0`；mask-normalized L1 加四类 soft-histogram loss | 由预测 x0 计算 posterior，最后只替换 union mask 内区域 |

exp012 的 histogram loss 权重在前 500 step 从 0 线性升至 `0.1`。

## 4. QA 设计

每种方法只使用同一组 8 个 held-out validation patch，运行以下五种变体：

| 目录名 | 条件 |
|---|---|
| `qa_original_real` | 保留真实四通道 mask，并使用该 patch 的 paired histogram |
| `qa_original_cluster_first` | 保留真实四通道 mask，使用训练集各类首个 cluster histogram |
| `qa_original_cluster_last` | 保留真实四通道 mask，使用训练集各类末个 cluster histogram |
| `qa_union_cluster_first` | 将完整 union mask 赋给一个目标类别，其余 mask/hist block 置零，使用首个 cluster |
| `qa_union_cluster_last` | 与上一项相同，但使用末个 cluster |

8 例在 union-as-single 模式中按 NETC/SNFH/ET/RC 循环，每类两例。每例五联图固定为：

```text
原始未挖图 | 挖空输入 | 生成结果 | absolute difference | conditioning mask
```

门槛为：

1. 至少 6/8 病例相对零填洞基线的 lesion MAE 改善至少 20%，且输出非零、非平坦并与 mask 对齐。
2. 至少 6/8 病例的输出 histogram 更接近请求 histogram，而不是交换后的 histogram。
3. union-as-single 至少 6/8 病例对目标类别呈现一致的 histogram/纹理响应。
4. 最终合成结果的 mask 外最大绝对误差不超过 `1e-5`。

## 5. 结果

| 方法 | lesion 通过 | hist 控制 | union 单病灶 | 原始真实 hist lesion MAE | 相对零填洞改善 | 背景最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| exp010：完整 T1c + ε | 8/8 | 7/8 | 7/8 | 0.138066 | 55.1% | 0 |
| exp011：lesion-only + ε | 7/8 | 7/8 | 5/8 | 0.164752 | 45.6% | 0 |
| exp012：lesion-only + x0 + hist loss | 8/8 | 8/8 | 8/8 | 0.163218 | 47.5% | 0 |

三种方法的零填洞基线平均 lesion MAE 均为 `0.323883`。

更换 histogram 后，exp012 的“交换 hist 距离减去请求 hist 距离”平均优势为 `0.18891`，
最小优势仍为正值 `0.09967`；union-as-single 的平均优势为 `0.60547`，最小优势为
`0.09521`。相比之下，exp010 对应均值为 `0.07063/0.18622`，且最小值为负；exp011
对应均值为 `0.05400/0.18843`，union 只通过 5/8。由此可见，exp012 对 histogram 条件的
依赖最稳定，不只是偶然生成纹理。

## 6. 结论与解释

- **当前最佳可控候选是 exp012。** 它是唯一在 lesion、hist 和 union-as-single 三项上均达到
  8/8 的方法，显式 soft-histogram loss 明显增强了条件控制。
- **paired 重建最好的方法是 exp010。** 它的 lesion MAE 最低，适合作为后续视觉真实性改进的
  重建质量对照，但 histogram 切换带来的定量与视觉差异弱于 exp012。
- **exp011 不建议仅靠增加步数继续。** lesion-only 噪声预测没有优于 exp010，且 union 门槛
  只有 5/8；若保留 lesion-only state，应采用 exp012 的直接 x0 与显式 histogram 约束路线。
- 三种方法的背景误差均为 0，主要由推理末端“只在 union mask 内粘贴”的合成契约保证，
  不能解释为 denoiser 自身学会了保护背景。
- 目检中 exp012 个别病例仍有偏黑或偏亮团块。因此“最佳”只表示当前 5k、8 例、单 seed
  条件控制最佳，不能称为稳定或医学可信的伪病灶生成器。

## 7. QA 图与结果目录

统一三方法横向图和指标已归档到独立比较实验：

```text
/workspace/LeFusion_v2/my_experiment/experiments/
└── 20260807_exp013_gli_short_method_comparison/
    └── outputs/
        ├── three_method_original_real_contact_sheet.png
        ├── case_metrics.csv
        └── summary.json
```

三种方法的全部 QA 原图保留在各自实验目录：

```text
/workspace/LeFusion_v2/my_experiment/experiments/
├── 20260807_exp010_gli_single_state_noise/outputs/patch_64x64x32/seed_20260805/
├── 20260807_exp011_gli_lesion_only_noise/outputs/patch_64x64x32/seed_20260805/
└── 20260807_exp012_gli_lesion_only_x0_hist/outputs/patch_64x64x32/seed_20260805/
```

每个 seed 目录下均有第 4 节列出的五种 `qa_*` 目录；每种变体中：

- `qa_contact_sheet.png`：该方法、该条件下的 8 例总览图。
- `qa/*.png`：每例五联 QA 图。
- `metrics.json`：该变体汇总指标。
- `run_contract.json`：checkpoint、EMA、seed、采样步数与输入契约。
- `progress.jsonl` 和 NPZ：逐例进度及数值产物。

统一比较输出含逐病例医学图像统计和 contact sheet，因此按项目安全策略保留在本地/远端，
不提交到 Git；可审计的配置、结论和路径由 exp013 实验卡与本文档记录。

## 8. 后续建议

以 exp012 为主线，仅做小规模视觉真实性改进，优先约束异常亮暗团块、边界连续性和局部平滑；
同时保留 exp010 作为 paired 重建对照。任何新增训练、其他 seed、p80 或扩大测试范围都应另建
实验并单独授权。
