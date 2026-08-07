# exp010–exp012 单通道病灶生成短程对比

## 1. 研究目的

exp008 在固定 8 例 validation QA 中退化为接近零填洞，说明仅加入挖空上下文、四通道 mask
和 histogram 条件仍不足以保证模型学习有效病灶。exp010–exp012 因此在相同数据划分、seed、
训练步数和 QA 样本上比较三种单通道扩散契约，目标是在不运行 519 例或全量 test 的前提下，
判断模型能否在 5000 optimizer step 内学习非零病灶，并响应 histogram 与 union-as-single 条件。

本轮只回答短程工程可行性、paired 重建和全局强度 histogram 响应，不回答医学真实性、
病灶亚型的视觉/语义一致性、跨 seed 稳定性或全量泛化能力。

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
3. union-as-single 至少 6/8 病例的输出 histogram 更接近本次请求，而不是交换后的请求。
4. 最终合成结果的 mask 外最大绝对误差不超过 `1e-5`。

## 5. 结果

| 方法 | lesion 通过 | 四类 hist counterfactual | union hist counterfactual | 原始真实 hist lesion MAE | 相对零填洞改善 | 背景最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| exp010：完整 T1c + ε | 8/8 | 7/8 | 7/8 | 0.138066 | 55.1% | 0 |
| exp011：lesion-only + ε | 7/8 | 7/8 | 5/8 | 0.164752 | 45.6% | 0 |
| exp012：lesion-only + x0 + hist loss | 8/8 | 8/8 | 8/8 | 0.163218 | 47.5% | 0 |

三种方法的零填洞基线平均 lesion MAE 均为 `0.323883`。

### 5.1 Histogram counterfactual 指标的准确含义

对同一个病例分别使用 first-cluster 和 last-cluster histogram 生成 `G_first` 与 `G_last`，
对应请求为 `H_first` 与 `H_last`。比较脚本只在 conditioning mask 内统计 `[-1,1]` 范围的
16-bin 归一化强度 histogram，并计算：

\[
D_{own}=\frac{L1(hist(G_{first}),H_{first})+L1(hist(G_{last}),H_{last})}{2}
\]

\[
D_{swapped}=\frac{L1(hist(G_{first}),H_{last})+L1(hist(G_{last}),H_{first})}{2}
\]

\[
margin=D_{swapped}-D_{own}
\]

多标签模式对每个非空标签的 first/last 两项共同取平均；union 模式只有被赋值的单一目标标签。
归一化 histogram 的 L1 距离范围为 `0–2`。`margin>0` 表示输出更接近自己的请求，
`margin<0` 表示反而更接近交换后的错误请求；当前通过条件仅为 `D_own<D_swapped`，没有额外
margin 阈值。union-as-single 使用完全相同的公式，只是先把完整 union mask 放入一个目标类别
通道，其余 mask/hist block 置零。

| 方法 | 四类 hist margin 均值 | 四类 hist 最小值 | union hist margin 均值 | union hist 最小值 |
|---|---:|---:|---:|---:|
| exp010 | 0.07063 | -0.03344 | 0.18622 | -0.08446 |
| exp011 | 0.05400 | -0.01769 | 0.18843 | -0.11003 |
| exp012 | 0.18891 | +0.09967 | 0.60547 | +0.09521 |

因此 exp012 对 first/last histogram 请求的边际强度分布响应更稳定：普通四类 hist margin
约为 exp010 的 `2.67` 倍，union hist margin 约为 `3.25` 倍，且 8 例最小 margin 都为正。
这个指标**不衡量**空间纹理排列、形态、边界、union 内视觉均匀性或医学亚型语义，不能称为
“同一种病灶生成通过率”。

### 5.2 exp010 与 exp012 的 paired 重建差异

exp010 的 lesion MAE 为 `0.138066`，exp012 为 `0.163218`，绝对差为 `0.025152`；8/8
病例均由 exp010 获得更低 MAE，逐例优势范围为 `0.014106–0.050797`，中位数为 `0.019769`。
统一以 exp010 为参照时，exp012 的 MAE 高 `18.2%`；若改用 exp012 作分母，则 exp010 低
`15.4%`。两个百分比不矛盾，只是分母不同。后续统一使用“exp012 比 exp010 高 18.2%”。

exp010/exp012 的生成病灶平均绝对强度为 `0.33958/0.33104`，平均标准差为
`0.27487/0.23682`。这进一步说明 exp010 的 paired 重建优势是一致的，而 exp012 在显式
histogram 约束下牺牲了一部分逐体素复原能力。

### 5.3 Union QA 的视觉复核与结论下调

重新复核 exp012 的 union-first/union-last contact sheet 后，只能确认 first/last 条件会改变
union 内亮暗比例和全局强度分布。肉眼不能确认不同病例在指定同一类别后具有统一病灶外观；
union 内仍有不同亮暗斑块和局部结构，生成结果明显受解剖背景、union 形状和采样噪声影响。

当前设计还把类别与病例混杂：每个病例只被分配一个目标类别，不是同一病例分别生成
NETC/SNFH/ET/RC 四类。因此现有 8/8 只证明 histogram 配对正确，不能证明四个 mask 通道
学出了四种视觉可辨、类内一致的病灶。

## 6. 结论与解释

- **当前最强 histogram 条件响应候选是 exp012。** 它在普通四类和 union 模式的 histogram
  counterfactual 中均为 8/8；这不等价于视觉或医学亚型控制已经通过。
- **paired 重建最好的方法是 exp010。** 它的 lesion MAE 最低，适合作为后续视觉真实性改进的
  重建质量对照；exp012 相对它增加 `18.2%` MAE。
- **exp011 不建议仅靠增加步数继续。** lesion-only 噪声预测没有优于 exp010，且 union hist
  counterfactual 只有 5/8。
- 三种方法的背景误差均为 0，主要由推理末端“只在 union mask 内粘贴”的合成契约保证，
  不能解释为 denoiser 自身学会了保护背景。
- 若目标是 paired 病灶复原，当前证据支持 exp010；若目标仅是按请求 histogram 改变边际强度
  分布，当前证据支持 exp012。对于“生成指定医学亚型的伪病灶”，两者目前都没有充分证据。
- 目检中 exp012 个别病例仍有偏黑或偏亮团块。因此不能把它称为稳定、类别明确或医学可信的
  伪病灶生成器。

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

该建议已由 `20260808_exp014_gli_four_class_counterfactual_qa` 执行：对同一个挖空输入、同一个
union mask 和同一完整采样随机序列分别生成 NETC/SNFH/ET/RC，除目标 mask 通道及对应 hist
block 外保持条件一致。每个病例均生成四类，并检查：

1. 输出对四类真实/cluster histogram 的目标类别排名，而不是只比较 first/last 两个中心。
2. 类内一致性是否高于类间一致性；指标需包含空间纹理或无标签泄漏的类别表征，不能只有
   全局 histogram。
3. 同时保留 exp010/exp012 paired MAE、异常亮暗团块、边界连续性和局部平滑对照。

exp014 的 histogram 目标 Top-1 为 `21/32`，NETC/SNFH/ET/RC 分别为
`3/8、2/8、8/8、8/8`；强度归一化 GLCM texture LOCO 为 `26/32`，但 RC 仅 `4/8`。目检只
稳定确认 ET 高亮以及类别相关亮暗变化，NETC/SNFH/RC 仍明显重叠。因此四类视觉/医学语义
可辨性尚未确认，不进入 `λhist=0.03/0.05/0.1` 消融。下一步先在真实 held-out T1c 上验证
无标签泄漏的四类可分性上限；任何新增训练、其他 seed、p80 或扩大测试范围都应另建实验并
单独授权。完整结果见 `experiments/20260808_exp014_gli_four_class_counterfactual_qa/result.md`。
