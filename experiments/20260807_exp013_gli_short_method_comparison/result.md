# exp013 exp010–exp012 短程横向比较

## 目标

在相同 seed、训练/验证划分、5000 optimizer step 和冻结 8 例 validation QA 下，统一比较
exp010、exp011、exp012 的病灶生成、histogram counterfactual、union histogram
counterfactual 与背景保持能力。

## 变更

- 使用统一脚本比较三种方法的五种冻结 QA，不修改训练模型或 checkpoint。
- 2026-08-08 修订指标解释：union 结果只衡量 16-bin 边际强度 histogram 的 own-vs-swapped
  配对，不能解释为同类病灶视觉/语义一致性。

## 配置

- 来源实验：`20260807_exp010_gli_single_state_noise`、
  `20260807_exp011_gli_lesion_only_noise`、
  `20260807_exp012_gli_lesion_only_x0_hist`。
- patch：`64×64×32`；seed：`20260805`；每种方法训练 `5000` optimizer step。
- QA：同一冻结 manifest 的 8 个 validation patch，每种方法五种条件变体，共 15 个变体，
  每个变体均完成 8/8。
- 统一比较脚本：`scripts/gli_compare_short_generation_experiments.py`。
- 训练与比较代码 commit：`d0c6aaa056b38e34e4cb929f85106cc90af05426`。
- 未运行 p80、其他 seed、519 例、test split 或任何全量 test。

## 结果

| 方法 | lesion | hist CF | union hist CF | lesion MAE | 相对零填洞改善 | 背景最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| exp010 | 8/8 | 7/8 | 7/8 | 0.138066 | 55.1% | 0 |
| exp011 | 7/8 | 7/8 | 5/8 | 0.164752 | 45.6% | 0 |
| exp012 | 8/8 | 8/8 | 8/8 | 0.163218 | 47.5% | 0 |

exp010 的 paired 重建最好；8/8 病例 MAE 都低于 exp012。以 exp010 为参照，exp012 的
lesion MAE 高 `18.2%`，绝对增加 `0.025152`。

exp012 的 histogram 配对最稳定：请求 hist 相对交换 hist 的平均 margin 为 `0.18891`，
union hist 平均 margin 为 `0.60547`，且 8 例最小 margin 均为正。该指标只衡量 mask 内
16-bin 强度 histogram，不衡量空间纹理、形态、边界或医学类别。exp011 未通过 union hist
counterfactual 6/8 门槛。

目检确认三种方法都在 mask 内生成非零、非平坦结构；exp012 的 first/last 亮暗分布响应最明显，
但 union-as-single 结果没有呈现明确的跨病例同类外观，且个别病例仍存在偏黑或偏亮团块。
背景误差为 0 主要由最终 mask 内粘贴契约保证。

## 结论

exp012 是当前最强 histogram 条件响应候选，exp010 是 paired 重建质量最佳方法。现有 QA
不能证明 exp012 能生成视觉统一、类别明确的同一种病灶，因此不再把 union 8/8 表述为医学
亚型控制通过。结论仅限固定 8 例 validation QA 和单 seed。

## 下一步

先在相同输入、相同 union mask 和相同噪声下分别生成 NETC/SNFH/ET/RC，验证类内一致性与
类间可分性；在此之前不依据当前 union 8/8 调整 `λhist` 或决定最终生成模型。

完整设定、指标解释、QA 目录和后续建议见：
`docs/20260807_001_gli_short_method_comparison.md`。

## 输出路径

- `outputs/three_method_original_real_contact_sheet.png`：三方法原始真实 hist 横向图。
- `outputs/case_metrics.csv`：120 条逐方法、逐变体、逐病例指标。
- `outputs/summary.json`：统一汇总、门槛结果与效应量。

上述输出含逐病例统计与医学图像 contact sheet，只保存在本地/远端，不提交 Git。
