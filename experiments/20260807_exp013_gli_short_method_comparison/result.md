# exp013 exp010–exp012 短程横向比较

## 目标

在相同 seed、训练/验证划分、5000 optimizer step 和冻结 8 例 validation QA 下，统一比较
exp010、exp011、exp012 的病灶生成、histogram 控制、union-as-single 控制与背景保持能力。

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

| 方法 | lesion | hist | union | lesion MAE | 相对零填洞改善 | 背景最大误差 |
|---|---:|---:|---:|---:|---:|---:|
| exp010 | 8/8 | 7/8 | 7/8 | 0.138066 | 55.1% | 0 |
| exp011 | 7/8 | 7/8 | 5/8 | 0.164752 | 45.6% | 0 |
| exp012 | 8/8 | 8/8 | 8/8 | 0.163218 | 47.5% | 0 |

exp010 的 paired 重建最好。exp012 的 histogram 与 union-as-single 控制最稳定：请求 hist
相对交换 hist 的平均优势为 `0.18891`，union 平均优势为 `0.60547`，且 8 例最小优势均为正。
exp011 未通过 union 6/8 门槛。

目检确认三种方法都在 mask 内生成非零、非平坦结构；exp012 的条件响应最明显，但个别病例
仍存在偏黑或偏亮团块。背景误差为 0 主要由最终 mask 内粘贴契约保证。

## 结论

exp012 是本轮当前最佳可控候选，exp010 是 paired 重建质量对照，exp011 作为 union 门槛失败
对照保留。结论仅限固定 8 例 validation QA 和单 seed，不外推到医学有效性或全量泛化。

完整设定、指标解释、QA 目录和后续建议见：
`docs/20260807_001_gli_short_method_comparison.md`。

## 输出路径

- `outputs/three_method_original_real_contact_sheet.png`：三方法原始真实 hist 横向图。
- `outputs/case_metrics.csv`：120 条逐方法、逐变体、逐病例指标。
- `outputs/summary.json`：统一汇总、门槛结果与效应量。

上述输出含逐病例统计与医学图像 contact sheet，只保存在本地/远端，不提交 Git。
