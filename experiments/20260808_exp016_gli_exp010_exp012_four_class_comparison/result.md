# exp016：exp010 与 exp012 同病例四类别配对比较

## 目标

为 exp010 补充与 exp012 完全相同的冻结 8 例 × 4 类 QA，并以逐例配对方式比较 histogram 与无标签泄漏 GLCM 纹理可分性。

## 固定契约

- 不重新训练；exp010 与 exp012 均使用各自 step 5000 EMA checkpoint、seed `20260805`。
- 同一 8 个验证病例、挖空输入、union mask、四个目标类别、train-only cluster 的每类首个 histogram 中心。
- 同一 `sampling.seed=20260807`、逐病例 sample seed、随机序列与 `t_T=300`。
- 仅新增 exp010 的 32 个结果；exp012 复用 exp014 已审计的 32 个结果。
- 不运行 exp011、p80、其他 seed、519 例、全量 test 或任何训练。

## 结果

待执行。

## 结论

待执行。

## 输出

`experiments/20260808_exp016_gli_exp010_exp012_four_class_comparison/outputs/`
