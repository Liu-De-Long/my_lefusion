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

- exp010 的 NETC/SNFH/ET/RC 均完成 `8/8`，共新增 32 个结果；exp012 直接复用 exp014 的
  32 个冻结结果。没有运行训练、exp011、p80、其他 seed、519 例或全量 test。
- 32 对原图、挖空输入、source/target mask、四通道 mask、64-bin 条件 hist 和 sample seed
  均逐数组完全一致；两种模型 mask 外最大绝对误差均为 `0`。
- histogram Top-1：exp010 `18/32`，exp012 `21/32`；平均目标 rank 分别为
  `1.625` 与 `1.34375`，平均 target-vs-next-best margin 分别为 `0.100785` 与 `0.329680`。
- histogram 分类别 Top-1（NETC/SNFH/ET/RC）：exp010 为 `0/8、3/8、8/8、7/8`，
  exp012 为 `3/8、2/8、8/8、8/8`。exp010 的 NETC 有 `7/8` 被 histogram 判为 RC，SNFH
  有 `5/8` 被判为 RC；exp012 的 NETC/SNFH 仍分别有 `5/8、6/8` 被判为 RC。
- 32 对 histogram rank 胜负为 exp010/exp012/tie `1/8/23`；margin 胜负为 `13/19`。
  按每病例四类平均 margin 汇总，exp012 在 `8/8` 病例全部胜出。
- 强度归一化 3D GLCM LOCO Top-1：exp010 `21/32`，exp012 `26/32`；分类别正确数分别为
  `5/8、4/8、8/8、4/8` 与 `6/8、8/8、8/8、4/8`。逐对 texture margin 胜负为
  `15/17`，病例级为 `4/4`，说明纹理 margin 优势弱于 histogram 优势。
- 目检中两种模型都稳定产生高亮 ET；exp012 的类别亮暗与局部纹理差异更明显，尤其改善 SNFH
  的 GLCM 可分性。两种模型的 NETC、SNFH、RC 仍明显重叠，跨病例形态继续受解剖背景和
  union 形状主导，不能解释为视觉统一或医学语义明确的四种病灶。
- GPU1 上四个输出互斥 worker 的 PID 为 NETC `30308`、SNFH `30312`、ET `30316`、RC
  `30304`；观察显存约 `7.0 GiB`、利用率最高 `100%`，单 worker 峰值约 `1090.5 MiB`。
  首次 detached 启动因 login shell 误用 `/opt/conda/bin/python`、缺少 `blobfile` 而在采样前
  退出，未写入结果；保留失败日志后改用明确的 LeFusion conda 解释器正常完成。

## 结论

在本次严格配对的四类别条件生成任务上，exp012 整体优于 exp010：histogram Top-1、目标 rank、
margin 和 GLCM Top-1 均更好，且 8 个病例的四类平均 histogram margin 全部胜出。因此后续若目标
是类别/histogram 可控伪病灶生成，应以 exp012 为主干；exp010 继续保留为 paired 重建质量更好的
对照。

该比较不能把增益单独归因于 soft-histogram loss，因为 exp010 与 exp012 同时还存在 full-T1c
noise prediction 与 lesion-only x0 prediction 的差异。若要隔离 `λhist` 的因果贡献，下一步应在
exp012 完全相同的 lesion-only x0 契约下只做 `λhist=0` 消融。当前结果仍未验证医学亚型语义，
不授权据此扩展到全量 test 或数据增强发布。

## 输出

`experiments/20260808_exp016_gli_exp010_exp012_four_class_comparison/outputs/`

主要文件：

- `exp010/summary.json`、`exp010/case_metrics.csv`、`exp010/texture_features.csv`
- `comparison/summary.json`、`comparison/paired_metrics.csv`、`comparison/per_case_comparison.csv`
- `comparison/exp010_hist_confusion.csv`、`comparison/exp012_hist_confusion.csv`
- `comparison/exp010_texture_confusion.csv`、`comparison/exp012_texture_confusion.csv`
- `comparison/case_contact_sheets/case_00.png` 至 `case_07.png`
- `comparison/exp010_exp012_four_class_contact_sheet.png`
