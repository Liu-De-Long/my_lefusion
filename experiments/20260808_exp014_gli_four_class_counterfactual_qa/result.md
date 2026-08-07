# exp014：同病例四类别 counterfactual QA

## 目标

在 exp012 的同一个冻结病例、同一个挖空输入、同一个 union mask 和同一采样随机序列下，分别请求 NETC、SNFH、ET、RC，解除“类别与病例绑定”的混杂。

## 变更

- 只扩展推理条件编排，不改变模型、训练契约或 checkpoint。
- 新增 `union_single_label_fixed`，每次只激活一个目标 mask 通道及对应 16-bin histogram block。
- 四类使用相同 `sampling.seed`；每例采样 seed 只由冻结 seed 与 source path 决定。
- 增加四类 histogram 排名、强度归一化 3D GLCM texture leave-one-case-out 指标和同病例横向图。

## 配置

- 来源模型：exp012 p64、seed `20260805`、step `5000`、EMA。
- 数据：原有冻结 validation 8 例，不运行 519 例或全量 test。
- 每例四类，共 32 个生成结果。
- histogram：每类固定使用 train-only cluster 的首个中心。

## 结果

- 四类均完成 `8/8`，总计 32 个生成结果；没有运行 519 例、test split 或全量 test。
- 同病例合同逐数组核对通过：原图、挖空输入、source union 和 `sample_seed` 在四类之间完全一致；
  每次只有目标 mask 通道和对应 hist block 非零。
- 四类 histogram 目标 Top-1 为 `21/32`，平均目标 rank 为 `1.34375`，平均 margin 为
  `0.329680`；mask 外最大绝对误差为 `0`。
- 分类别 histogram Top-1：NETC `3/8`、SNFH `2/8`、ET `8/8`、RC `8/8`；平均 rank 为
  `1.625 / 1.750 / 1.000 / 1.000`，平均 margin 为
  `-0.043626 / 0.073016 / 0.653920 / 0.635411`。
- 强度归一化 3D GLCM texture leave-one-case-out 为 `26/32=81.25%`；NETC/SNFH/ET/RC 分别
  `6/8、8/8、8/8、4/8`。该分类只使用生成病灶内部的归一化邻接纹理，不读取 mask/hist/路径，
  但训练和测试都来自 32 个生成结果，因此只能作为描述性统计，不能证明医学语义。
- 单 worker 峰值显存约 `1091 MiB`；经用户授权使用 GPU1 四个输出互斥 worker，并行观察显存
  约 `7.0 GiB`、利用率 100%。四个 worker 均正常退出。
- 首次 1 例 smoke 在采样前暴露直接入口 import 遗漏，未产生病例结果；修复后 focused tests
  `2/2` 通过，成功 smoke 的 NETC 1 例由正式 worker 原目录恢复，不重复生成。

目检总览图显示：

- ET 在 8 个病例中均呈显著高亮，是最稳定、最容易区分的条件响应；
- NETC 与 RC 多为较暗结构，SNFH 多处于中间强度，但三者视觉重叠明显；
- 相同目标类别在不同病例中没有统一形态，局部结构仍明显跟随原解剖背景和 union 形状；
- 因此模型主要学到了类别相关的亮暗比例和部分纹理统计，没有充分证据证明四种病灶语义。

## 结论

exp012 的 histogram/亮暗控制得到进一步证实，但四类别控制没有通过严格验证：总体 histogram
Top-1 仅 `21/32`，NETC/SNFH 明显不足；GLCM-LOCO 虽为 `26/32`，RC 仍只有 `4/8`，且该指标
不能连接到真实医学类别。现阶段仍不能称 union-as-single 能生成视觉统一、类别明确的同一种
病灶，也不能据此选定最终模型或调整 `λhist`。

## 下一步

先在真实 held-out T1c 病灶上建立无标签泄漏的四类可分性上限，并加入 histogram-only 基线；
如果真实 T1c 确实可分，再针对 NETC/SNFH/RC 的重叠设计空间纹理或类别表征约束。本实验不调整
`λhist`，不扩大 seed 或测试规模。

## 输出路径

`experiments/20260808_exp014_gli_four_class_counterfactual_qa/outputs/`

主要文件：

- `summary.json`
- `case_metrics.csv`
- `texture_features.csv`
- `four_class_contact_sheet.png`
- `case_contact_sheets/case_00.png` 至 `case_07.png`
- `target_1_netc/`、`target_2_snfh/`、`target_3_et/`、`target_4_rc/`
