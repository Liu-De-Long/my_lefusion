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

待远端 QA 完成后填写。

## 结论

待定。四类 histogram 排名正确也不能单独证明医学亚型语义。

## 下一步

先完成本 QA，再决定是否需要真实 T1c 的无泄漏类别可分性上限验证；本实验不调整 `λhist`。

## 输出路径

`experiments/20260808_exp014_gli_four_class_counterfactual_qa/outputs/`
