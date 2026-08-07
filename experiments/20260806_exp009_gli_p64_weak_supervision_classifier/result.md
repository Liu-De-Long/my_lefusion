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
- test 入口 fail-closed：只有 `best_val_metrics.json` 的五项门禁全部通过，且 checkpoint、config、
  subset SHA-256 与验证记录一致时，才允许读取冻结的 `best.pt` 运行 test。
- 新增真实 p64 CPU preflight：覆盖八个 `anchor × role` 分层，执行一次前反向但固定
  `optimizer_steps=0`；C0 使用真实病灶中心小裁块控制 CPU 成本，正式完整 p64 仍须等待 GPU。

## 配置

- 数据：p64 `[X,Y,Z]=[64,64,32]`，模型使用 `[D,H,W]=[32,64,64]`。
- 有标签训练集：冻结 train pool 中 1000 patch，每个 anchor 类 250，每类
  interior/boundary 各 125。
- 验证与测试：复用 `splits_v2.json`，按 subject 隔离；模型选择只看 val，test 只在最终冻结后运行。
- 配置文件：`config_m0.yaml`、`config_m1.yaml`、`config_m2.yaml`、`config_c0.yaml`。
- 冻结子集：`labeled_subset_1000.json`，作为小型 provenance 资产纳入 Git，不放入忽略的
  `outputs/`。

## 结果

当前处于实现与 CPU 验证阶段，尚未启动训练。exp008 正在占用两张 GPU，本实验不得与其争抢 GPU。

## 结论

尚无模型质量结论，85% 门槛未验证。

## 下一步

完成远端真实数据 subset 冻结、全量非训练测试和 CPU smoke；待 exp008 释放资源后，依次运行
M0、M1、M2，若 MLP 未达门槛则运行 C0。验证集冻结模型后只运行一次 test。

## 输出路径

`experiments/20260806_exp009_gli_p64_weak_supervision_classifier/outputs/`
