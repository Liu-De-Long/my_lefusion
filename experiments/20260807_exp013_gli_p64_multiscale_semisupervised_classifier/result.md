# exp013 p64 多尺度与半监督逐体素亚区分类

## 目标

在 exp009 冻结的同一 1000 个有标签 p64 patch 上，将 ET/RC 患者等权 focus mIoU 从轻量
C0 的 0.5436 提升至至少 0.85；监督阶段使用多尺度残差 3D U-Net，随后允许剩余 train
patch 以无标签方式进入 EMA teacher 一致性与高置信伪标签训练。

## 变更

- 待完成。

## 配置

- `config_supervised.yaml`：冻结 1000 patch 的监督多尺度残差 3D U-Net。
- `config_mean_teacher.yaml`：从监督最佳 checkpoint 初始化，使用剩余 train patch 的
  mean-teacher 半监督阶段。
- 输入白名单仍为 T1c、总病灶 mask 及其无标签派生；未标签 patch 不加载四分类 target。
- 所有正式训练必须 W&B online fail-closed。

## 结果

待测试与训练。

## 结论

待验证集结果。

## 下一步

先完成代码回归、真实数据 CPU preflight 与 GPU 显存 preflight，再顺序运行监督和半监督阶段。
只有 focus mIoU、ET IoU、RC IoU、mask 外背景和 union 一致性五项验证门禁全部通过，才允许
对冻结 test 运行一次。

## 输出路径

`experiments/20260807_exp013_gli_p64_multiscale_semisupervised_classifier/outputs/`
