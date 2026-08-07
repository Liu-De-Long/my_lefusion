# exp016 p64 四模态逐体素亚区分类

## 目标

在 exp009–exp015 相同的冻结 1000 个有标签 train p64 patch、相同患者级划分和相同验证门禁下，验证
T1c/T1n/T2f/T2w 四模态是否能突破 T1c-only 的可分性瓶颈，并将患者等权 ET/RC focus mIoU 提升至
`0.85` 以上。

## 变更

- 输入改为按模态独立归一化的 T1c/T1n/T2f/T2w 与总病灶 mask，共 5 通道；输出保持 mask 内四分类 logits。
- 新增冻结窗口重放脚本，逐条复用原 manifest 的 `relative_path`、患者、XYZ origin、padding 与标签 patch；输出
  manifest 保持字节级一致，因此冻结 subset 与患者划分哈希不变。
- 新 NPZ 只保存四模态、scalar `seg` 和 affine，不保存 `hist`；loader 只返回影像、总 mask、可选 target 和样本 ID。
- 从 exp014 best warm-start：共享层完全复用，stem 仅映射 T1c 与总 mask；T1n/T2f/T2w 权重初始化为 0。
- loss、网络宽度和 val-only 选模契约保持 exp014，隔离“多模态输入”这一主要变量。

## 配置

- 输入/输出：`[B,5,32,64,64] -> [B,4,32,64,64]`。
- 模型：base channels 32 的两级多尺度残差 3D U-Net，约 306 万参数。
- 有标签数据：同一冻结 1000 train patch；验证集按完整 p64 重建后患者等权聚合。
- batch 8，初始学习率 `2e-4`，最多 50 epoch，early-stopping patience 10。
- loss：CE/Focal/Dice/Lovász=`0.25/0.10/0.25/0.40`，ET/RC multiplier `1.25`，类间边界 multiplier `1.5`。
- W&B online fail-closed run：`exp016-multimodal-s20260808`。

## 结果

- 已获得用户对 exp016 四模态输入的明确授权。
- 本地独立分支/worktree 已建立；实现、测试、远端数据重建与 preflight 正在进行。
- 当前未启动正式 W&B run，未执行 optimizer step，未运行 test。

## 结论

待 val-only 正式训练完成后填写。exp014 的患者等权 focus mIoU `0.592211` 是本实验的直接对照；只有五项门禁
全部通过，才允许运行一次冻结 test。

## 下一步

1. focused tests 与单 case 重放 smoke；
2. 物化 train+val 四模态 p64 并审计 manifest/subset/split、NPZ 键、轴序和 test 未物化状态；
3. CPU 与物理 GPU0 零 optimizer-step preflight；
4. W&B online val-only 正式训练；
5. 门禁未通过则继续封存 test。

## 输出路径

`experiments/20260808_exp016_gli_p64_multimodal_classifier/outputs/supervised/`
