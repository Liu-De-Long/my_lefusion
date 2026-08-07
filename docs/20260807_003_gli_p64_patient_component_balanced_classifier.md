# p64 患者与组件等权分类方案

## 1. 问题定义

exp014 best 的患者等权 ET/RC focus mIoU 为 `0.592211`，但仅作参考的 pooled ET/RC IoU
为 `0.813983/0.775111`。`>1000` 体素 ET/RC patch Dice 已为 `0.877092/0.808596`，
而 `1–100` 体素仅 `0.124810/0.196575`。这不是继续增加同配方 epoch 能解决的普通收敛问题，
而是训练 reduction 与患者等权评价目标不一致。

## 2. exp014 目标失配

exp014 的 CNN 随机遍历全部 patch；CE/Focal 在整个 batch 内按体素归一，Dice/Lovász 跨 batch
聚合。class weight 能平衡总体类别频率，却不能阻止同类的大区域覆盖小区域梯度；同一患者的多个
patch 也会比单 patch 患者贡献更多 optimizer signal。

## 3. exp015 对齐方法

### 3.1 患者等权 epoch

每个 epoch 每位训练患者只贡献一个 patch，并用 `seed、subject_id、relative_path、epoch`
确定性轮换。该 sampler 不读取四分类 target；跨 epoch 覆盖同一患者的全部冻结 patch。

### 3.2 样本类别与组件等权

CE/Focal 先在每个 `样本×类别` 单元内归一，再在样本间和类别间聚合。ET/RC 按真实目标体素量
只在 loss 中设置 `1–100/101–1000/>1000 = 4/2/1` 权重。同类存在多个 26 邻域连通组件时，
每个组件按体积倒数分配 mass，并以 `16×` 封顶。

### 3.3 小区域检测辅助

sample-wise Tversky 使用 `alpha=0.3、beta=0.7`，优先降低小区域 false negative；patch
presence 使用 mask 内每类 top-32 概率均值做二元辅助，显式区分“该 patch 是否存在该类”。
推理仍只使用四类 voxel logits，不增加标签派生输入或推理元数据。

## 4. 泄漏边界

模型输入继续严格限定为 T1c 与总 mask 推导的 18 通道。真实四分类 target 只允许用于监督 loss、
组件标记、区域大小分带和训练采样审计；禁止将其计算结果送入网络 forward、验证输入或 test。

## 5. 验收顺序

1. focused tests 验证患者轮换覆盖、组件 mass、大小区域梯度等权、同构 warm-start 与 test seal；
2. CPU 零 optimizer-step preflight；
3. 获得新 GPU 授权后再做 GPU0 零步 preflight；
4. 只运行一个 W&B online 正式 run，仅按完整患者级 val 选模；
5. focus mIoU、ET IoU、RC IoU 与两项空间门禁全部通过后，才允许一次冻结 test。

## 6. 2026-08-08 训练前验收

- 用户已授权推送 exp015，并在远端恢复后执行 focused tests、CPU/GPU0 preflight 和 val-only
  正式训练；冻结 test 的五项门禁没有变化。
- 分支已推送，并在 `/workspace/LeFusion_v2/my_experiment_exp015` 创建独立 worktree。本地、
  origin 与远端训练代码提交为 `4bd5f8e5a1d0c4102c403e21d5fa470833385bdb`。
- focused tests 首次为 19/20；唯一失败来自测试样例每个样本类别只有一个组件，无法触发归一化后
  大于 1 的小组件权重。补充同类大小两个分离组件后，完整 focused tests 为 20/20。
- CPU preflight 为零 optimizer step，覆盖 8 个 `类别×boundary/interior` 分层；输入/输出为
  `[1,18,8,16,16] -> [1,4,8,16,16]`，loss `0.748481`，gradient norm `23.960320`。
- GPU preflight 只暴露物理 GPU0，完整 p64 batch 8 输入/输出为
  `[8,18,32,64,64] -> [8,4,32,64,64]`；loss `0.462901`，gradient norm `4.351939`，
  峰值 allocated/reserved 显存 `2865.985/3946 MiB`，optimizer steps 为 0。GPU1 既存进程未被干扰。
- 两份 preflight 的冻结 subset SHA-256 均为
  `aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a`，config SHA-256 均为
  `66569653115af42e26667e3b3abe01120951704edfa87d0588c4dbfdcd259abd`，exp014 initial checkpoint
  SHA-256 均为 `b293e5d35b078f6a290a24695f0081e0ce0599154513bf6bcbf7372a007898c2`。
- 截至该训练前验收节点，尚未初始化 exp015 正式 W&B run、未执行 optimizer step、未读取或运行 test；
  后续正式结果见第 7 节。

## 7. 正式结果与方法结论

正式 run 完成 epoch 0–9；epoch 2–9 的患者等权 focus mIoU 长期在 `0.54–0.56` 波动，学习率
降至 `5e-5` 后仍无改善。用户在 epoch 10 训练中触发方法级 early stopping，未形成 epoch 10
history 或 checkpoint。最终只对 epoch 5 best 做完整 1032-patch、73-patient val 与 1000 次
患者 bootstrap：

- focus mIoU `0.555747`，95% CI `[0.510204,0.600693]`；
- ET/RC IoU `0.503885/0.607609`；
- `1–100` 体素 ET/RC patch Dice `0.167198/0.232198`；
- outside nonzero `0`，union Dice `1`；
- 三项性能门禁失败，冻结 test 未运行。

相对 exp014 的 focus mIoU `0.592211`，exp015 下降 `0.036464`。小区域 Dice 的局部提高没有转化为
患者等权总体收益，说明患者/组件重加权、偏召回 Tversky 和 patch presence 主要重新分配错误，
并未提高 T1c+总 mask 对四类亚区的可分信息。继续训练或叠加同类 loss 权重不再是合理路径。

## 8. 下一方法决策

下一步必须改变表征或输入信息：若坚持 T1c-only，应先审计 train-fit 上限和小区域标签可辨识性，
再考虑患者多 patch 联合上下文、全量 train T1c/mask 自监督预训练与 uncertainty-aware refinement；
若允许扩展输入，则优先使用已对齐 T1n/T2f/T2w，构建 T1c+多模态+总 mask 的 p64 分割器。
exp013 已证明当前 mean-teacher 配方无增益，不应原样重复。新路线必须使用新实验 ID 并由用户确认。
