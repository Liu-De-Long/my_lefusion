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
