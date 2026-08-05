# GLI patch 级 inference 闭环接口

## 文档目的

本文记录可由后续训练、推理和数据增强实验复用的 GLI patch 级接口约定。单次 smoke 数值和运行产物记录在 `20260805_exp004_gli_inference_closed_loop` 的 `result.md`，不在本文重复维护。

## 数据与划分

- scalar segmentation `[B,1,D,H,W]` 是 label 事实源，合法值为 `0,1,2,3,4`。
- lesion mask 仅由 `lesion_mask[:,c] = segmentation == c+1` 确定性展开，shape 为 `[B,4,D,H,W]`。
- versioned split 保留 584 个 train subject，并将原 147 个 holdout subject 确定性分为 73 个 val 和 74 个 test；两种 patch 共用同一映射。
- validation checkpoint 和 histogram cluster 只能读取 train；闭环 inference smoke 只能读取 test。

## Histogram condition

四个 label 的 16-bin block 分别在 train patch 上聚类，空 block 不参与拟合。每个 `(subject,label)` 的所有 patch 权重之和为 1，避免多 patch subject 过度加权。推理时，不存在的 label 使用零 block；存在的 label 使用与 source block 最近的 train cluster center。四个 block 按 label 1 至 4 顺序拼接为 64-D condition。

## RePaint 语义

四个 lesion channel 共用一条由原始单通道 T1c 和固定 noise 构造的前向扩散背景轨迹。每个反向转换的模型输入和输出状态都重新执行病灶/背景融合：各 channel 仅在自身 lesion mask 内保留生成状态，其余位置取同一个 timestep 的共享背景状态。最终单通道背景直接取 sampler 的终态共享背景，病灶区按 scalar label 选择相应生成 channel。

禁止以下操作：

- 采样完成后再用原始 T1c 硬覆盖健康区；
- 对四个输出 channel 求平均；
- 使用任一病灶 channel 的未监督区域作为独立背景预测；
- 将 source segmentation 描述为生成 segmentation。

## 空间与保存

- `patch_size_xyz=[64,64,32]` 对应模型 `DHW=[32,64,64]`。
- `patch_size_xyz=[80,96,80]` 对应模型 `DHW=[80,80,96]`。
- 模型内部始终使用 CDHW，NPZ/NIfTI 始终显式转换为 XYZ 并保留 patch affine。
- checkpoint 必须携带 schema、模型结构、空间 shape、训练步数、随机种子和 Git SHA；inference 加载时不一致立即报错。

## 显式 brain support

`explicit_brain_support_mask` 由“四模态至少两个原始体素非零、最大 3D 连通域、孔洞填充、并入 segmentation”构成。它不是人工金标准，不参与当前 RePaint 混合，仅用于 lesion 脑内合法性、healthy brain/outside 分区指标和 normalization QA。禁止使用 `normalized_t1c != 0` 反推 brain mask。

## 验收边界

本闭环验证 patch 级接口、shape、条件、采样和保存，不验证医学有效性，不生成新 segmentation，不覆盖全脑 sliding-window 拼接，也不构成启动正式训练的授权。
