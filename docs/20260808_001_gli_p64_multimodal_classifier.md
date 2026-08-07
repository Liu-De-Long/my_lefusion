# GLI p64 四模态亚区分类数据与训练契约

## 1. 决策背景

exp014 在 T1c 与总病灶 mask 派生的 18 通道无泄漏输入上达到患者等权 ET/RC focus mIoU
`0.592211`；exp015 的患者/组件重加权降至 `0.555747`。这说明继续调整同一 T1c 表征下的 loss 主要是在
重新分配错误，不能提供达到 `0.85` 所需的新可分信息。

既有 100-case 原始强度统计显示四模态提供互补对比：T1c 对 ET 最显著，T2f 对 SNFH 有独立响应，T2w
对 RC 更敏感，T1n 提供不同于 T1c 的组织对比。因此 exp016 将主要变量改为四模态输入，而不同时引入新的
采样或半监督配方。

## 2. 冻结数据契约

- 模态顺序固定为 `T1c, T1n, T2f, T2w`，每个模态分别在全病例非零前景上用 0.5/99.5 百分位裁剪并映射至
  `[-1,1]`；背景保持 0。
- 新 NPZ 的精确键为 `t1c,t1n,t2f,t2w,seg,affine`，不生成 per-class `hist`。
- `seg` 只用于监督 target 与推导 `total_mask = seg > 0`，不会生成任何 per-class 输入通道。
- patch 存储轴序保持 XYZ=`[64,64,32]`；loader 转换为 CDHW=`[4,32,64,64]`。
- 重建器逐条重放原 manifest 的 `relative_path`、`case_id`、`subject_id`、`origin_x/y/z` 与 padding，并逐 patch
  断言重建 T1c、seg 和 affine 与原数据一致。
- 输出 manifest 直接复制原始字节，目标 SHA-256 固定为
  `42f71687b3edfe1f54819a01490224e3bd9b6a56be25125aed0d311b99e683ad`；因此原冻结 1000 patch 文件
  `aa7cd984...fa3a` 可原样复用，不重新抽样。
- `splits_v2.json` SHA-256 固定为
  `7fd0ea31dbbc7d68142852df49abb63b3cc5bdedeceeb1baef53adfdc894e8b1`，患者级 train/val/test
  隔离不变。
- 首次只物化 train+val。test 行仍保留在 manifest 供哈希校验，但 test NPZ 不生成；只有 val 五项门禁通过后，
  才能在单独授权步骤中物化并运行一次 test。

## 3. 模型契约

网络输入为 `[B,5,32,64,64]`：四个 MRI 通道加一个总病灶二值 mask；输出为
`[B,4,32,64,64]`，依次对应 NETC/SNFH/ET/RC。训练与验证 loss 只在总 mask 内计算；重建时 mask 内
argmax 加 1，mask 外强制写 0，因此类别 union 必须与输入总 mask 完全一致。

模型使用 exp014 相同的 base-32 两级多尺度残差 3D U-Net 和相同 CE/Focal/Dice/Lovász/边界监督。加载
exp014 best 时，共享层严格复用；输入 stem 的 T1c 与 mask 权重分别映射到新通道 0 与 4，新增三个模态通道
初始化为 0。该映射保留初始 T1c-only 行为，同时允许优化器学习互补模态。

强度增强对每个样本、每个模态独立采样 scale/shift，并加入逐体素噪声；空间翻转对所有模态、总 mask 和 target
同步执行。任何模态缺失、顺序错误、shape/dtype/range 不符均 fail closed。

## 4. 训练与评价门禁

- 有标签训练仍只用冻结 1000 个 train patch，不读取真实四分类 hist 或 manifest 类别统计作为输入。
- 仅按完整 p64 重建后的患者等权 val focus mIoU 选择 best；不允许 pooled voxel 指标替代选模指标。
- 正式训练必须使用 W&B online fail-closed，并记录 config/subset/initial checkpoint/Git SHA。
- 物理设备只允许 GPU0；GPU1 上的既有进程不得检查、终止或迁移。
- test 五项门禁保持：focus mIoU≥`0.85`、ET IoU≥`0.80`、RC IoU≥`0.80`、mask 外非零率=`0`、
  union Dice=`1`。全部通过后最多运行一次冻结 test。

## 5. 可复现命令

重建 train+val 数据：

```bash
python scripts/brats_gli_rebuild_multimodal_p64.py \
  --raw-root /workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI \
  --raw-split train \
  --source-dataset-root /workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches \
  --split-file experiments/20260805_exp004_gli_inference_closed_loop/splits_v2.json \
  --output-root /workspace/LeFusion_v2/dataset/brats2024_gli_multimodal_local_patches \
  --include-split train --include-split val --workers 4
```

CPU/GPU0 preflight 与正式训练继续使用统一入口：

```bash
python scripts/gli_p64_classifier.py preflight --config experiments/20260808_exp016_gli_p64_multimodal_classifier/config.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/gli_p64_classifier.py gpu-preflight --config experiments/20260808_exp016_gli_p64_multimodal_classifier/config.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/gli_p64_classifier.py train --config experiments/20260808_exp016_gli_p64_multimodal_classifier/config.yaml
```

上述命令中的 test 不属于默认流程；门禁通过前禁止调用 `evaluate --split test`。
