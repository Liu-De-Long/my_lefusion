# 实验结果

## 目标

构建 BraTS 2024 GLI 的 T1c 单模态局部病灶 patch 数据集，为 NETC、SNFH、ET、RC 四类病灶分别提供内部纹理和边界纹理样本。

## 变更

- 新增可复用的确定性 patch 裁剪器。
- 增加 `64×64×32` 与 `80×96×80` 两份配置。
- 使用患者级 80/20 划分，并在两种尺寸间共享同一划分。

## 配置

- `config_64x64x32.yaml`
- `config_80x96x80.yaml`

## 结果

- 输出目录：`/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches`
- 输入病例：1621；患者组：731；失败病例：0；归一化退化病例：0。
- 患者级划分：train 584，val 147；两种尺寸共享同一份 `splits.json`。
- 每种尺寸均生成 9842 个 patch，总计 19684 个 NPZ。
- 每种尺寸的采样角色均为 interior 4921、boundary 4921。
- 每种尺寸的 anchor 数均为 NETC 1412、SNFH 3236、ET 2446、RC 2748。
- `64×64×32`：padding patch 201（2.04%），NPZ 逻辑压缩容量 3,670,376,588 bytes。
- `80×96×80`：padding patch 1545（15.70%），NPZ 逻辑压缩容量 13,538,937,319 bytes。
- 两种尺寸 NPZ 合计约 16.0 GiB；JuiceFS `du` 分配量约 65 GiB。
- 每种尺寸生成 16 张 QA 图；manifest 各 9842 行数据；无临时文件残留。
- 远端 7 项单元/集成测试通过，5 例 staging 冒烟测试通过，最终逐 NPZ 完整性验证通过后原子发布。
- 远端环境缺少 `openpyxl`，metadata XLSX 未读取，患者归组使用确定性的 case ID 回退规则，例如将 `BraTS-GLI-00008-100` 归为 `BraTS-GLI-00008`。

## 结论

数据集已完成并通过发布门禁。两种 patch 尺寸使用相同病例、患者划分、标签定义与采样策略，可以直接用于后续 patch size 对照和四通道 GLI loader 接入。

## 下一步

实现 GLI loader：读取单份 T1c 与标量 segmentation，在加载阶段展开 NETC/SNFH/ET/RC 四个 lesion channel，并转换 NIfTI `XYZ` 为模型 `[C,D,H,W]`。
