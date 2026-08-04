# 实验结果

## 目标

实现并注册 BraTS2024 GLI T1c 局部 patch loader，保留原始 LeFusion 的标量 segmentation 语义，同时提供 NETC、SNFH、ET、RC 四通道 lesion mask 和 64 维 histogram 条件。

## 变更

- 新增 `LeFusion/dataset/gli_hist.py`，读取 NPZ 和 manifest。
- 保留 `label` 为 `[1,D,H,W]` 的标量 segmentation。
- 新增 `lesion_mask` 为 `[4,D,H,W]` 的 NETC/SNFH/ET/RC 二值 mask。
- 将 T1c 从 `[X,Y,Z]` 转换为 `[C,D,H,W]`，并复制到四个 lesion 对齐通道。
- 在 `get_dataset.py` 注册 `gli` 训练 dataset。
- 暴露 manifest 中的 padding、origin、anchor 和归一化 metadata。
- 增加两种 patch 尺寸的 focused tests。

## 配置

- 配置文件：`config.yaml`
- 模态：`t1c`
- lesion channel：`NETC/SNFH/ET/RC`
- 标签值：`1/2/3/4`
- histogram：`4×16=64`
- patch：`64×64×32`、`80×96×80`
- 训练：本实验未启动训练。

## 结果

本地测试 fixture 覆盖轴转换、scalar label、四通道 lesion mask、histogram 顺序、batch shape、padding metadata 和 factory 注册。远端测试将使用已发布的真实 patch 数据集验证两种尺寸迭代。

## 结论

GLI loader 可以在不破坏原始 EMIDEC scalar-label 约定的前提下提供四通道 lesion mask。当前 diffusion loss 尚未实现 `gli` 分支，因此本实验只确认数据接口，不宣称已具备训练能力。

当前空间截断 metadata 定义为 patch padding 比例；由于 manifest 没有整病灶原始体素数，未计算 lesion coverage 截断比例。

## 下一步

单独实现 GLI 四通道 loss、Trainer 的 `lesion_mask` 传递，以及矩形 `80×96×80` patch 的 diffusion shape 支持；之后再设计 GLI inference loader 和 RePaint keep-mask。

## 输出路径

- loader：`LeFusion/dataset/gli_hist.py`
- 测试：`tests/test_gli_loader.py`
- 数据集：`/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches`
