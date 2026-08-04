# 实验结果

## 目标

整理 BraTS2024 GLI 数据审计与预处理统计产物，将跨实验复用的统计结果从根目录 `results/` 收拢到实验卡片目录，避免形成第二套实验管理入口。

## 变更

- 将全量病灶 bbox 统计结果迁移到 `outputs/region_patch_stats_full/`。
- 将 100 例模态强度统计结果迁移到 `outputs/modality_intensity_stats_100cases/`。
- 同步更新 `README.md`、`STATUS.md`、`CHANGELOG.md` 和 `docs/` 中的旧路径引用。
- 如果根目录 `results/` 迁移后为空，则移除空目录。

## 配置

本次为项目结构整理，不新增训练配置。原统计命令和参数见 `docs/20260804_002_brats2024_gli_patch_preprocessing_plan.md`。

## 结果

- 全量病灶 bbox 统计：`experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/region_patch_stats_full/`
- 100 例模态强度统计：`experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/modality_intensity_stats_100cases/`

## 结论

统计产物已归入实验目录，后续以 `experiments/` 作为实验配置、结果记录和轻量输出的主入口；`docs/` 继续保留跨实验长期说明，`scripts/` 保留可复用工具脚本。

## 下一步

继续实现 T1c 单模态输入和 NETC/SNFH/ET/RC 四通道 lesion mask 的 GLI loader，并在对应实验目录下管理配置与输出。
