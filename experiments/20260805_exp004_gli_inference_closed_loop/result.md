# 实验结果

## 目标

在不启动正式训练的前提下，完成 GLI patch 级 validation checkpoint、train-only histogram cluster、逐步 RePaint、四通道病灶合成和单通道 NPZ/NIfTI 保存闭环，并审计当前 normalization 所依赖的原始 T1c 非零区域。

## 变更

- 将原 147 个 holdout subject 确定性拆分为 73 个 val 和 74 个 test，保留原 584 个 train subject。
- 新增 GLI inference dataset adapter、显式多模态 brain support、train-only 分标签 histogram clustering 和可视化工具。
- 按 scalar segmentation 确定性展开四通道 lesion mask；RePaint 每个反向转换均使用共享的前向扩散 T1c 背景状态。
- sampler 内部将四通道终态折叠为单通道；不执行采样结束后的原始健康区硬覆盖。
- 新增严格 checkpoint schema、矩形 shape、DHW/XYZ、NIfTI 回读、分区指标和 provenance 验证。

## 配置

- `config_64x64x32.yaml`：训练 smoke 20 step，inference batch size 1，缩减 RePaint schedule `t_T=5`。
- `config_80x96x80.yaml`：训练 smoke 1 step，inference batch size 1，缩减 RePaint schedule `t_T=5`。
- histogram condition：每个 label 独立使用 train patch 聚类，空 label 保持零 block，存在 label 选择与 source histogram 最近的 train center。
- 显式 brain support：四模态至少两个非零，保留最大 3D 连通域并填洞，最后并入 `segmentation > 0`；仅用于 QA 和合法性检查。

## 结果

- 数据划分：`splits_v2.json` 为 train/val/test `584/73/74`，三组 subject 零交集；cluster 和 checkpoint 仅使用 train，inference 仅使用 test。
- cluster：两种 patch 的 NETC/SNFH/ET/RC 均选择 `k=[5,3,2,2]`；完整保留 membership、centers、metrics、provenance、k-sweep、PCA、中心分位带、cluster size 和代表 patch montage。
- normalization 审计：64 个分层 train subject 的 median Dice `0.999997`、p05 Dice `0.990682`、median extra fraction `0`、p95 extra fraction `0.002444`，四项自动门禁全部通过。最差 16 例和随机 16 例视觉检查未见系统性颅外伪影；个别低位病例存在局部原始非零/多模态 support 差异，已作为限制保留。
- 64 checkpoint：真实 batch `[1,4,32,64,64]`，20 step，首 5 步 loss `0.785975`、末 5 步 `0.360158`，峰值显存 `6309.063 MiB`。
- 矩形 checkpoint：真实 batch `[1,4,80,80,96]`，1 step loss `0.852194`，峰值显存 `33944.751 MiB`。
- 64 inference：test sample 1，内部 `[1,4,32,64,64]`，最终 `[1,1,32,64,64]`，5 次模型调用，`2.6067 s`，峰值显存 `1083.212 MiB`；NIfTI XYZ `[64,64,32]`。
- 矩形 inference：test sample 1，内部 `[1,4,80,80,96]`，最终 `[1,1,80,80,96]`，5 次模型调用，`4.6900 s`，峰值显存 `6518.637 MiB`；NIfTI XYZ `[80,96,80]`。
- 两种 inference 的 healthy-brain MAE、outside residual 和 boundary outer-shell MAE 均为 `0`；病灶 change MAE 分别为 `0.6960` 和 `0.7044`。该精确保持来自每个反向步的 sampler 状态融合，不是采样结束后的原图覆盖。
- 全量测试在真实数据根目录下 22/22 通过；四份 Hydra 配置均完整解析且无 `???`。
- W&B：首次 online 初始化因远端无凭据而在计算前失败；两次 checkpoint smoke 随后以 offline 模式完成，run ID 为 `30rwobgb` 和 `30wigck6`，尚无在线 URL。
- Hydra 解析生成的远端临时 `.hydra/` 已删除；checkpoint、offline run、cluster 和生成体积均保留在隔离 outputs，不进入 Git。

## 结论

patch 级训练 checkpoint→train-only cluster→GLI RePaint→四通道病灶合成→单通道保存闭环已经打通，64 和矩形 shape 均可行。当前 normalization 可保留，但显式 brain support 仍必须独立传递，不能由 normalized value 反推。当前 validation checkpoint 与 `t_T=5` 结果只可用于接口验收，不能作为正式模型或生成质量结论；正式训练仍受标签医学语义和 W&B 在线记录门禁约束。

## 下一步

1. 用官方说明或 metadata 复核 `1=NETC、2=SNFH、3=ET、4=RC`。
2. 通过远端环境配置新的 W&B 凭据，并按需同步两个 offline smoke run。
3. 用户另行确认正式训练方案后再启动训练；本实验不启动正式训练。
4. 获得正式 checkpoint 后，再使用完整 RePaint schedule 做生成质量和下游医学评估。

## 输出路径

- 远端实验根目录：`/workspace/LeFusion_v2/my_experiment/experiments/20260805_exp004_gli_inference_closed_loop`
- checkpoint、W&B 缓存和生成体积位于该目录的 `outputs/` 下，不进入 Git。
- versioned split、配置、结果记录和代码版本进入 Git。
- normalization QA 图：`outputs/normalization_audit/`。
- cluster 图与 provenance：`outputs/patch_64x64x32/clusters/`、`outputs/patch_80x96x80/clusters/`。
- inference 结果：两个 patch 目录下的 `inference_cluster/`。
