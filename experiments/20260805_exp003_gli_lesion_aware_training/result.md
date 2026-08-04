# 实验结果

## 目标

为 BraTS2024 GLI T1c 四通道局部 patch 接入 lesion-aware diffusion loss、训练 batch 传递和矩形三维空间 shape，并在不启动正式训练的前提下完成真实 GPU smoke 验收。

## 变更

- `train.py` 接受 `gli`，新增配置校验和可复用模型构建入口。
- 使用独立的 `base_dim` 和 `spatial_shape_dhw`，保留 LIDC/EMIDEC legacy 配置兼容。
- Trainer 对 GLI 使用四通道 `lesion_mask`，histogram 移动到当前 device。
- GLI loss 对所有非空 `(sample, lesion channel)` 单元等权平均，不按病灶体素数或通道类别出现频率加权。
- diffusion 支持严格 `[B,C,D,H,W]` 校验、固定 timestep/noise 测试和统一 sample shape。
- 推理调用方改用完整 sample shape；GLI RePaint 语义仍未开放。

## 配置

- `config_64x64x32.yaml`，runtime override：`+experiment=gli_64x64x32`。
- `config_80x96x80.yaml`，runtime override：`+experiment=gli_80x96x80`。
- 图像通道：4；histogram 条件维度：64；UNet `base_dim=64`；`temporal_max_distance=128`。
- 两种 smoke 均为 batch size 1 和 AMP；64 patch 执行固定 batch 20 步，矩形 patch 执行单步前向/反向。

## 结果

- 集成测试：5/5 通过。
- 全量测试：16 项通过。
- 真实发布数据 loader：4/4 通过。
- Hydra：两份配置均完整解析，无 `???`。
- `64×64×32`：batch `[1,4,32,64,64]`；首 5 步 loss 均值 `0.785975`；末 5 步 `0.360158`；峰值显存 `6309.063 MiB`。
- `80×96×80`：batch `[1,4,80,80,96]`；单步 loss `0.852194`；前向和反向通过；峰值显存 `33944.751 MiB`。
- W&B：
  - `patch_64x64x32`：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/23hogegm>
  - `patch_80x96x80`：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/228g4h4j>
- smoke 没有保存 checkpoint；同步 W&B 后已删除远端 offline 临时缓存和空 outputs 目录。

## 结论

两种 patch 的训练接口均已通过限定 smoke。当前四层 resolution stage 对 H/W 实际执行三次 stride-2 下采样，`64×64` 和 `80×96` 都能被 8 整除并精确恢复；depth 保持不变。矩形 patch 在 A100 80GB 上以 batch size 1 可完成反向，但显存明显高于第一阶段，正式 batch size 不应在本实验中外推。

## 下一步

1. 用官方说明或 metadata 复核 `1=NETC、2=SNFH、3=ET、4=RC`。
2. 单独确认正式训练的 batch size、gradient accumulation、验证策略和 W&B 命名后再启动正式训练。
3. 将 GLI inference loader、RePaint keep-mask、histogram cluster 和输出合并作为后续方法实验。

## 输出路径

- 运行配置：`LeFusion/train/config/experiment/gli_64x64x32.yaml`、`LeFusion/train/config/experiment/gli_80x96x80.yaml`
- smoke 工具：`scripts/gli_training_smoke.py`
- 测试：`tests/test_gli_training_integration.py`
- W&B：上述两个在线 run URL
- 本地/远端 checkpoint：无

