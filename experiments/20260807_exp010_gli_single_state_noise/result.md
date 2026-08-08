# exp010 单一完整 T1c 扩散状态

## 目标

删除 exp008 的四份重复 T1c 状态，验证单一完整 T1c 扩散状态能否在 5000 step 内学习挖空区域的病灶生成，并响应四组 histogram 条件。

## 变更

- 扩散状态和输出均改为单通道 T1c。
- 空间条件固定为挖空 T1c 加四通道 lesion mask。
- 预测目标为噪声，loss 按非空病灶类别区域逐体素归一化。
- histogram condition dropout 为 0.1，推理使用 CFG。

## 配置

- patch：`64×64×32`
- seed：`20260805`
- 最大 optimizer step：`5000`
- W&B run：`exp010-p64-s20260805`
- QA：固定 8 个 validation patch，五种条件变体；不运行 test split。

## 结果

- 正式运行代码 commit：`d0c6aaa056b38e34e4cb929f85106cc90af05426`；远端完整回归测试 `39/39` 通过。
- BF16 正式预检（零 optimizer update）通过：loss `1.069728`，梯度范数 `16.731850`，反向峰值显存 `23324.62 MiB`；checkpoint 恢复成功。
- 预检 validation：`val/ema/total_loss=0.996760`，覆盖 `1032` patches、`73` subjects、`3207` 个有效类单元及全部 4 类。
- 正式训练仅使用 GPU1、seed `20260805`，完成 `5000` optimizer step；W&B run
  [`exp010-p64-s20260805`](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp010-p64-s20260805)
  已 finished。首次后台启动因登录 shell 丢失 conda PATH，在 step 0 导入 `einops` 时退出；保留失败日志后改用 conda Python 绝对路径启动，未产生重复训练或 checkpoint。
- `milestone-500/1000/2500/5000.pt` 与 `latest.pt` 齐全。step 5000 的 EMA validation total loss
  为 `0.135519`，覆盖 `1032` patches、`73` subjects、`3207` 个有效类单元及全部 4 类。
- 五种 QA 均只运行冻结的 8 个 validation patch，全部为 `8/8`，NPZ、`metrics.json`、
  `run_contract.json` 与 contact sheet 齐全；没有运行 519 例、test split 或全量 test。
- `qa_original_real` 的平均 lesion MAE 为 `0.138066`，相对零填洞基线 `0.323883` 平均改善
  `55.1%`；8/8 病例达到“改善至少 20% 且非零、非平坦”的门槛。
- histogram counterfactual 为 `7/8`，union histogram counterfactual 为 `7/8`，mask 外最大绝对误差为 `0`；按原定四项定量门槛全部通过。
- counterfactual 只比较 mask 内 16-bin 强度 histogram，不衡量空间纹理、形态或医学类别语义。
- 目检显示病灶生成与 mask 对齐、洞外严格不变，paired 重建在三方法中最好；但 first/last cluster 的亮暗分布响应弱于 exp012。

## 结论

exp010 已证明单一完整 T1c 状态能在 5k 内学习挖空区域的非平坦病灶生成，并通过原定短程门槛。它是 paired 重建质量最好的方法；histogram 配对响应弱于 exp012，但现有 QA 不能据此判断两者的医学亚型控制能力。

## 下一步

不扩大到其他 seed、p80、519 例或全量 test。若后续要提高视觉真实性，优先将 exp010 作为重建质量对照，与 exp012 的 histogram 控制能力联合设计；须另建实验并重新授权。

## 输出路径

- 实验输出：`experiments/20260807_exp010_gli_single_state_noise/outputs/`
- 统一比较：`experiments/20260807_exp013_gli_short_method_comparison/outputs/`

## 50k 双 GPU 正式训练扩展

### 目标

在短程 5k 已证明能够生成非零、非平坦病灶后，保持 exp010 方法契约不变，从随机初始化开始运行
完整 train split 的 50,000 optimizer step 正式训练。exp012 仅作为后续对照，本次不启动。

### 变更

- 方法、loss、seed、全局 batch 与学习率不变。
- 启用 GPU0/1 `DataParallel`，全局 batch 4、每卡 micro-batch 2。
- train/validation DataLoader worker 均提高到 8；W&B 使用新的 online fail-closed run。
- 新输出目录与 5k 资产隔离，不从 5k checkpoint 强行恢复。

### 配置

- Hydra：`LeFusion/train/config/experiment/gli_exp010_single_state_noise_64x64x32_full.yaml`
- 轻量记录：`config_full_50k_2gpu.yaml`
- W&B run：`exp010-p64-full50k-2gpu-s20260805`
- 最大步数：50,000；validation 每 2,000 step；latest 每 500 step；milestone 每 5,000 step。

### 结果

- 双 GPU online preflight 已通过，optimizer update 为 `0`；W&B：
  [exp010-p64-full50k-2gpu-preflight-s20260805-r1](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp010-p64-full50k-2gpu-preflight-s20260805-r1)。
- 真实 batch shape 为 `[4,1,32,64,64]`，loss `1.069795`，梯度范数 `16.795513`。
- GPU0/1 反向峰值显存为 `11855.53/11693.59 MiB`；双卡均有实际计算与参数副本。
- 完整 validation 覆盖 1032 patch、73 subject、3207 个有效类单元和全部四类，EMA total loss
  `0.996759`；checkpoint 保存/恢复和下一 batch 读取通过，临时 checkpoint 已自动删除。
- 正式训练已于 `2026-08-07 19:58:48 UTC`（北京时间 `2026-08-08 03:58:48`）启动，父 PID
  `39061`，运行 HEAD `1392bffed5691fae7651e50d14a080b228fb7aff`。
- 正式 W&B：
  [exp010-p64-full50k-2gpu-s20260805](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp010-p64-full50k-2gpu-s20260805)，远端状态为 `finished`。
- 启动审计到 optimizer step `649`，最近 loss `0.280569`、grad norm `1.472375`，均为有限值；
  GPU0/1 显存约 `13369/13165 MiB`，两卡均有计算利用率。
- step 500 的首个 `latest.pt` 已写入，约 554 MiB；首次观察 SHA-256 为
  `4a80b4a3a472e99ae1787468223a660f4c51edd41961acb27cf33317238ce98c`。
- 本次没有正常达到 50k 或 early stopping：训练 loss/grad 在 step `27838` 首次变为 NaN，
  step `28000` 的 EMA validation 也为 NaN，随后有限性门禁以
  `ValueError: early-stopping metric is not finite: nan` 退出。`latest.pt` 的 model/EMA 各有
  `291` 个含非有限值的 tensor，禁止用于推理或恢复。
- 最佳有限 checkpoint 为 step `14000` 的 `best.pt/EMA`，validation total loss
  `0.12645006954837223`，SHA-256 为
  `58e939798fc8d3427b70aafdd61877a14984d2478cb7a39796c6019407f07358`；model/EMA 共
  `36,235,313` 个参数元素均为有限值。step 20000 虽得到略低绝对 loss `0.1263959`，但未满足
  0.5% 相对改善门槛，因此正式 `best.pt` 仍冻结在 step 14000。

### 结论

双 GPU训练在 28k 暴露后期数值失稳，不能记为“50k 成功完成”。step 14k 的 best/EMA 在失稳前、
参数全有限且验证最优，保留为后续冻结评估入口；污染的 latest 不再使用。

### 下一步

不擅自恢复或补跑训练。按用户授权，仅用 step 14k best/EMA 在原冻结 test split 中运行确定性
50% 子集（519/1038），GPU0/1 各一个输出互斥分片；不运行全量 test。

## step 14k best/EMA 的 test 50% 评估

- 复用已冻结的 test 50% manifest，SHA-256 为
  `295b20a01327dcd0071058efd8fe854135688a843c1888ef33242a908b6c3698`。
- 全 test 为 `1038` patch，本次只选 `519`，两个固定 shard 为 `260/259`；selection seed
  `20260806`，不存在运行时重新抽样。
- 输入为 union 病灶区严格置零的 T1c 加原始四通道 mask；hist 使用训练集 cluster 中与真实
  四类 hist 最近的中心，CFG scale `2.0`，`t_T=300`，checkpoint 固定为 step 14k EMA。
- GPU0/GPU1 各一个独立进程，每进程 4 个 DataLoader worker，输出目录互斥且启用严格 resume；
  不允许第三个生成进程、重复病例或全量 test。
- 运行状态与 PID 见 `config_test_subset50_2gpu.yaml`；最终指标和审计在两个 shard 完成后补记。

### 输出路径

`experiments/20260807_exp010_gli_single_state_noise/outputs/full_50k_2gpu/`
