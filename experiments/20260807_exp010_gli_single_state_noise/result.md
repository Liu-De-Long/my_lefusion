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
  [exp010-p64-full50k-2gpu-s20260805](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp010-p64-full50k-2gpu-s20260805)，状态 running。
- 启动审计到 optimizer step `649`，最近 loss `0.280569`、grad norm `1.472375`，均为有限值；
  GPU0/1 显存约 `13369/13165 MiB`，两卡均有计算利用率。
- step 500 的首个 `latest.pt` 已写入，约 554 MiB；首次观察 SHA-256 为
  `4a80b4a3a472e99ae1787468223a660f4c51edd41961acb27cf33317238ce98c`。训练继续运行。

### 结论

双 GPU、W&B、DataLoader、optimizer 和 checkpoint 正式闭环已进入稳定运行；模型质量结论等待
后续 validation、50k 完成及冻结少量 QA。

### 下一步

只启动这一份 exp010 正式训练并监控 W&B；exp012 保持未启动。

### 输出路径

`experiments/20260807_exp010_gli_single_state_noise/outputs/full_50k_2gpu/`
