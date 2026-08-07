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
- histogram counterfactual 为 `7/8`，union-as-single counterfactual 为 `7/8`，mask 外最大绝对误差为 `0`；四项定量门槛全部通过。
- 目检显示病灶生成与 mask 对齐、洞外严格不变，paired 重建在三方法中最好；但 first/last cluster 的视觉纹理差异弱于 exp012。

## 结论

exp010 已证明单一完整 T1c 状态能在 5k 内学习挖空区域的非平坦病灶生成，并通过全部短程门槛。它是 paired 重建质量最好的次优方案，但 histogram/union 控制一致性略弱于 exp012，因此不作为当前首选可控生成方法。

## 下一步

不扩大到其他 seed、p80、519 例或全量 test。若后续要提高视觉真实性，优先将 exp010 作为重建质量对照，与 exp012 的 histogram 控制能力联合设计；须另建实验并重新授权。

## 输出路径

- 实验输出：`experiments/20260807_exp010_gli_single_state_noise/outputs/`
- 统一比较：`results/20260807_exp010_exp012_short_comparison/`
