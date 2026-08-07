# exp015 p64 患者与组件等权分类

## 目标

针对 exp014 已确认的“小区域 ET/RC 与患者等权聚合”瓶颈，在不改变冻结 1000 patch、患者划分、
输入白名单和验证门禁的前提下，使训练目标从大区域体素主导改为患者、样本类别和连通组件等权，
继续冲击患者等权 ET/RC focus mIoU `0.85`。

## 变更

- 每个 epoch 每位训练患者确定性轮换抽取一个 patch，避免多 patch 患者重复加权；跨 epoch 循环
  覆盖该患者的全部冻结 patch。
- CE/Focal 先在每个 `样本×类别` 内归一，再按类别聚合；ET/RC 的 `1–100/101–1000/>1000`
  体素区域使用 `4/2/1` 的训练权重。
- 同类多连通组件按体积倒数分配 loss mass，并以 `16×` 上限防止单体素噪声无限放大。
- 新增偏召回的 sample-wise Tversky 与 top-32 patch presence 辅助；所有标签派生量只参与
  sampler/loss，不进入模型输入或推理。
- 支持从 exp014 `geometry_unet3d` best 严格同构 warm-start；subset 或参数 shape 不一致时
  fail closed。

## 配置

- 模型与输入保持 exp014：`[B,18,32,64,64] -> [B,4,32,64,64]`，约 307 万参数。
- loss：CE/Focal/Dice/Lovász/Tversky/Presence=`0.25/0.10/0.15/0.10/0.30/0.10`；
  Tversky `alpha/beta=0.3/0.7`，ET/RC multiplier `1.25`。
- batch 8，初始学习率 `1e-4`，最多 60 epoch；只依据完整 p64 患者等权 val 选模。
- W&B online fail-closed run 预留为 `exp015-patient-component-s20260807`。

## 结果

- 本地静态编译已通过。
- 分支已推送，并在远端独立 worktree 同步相同代码提交 `4bd5f8e5a1d0c4102c403e21d5fa470833385bdb`。
- 远端 focused tests 20/20 通过。
- CPU 零 optimizer-step preflight 通过：subset 1000，loss `0.748481`，gradient norm `23.960320`。
- 物理 GPU0 零 optimizer-step preflight 通过：完整 p64 batch 8，loss `0.462901`，峰值
  allocated/reserved 显存 `2865.985/3946 MiB`；GPU1 既存进程未被干扰。
- subset/config/initial checkpoint SHA-256 分别为 `aa7cd9844550c826f17ebe4cb43718d5c8ff61759c3b0ae81b7935c79886fa3a`、
  `66569653115af42e26667e3b3abe01120951704edfa87d0588c4dbfdcd259abd`、
  `b293e5d35b078f6a290a24695f0081e0ce0599154513bf6bcbf7372a007898c2`。
- 唯一正式 GPU0/W&B online run 为
  <https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp015-patient-component-s20260807>。
- 正式训练完成 epoch 0–9；epoch 2–9 的患者等权 focus mIoU 长期在约 `0.54–0.56` 波动，
  学习率降至 `5e-5` 后仍无改善。用户据此在 epoch 10 训练中触发方法级 early stopping；
  epoch 10 未完成、未写入 history 或 checkpoint。
- best 为 epoch 5：患者等权 ET/RC focus mIoU `0.555747`，95% CI
  `[0.510204,0.600693]`；ET/RC IoU `0.503885/0.607609`，macro IoU `0.561770`，
  macro Dice `0.632656`，balanced accuracy `0.824417`，mask 内错误率 `0.092349`。
- 各类 Dice 为 NETC/SNFH/ET/RC=`0.338159/0.922263/0.575203/0.695000`；precision 为
  `0.338348/0.936579/0.548878/0.686040`，recall 为
  `0.691949/0.932666/0.821744/0.851309`。
- `1–100` 体素 ET/RC patch Dice 为 `0.167198/0.232198`，相对 exp014 的
  `0.124810/0.196575` 有局部提升；但总体 focus mIoU 比 exp014 `0.592211` 下降 `0.036464`，
  说明该配方主要重新分配错误，而非提高可分性。
- mask 外非零率 `0`、union Dice `1`；三项性能门禁失败，两项空间门禁通过。冻结 test 未运行。
- best checkpoint SHA-256 为
  `88684ad7e925641a26cba0fa22d4237f70bcaf36e30c7be3954f471cc1996451`；最终 val 审计文件
  `best_val_metrics.json` SHA-256 为
  `39ab1b7296a5d920f79f9a11f1411a8c7861a1167a6a705b0287bc82256549e9`。
- `SIGTERM` 后 W&B 一度保持 running；只恢复同一 run 补记最终 val、门禁、`test/ran=0` 和
  `user_triggered_method_early_stop`，未训练、未使用 GPU，随后已核验状态为 `finished`。

## 结论

exp015 没有产生实质收益，不能替代 exp014 best，也不能晋级 test。患者/组件等权、偏召回
Tversky 和 patch presence 能偶尔提高小区域 Dice，但没有把信息转化为患者等权总体收益；继续
训练或继续叠加同类 loss 权重不具备达到 `0.85` 的合理预期。

## 下一步

继续封存 test，不恢复 exp015。下一方法需改变有效信息或表征能力，而不是继续重加权：

1. 若保持 T1c-only，优先验证患者多 patch 联合上下文、全量 train T1c/mask 自监督预训练与
   uncertainty-aware refinement；在正式训练前先审计 train-fit 上限和小区域标签可辨识性。
2. 若允许扩展影像输入，优先引入已对齐的 T1n/T2f/T2w，与 T1c、总 mask 共同训练 p64
   多模态分割器；这是比继续修改 loss 更可能显著提高 ET/RC 可分性的路径。
3. exp013 已证明当前 mean-teacher 配方无增益，下一实验不得原样重复伪标签一致性训练。
4. 上述任一路线均使用新实验 ID、独立 worktree、val-only 选模和原五项 test 门禁，实施前需用户确认。

## 输出路径

`experiments/20260807_exp015_gli_p64_patient_component_balanced_classifier/outputs/patient_component_balanced/`
