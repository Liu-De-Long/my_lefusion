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
- 尚未启动正式训练或任何 test 评估。

## 结论

训练前工程门禁已通过；尚无 exp015 验证性能证据，不能替代 exp014 best，也不能晋级 test。

## 下一步

同步本记录后，先核验远端 HEAD/工作树、W&B online fail-closed 认证、GPU0 空闲与输出目录，
再启动唯一一个 val-only 正式 run。只按患者等权 val 选模；五项门禁全部通过前继续封存 test。

## 输出路径

`experiments/20260807_exp015_gli_p64_patient_component_balanced_classifier/outputs/patient_component_balanced/`
