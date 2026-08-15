# 实验变更记录

## 2026-08-15 — exp023 完成 Med-DDPM 数据域修复、5k 短训与 test10 失败审计

- 从 `a12db7e` 建立独立分支/工作树，仅修复旧 Med-DDPM 训练域与当前 v2 `[-1,1]` 不一致的问题；保持 mask-only、网络结构、L1 noise、EMA、NFE=250 不变，并从头初始化模型、EMA 与 optimizer。
- 当前 p64 train/val/test 数据审计为 `7772/1032/1038 patch`、`584/73/74 patient`，无 split 泄漏；新增训练、validation、test10 统一入口及不依赖 pytest 的最小回归测试，远端 `2/2` 通过。
- step 500 门禁通过；EMA validation loss 在 step 0/500/3000/5000 为 `0.7978831/0.7516929/0.2489628/0.1365685`。按完整 val loss 冻结 step 5000，checkpoint SHA-256 为 `e69be574a526c5cc46932386cbe16fd418c232cd746e7dd2e38847cf1de59bc1`。
- 固定 exp022 10 例、shared mask、seed `20260806` 与 NFE=250 生成 10 个有限 NPZ；区域外最大误差均为 0。最终指标为 `13.4748/0.3363/0.39412/0.33929`，相对旧结果四项均未改善。
- 候选拼图视觉 QA 仍见普遍高频颗粒噪声，故按预设门禁不替换论文。figure protocol 标注 `domain-fixed short-budget` 与 `paper3_modified=false`，现有 Fig. 2、PDF 和定量表保持不变。

## 2026-08-15 — exp022 完成 LeFusion_v2 Fig. 2 十例五方法重跑

- 不训练模型；从当前 p64 test split 的 `1038 patch / 74 patient` 中按 GT 元数据冻结 10 位患者各 1 个 patch，label 配额 `2/4/2/2`、role 配额 `5/5`，体积百分位近似 `0.05, 0.15, …, 0.95`。
- 使用 exp016 classifier best checkpoint 与冻结阈值 `0.964` 生成 direct/filtered 伪四类 mask；本次 10 例 filtered union 均非空，fallback 为 0。
- 只读加载旧 Fig. 2 的 RePaint EMA step 4000、Med-DDPM model-20、Pix2Pix step 20000、Latent RFlow/VAE step 20000；五个 checkpoint 与九个旧推理源码/配置均通过固定 SHA-256、权重键、形状和有限性审计。
- Filtered (v2) 使用 step 46000 best/EMA、FP32、300-step、CFG 2.0；五方法正式结果各 10 个，统一 seed root `20260806` 和同一 shared filtered-retained union。
- 新增统一选择器、旧模型隔离适配入口、legacy 输入打包、五方法评价/拼图脚本与 focused 单元测试；本地测试 `2/2`，远端脚本编译及五方法单例 preflight 通过。
- 统一适配为 v2 `[-1,1]` 后，在 shared union 上 Filtered (v2) 的 PSNR/三视图局部 SSIM/MAE/Hist-W1 为 `23.1414/0.7408/0.11079/0.07269`；区域外背景最大误差为 0。
- 生成标注版/清洁版 PNG、PDF、figure protocol、comparison manifest、checkpoint contracts、50 条 paired metrics 和五方法各 10 个 adapted NPZ；未计算或替换 FID/FSD/KID/Rad-MMD。标注版随后已按用户要求写入 `paper3`。

## 2026-08-13 — exp020 完成 direct/filtered Test 伪 Mask 条件配对重评

- 仅为 exp019 冻结的 200 例按原 crop origin 重建 exp016 多模态输入；direct/filtered sidecar 均精确覆盖 selection manifest，未访问其余 838 例。
- direct/filtered 各完成两个 100 例 shard、EMA、FP32、300-step 推理；exp010 只读复用 exp019 GT 条件输出。
- 新增 GT 四类、GT union、脑内 union 外/安全区、common filtered-retained、每类 retained 交集和 rejected copied-region 的 3D SSIM-map 与区域 PSNR 统计。
- direct union Dice/IoU 为 1；filtered union coverage 0.8090。共同 retained 区域 pooled PSNR/SSIM 为 `19.1322/0.4988`、`20.7710/0.5977`、`21.4968/0.6636`。
- 增加 GT-union 与 common-retained 两套 FID/SwAV-FSD/KID、Hist-W1 双 mask 口径、患者 bootstrap、配对差值 CSV 和固定 8 例 QA。
- 修正保活恢复时工作目录漂移与 NumPy scalar JSON 序列化；所有 GPU 任务结束后恢复双卡保活 PID `806844/806845`。
- 补充完整 `64×64×32` patch PSNR 与 3D SSIM-map：逐 patch PSNR 均值为 `26.6975/28.4252/32.1061`，患者等权 SSIM 均值为 `0.8501/0.8826/0.9154`；明确其受精确复制区域抬高，仅作辅助指标。

## 2026-08-12 — exp019 完成三种 exp010 模型的冻结 200 例配对 test

- 精确分层冻结 200/1038 个 test patch、66 个 patient，manifest SHA-256 为
  `b7c2802dd24193e950e505700eeeb374435c19eac539b6e263b470283210ebae`；三模型使用同一真实 mask、
  真实 histogram、sampling seed `20260806`、FP32 和 300-step RePaint。
- 原始 exp010、direct、filtered 各完成 100/100 双 shard，共 600 个唯一结果；没有访问未选择的
  838 个 patch，背景外变化严格为 0，所有输出和指标有限。
- filtered 的 lesion PSNR/SSIM/MAE 为 `21.8071/0.7277/0.12087`，优于 direct 的
  `21.5548/0.7053/0.12628` 和 exp010 的 `19.9992/0.6451/0.15482`。
- FID/SwAV-FSD 为 exp010 `60.4132/3.3253`、direct `49.7290/2.6669`、filtered
  `47.0280/2.4970`；Hist-W1 macro 则 direct 最低：`0.06749`。
- batch=1 正式链在 14+14 后因预算外推安全停止并隔离保留；最终采用 batch=8，端到端
  `12276` 秒（`3:24:36`），满足四小时约束。固定 8 例 QA 补齐三模型绝对误差图。

## 2026-08-08 — exp010 step 14k 完成冻结 test 50% 效果审计

- GPU0/1 两个输出互斥 worker 正常完成 `260/259`，共 519 个唯一 patch、73 个 subject；分片
  交集为 0、并集严格等于冻结 manifest，日志无 traceback/error，没有运行剩余 50% 或全量 test。
- 固定 step 14000 `best.pt/EMA`、nearest train-cluster hist、CFG 2.0、`t_T=300` 和逐例 seed；
  共 155,700 次模型调用，墙钟约 2 小时 38 分 48 秒，两卡峰值显存约 1087.12 MiB。
- 平均/中位 lesion MAE 为 `0.157524/0.152091`，零填洞基线平均为 `0.406125`；逐例改善平均
  `55.38%`，`493/519` 至少改善 20%，`502/519` 为正改善，仍有 17 例负改善。
- 519/519 输出非零且非平坦；mask 内输入严格为 0，mask 外最大误差为 0。背景严格保持来自
  inference 的 post-sample restore 合同，不能单独作为模型学会背景的证据。
- 生成 histogram 相对零填洞明显更接近请求 cluster，但相对原真实病灶，仅 NETC/SNFH/ET/RC
  `97/243、234/519、177/402、231/408` 更近；因此只确认 paired 修复有效，不确认 histogram
  或四类医学语义控制。
- 目检最好/最差病例：模型普遍能填入 mask 对齐结构，但最差例仍有亮度偏差、局部过平滑和
  纹理错配；低强度 SNFH/RC 中零填洞基线天然较强，是部分负改善的重要混杂因素。
- 新增轻量审计输出 `summary.json`、`case_metrics.csv`、`histogram_target_comparison.csv` 和
  `qa_best_worst_by_anchor.png`，位于
  `experiments/20260807_exp010_gli_single_state_noise/outputs/full_50k_2gpu/seed_20260805/`
  `test_subset_50_best_step14000/audit/`。
- 删除一次性本地/远端审计 helper、本地目检副本和远端运行时 `.hydra/`；正式 QA、NPZ、NIfTI、
  worker 日志和审计输出均保留。`summary.json` SHA-256 为
  `24e80306e11d3d58d04886ec7c2e9bbcb7b90b3806d0d7f0cb2c68d12bdeb8c1`。

## 2026-08-08 — exp010 长程训练数值失稳并冻结半量 test 入口

- 终态审计确认正式训练未正常达到 50k：step 27838 首次记录 NaN，step 28000 EMA validation
  为 NaN，随后因 early-stopping metric 非有限而异常退出；GPU0/1 已释放，没有重复训练进程。
- W&B run 虽标记 `finished`，但本地 traceback、checkpoint 和 history 一致证明是数值异常终止；
  `latest.pt` 的 model/EMA 各有 291 个含非有限值 tensor，禁止推理和 resume。
- 冻结 step 14000 `best.pt/EMA`：validation total loss `0.1264500695`，checkpoint SHA-256
  `58e939...07358`，model/EMA 参数逐 tensor 有限性审计通过。
- 按用户授权准备运行 test 确定性 50% 子集：复用历史冻结 manifest（519/1038，shard
  260/259，SHA-256 `295b20...c3698`），不重新抽样、不运行全量 test。
- 新增 GPU0/1 输出互斥推理配置；每进程 4 个 DataLoader worker，绑定 step 14k EMA、
  original-multilabel 挖空输入、nearest cluster hist、CFG 2.0 与 `t_T=300`。
- shard 0 单例可恢复 smoke 通过：300 次调用、mask 内输入严格为 0、mask 外误差为 0，随后以
  父 PID `2679/2680` 启动 GPU0/1 正式分片。首轮进度 `2/260、1/259`，两卡利用率
  `93%/92%`；当前仍是运行中，不记作完成或全量 test。

## 2026-08-08 — 授权准备 exp010 50k 双 GPU 正式训练

- 用户决定以 exp010 作为首个全量训练方法，exp012 仅保留为后续对照，本次不启动 exp012。
- 将“全量训练”冻结为完整 train split、seed `20260805`、从随机初始化运行最多 50,000 optimizer
  step；不把已 finished 的 5k 单卡 checkpoint 跨配置、Git 和 W&B 身份强行续接。
- 保持 exp010 的 full-T1c/pred-noise、global batch 4、学习率 `1e-4`、CFG dropout 0.1 与 lesion
  mask-normalized loss 不变；启用 GPU0/1 `DataParallel`，每卡 micro-batch 2，train/validation
  DataLoader worker 均为 8。
- 新增独立 Hydra/轻量配置和输出目录，W&B 正式 run ID 为
  `exp010-p64-full50k-2gpu-s20260805`，online fail-closed。
- 扩展正式 preflight，使新配置可显式执行双 GPU、零 optimizer update 的反向/validation/resume
  门禁，并记录每卡峰值显存；历史配置默认仍保持单 GPU preflight。
- 启动前远端两张 A100 80GB 均为空闲，W&B 本机凭据可用；正式训练须在双 GPU preflight、
  配置解析、focused tests 和文档同步全部通过后才启动。
- 远端 focused tests `2/2`、完整 Hydra 解析和双 GPU online preflight 均通过；preflight 未执行
  optimizer update，loss/grad norm 为 `1.069795/16.795513`，GPU0/1 反向峰值显存为
  `11855.53/11693.59 MiB`，完整 validation loss 为 `0.996759`，覆盖 1032 patch、73 subject
  和全部四类，checkpoint 恢复成功。W&B preflight run 已 finished 并同步。
- 正式训练于 `2026-08-07 19:58:48 UTC` 启动，父 PID `39061`，运行 HEAD `1392bff`；W&B run
  `exp010-p64-full50k-2gpu-s20260805` 已 online/running。启动审计到 step 649，loss/grad norm 为
  `0.280569/1.472375`，GPU0/1 显存约 `13369/13165 MiB`，两卡均实际计算。
- step 500 首个 `latest.pt` 已成功原子写入，约 554 MiB；exp012 未启动，也没有第二份 exp010
  训练进程。训练继续运行，质量结论等待 validation 与冻结 QA。

## 2026-08-08 — exp016 完成 exp010/exp012 同病例四类别配对比较

- 仅为 exp010 step 5000 EMA 补充与 exp014 完全相同的冻结 8 例 × NETC/SNFH/ET/RC，共 32 个
  结果；复用 exp014 的 exp012 输出，没有重新训练、运行 exp011、p80、其他 seed、519 例或全量 test。
- 远端 focused tests `2/2` 与 Hydra 契约解析通过；1 例 smoke 验证 300 次模型调用、洞内输入严格为
  `0`、mask 外误差 `0` 和峰值显存约 `1089 MiB`。
- 经用户授权在 GPU1 使用四个输出互斥 worker，PID 为 `30308/30312/30316/30304`；观察显存约
  `7.0 GiB`、利用率最高 100%，四类最终均为 `8/8`。首次 detached 启动因 login shell 误用基础
  Python、缺少 `blobfile` 而在采样前退出；未产生病例结果，保留失败日志后以明确 conda 解释器完成。
- 32 对原图、挖空输入、source/target mask、四通道 mask、条件 hist 与 sample seed 逐数组一致；
  exp010/exp012 的 mask 外最大误差均为 `0`。
- exp010/exp012 的 histogram Top-1 为 `18/32、21/32`，均值 rank 为 `1.625/1.34375`，平均
  margin 为 `0.100785/0.329680`；exp012 在 8/8 病例的四类平均 histogram margin 上均胜。
- GLCM-LOCO 为 `21/32` 对 `26/32`；exp012 主要改善 SNFH（`4/8 -> 8/8`），ET 两者均 `8/8`，
  RC 两者均 `4/8`。texture margin 逐对胜负为 exp010/exp012 `15/17`，病例级为 `4/4`。
- 目检确认 exp012 的亮暗/局部纹理条件响应整体更强，因此选为后续条件生成主干；exp010 继续作为
  paired 重建最佳对照。NETC/SNFH/RC 仍明显重叠，四类医学语义未验证，且两方法不只差 `λhist`，
  不能把增益单独归因于 soft-histogram loss。
- 新增 exp010 四类别配置、通用四类分析参数和严格配对比较脚本；输出包括两模型 confusion matrix、
  逐对/逐病例 margin、8 张双模型横向图与总览图，保存在 exp016 `outputs/`。

## 2026-08-08 — exp014 完成同病例四类别 counterfactual QA

- 新建 `20260808_exp014_gli_four_class_counterfactual_qa`，只扩展推理条件编排：新增
  `union_single_label_fixed`，不修改 exp012 模型、训练契约、checkpoint 或 `λhist`。
- 固定 exp012 p64、seed `20260805`、step `5000` EMA、原 8 例 validation manifest、
  `sampling.seed=20260807` 和 `t_T=300`；每个病例分别请求 NETC/SNFH/ET/RC，共生成 32 个结果。
- 首次 1 例 smoke 在生成前因直接脚本入口漏导入 helper 退出，未写入病例进度；修复 commit 为
  `1b7a946f88b936c33a417696fb7096a25d281238`，补充直接入口测试后远端 focused tests `2/2` 通过。
- 经用户明确授权，使用 GPU1 四个输出互斥 worker 加速；父 PID 为
  `23965/24388/24390/24392`，NETC 从成功的 1 例 smoke 恢复，其余三类各自独立运行，最终均为 `8/8`。
  峰值显存按单 worker 约 `1091 MiB`，并行观察值约 `7.0 GiB`，GPU1 利用率达到 100%；GPU0 未占用。
- 同病例合同审计通过：原图、挖空输入、source union、sample seed 在四类间逐数组一致；每次只激活
  对应 mask 通道和 16-bin hist block；mask 外最大误差为 `0`。
- 四类 histogram 目标 Top-1 为 `21/32`、均值 rank `1.34375`、平均 margin `0.32968`；
  NETC/SNFH/ET/RC 分别为 `3/8、2/8、8/8、8/8`。强度归一化 3D GLCM texture LOCO 为
  `26/32`，各类为 `6/8、8/8、8/8、4/8`。
- 目检只稳定确认 ET 高亮及类别条件引起的亮暗/纹理变化；NETC、SNFH、RC 仍大量重叠，形态受
  原病例背景和 union 形状主导。因此 exp012 仍只是 histogram 响应候选，四类视觉/医学语义控制未通过。
- 输出保存在 `experiments/20260808_exp014_gli_four_class_counterfactual_qa/outputs/`，包括四类各 8 个
  NPZ/NIfTI/五联图、`summary.json`、`case_metrics.csv`、`texture_features.csv`、8 张逐病例图和总览图。
- 同步使用仅包含新增 commit 的一次性增量 Git bundle；远端原有 `.hydra/` 先保存为
  `stash@{0}: pre-exp014-existing-hydra-20260808`，所有临时 bundle 均已从本地和服务器删除。
- 删除 exp014 smoke/配置解析在远端仓库根重新生成的 `.hydra/`；它只是临时 Hydra 运行产物，
  正式配置、合同、日志和 QA 输出均保留，旧 `.hydra/` 备份仍在上述 stash 中可恢复。
- 本次没有训练、W&B 新 run、p80、其他 seed、519 例、test split 或全量 test；下一步先验证真实
  held-out T1c 四类的无标签泄漏可分性上限，不立即调整 `λhist`。

## 2026-08-08 — 修订 exp010–exp012 histogram 与 union 指标解释

- 更新 `docs/20260807_001_gli_short_method_comparison.md`，补充 own-vs-swapped histogram
  L1、margin 公式、`0–2` 距离范围和通过条件，明确 union 8/8 只表示 histogram 配对正确。
- 记录 exp010/exp012 paired MAE 的绝对差 `0.025152`：以 exp010 为参照，exp012 高
  `18.2%`；exp010 在 8/8 病例上均获得更低 MAE。
- 重新解释 union QA 目检：不同病例指定同一类别后没有明确统一外观；当前设计把类别与病例、
  解剖背景、mask 形状混杂，不能证明四通道学出了视觉可辨的医学亚型。
- 将 exp012 定位从“最佳可控伪病灶候选”收窄为“最强 histogram 条件响应候选”；exp010
  继续作为 paired 重建最佳方法。下一步先做同一病例、同一 mask/噪声下的四类别
  counterfactual QA，再考虑 `λhist` 消融。
- 修正 exp012/exp013 配置元数据：取消 `selected_best/selected_method`，明确
  `semantic_subtype_control_validated=false`、`final_model_selected=false`；当前没有已完成
  语义控制验证的最终模型。
- 同步修订 `README.md`、`STATUS.md`、exp010–exp013 `result.md` 与实验地图；本次没有运行
  训练、QA、p80、其他 seed、519 例或全量 test。

## 2026-08-07 — 建立 exp013 横向比较实验卡与长期说明文档

- 新建 `20260807_exp013_gli_short_method_comparison`，将 exp010–exp012 的统一比较作为独立的
  跨实验分析归档，不将比较结果附属于任一训练方法。
- 将 `case_metrics.csv`、`summary.json` 和三方法横向 contact sheet 从仓库根 `results/`
  迁移到 exp013 的 `outputs/`；文件内容和三种方法原始 QA 输出不变。
- 新增 `docs/20260807_001_gli_short_method_comparison.md`，完整记录共同训练契约、三方法差异、
  五种固定 8 例 QA、四项门槛、定量结果、视觉结论、局限和精确 QA 图目录。
- exp012 仍是当前最佳可控候选，exp010 仍是 paired 重建对照；未运行任何新增训练、p80、
  其他 seed、519 例或全量 test。

## 2026-08-07 — exp010–exp012 完成 5k、冻结 8 例 QA 与统一比较

- 三种方法均固定 seed `20260805`、仅用 GPU1 完成 `5000` optimizer step；W&B run
  `exp010-p64-s20260805`、`exp011-p64-s20260805`、`exp012-p64-s20260805` 均 finished，
  `milestone-500/1000/2500/5000.pt`、`latest.pt` 与最终 validation 齐全。
- step 5000 EMA validation total loss 分别为 `0.135519 / 0.136704 / 0.144356`；exp012 的
  base/hist loss 为 `0.111711 / 0.326442`，hist 权重为 `0.1`。运行代码固定为
  `d0c6aaa056b38e34e4cb929f85106cc90af05426`。
- 每种方法只运行同一冻结 manifest 的 8 个 validation patch 与五种既定 QA 变体；15 个变体
  均为 `8/8`，NPZ、metrics、contract 与 contact sheet 完整。未运行 p80、其他 seed、519 例、
  test split 或任何全量 test。
- exp010 五变体先串行完成。用户随后授权为提高 GPU1 利用率，将剩余输出互斥的 QA 切换为
  两个 worker：父 PID/PGID `31788` 与 `31790`；日志为
  `experiments/20260807_exp010_gli_single_state_noise/outputs/short_qa_parallel_worker_a.log` 和
  `short_qa_parallel_worker_b.log`。未启动第三个 worker，未重复已完成变体，两个进程均正常结束。
- 仅运行一次 `scripts/gli_compare_short_generation_experiments.py`。统一结果写入
  `experiments/20260807_exp013_gli_short_method_comparison/outputs/`，包括 `case_metrics.csv`、`summary.json` 和
  `three_method_original_real_contact_sheet.png`。
- 四项门槛结果：exp010 为 lesion/hist/union `8/7/7`、背景误差 `0`，全部通过；exp011 为
  `7/7/5`、背景误差 `0`，因 union 低于 `6/8` 失败；exp012 为 `8/8/8`、背景误差 `0`，
  全部通过。原始真实 hist 模式平均 lesion MAE 为 `0.138066 / 0.164752 / 0.163218`，零填洞
  基线为 `0.323883`。
- 横向原始模式与六张 union first/last contact sheet 均已目检：三者均在 mask 内生成非零、
  非平坦结构且洞外不变；exp012 的 hist 切换响应最明显，exp010 paired 重建最好，exp011
  union 响应不稳定。exp012 个别样本仍有偏黑/偏亮团块，不能外推为医学有效。
- 本地目检临时拉取的六张 union contact sheet 在结论记录后删除；远端各实验正式 QA 原图未删除。
- exp012 选为当前最佳可控伪病灶候选，exp010 保留为 paired 重建次优对照，exp011 保留为
  union 门槛失败对照。三份 `result.md`、`STATUS.md`、实验地图与 config 状态已同步更新。

## 2026-08-07 — exp008 完成训练并仅执行两组 8 例 QA

- exp008 p64 seed `20260805` 正常完成 50,000 step；最终 EMA validation total loss 为
  `0.091035`，W&B run `exp008-p64-s20260805` 已同步完成。
- 正式 `best.pt` 为 step `48000`、EMA，SHA-256 为
  `660e23533064eaf9f32310b42500a8067c476dbd2bddbfd88a88dbc0f2c64857`。
- 新增 `masked_multilabel` 与 `masked_anchor_union` 两份 QA 配置，固定复用同一 8 例 val
  manifest 和完整 `t_T=300`；focused 配置测试通过，配置 commit 为 `16d0a4e`。
- 两组均完成 `8/8`、每组 2,400 次模型调用；没有运行 519 例、test split 或全量 test。
- 工程门禁全部通过：挖空值严格为 0、mask/hist 契约正确、数组有限、显存稳定、逐例
  NPZ/NIfTI/五联图和两张总览图齐全。
- 删除 focused Hydra 配置解析生成的仓库根 `.hydra/config.yaml`、`hydra.yaml` 和
  `overrides.yaml`；它们只是一次性解析副产物，正式 QA 配置和输出均已保留。
- 质量验收失败：病灶区生成绝对强度均值仅 `0.00834/0.00949`，原始病灶为 `0.32388`；
  histogram L1 为 `1.49–1.98`，目检显示输出基本退化到挖空后的均匀灰值。
- exp008 降级为失败审计，不作为有效伪病灶 checkpoint；下一步只允许先做少量 timestep/
  `x0` 重建诊断，未经授权不重新训练或扩大测试规模。

## 2026-08-06 — 记录 p64 少标签逐体素亚区分类方案

- 新增长期方案 `docs/20260806_001_gli_p64_weak_supervision_classifier_plan.md`。
- 将“10% 且约 1000 Case”解释为从冻结 train pool 中确定性选择 1000 个 p64 patch，
  每个 anchor 类 250，并在类别内平衡 interior/boundary 和患者贡献；同时保留严格 984 个
  patch 作为实施前可选口径。
- 首个推荐模型改为无标签泄漏的空间特征 MLP，并保留单强度 MLP、局部邻域 MLP 和轻量
  3D CNN 对照；不因模型较小而假设单体素 T1c 足以区分四类。
- 暂定 ET 与 RC 为 T1c 下两个重点亚型，成功门禁为患者等权 focus mIoU 不低于 85%，
  同时继续报告四类完整指标和小区域表现。
- 明确 NPZ histogram、四通道 lesion mask、anchor label 和 per-class voxel count 不得进入
  模型输入；剩余未标签 train patch 首阶段不参与优化。
- 本次只修改中文方案与状态文档，没有修改代码、创建实验卡、登录 W&B、启动训练或占用 GPU。

## 2026-08-06 — 停止错误训练契约并启动 exp008 条件式修复方法

- QA 确认 exp005 的 denoiser 没有接收挖空 T1c 和四通道 lesion mask 空间条件；旧 p64/p80
  checkpoint 不满足伪病灶生成目标，exp006/exp007 结果同步降级为失败审计。
- 按用户要求停止 p80 主进程与两个数据加载子进程；最后完整 checkpoint 为 step `11500`，
  GPU 0/1 已释放至 `1 MiB`、`0%`，未删除任何 checkpoint。
- 新建方法实验 `20260806_exp008_gli_conditional_inpainting_training` 和分支
  `feature/20260806-exp008-gli-conditional-inpainting`。
- 数据 loader 新增病灶 union 内置零、洞外保持原值的单通道 `masked_context`；denoiser 空间输入
  改为四通道 `x_t`、一通道 masked T1c 与四通道 lesion mask，输出仍为四通道噪声预测。
- 训练、validation、preflight 与 RePaint 共用该空间条件，hist 保持四组 16-bin、loss 保持
  仅在对应病灶 mask 内计算。
- 实现 commit：`c117bc74519c4329ff721817d31b3e2171be90b2`。
- 兼容与 preflight 门禁修复后的运行版本：`6e29f355294c762cfb8928f73abcb350200c2e70`。
- 远端真实 patch 全套回归 `34/34` 通过；online preflight 完成 0 次 optimizer update 的真实
  batch 前/反向、完整 validation 和 checkpoint resume，反向峰值显存 `23344.58 MiB`。
- preflight W&B run：`exp008-p64-preflight-s20260805`；临时 checkpoint 已自动删除。
- 首次 preflight 被旧 `exp005-` ID 硬编码在计算前拦截，已泛化门禁并补回归测试。
- `2026-08-06 08:28:25 UTC` 启动且只启动 exp008 p64 seed `20260805` 正式训练；W&B run
  `exp008-p64-s20260805` 已 online，同步流已记录约 164 个 optimizer step 和有限 loss。
- 启动核验时两张 GPU 均参与计算，显存约 `13383/13161 MiB`；未启动 p80、其他 seed 或 test。
- 首个约 `580.4 MB` 的 `latest.pt` 已按 500-step 规则写入，写入后双 GPU 训练继续正常。
- 清理旧 exp007 隔离 worktree 中仅存的 `.hydra/config.yaml`、`hydra.yaml`、`overrides.yaml`
  临时解析产物，并注销 `/workspace/LeFusion_v2/.exp007_qa_worktree`；该 worktree 无源码改动，
  旧实验输出与 checkpoint 均未删除。
- 已删除本地与远端一次性 `.tmp_inspect_checkpoint.py`；它仅用于只读记录停止点，不是实验资产。

## 实验 ID

20260804_init_docs_v1

## 日期

2026-08-04

## 目标

初始化 BraTS2024 GLI 迁移实验工作区的项目管理文档。

## 方法

- 检查复制后的 `my_experiment` 仓库结构。
- 检查已有 README。
- 确认当前训练和推理入口。
- 检查数据集工厂支持情况。
- 检查 EMIDEC 训练、推理、数据集和配置文件，将其作为迁移参照。
- 创建 `README.md`、`STATUS.md` 和 `CHANGELOG.md`。

## 结果

已创建项目管理入口文档。本次实验未运行模型训练，也未执行 GLI 迁移代码。

## 结论

当前工作区已经具备可追踪实验管理入口。下一项技术任务是实现并注册 GLI 数据集，同时定义模态、标签、通道和直方图约定。

---

## 实验 ID

20260804_gli_data_audit_v1

## 日期

2026-08-04

## 目标

检查 BraTS2024 GLI 数据集是否已经存在、目录结构是否可用于迁移、数据规模和样本格式是否适配 LeFusion 后续数据集实现。

## 方法

- 使用 `ssh-remote-python-workspace` 在远端只读检查 `/workspace/LeFusion-main/BraTS-2024-Complete/BraTS-GLI`。
- 统计顶层结构、数据大小、病例数量、NIfTI 文件数量和各模态数量。
- 抽样读取 NIfTI header，确认尺寸、spacing 和数据类型。
- 抽样读取训练分割标签唯一取值。
- 使用一次性只读检查脚本和远端 shell 命令完成检查，并将详细结果记录到 `docs/20260804_001_brats2024_gli_data_audit.md`。
- 检查结论记录完成后，删除不会复用的临时脚本，避免干扰项目结构。

## 结果

- 数据集总大小约 `182G`，总文件数 `8859`，其中 NIfTI 文件 `8857`。
- `train` 有 `1621` 个病例目录，每例包含 `seg/t1c/t1n/t2f/t2w`。
- `val` 有 `188` 个病例目录，每例包含 `t1c/t1n/t2f/t2w`，未发现 `seg`。
- 抽样尺寸为 `182 x 218 x 182`，spacing 为 `1.0 x 1.0 x 1.0`。
- 抽样标签取值包含 `0,1,2,3,4`，单个病例不一定包含所有病灶类别。
- 未发现 `.npy`、`.npz`、`.pkl`、`.pt`、`.pth`、`.h5`、`.hdf5` 等 LeFusion 预处理缓存格式。

## 结论

该数据目录是 BraTS 发布格式的 NIfTI 数据，不是 LeFusion 已处理缓存。后续迁移应基于 `train` split 实现 GLI dataloader，并在配置中固定模态顺序为 `t1c, t1n, t2f, t2w`。在实现前还需要确认标签 `1,2,3,4` 的官方语义，并决定 GLI 标签到 LeFusion 多通道病灶表示的映射方式。

## 下一步

确认 GLI 标签语义和通道映射，然后实现 `LeFusion/dataset` 下的 GLI 数据集类与 `get_dataset.py` 注册逻辑。

---

## 实验 ID

20260804_gli_patch_stats_preprocess_v1

## 日期

2026-08-04

## 目标

统计 BraTS2024 GLI/PTG 训练集病灶区域大小和模态病灶强度分布，为 LeFusion 数据接入选择 patch 尺寸、裁剪方式、缩放方式和 histogram 条件设计。

## 方法

- 新增正式可复用脚本 `scripts/brats_gli_lesion_patch_stats.py`。
- 对 `train` split 的 `1621` 个病例做全量 mask/bbox 统计，输出到 `experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/region_patch_stats_full`。
- 对前 `100` 个病例做四模态病灶强度抽样统计，输出到 `experiments/20260804_exp000_brats_gli_data_preprocess_audit/outputs/modality_intensity_stats_100cases`。
- 按 BraTS post-treatment glioma 语义暂定标签：`1=NETC`、`2=SNFH`、`3=ET`、`4=RC`。
- 将 LeFusion 论文中的 lesion-focused loss、histogram texture control 和 multi-channel decomposition 约束整理为预处理方案。

## 结果

- `WT=1+2+3` 的 p90 bbox 为 `95 x 120 x 91`，p95 bbox 为 `103 x 129 x 98`。
- `TC=1+3` 的 p90 bbox 为 `59 x 74 x 64`，p95 bbox 为 `68.6 x 86 x 73.6`。
- `ET=3` 的 p90 bbox 为 `59 x 74 x 64`，p95 bbox 为 `68 x 86 x 73`。
- 包含 `RC` 的整体异常区域 p95 patch 建议为 `112 x 144 x 104`。
- 100 例模态强度抽样显示四个 MRI 模态强度尺度差异明显，不能直接共享一个全局 intensity range。
- 详细方案记录到 `docs/20260804_002_brats2024_gli_patch_preprocessing_plan.md`。

## 结论

GLI 第一版接入建议使用 3D mask-centered crop，不做空间重采样；主配置采用 `patch_size=[112,144,104]` 覆盖 `WT` p95。强度预处理应按病例、按模态对非零脑区做 clip 和归一化。LeFusion v1 建议先使用 `WT` 单通道和四模态 histogram 条件，`cond_dim=64`，通过小规模 overfit 验证流程后再扩展到 `SNFH/TC/RC` 或 `NETC/SNFH/ET/RC` 多通道分解。

## 下一步

复核官方标签语义后，实现 GLI dataset v1，并在 dataloader 中记录 patch 截断比例和 histogram 条件维度。

---

## 实验 ID

20260804_exp001_t1c_local_patch_dataset

## 日期

2026-08-04

## 目标

构建 BraTS 2024 GLI 的 T1c 单模态局部病灶 patch 数据集，为 NETC、SNFH、ET、RC 四类病灶分别提供内部纹理和边界纹理样本。

## 方法

- 新增可复用的确定性 patch 裁剪器 `scripts/brats_gli_crop_local_patches.py`。
- 使用 `train` split 的 1621 例有标签病例，按患者组做 80/20 划分。
- 固定输入模态为 `t1c`，保留标量 segmentation，后续 loader 再展开为 NETC/SNFH/ET/RC 四通道 lesion mask。
- 生成 `64×64×32` 与 `80×96×80` 两种 patch，每种尺寸共享同一份患者划分。
- 每个存在的 anchor label 生成 interior 和 boundary 两类样本。
- 按非零 foreground 做 `p0.5-p99.5` clip 并归一化到 `[-1,1]`。
- 每个标签计算 16-bin histogram，保存为四标签 histogram 条件。
- 使用 staging 目录完成构建、QA 和逐 NPZ 完整性验证后原子发布。

## 结果

- 输出目录：`/workspace/LeFusion_v2/dataset/brats2024_gli_t1c_local_patches`。
- 输入病例 1621 例，归组为 731 个患者；train 584，val 147，无患者泄漏。
- `64×64×32`：9842 个 patch，padding patch 201（2.04%）。
- `80×96×80`：9842 个 patch，padding patch 1545（15.70%）。
- 两种尺寸合计 19684 个 NPZ；NPZ 逻辑压缩容量约 16.0 GiB，JuiceFS `du` 分配量约 65 GiB。
- 每种尺寸 anchor 数一致：NETC 1412、SNFH 3236、ET 2446、RC 2748。
- 每种尺寸 interior/boundary 各 4921 个样本。
- 失败病例 0，归一化退化病例 0。
- 每种尺寸生成 16 张 QA 图，最终逐 NPZ 验证通过，无临时文件或 staging 残留。

## 结论

数据集已完成并通过发布门禁。当前有效路线从旧的 `WT` 单通道四模态方案调整为 T1c 单模态图像加 NETC/SNFH/ET/RC 四通道 lesion mask 的局部 patch 方案；两种 patch 尺寸可直接用于后续 patch size 对照和 GLI loader 接入。

## 下一步

实现 GLI loader：读取单份 T1c 与标量 segmentation，在加载阶段展开 NETC/SNFH/ET/RC 四个 lesion channel，并验证 `[X,Y,Z]` 到 `[C,D,H,W]` 的轴转换与 histogram 条件拼接。

---

## 实验 ID

20260805_exp002_gli_loader_t1c_multilesion

## 日期

2026-08-05

## 目标

实现并注册 BraTS2024 GLI T1c 局部 patch loader，参考原始 EMIDEC 的 scalar label 语义，同时暴露四通道 lesion mask 和 64 维 histogram 条件。

## 方法

- 在 `LeFusion/dataset/gli_hist.py` 实现可复用 loader。
- 保留 `label` 为 scalar segmentation，新增 `lesion_mask` 为 NETC/SNFH/ET/RC 四通道二值 mask。
- 将 NPZ 的 `[X,Y,Z]` 按 `[channel,z,x,y]` 转换为 LeFusion 的 `[C,D,H,W]`。
- 在 `get_dataset.py` 注册 `gli`，新增 focused tests 和实验配置。
- 本轮不启动训练。

## 结果

代码已提交到 `feature/20260805-exp002-gli-loader`，commit 为 `fc7d7e1`。本地语法检查通过；远端 focused tests 4/4 通过，包含真实发布数据两种 patch 尺寸迭代；全量测试 10 项通过、1 项因本地未设置真实数据路径跳过。

## 结论

GLI loader 的 scalar label 与四通道 lesion mask 语义已分离，避免将原始 EMIDEC 的前景选择逻辑错误套用到 GLI。当前训练 loss 和推理入口仍未改造。

## 下一步

后续将 GLI loss、Trainer 传递和矩形 patch 支持作为独立方法实验；本次未启动训练。

---

## 文档整理

## 日期

2026-08-05

## 目标

将已实施的 GLI loader 设计、scalar label 兼容语义、四通道 lesion mask、轴转换、padding metadata 和验证标准整理为可复用方案文档。

## 结果

新增：`docs/20260805_001_gli_loader_implementation_plan.md`。

## 结论

方案文档已与 `20260805_exp002_gli_loader_t1c_multilesion` 实验记录关联，后续 GLI loss、训练和推理接入可按该文档继续拆分实验。

---

## 文档整理

## 日期

2026-08-05

## 目标

将 GLI lesion-aware 训练接入、四通道 loss、DHW shape 约定和验收门禁整理为跨实验可复用的长期方案。

## 结果

新增：`docs/20260805_002_gli_lesion_aware_training_integration_plan.md`。

文档覆盖训练入口、Trainer 数据传递、非空 `(sample, lesion channel)` 等权 loss、64×64×32 与 80×96×80 shape 约束、配置隔离和 smoke 验收标准。

## 结论

该方案文档作为后续 GLI 训练和矩形 patch 实验的共同接口说明；单次运行结果仍保留在各实验目录的 `result.md` 中。

## 下一步

后续 GLI inference loader、RePaint keep-mask 和输出合并继续使用独立方法实验管理。

---

## 实验 ID

20260805_exp003_gli_lesion_aware_training

## 日期

2026-08-05

## 目标

实现 GLI 四通道 lesion-aware loss、Trainer 的 `lesion_mask` 传递和完整三维空间 shape 支持，并验证 `64×64×32` 与 `80×96×80` 两种 patch 的训练接入可行性。

## 方法

- 训练入口接受 `gli`，解耦 `base_dim` 与 `spatial_shape_dhw`。
- GLI loss 对所有非空 `(sample, lesion channel)` 单元等权平均，每个单元先按自身病灶 voxel 数归一化。
- histogram 使用当前训练 device，不再写死 `.cuda()`。
- 参数化 temporal relative-position `max_distance`，GLI 配置统一使用 `128`。
- 增加两份 Hydra 配置、可复用 smoke 工具和 loss/shape/Hydra 集成测试。
- 推理入口改用 diffusion 的完整 sample shape，但没有开放 GLI RePaint 推理。

## 结果

- 代码 commit：`dbeabd5e1d9f6f35fc0348031c800043e554585f`。
- 新增集成测试 5/5 通过；项目全量测试 16 项通过；真实发布数据 loader 4/4 通过。
- 两份 Hydra 配置均通过 `--cfg job --resolve`，无未解析字段。
- `64×64×32`：真实输入 `[1,4,32,64,64]`，固定 batch 20 步首 5 步 loss 均值 `0.785975`、末 5 步 `0.360158`，峰值显存 `6309.063 MiB`。
- `80×96×80`：真实输入 `[1,4,80,80,96]`，单步 loss `0.852194`，前向/反向通过，峰值显存 `33944.751 MiB`，未 OOM。
- W&B 64 patch run：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/23hogegm>
- W&B 矩形 patch run：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/228g4h4j>
- 初次在线启动因远端没有已配置凭据而在模型计算前失败；随后以 offline 模式完成 smoke，并在用户提供临时环境凭据后同步成功。该凭据未写入项目文件、配置或 Git。
- 同步完成后删除 `experiments/20260805_exp003_gli_lesion_aware_training/outputs/` 下的 W&B offline 临时缓存；未生成 checkpoint，因此整个临时 outputs 目录可安全删除。

## 结论

`64×64×32` 已通过固定 batch overfit smoke；`80×96×80` 已通过真实单 batch 前向和反向，当前 A100 80GB 上 batch size 1 的峰值显存约 `33.15 GiB`。两种尺寸的训练接口均可行，但正式训练仍未启动，当前结果不属于模型性能结论。

## 下一步

先复核 GLI 标签医学语义，再确认正式训练 batch size、gradient accumulation、验证策略和在线 W&B 环境；GLI RePaint inference 作为后续独立方法实验实施。

---

## 实验 ID

20260805_exp004_gli_inference_closed_loop

## 日期

2026-08-05

## 目标

在不启动正式训练的前提下，完成 GLI patch 级 validation checkpoint、train-only histogram cluster、逐 timestep RePaint、四通道病灶合成、单通道保存和 normalization QA 闭环。

## 方法

- 保留原 584 个 train subject，将原 147 个 holdout subject 确定性拆分为 73 val 和 74 test。
- scalar segmentation 作为事实源，确定性展开四通道 lesion mask；逐 batch 校验标签映射和互斥性。
- 四个 lesion channel 在每个反向转换中共享同一条固定 noise 的前向扩散 T1c 背景轨迹；终态背景直接取 sampler 的共享状态，不执行采样结束后的原图硬覆盖。
- 在 train patch 上按 label 独立聚类 16-bin histogram，并按 `(subject,label)` 归一权重；inference 选择最近 cluster center。
- 从四模态原始体积构建显式 support，仅用于 QA、分区指标和 lesion 合法性，不参与 RePaint。
- 分别生成 64 patch 20-step 和矩形 patch 1-step validation checkpoint，并在 test split 上执行缩减 `t_T=5` inference smoke。

## 结果

- 实现 commit：`bcf9097a6ccbd20db6ab6d992ae72872c5ea65dd`；分支：`feature/20260805-exp004-gli-inference`。
- 全量测试 22/22 通过；四份 Hydra 配置解析通过且无未解析字段。
- split 为 `584/73/74` 且零交集；两种 patch 的 NETC/SNFH/ET/RC cluster 均选择 `k=[5,3,2,2]`。
- 64-subject normalization 审计四项门禁全部通过：median Dice `0.999997`、p05 `0.990682`、median extra `0`、p95 extra `0.2444%`。最差和随机 montage 的视觉检查未见系统性颅外伪影，保留当前 normalization。
- 64 validation checkpoint：20-step overfit 首 5 步 loss `0.785975`、末 5 步 `0.360158`，峰值显存 `6309.063 MiB`。
- 矩形 validation checkpoint：1-step loss `0.852194`，峰值显存 `33944.751 MiB`。
- 64 inference：5 次模型调用、`2.6067 s`、`1083.212 MiB`；矩形 inference：5 次模型调用、`4.6900 s`、`6518.637 MiB`。两者四通道/单通道 shape、XYZ/DHW、affine 和 NIfTI round-trip 均通过。
- 首次 online W&B 初始化因远端没有环境凭据而在模型计算前失败；随后以 offline 模式完成，run ID 为 `30rwobgb` 和 `30wigck6`，暂无线上的 run URL。

## 清理

- 删除 Hydra 配置解析在远端仓库根生成的临时 `.hydra/config.yaml`、`.hydra/hydra.yaml` 和 `.hydra/overrides.yaml`；这些文件仅是解析副产物，不是实验资产。
- 未删除 validation checkpoint、W&B offline run、cluster、normalization QA 和 inference 输出；它们仍是当前闭环的有效远端资产，并由 `.gitignore` 排除。

## 结论

GLI patch 级训练 checkpoint→cluster condition→RePaint→四通道合成→单通道保存闭环已打通，64 与矩形 patch 均可行。当前 checkpoint 和缩减 schedule 只证明接口正确，不能作为正式模型质量结论；正式训练仍需官方标签语义与 W&B 在线记录。

## 下一步

复核标签医学语义并配置新的远端 W&B 凭据；只有在用户另行确认正式训练配置后，才启动长期训练和完整 RePaint 质量评估。

---

## 文档整理

## 日期

2026-08-05

## 目标

记录 BraTS2024 GLI 正式训练前的标签确认、normalization、类别平衡、训练参数、validation/test 隔离、checkpoint/resume、W&B 安全接入和项目管理决策。

## 方法

- 使用 BraTS 官方评测说明确认 `0=background、1=NETC、2=SNFH、3=ET、4=RC`。
- 核对 patch、四通道 mask、histogram、loss、cluster 和 inference channel 顺序。
- 复核 exp004 normalization support audit，决定保留当前 normalization。
- 对两种 patch 的 train/val/test 做 anchor、lesion、subject、role 和多标签分布统计。
- 比较自然采样、anchor 平衡、`(subject,label)` 平衡和分层 sampler，推荐分层 sampler baseline。
- 设计两阶段正式训练参数、validation、checkpoint、resume、early stopping、三 seed 复现和 W&B online 安全流程。

## 结果

- 新增长期方案：`docs/20260805_004_gli_formal_training_plan.md`。
- 标签语义证据充分，不再是 blocker。
- 保留当前 `T1c != 0 -> p0.5/p99.5 -> clip -> [-1,1]` normalization。
- 推荐先训练 `64×64×32`，再将 `80×96×80` 作为第二阶段对照。
- 推荐新实验 ID：`20260805_exp005_gli_formal_training_baseline`。
- 推荐分层 sampler；保持现有 loss 不变，但该训练流程变化需要新 branch 和 Git commit。
- 用户已说明在 F 盘准备新的 W&B key；本次未读取、未显示、未写入项目，也未登录或创建 run。

## 结论

当前仍不能直接启动正式训练。正式训练前必须先实现 sampler、validation、完整 checkpoint/resume、W&B online fail-closed 和 preflight，并统一 normalization audit provenance。

## 下一步

等待用户确认方案文档中列出的 experiment ID、branch、sampler、训练参数、复现次数和 W&B entity；确认后先实施代码与测试，不自动启动训练。

---

## 文档整理

## 日期

2026-08-05

## 目标

保留并同步扩写后的 GLI patch 级 inference 闭环长期接口文档。

## 结果

- 扩写 `docs/20260805_003_gli_inference_closed_loop.md`，补全 test-only loader、显式 brain support、histogram cluster provenance、逐 timestep RePaint、四通道输出合成、NPZ/NIfTI 保存回读和分阶段验收方案。
- 将文档中旧的标签待确认描述更新为已经官方确认的 `0=background、1=NETC、2=SNFH、3=ET、4=RC` 契约。
- 本次只整理并同步文档，不修改 inference 代码、不登录 W&B、不创建 run、不启动训练。

## 结论

该文档作为 exp005 正式 checkpoint 验证及后续生成质量评估的长期接口依据继续保留。

---

## 实验 ID

`20260805_exp005_gli_formal_training_baseline`

## 日期

2026-08-05

## 目标

在不登录 W&B、不创建 run、不启动训练的前提下，实现 GLI 正式训练所需的采样、验证、恢复和在线日志门禁。

## 方法

- 实现按 `anchor_label × sample_role` 分层、层内按 subject 平衡的确定性 sampler，保持现有 lesion-aware loss 不变。
- 新增固定 timestep/noise 的监督 val，记录 EMA 总 loss、NETC/SNFH/ET/RC loss、各通道和总有效单元数、patch/subject/anchor label 覆盖。
- 实现 early stopping、原子 `latest.pt`、`best.pt`、最近三个 milestone 和完整 checkpoint/resume。
- W&B 使用固定 entity/project/run ID 与 `resume=never/must`，online 初始化失败时 fail closed，不读取或保存 key。
- 新增两种 patch 的正式 Hydra 配置；参数保持可配置，50,000 step 作为上限而非必须跑满。

## 结果

- 实现代码版本：`7a288dc2f59947db2c8bf00200a59f19f6e0bc7a`。
- focused tests 首轮通过；全量测试曾发现旧 `SimpleNamespace` factory 兼容问题，已在 `cd2affb` 修复。
- 直接训练脚本配置解析曾发现 `train.tracking` 包解析问题，已在 `4ee7ebf` 修复。
- 最终远端全量非训练测试 28/28 通过；两份正式配置解析通过且无 `???`。
- exp004 normalization audit stale 字段已改为人工 QA 完成并保留当前 normalization，JSON 回读通过。
- 本轮没有登录 W&B、创建 run、执行 GPU preflight 或启动训练。

## 结论

exp005 已具备进入独立 preflight 的代码基础，但仍不允许正式全量训练。micro-batch 可根据显存调整并用 accumulation 保持 effective batch；三个 seed 是本项目复现候选，不是原始 LeFusion 的强制规则；训练可在 validation 收敛时提前停止。

## 下一步

用户安全配置远端 W&B key 后，另行授权 W&B online、显存、validation 和 resume preflight；preflight 通过后仍需再次确认才能启动首个正式 run。

---

## 实验 ID

`20260805_exp005_gli_formal_training_baseline`

## 日期

2026-08-05

## 目标

在不启动正式训练的前提下，执行已授权的 W&B online、64 patch 显存、完整 validation 和 checkpoint resume preflight。

## 方法

- 新增可复用 `scripts/gli_formal_training_preflight.py`，使用独立 preflight W&B run ID。
- 对 `64×64×32` 的 `batch=4/accum=1` 仅执行一次 fixed-batch forward/backward，不调用 optimizer/scaler step；随后运行完整固定 validation、保存/重载临时 checkpoint 并验证 resume。
- 成功后自动删除临时 `resume_preflight.pt`，保留轻量 metrics 和日志；不生成正式 checkpoint。

## 结果

- 新 W&B key 已在远端安全可用，成功 run：<https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-preflight-s20260805-r3>。
- 0 次 optimizer update；batch `[4,4,32,64,64]`；反向峰值显存 `23330.56 MiB`；完整 validation 峰值 `5482.45 MiB`。
- validation：1032 patch、73 subject、4 label、3207 有效单元；EMA total loss `0.959372`；NETC/SNFH/ET/RC loss `0.893387/0.904111/1.140945/0.887598`。
- checkpoint resume 恢复成功；临时 checkpoint 已删除。
- 首次 preflight 因本地 SSH 会话传输中断而留下未完成 online run；第二次 retry 暴露合法负 `origin_xyz` 被 loader 拒绝。已修复该 bug（`b34d867f944343f9f6ff6e4edde5abe3a0b0805b`），修复后真实 patch 根目录下远端全量测试 28/28 通过。

## 结论

64 patch 的所有技术门禁已通过。该 preflight 不构成正式训练；正式 50,000-step seed `20260805` 仍需用户单独授权。

## 下一步

等待用户决定是否启动首个正式 `64×64×32` run；不自动启动另外两个 seed 或 80 patch 对照。

---

## 正式训练授权

## 日期

2026-08-05

## 结果

- 用户授权仅启动 `64×64×32` 的正式 seed `20260805`，明确禁止自动扩展至其他 seed 或 `80×96×80` 对照。
- 远端确认 GPU 0、1 均为空闲 A100 80GB；正式配置改为 DataParallel，保持全局 batch 4、每卡 micro-batch 2、accumulation 1、effective batch 4。
- 运行代码版本固定为 `75b187ce6664d4fa44ddad32dd328112998ff5c4`。
## 2026-08-05 — 启动 exp005 首个正式双 GPU 训练

- 经用户授权，已启动且仅启动 `20260805_exp005_gli_formal_training_baseline` 的 `64×64×32`、seed `20260805` 正式训练；未启动其他 seed 或 `80×96×80` 对照。
- 运行使用 GPU 0、1 的 `DataParallel`，全局 batch 为 4（每卡 2），运行时 Git HEAD 为 `ea88464f3bf51350f5bd7d33f1bfcc4d7f80b6c1`，实现/配置版本为 `75b187ce6664d4fa44ddad32dd328112998ff5c4`。
- 已确认 W&B online run：`exp005-p64-s20260805`，链接为 https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp005-p64-s20260805 。
- 正式输出目录为 `experiments/20260805_exp005_gli_formal_training_baseline/outputs/patch_64x64x32/seed_20260805/`；运行日志为其中的 `train.run.log`。

## 2026-08-06 — 修订 exp005 正式闭环 QA 与半量 test 方案

- 用户固定使用正式 `best.pt` 的 `ema` 权重，并授权 schema 2 checkpoint inference 兼容、QA 指标、双 GPU test 分片和 Git 提交。
- 将正式门禁写入 `docs/20260805_003_gli_inference_closed_loop.md`：先复核 best EMA validation 和 actual-checkpoint 零更新 resume，再在 val 上执行完整 `t_T=300` 闭环 QA。
- 只有全部 val 门禁通过才自动开始 test；test 范围由全量改为确定性 50% patch 子集。
- 半量子集按 `anchor_label × sample_role` 分层，以固定 seed `20260806` 和稳定路径 SHA-256 排序选择精确 `floor(N/2)`，冻结 subset manifest 后由 GPU 0、1 独立分片运行。
- test 输出必须标记为“test 50% 确定性子集”，不得外推或表述为全量 test。
- 实现提交：`9662ba1e3695191c0c368f55f3f62a7bca1a080a`；该提交只扩展正式 checkpoint 评估基础设施，不改变已完成训练的模型和 loss。

## 2026-08-06 — exp005 正式 val QA 通过并启动 test 50% 子集

- 固定使用 schema 2、step 46000 的 `best.pt/ema`；固定 val 复算 loss 为
  `0.0965080350`，与 checkpoint 记录误差 `1.16e-7`，完整覆盖与 resume 门禁通过。
- 8 个分层 val patch 的正式 `t_T=300` QA 通过：四通道顺序、shape、explicit support、
  RePaint 闭环、有限值、背景不变性和显存稳定性均满足硬门禁；代表性 montage 目检无推理
  新增明显背景污染。
- 冻结 test 50% manifest：1038 中选择 519，SHA-256 为
  `295b20a01327dcd0071058efd8fe854135688a843c1888ef33242a908b6c3698`；两分片
  260/259、交集为 0、并集等于冻结子集。
- 已于 2026-08-06 03:42 CST 启动 GPU 0/1 两个独立 shard；未启动全量 test、其他 seed
  或 80 patch，也未为 inference 创建 W&B run。

## 2026-08-06 — 启动 exp006 原始 LeFusion RePaint 语义对齐

- 经用户确认，将 GLI inference 中额外的 post-denoiser `target_background` hard clamp 删除；
  保留 denoiser 前的真实背景前向加噪注入。
- 对齐原始 LeFusion 分支：背景噪声改为每次反向调用重新采样，单次调用内四个 lesion channel
  共享，不再缓存整条轨迹的固定 noise。
- 不重新训练，固定复用 exp005 step 46000 `best.pt/ema`、train-only cluster、8 例 val
  manifest 和 519 例 test 50% manifest。
- 新增 exp006 独立配置与输出目录；exp005 的 519 例 hard-clamp 输出保留作错误版本对照，
  不删除、不覆盖。
- QA 从“病灶外必须严格零变化”改为记录 healthy/support 外/outer-shell 的 MAE、p95、最大值、
  变化比例和 lesion/healthy 边界 jump；exact 字段只保留用于新旧对照。
- 新增可复用 `scripts/gli_audit_inference_subset.py`，用于审计两个 shard 的无重叠无遗漏、
  channel/role 分组、support 约束、异常样本和 exp005/exp006 同路径配对差异。

## 2026-08-06 — exp006 完整 val 与 test 50% 子集验收通过

- 远端真实发布 patch 环境全量回归 `31/31` 通过；8 例完整 `t_T=300` val QA 的 healthy、
  support 外和 outer-shell MAE 均约 0.009，边界 jump p95 最大增量 `0.004737`，显存稳定。
- 同一冻结 manifest 的 519 例 test 在 GPU 0/1 完成 260/259 两个 shard；交集 0、并集严格
  等于 manifest，每例 300 calls，checkpoint/EMA/cluster/channel/shape provenance 一致。
- test healthy/support 外/outer-shell 的变化大于 0.1 比例均为 0；边界 jump p95 增量
  mean/p95/max 为 `-0.001774/0.004392/0.014357`，最差样本图未见明显背景污染或接缝。
- NETC/SNFH/ET/RC 的 lesion change MAE 为
  `0.010038/0.010379/0.010303/0.010995`；histogram L1 为
  `0.476755/0.510642/0.439694/0.688289`。
- exp005 旧版 519 例（5.4 GiB）未删除；exp006 新版约 6.6 GiB。step 46000
  `best.pt/ema` 与 exp006 inference 语义冻结为 64 patch 工程 baseline，不外推为医学有效性
  或全量 test 结论。

## 2026-08-06 — 启动 exp007 显式挖空输入 8 例 QA

- 用户授权只复用冻结的 8 例 validation QA，禁止运行 519 例 test 子集或全量 test。
- 新建方法实验 `20260806_exp007_gli_masked_input_qa`：两个变体均将真实病灶 union 区域在送入 LeFusion 前显式置零。
- `masked_multilabel` 保留原四通道 mask 与逐标签最近 train-only cluster；`masked_anchor_union` 将完整病灶 union 赋给 patch 的 `anchor_label` 通道，其他 mask 通道和 condition block 置零。
- QA 固定输出原始输入、挖空输入、生成输出、绝对 difference 和 conditioning mask 五联图。
- 新增可复用 `scripts/gli_build_qa_contact_sheet.py`，将每个变体的 8 张五联图整理为 2×4 总览图。
- 实现 commit：`000e3735ea1604981a316cfee3a97eb3f17c6e5a`。
- 检查时 p80 正在占用 GPU 0/1；在其结束前只实施与测试代码，不启动本实验 GPU 推理，不切换远端训练工作树。

## 2026-08-07 — 实施 exp010–exp012 单通道短程对比

- 用户授权三个方法全部尝试，每种只运行 seed `20260805`、最多 5000 optimizer step；禁止
  p80、其他 seed、519 例和任何全量 test。
- exp010 使用单一完整 T1c 状态预测噪声；exp011 使用 lesion-only 状态预测噪声；exp012
  使用 lesion-only 状态预测 x0，并将 soft-histogram loss 在前 500 step 升至 0.1。
- 数据 loader 新增单通道 `target_t1c`，旧四通道 `data` 仅为历史兼容保留；单通道预测仍按
  四类 mask 分别归一化和统计，不在扩散状态中复制 T1c。
- 新增五种固定 8 例 QA：原始四类真实 hist、原始四类首/末 cluster，以及 union-as-single
  首/末 cluster；新增统一指标和三方法横向 contact sheet 工具。
- 启动审计发现 GPU 0 正在运行 exp009 队列、GPU 1 空闲；不得切换会影响 exp009 后续配置
  读取的远端工作树，也不得占用 GPU 0。
- exp010 首次 online preflight 在 0 optimizer update 时检测到 FP16 初始反向梯度范数非有限；
  临时 checkpoint 已清理，正式训练未启动。三个方法统一改用 A100 BF16 autocast 且不启用
  GradScaler，exp010 使用新的 `exp010-p64-preflight-s20260805-r2` 身份重跑。

## 2026-08-11 — exp018 伪四通道 mask 与双路线训练预检

- 从 exp010 全量分支建立 exp018，不合并会删除 exp010 文件的分类器分叉历史；仅引入 exp016
  checkpoint-compatible 推理代码、overlay loader、可复用导出/阈值/过滤/审计脚本和回归测试。
- exp016 best 在 train+val 生成 `8804` 份 direct sidecar；CRR–ERR 交点冻结为 `0.964`，并生成
  `8804` 份 filtered sidecar。两套 contract 均记录文件、checkpoint、config、subset、split 与 manifest 哈希，未访问 test。
- direct/filtered 全 train 患者等权 focus mIoU 为 `0.671966/0.433933`；filtered 保留覆盖率
  `0.842106`，共有 `334` patch 使用空 mask 兜底。
- 新增两份仅 mask overlay 与运行标识不同的 exp010 FP32 50k 配置；49 项相关回归通过、1 项远端数据环境测试按设计跳过。
- direct/filtered online preflight 均通过零更新反向、完整真实-mask val 和 checkpoint resume；W&B run
  分别为 `exp018-direct-mask-fp32-preflight-s20260805` 与
  `exp018-filtered-mask-fp32-preflight-s20260805`。
- 下一步严格顺序启动 direct、filtered 正式 run；任何 NaN/Inf 均 fail closed，不跳 batch、不自动改学习率。

## 2026-08-11 — exp018 双路线 50k 与最终审计完成

- direct/filtered 均以 FP32 到达 50k；best 分别为 step `44000/46000`，真实-mask val loss 为
  `0.1126854883/0.1122985579`，训练期间未出现 NaN/Inf。
- 修正正式 checkpoint gate 对条件模型 metadata 的错误默认值，并增加 objective、GLI state mode
  与 overlay contract hash 校验；修复提交为 `3ffc9bae421ec9747741507f446e4cacf0ec43bb`。
- 修复后两份 checkpoint 数值复算、零 optimizer-update resume、下一 batch 可复现性均通过；相关
  远端测试 `14/14` 通过。
- 完成 7772-patch direct/filtered mask 复审、完整 1032-patch real/overlay val、固定 8-patch
  paired QA 和最终 montage/summary 聚合；审计链正常结束且未访问 test。
