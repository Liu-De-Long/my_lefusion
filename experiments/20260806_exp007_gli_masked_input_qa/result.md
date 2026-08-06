# 20260806_exp007_gli_masked_input_qa

## 目标

仅使用 exp005 已冻结的 8 例 validation QA，检查显式挖空病灶输入后的生成结果，并比较原始多标签病灶条件与“全部病灶 union 统一为 anchor label”条件。禁止运行 519 例 test 子集或全量 test。

## 变更

- 在送入 LeFusion 前将全部真实病灶 union 区域显式置为 `0.0`，同时保留原始 T1c 只用于指标和 QA。
- 变体一保留原 NETC/SNFH/ET/RC 四通道 mask 和逐标签最近 train-only cluster condition。
- 变体二以 patch 的 `anchor_label` 为唯一目标类型，把完整病灶 union 放入该通道，其他三个 mask 通道及 condition block 置零；目标通道使用其原始 hist block 匹配到的最近 train-only cluster center。
- QA 固定展示原始输入、挖空输入、生成输出、相对原始输入的绝对 difference 和最终 conditioning mask。

## 配置

- 实现 commit：`000e3735ea1604981a316cfee3a97eb3f17c6e5a`。
- 基线：exp006 原始 LeFusion 对齐的 pre-denoiser RePaint，禁止 post-denoiser hard clamp。
- checkpoint：exp005 p64 seed `20260805`、step 46000 `best.pt/ema`。
- schedule：`t_T=300, n_sample=1, jump_length=1, jump_n_sample=1`。
- 样本：同一份冻结 validation QA manifest，共 8 例，覆盖四标签的 interior/boundary。
- sampling seed：`20260806`。
- p80 训练运行期间不得启动本实验 GPU 推理。
- QA contact sheet：`scripts/gli_build_qa_contact_sheet.py` 将每个变体的 8 张五联图整理为 2×4 总览图。

## 结果

QA 暴露旧 exp005 checkpoint 没有学习“挖空背景 + 目标位置”条件下的病灶修复。旧模型训练时
denoiser 只接收由完整病灶图加噪得到的 `x_t` 与 histogram，四通道 mask 只用于 loss；因此把
输入图像挖空并不能让该 checkpoint 获得训练阶段从未见过的空间条件能力。

## 结论

本实验停止，不再继续运行 519 例或全量 test。exp005/exp006 不再是伪病灶生成的有效 baseline；
后续由 exp008 重新训练条件式修复模型。

## 下一步

转入 `20260806_exp008_gli_conditional_inpainting_training`。

## 输出路径

- `experiments/20260806_exp007_gli_masked_input_qa/outputs/patch_64x64x32/seed_20260805/val_qa_masked_multilabel`
- `experiments/20260806_exp007_gli_masked_input_qa/outputs/patch_64x64x32/seed_20260805/val_qa_masked_anchor_union`
