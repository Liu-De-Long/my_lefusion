# exp016 p64 四模态逐体素亚区分类

## 目标

在 exp009–exp015 相同的冻结 1000 个有标签 train p64 patch、相同患者级划分和相同验证门禁下，验证
T1c/T1n/T2f/T2w 四模态是否能突破 T1c-only 的可分性瓶颈，并将患者等权 ET/RC focus mIoU 提升至
`0.85` 以上。

## 变更

- 输入改为按模态独立归一化的 T1c/T1n/T2f/T2w 与总病灶 mask，共 5 通道；输出保持 mask 内四分类 logits。
- 新增冻结窗口重放脚本，逐条复用原 manifest 的 `relative_path`、患者、XYZ origin、padding 与标签 patch；输出
  manifest 保持字节级一致，因此冻结 subset 与患者划分哈希不变。
- 新 NPZ 只保存四模态、scalar `seg` 和 affine，不保存 `hist`；loader 只返回影像、总 mask、可选 target 和样本 ID。
- 从 exp014 best warm-start：共享层完全复用，stem 仅映射 T1c 与总 mask；T1n/T2f/T2w 权重初始化为 0。
- loss、网络宽度和 val-only 选模契约保持 exp014，隔离“多模态输入”这一主要变量。

## 配置

- 输入/输出：`[B,5,32,64,64] -> [B,4,32,64,64]`。
- 模型：base channels 32 的两级多尺度残差 3D U-Net，约 306 万参数。
- 有标签数据：同一冻结 1000 train patch；验证集按完整 p64 重建后患者等权聚合。
- batch 8，初始学习率 `2e-4`，最多 50 epoch，early-stopping patience 10。
- loss：CE/Focal/Dice/Lovász=`0.25/0.10/0.25/0.40`，ET/RC multiplier `1.25`，类间边界 multiplier `1.5`。
- W&B online fail-closed run：`exp016-multimodal-s20260808`。

## 结果

- 已获得用户对 exp016 四模态输入的明确授权。
- 本地静态编译与 `git diff --check` 通过；远端 `lefusion` 环境 focused tests 31/31 通过。
- 单 case 真数据重放 smoke 通过：4 个 patch 的 T1c/seg/affine 与原数据一致，manifest SHA 不变，四模态
  均非退化，且 `test_materialized=false`、`label_derived_input_keys=[]`；专用 smoke 目录已按规则删除。
- 分支已推送，远端独立 worktree 与实现提交 `a5be708259a6c6860d962c3c6b23ef8ca31d4429` 对齐；
  正式 train+val 四模态数据重建已完成。
- 正式数据共 8804 patch/1447 cases：train 7772、val 1032；四个模态的退化病例均为 0。压缩字节总量约
  12.98 GB，远端文件系统占用约 48 GiB。
- 独立审计确认原 manifest 9842 条，其中 test 1038 条；train+val 缺失 0、额外文件 0、冻结 1000 subset
  缺失 0、test 物化 0。随机 64 patch 的六键、shape/dtype/range、seg 标签和 T1c 复现均通过。
- 数据 audit SHA-256 为 `cb34a00e963f77a0dd0df74c84d6937d635d2a4606dd585f4b0e118d7e3996d1`。
- CPU 零 optimizer-step preflight 通过：参数量 3065600，输入/输出
  `[1,5,8,16,16] -> [1,4,8,16,16]`，loss `0.606481`，梯度范数 `8.815803`，覆盖 8 个
  `anchor×role` 分层；config/subset/initial checkpoint SHA 全匹配。CPU preflight 文件 SHA-256 为
  `9b75211a794f822f375e4556212c14afe13b026b2ee424b6c849e0282f378e99`。
- 一次性 `data_build.log` 已由正式 JSON audit 取代并删除；未删除数据集、audit 或 CPU preflight。
- 物理 GPU0 完整 p64 零 optimizer-step preflight 通过：batch 8，输入/输出
  `[8,5,32,64,64] -> [8,4,32,64,64]`，loss `0.190637`，梯度范数 `1.868516`，峰值
  allocated/reserved 显存 `2777.291/3920 MiB`；GPU1 既有任务未干扰。
- GPU preflight 文件 SHA-256 为 `4ae883839b9912451c30a3d9f24bc9dfcaf0c2f6fd5bc48ef9d3d227ee530d00`，
  config/subset/initial checkpoint SHA 全匹配。
- 唯一 W&B online 正式 run
  [exp016-multimodal-s20260808](https://wandb.ai/jinyuanbao719-xi-an-jiaotong-university-/lefusion-brats2024-gli/runs/exp016-multimodal-s20260808)
  使用代码提交 `ff288093578b4ade4c34a0318021574254225691`。用户在 epoch 21 后暂停，随后从完整
  `latest.pt` 恢复同一 run；optimizer、scheduler、early-stopping 状态和 RNG 均连续。
- 训练在 epoch 29、`bad_epochs=10` 时自然 early stop；绝对 best 为 epoch 24。最终患者等权 focus mIoU
  为 `0.664752`，95% CI `[0.619226,0.706501]`；ET/RC IoU 为 `0.589268/0.740236`，macro IoU
  `0.641886`，macro Dice `0.702504`，balanced accuracy `0.809577`，mask 内错误率 `0.051375`。
- 仅作参考的 pooled ET/RC IoU 为 `0.858449/0.840951`；`>1000` 体素 ET/RC patch Dice 为
  `0.905619/0.853121`，而 `1–100` 体素仅为 `0.131446/0.216415`。患者等权与小区域仍是主要瓶颈。
- 三项性能门禁均失败；mask 外非零率 `0`、union Dice `1` 两项空间门禁通过。冻结 test 未运行，
  `test_metrics.json` 不存在。
- W&B 最终状态为 `finished`，线上 history 为 epoch `0–20,22–29`；暂停前 epoch 21 已完整写入本地
  checkpoint/history，但未在 SIGTERM 前上传，是唯一在线日志缺口。本地 history 完整覆盖 `0–29`。
- `best.pt` SHA-256 为 `3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617`；
  `latest.pt` 为 `50f086debed63226cbfc652e4f95f804cbf403ed29dd22909cb828f78f027328`；最终 val 指标文件为
  `6069fe97e7ebb21f810d0bda89b9b04dd26f1e2cbfb649ccb5ca84c911333c5c`。

## 结论

四模态将患者等权 focus mIoU 从 exp014 的 `0.592211` 提升到 `0.664752`，绝对增益 `0.072541`，证明新增
模态提供了有效信息；但仍比 `0.85` 低 `0.185248`，且 ET 与小区域性能不足。当前共享 stem 从 T1c/总 mask
warm-start、其余模态置零的融合方式没有充分利用 T1n/T2f/T2w，继续相同配方训练不具备达到门禁的合理预期。

## 下一步

1. 保持冻结 test 封存，先对 exp016 best 做 val-only 逐模态 occlusion，量化四模态真实贡献；
2. 下一方法使用各模态独立浅 stem 与平衡融合，避免新增模态长期受零初始化 stem 压制；
3. 允许利用剩余 train patch 做无标签多模态自监督预训练，再在同一冻结 1000 patch 上微调；
4. 仅在总 mask 几何触发的 component/ROI 小区域分支上验证局部 refinement，不再重复堆叠 loss 权重。

## 输出路径

`experiments/20260808_exp016_gli_p64_multimodal_classifier/outputs/supervised/`
