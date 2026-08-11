# exp019：三种 exp010 模型的 200 例配对 test 对比

## 结论

原始 exp010、direct 伪四通道训练和 filtered 伪四通道训练的 best/EMA checkpoint 已在同一组
200 个 test patch 上完成严格配对比较。三者使用完全相同的真实四通道 mask、真实 histogram、
sampling seed `20260806`、FP32 和 300-step RePaint；没有生成 test 伪 mask，也没有访问未入选的
838 个 test patch。

两种伪 mask 训练路线都明显优于原始 exp010。filtered 的 lesion PSNR、SSIM、MAE 以及
FID/SwAV-FSD 最好；direct 的 Hist-W1 最低。因样本量只有 200，FID/FSD/KID 只作点估计或
重复子集描述，不作显著性结论。

| 模型 | lesion PSNR ↑ | lesion SSIM ↑ | lesion MAE ↓ | Hist-W1 macro ↓ | FID ↓ | SwAV-FSD ↓ | KID mean ± std ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp010 step 14000 | 19.9992 | 0.6451 | 0.15482 | 0.10094 | 60.4132 | 3.3253 | 0.01225 ± 0.00354 |
| direct step 44000 | 21.5548 | 0.7053 | 0.12628 | **0.06749** | 49.7290 | 2.6669 | 0.00370 ± 0.00229 |
| filtered step 46000 | **21.8071** | **0.7277** | **0.12087** | 0.08346 | **47.0280** | **2.4970** | **0.00131 ± 0.00162** |

完整 patch PSNR/SSIM 分别为 exp010 `26.6975/0.8184`、direct `28.2530/0.8539`、filtered
`28.5053/0.8653`；它们会被精确恢复的背景抬高，仅作辅助指标。患者等权 Hist-W1 为
`0.09661/0.06605/0.07892`。分类 Hist-W1 见 `per_class.csv`。

患者聚类配对结果显示：direct 相对 exp010 的 lesion PSNR/SSIM 平均提升
`+1.6586/+0.06183`，95% CI 为 `[1.3349,1.7829]` 和 `[0.05268,0.06837]`；filtered 相对
exp010 提升 `+1.9806/+0.08643`，95% CI 为 `[1.4969,2.1177]` 和
`[0.07449,0.09182]`。filtered 相对 direct 的 SSIM 提升 `+0.02460`，95% CI
`[0.01611,0.02886]`；PSNR 差 `+0.3220` 的 CI `[-0.03335,0.50638]` 跨 0。

## 数据、分布指标与审计

- manifest SHA-256：`b7c2802dd24193e950e505700eeeb374435c19eac539b6e263b470283210ebae`。
- 精确选择 200 patch、66 patient；两个互斥 shard 均为 100，三模型共同交集为 200。
- 每模型恰好 200 个唯一 NPZ；每例 300 次模型调用，背景外最大变化严格为 0，全部输出有限。
- 三个正交最大病灶面积切片组成 600 张分布评估图；真实特征只缓存一次并由三模型共享。
- real-vs-real split baseline：FID `92.2488`、SwAV-FSD `5.2201`、KID
  `0.001285 ± 0.001716`。该 baseline 反映小样本拆分偏差，不能据此把不同特征距离横向比较。
- 主要产物：`summary.json`、`distribution_metrics.json`、`per_patch.csv`、`per_class.csv`、
  `paired_comparisons.csv`、feature cache 和固定 8 例 QA montage。
- `summary.json` SHA-256：`b83ecc8d0419a095e5415ba705db11edc81f8b3ef0af60e2a144e8fe256c3d07`。

## 运行记录

原始 batch=1 正式链在 14+14 例后因四小时预算外推失败而安全停止，结果保存在
`exp010_aborted_batch1_20260811T162335Z`，没有混入正式输出。随后用 batch=4、batch=8 做真实
吞吐基准，冻结为 FP32 batch=8；三模型正式耗时各约 58 分钟。端到端从首次 preflight 到最终
指标完成共 `12276` 秒，即 `3:24:36`，低于四小时上限。正式结束后 GPU 已释放。

推理实现提交为 `59919b34ec6fe14e3d7b1ce4ef5cb34ca7ae5e78`；固定哈希样本和三模型
绝对误差图 QA 修复提交为 `03ae6b7`。大型 NPZ、权重、特征缓存、日志与 QA 图不进入 Git。
