# exp020：direct/filtered Test 伪 Mask 条件下的配对重评

## 结论

exp020 修正了 exp019 在 direct/filtered 推理时仍统一使用 GT mask 的设计。冻结的 200 例、源 T1c、路径顺序、sample seed、FP32 和 300-step RePaint 全部复用；只有条件 mask/histogram 改为与训练路线匹配：exp010 使用 GT，direct 使用 exp016 direct sidecar，filtered 使用阈值 0.964 的 filtered sidecar。

direct 与 filtered 各完成 200 个唯一结果。direct union 与 GT total mask 精确一致；filtered union 是 direct union 子集，覆盖 GT union 的 80.9007%，其余 19.0993% 为 copied region，不能计入 generated-only 排名。

## GT 区域指标

下表为体素 pooled PSNR 与 3D SSIM-map 的体素 pooled 区域均值。PSNR 的 data range 固定为 2。

| 模型 | NETC PSNR / SSIM | SNFH PSNR / SSIM | ET PSNR / SSIM | RC PSNR / SSIM | GT union PSNR / SSIM |
|---|---:|---:|---:|---:|---:|
| exp010 GT 条件 | 19.2273 / 0.5121 | 21.2991 / 0.5111 | 14.3839 / 0.3785 | 19.6323 / 0.5416 | 18.7550 / 0.4960 |
| direct 伪 mask 条件 | 21.1958 / 0.6268 | 22.9841 / 0.6002 | 16.1249 / 0.5272 | 21.3886 / 0.6356 | 20.4882 / 0.5972 |
| filtered 伪 mask 条件 | 26.7716 / 0.8105 | 24.6609 / 0.6990 | 17.8071 / 0.6854 | 24.0289 / 0.7681 | 22.4173 / 0.7171 |

filtered 的 GT 区域总体数字包含 19.0993% 未挖空、精确复制的 GT lesion，因此不能据此单独宣布 filtered 最优。

## 公平 generated-only 与 copied-region 拆分

| 模型 | common filtered-retained PSNR / SSIM | rejected GT union PSNR / SSIM | rejected exact-copy |
|---|---:|---:|---:|
| exp010 | 19.1322 / 0.4988 | 17.4440 / 0.4841 | 0.0561% |
| direct | 20.7710 / 0.5977 | 19.4605 / 0.5952 | 0.1628% |
| filtered | **21.4968 / 0.6636** | +∞ / 0.9435 | **100%** |

common filtered-retained 是三模型在同一真正生成区域上的比较。filtered rejected GT union 的 MSE 为 0、PSNR 为 +∞，这是 post-sample exact copy；SSIM 未等于 1 是因为完整 3D SSIM 窗口会跨入相邻生成区域。

三模型在有效脑区内 GT-union 外均精确不改变：MSE 0、PSNR +∞、变化体素率 0、最大绝对变化 0。全部脑内 union 外的 pooled SSIM 分别为 0.9653、0.9753、0.9846；距 GT union 至少 3 体素的安全区为 0.9973、0.9982、0.9990。

## 完整 64×64×32 patch 指标

完整 patch 包含固定恢复的病灶外体素，因此只作为辅助指标。`patch PSNR mean` 是先逐 patch 计算 PSNR 再取均值；`pooled PSNR` 是汇总全部 26,214,400 个体素的 SSE/MSE 后再换算，两者不可混用。SSIM 使用完整 3D SSIM map（win_size=7、data range=2）。

| 模型 | patch PSNR mean / median | pooled PSNR | patient-equal PSNR mean [95% CI] | pooled SSIM | patient-equal SSIM mean [95% CI] |
|---|---:|---:|---:|---:|---:|
| exp010 | 26.6975 / 26.1221 | 24.4695 | 26.5953 [25.5376, 27.7810] | 0.8439 | 0.8501 [0.8267, 0.8706] |
| direct | 28.4252 / 27.5442 | 26.2027 | 28.4751 [27.3733, 29.7770] | 0.8768 | 0.8826 [0.8647, 0.8990] |
| filtered | 32.1061 / 30.5484 | 28.1318 | 30.9005 [29.6025, 32.3484] | 0.9145 | 0.9154 [0.9006, 0.9296] |

filtered 的完整 patch 指标最高不能单独解释为生成质量最高：其过滤后未挖空区域和全部 mask 外区域被精确复制。公平的 generated-only 比较仍应使用上一节的 common filtered-retained 指标。

## 伪 mask 审计

- selection manifest SHA-256：`b7c2802dd24193e950e505700eeeb374435c19eac539b6e263b470283210ebae`。
- exp016 checkpoint SHA-256：`3f460bbd245fd69ba6e2c6cf71f806bc28e4ebb3862a2e3fc5c411b7e66f3617`。
- direct union：Dice/IoU/precision/recall 均为 1。
- filtered union：Dice 0.8944、IoU 0.8090、precision 1、recall/coverage 0.8090。
- direct 每类 Dice：NETC 0.5972、SNFH 0.9718、ET 0.9260、RC 0.9124。
- filtered 每类 Dice：NETC 0.2344、SNFH 0.9488、ET 0.7270、RC 0.8155。
- filtered 共保留 480 个组件、移除 3019 个组件；11 个 patch 触发非空兜底。
- 两套 sidecar 路径并集均精确等于 200 例 manifest，histogram 均由对应伪 mask 重算；`unselected_test_accessed=false`。

## Histogram 与分布指标

GT 类别 Hist-W1 macro 为：exp010 0.10094、direct 0.06904、filtered 0.03469。filtered 的该数值同样受 copied region 影响。按模型 conditioning 类别计算时为 0.10094、0.06903、0.05083，filtered 有效类别单元为 328。

| 区域口径 | 模型 | FID | SwAV-FSD | KID mean ± std |
|---|---|---:|---:|---:|
| GT-union composite | exp010 | 60.4132 | 3.3253 | 0.01225 ± 0.00354 |
| GT-union composite | direct | 49.4853 | 2.6392 | 0.00359 ± 0.00231 |
| GT-union composite | filtered | 39.5898 | 1.9807 | 0.00034 ± 0.00157 |
| common filtered-retained | exp010 | 61.5189 | 2.6948 | 0.01114 ± 0.00299 |
| common filtered-retained | direct | 52.6213 | 2.2186 | 0.00324 ± 0.00191 |
| common filtered-retained | filtered | 49.6937 | 2.0737 | 0.00208 ± 0.00171 |

GT-union 与 retained 口径的 real-vs-real split baseline 分别为 FID 92.2488/88.4913、FSD 5.2201/3.6582；小样本分布距离只作描述性点估计，不作显著性结论。

## 运行与验收

- preflight：direct step 44000 与 filtered step 46000 均为 EMA、FP32、每例 300 calls、背景精确不变、memory stable。
- 正式推理：direct 58:03，filtered 58:04；各 shard 100/100，三模型共同交集 200。
- sidecar、推理与指标入口均核验 selection SHA、checkpoint SHA、overlay contract SHA、路径与 sample seed。
- 最终汇总、CSV、分布指标和固定 8 例 QA 已生成；加入完整 patch 指标后的 `summary.json` SHA-256 为 `1ee07848a4eebffe9c806d1c012c23c3ac9002303d95bab34d7c681d181cf236`。
- GPU 任务结束后双卡保活已恢复为 PID `806844/806845`；PID 仅记录本次运行，不作为未来固定值。
- 大型 sidecar、推理 NPZ、特征缓存、checkpoint、日志和 QA 图不进入 Git。

完整 patch 指标实现版本：`b8697c5d`。
