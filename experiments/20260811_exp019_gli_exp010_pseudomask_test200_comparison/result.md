# exp019：三种 exp010 模型的 200 例配对 test 对比

本实验比较原始 exp010、direct 伪四通道训练和 filtered 伪四通道训练的 best/EMA checkpoint。
三模型固定使用相同的 200 个 test patch、真实四通道 mask、真实 histogram、sampling seed
`20260806` 和 300-step RePaint。计划报告 lesion/full-patch PSNR、SSIM、Hist-W1、FID、
SwAV-FSD 和 KID；正式结果待运行后填写。
