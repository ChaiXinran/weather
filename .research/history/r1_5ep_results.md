# R1 5-epoch 结果

运行时间：2026-07-31  
实验目录：`work_dirs/bth_simvp_gsta_r1_5ep_seed0`  
日志：`lightning_logs/version_12/metrics.csv`

## 协议

保持 Tiny-SimVP、MSE、递归 10→20、manifest、batch size 16、
val batch size 8、学习率 1e-3、OneCycle 和 seed 0 不变。
验证期增加 CSI/POD/FAR/Bias、Intensity Ratio 和 Area Ratio，
并用下式选择 CSI-best：

```text
CSI16(0-1h) + CSI32(0-1h) + CSI16(1-2h) + 2*CSI32(1-2h)
```

## 逐轮结果

| Epoch | Val loss | CSI score | CSI16 0-1h | CSI32 0-1h | CSI16 1-2h | CSI32 1-2h |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.011683 | **0.331482** | 0.171883 | 0.070569 | 0.067435 | 0.010797 |
| 1 | **0.010082** | 0.240884 | 0.168245 | 0.069080 | 0.003528 | 0.000016 |
| 2 | 0.010733 | 0.220303 | 0.152591 | 0.054442 | 0.012740 | 0.000265 |
| 3 | 0.011513 | 0.256728 | 0.172368 | 0.069941 | 0.013485 | 0.000467 |
| 4 | 0.011883 | 0.314977 | 0.198812 | 0.084490 | 0.026775 | 0.002450 |

## Best checkpoint 诊断

CSI-best（epoch 0）的强降水诊断：

| Period / threshold | POD | FAR | Bias | Intensity ratio |
|---|---:|---:|---:|---:|
| 0-1h / 16 | 0.214504 | 0.536176 | 0.462468 | 0.765243 |
| 0-1h / 32 | 0.085364 | 0.710644 | 0.295015 | 0.765243 |
| 1-2h / 16 | 0.093051 | 0.803236 | 0.472905 | 0.794772 |
| 1-2h / 32 | 0.014092 | 0.955855 | 0.319216 | 0.794772 |

MSE-best（epoch 1）第二小时明显塌缩：

- CSI16 1-2h：0.003528
- CSI32 1-2h：0.000016
- Bias16 1-2h：0.013089
- Bias32 1-2h：0.000016
- Intensity Ratio 1-2h：0.489606

## 初步结论

R1 已证明 MSE-best 与 CSI-best 不是同一 checkpoint。epoch 1 相比 epoch 0
降低了 Val MSE，但第二小时 CSI、Bias 和强降水检出率几乎崩溃。

不过 epoch 0 的 CSI-best 仍伴随很高的 FAR，尤其第二小时 FAR32 为 0.955855。
因此这个 checkpoint 只能说明“checkpoint 选择确实错位”，不能说明当前 MSE 模型
已经获得可用的强降水预报能力。下一步应在 R2 中修改 loss，并继续联合监控
CSI、Bias 与 FAR。

本次只要求完成 5 个 epoch。`tools/train.py` 随后自动启动的完整 test
预计还需约 45 分钟，已在训练和验证 checkpoint 全部落盘后终止。
