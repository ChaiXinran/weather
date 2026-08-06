# EvolutionTemporalUNet Motion-Only 10-Epoch Report

Date: 2026-08-06

## 1. Executive Summary

本实验完成了第一阶段 `EvolutionTemporalUNet` motion-only 的 10 epoch
训练，并对验证 CSI 最佳的 epoch 6 checkpoint 运行了冻结验证集全诊断：

```text
work_dirs/bth_temporal_unet_motion_10ep_seed0/checkpoints/
val-csi-epoch=06-val_csi_score=0.439920.ckpt
```

新主干可以稳定训练，显存和运行时间均满足工程要求，但效果没有通过 motion
gate。最佳验证综合 CSI 为 `0.4399`，明显低于同为 normalized-dBZ、2-pixel、
no-stop 的 ConvLSTM motion 对照 `0.6225`。冻结验证诊断中，整体 CSI16/32
分别为 `0.1533/0.0734`，而该 ConvLSTM 对照为 `0.2111/0.1022`。

主要问题是强回波面积和强度随 rollout 快速衰减。第一小时强度比为 `0.7925`，
第二小时降至 `0.6153`；第二小时 CSI32 仅 `0.00357`。FSS、object POD 和
16 mm/h 质心误差也没有显示多尺度 U-Net 主干带来运动表示收益。

结论：第一版 Temporal U-Net/FPN motion-only 是一个工程上成功、效果上失败的
架构实验。当前 checkpoint 可保留用于消融，但不应直接进入 source 阶段，也不应
通过继续追加相同 OneCycle epoch 来补救。

## 2. Fixed Protocol

| Item | Setting |
|---|---|
| Split | Event-level validation only |
| Samples | 932 |
| Input/output | 10 history frames -> 20 forecast frames |
| Interval | 6 minutes/frame |
| Model | Shared frame encoder + temporal fusion E1--E3 + 96-channel FPN |
| Motion head | Coarse + bottleneck feature, 20 incremental flows |
| Evolution space | normalized dBZ |
| Displacement bound | 2 pixels/step |
| Source / flow gate | Disabled / disabled |
| Stop gradient | Disabled |
| Training | 10 epochs, batch 4, seed 0, OneCycle, LR 2e-4 |
| Loss | Precipitation R2 + TF transport + flow regularization |
| Test split | Not accessed |

The primary architecture comparison is the earlier normalized-dBZ/no-stop
ConvLSTM motion run (`bth_r4b_motion_pre0788_5ep_seed0`), because it uses the
same field space and displacement bound. The later `0.640662` R4-b operational
baseline uses rain-rate warping and a 1-pixel calibrated bound, so comparison
against it is informative but not a pure backbone ablation.

## 3. Training and Checkpoint Selection

| Epoch | Train loss | Val loss | Val CSI score |
|---:|---:|---:|---:|
| 0 | 0.05456 | **0.01278** | 0.35574 |
| 1 | 0.04791 | 0.01372 | 0.40934 |
| 2 | 0.04252 | 0.01358 | 0.39997 |
| 3 | 0.03891 | 0.01325 | 0.42119 |
| 4 | 0.03650 | 0.01360 | 0.40683 |
| 5 | 0.03465 | 0.01338 | 0.43367 |
| 6 | 0.03316 | 0.01374 | **0.43992** |
| 7 | 0.03196 | 0.01381 | 0.43602 |
| 8 | 0.03119 | 0.01382 | 0.43734 |
| 9 | **0.03085** | 0.01387 | 0.43632 |

训练 loss 下降约 43.5%，但最低验证 loss 出现在 epoch 0，此后始终没有恢复。
验证 CSI 在 epoch 6 达到峰值，随后三轮稳定在 `0.436--0.437`，同时学习率已经
衰减到接近零。这说明当前 10-epoch 调度已基本收敛到一个泛化较差的解，而不是
在结束时仍保持明确上升趋势。

报告选择 epoch 6，而不是 val-loss 最佳的 epoch 0，因为本阶段的核心问题是强
降水运动 skill。`best_val_loss.ckpt` 代表更保守的早期状态，不适合作为主结果。

## 4. Overall Forecast Metrics

| Metric | Temporal U-Net | ConvLSTM norm/2px | R4-b rain/1px |
|---|---:|---:|---:|
| MAE (mm/h) | 0.3614 | **0.3224** | 0.3264 |
| RMSE (mm/h) | 2.0801 | 1.9537 | **1.9336** |
| Mean error (mm/h) | -0.1341 | -- | -0.0839 |
| Intensity ratio | 0.7046 | 0.7712 | **0.8151** |
| CSI16 | 0.1533 | **0.2111** | 0.2184 |
| CSI32 | 0.0734 | 0.1022 | **0.1057** |

| Threshold | CSI | POD | FAR | Frequency bias |
|---:|---:|---:|---:|---:|
| 0.1 mm/h | 0.5550 | 0.6982 | 0.2698 | 0.9562 |
| 2.5 mm/h | 0.3483 | 0.4627 | 0.4151 | 0.7912 |
| 8 mm/h | 0.2438 | 0.3153 | 0.4818 | 0.6086 |
| 16 mm/h | 0.1533 | 0.1874 | 0.5424 | 0.4096 |
| 32 mm/h | 0.0734 | 0.0783 | 0.4585 | 0.1446 |

FAR32 看起来低于一些历史模型，但不能单独解释为改进。Bias32 只有 `0.145`，
说明模型主要通过少预测强回波来降低 false alarm，POD32 也仅为 `0.078`。

## 5. Period Metrics

| Period | Threshold | CSI | POD | FAR | Bias | Intensity ratio |
|---|---:|---:|---:|---:|---:|---:|
| 0--1 h | 16 | 0.2487 | 0.3121 | 0.4496 | 0.5671 | 0.7925 |
| 0--1 h | 32 | 0.1343 | 0.1495 | 0.4304 | 0.2625 | 0.7925 |
| 1--2 h | 16 | 0.0497 | 0.0591 | 0.7611 | 0.2475 | 0.6153 |
| 1--2 h | 32 | 0.00357 | 0.00363 | 0.8273 | 0.0210 | 0.6153 |

相对 normalized-dBZ ConvLSTM 对照：

| Metric | Temporal U-Net | ConvLSTM norm/2px | Delta |
|---|---:|---:|---:|
| CSI16 0--1 h | 0.2487 | 0.3132 | -0.0645 |
| CSI32 0--1 h | 0.1343 | 0.1783 | -0.0440 |
| CSI16 1--2 h | 0.0497 | 0.1060 | -0.0563 |
| CSI32 1--2 h | 0.0036 | 0.0125 | -0.0089 |

四个核心分时段 CSI 全部下降，超过建议 Gate 中允许的 `0.02` 波动。第二小时
Bias32 仅 `0.021`，表示绝大多数极端回波在 rollout 中已经消失。

## 6. Spatial and Object Diagnostics

下表中的均值采用相同 evaluator 的 sample-lead 聚合方式：

| Model | Thr. | FSS1 | FSS3 | FSS5 | Centroid km | Area ratio | Object POD | Object FAR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Temporal U-Net | 16 | 0.1611 | 0.2475 | 0.2858 | 72.46 | 0.3672 | 0.1237 | 0.4259 |
| R4-b rain/1px | 16 | **0.2232** | **0.3227** | **0.3585** | **65.42** | **0.5038** | **0.1673** | **0.3030** |
| Temporal U-Net | 32 | 0.0608 | 0.0931 | 0.1054 | **48.53** | 0.1139 | 0.0666 | 0.3810 |
| R4-b rain/1px | 32 | **0.0871** | **0.1284** | **0.1440** | 54.46 | **0.1833** | **0.0954** | **0.3484** |

Temporal U-Net 在两个阈值的所有 FSS 窗口均下降。16 mm/h 质心误差也增加约
7 km。32 mm/h 质心误差表面改善约 6 km，但 area ratio 同时从 `0.1833`
下降到 `0.1139`，object POD 也下降。这个结果更符合“只有少量位置较接近的强
回波幸存”，不能证明运动场更准确。

## 7. Rollout Stability

| Lead range | MAE | RMSE | Mean error | Intensity ratio |
|---|---:|---:|---:|---:|
| 0--1 h | 0.3009 | 1.8409 | -0.0949 | 0.7925 |
| 1--2 h | 0.4220 | 2.2946 | -0.1734 | 0.6153 |

逐 lead 强度比从 6 分钟的 `0.9307` 单调下降到 120 分钟的 `0.5593`。预测值
始终位于 `[0,1]`，因此问题不是输出越界或 clipping，而是反复平流中的面积、
峰值和能量损失。当前主干没有学到足以抵消这种衰减的更有效位移序列。

工程稳定性没有问题：训练峰值 GPU 显存约 `890 MiB`，单 epoch 约 176--183
秒，10 epoch 总训练时间约 30 分钟，没有 NaN、OOM 或流场数值爆炸记录。

## 8. Gate Decision

| Gate | Result |
|---|---|
| CSI16/32 drop <= 0.02 | **Fail**; first-hour drops are 0.0645/0.0440 |
| FSS 3x3/5x5 not worse | **Fail** at both 16 and 32 mm/h |
| Centroid error not worse >10% | Mixed; 16 mm/h worsens, 32 mm/h is confounded by area collapse |
| Frequency bias / area not inflated | Pass for inflation, but fails from severe under-coverage |
| Flow/output numerical stability | Pass |
| Batch 4 engineering stability | Pass with about 890 MiB peak allocation |

Decision: **P1 motion gate failed. Do not add the source head to this checkpoint.**

## 9. Interpretation and Next Step

本实验不能证明“U-Net/FPN 不适合雷达运动预测”，但可以否定当前这一组具体设计：

- 所有尺度先逐帧编码，再把 E1--E3 压成单个加权时间特征；
- E0 只保留最后一帧；
- FPN 所有尺度统一到 96 通道；
- 从 coarse+bottleneck 一次性回归 20 个独立增量流；
- 从零初始化、完全 scratch 训练 10 epoch。

与 ConvLSTM 对照相比，最大的实验差异不只是卷积主干。ConvLSTM 使用
`0.788316` checkpoint 初始化并在前两轮冻结 encoder，而 Temporal U-Net 是完全
scratch。这意味着当前结果同时包含架构差异和初始化/预训练差异。它足以作为
当前实现的 no-go，但还不足以宣称多尺度架构本身失败。

建议下一轮仍停留在 motion 阶段，只做一个受控修订版本：

1. 先增加 flow teacher/distillation 或单步 teacher-forced motion warm-up，避免
   scratch 模型直接从 20-step rollout 学习运动。
2. 把 20 个完全独立的 flow 输出改为 lead-conditioned/refined flow，减少后期流场
   缺少时间结构的问题。
3. 在不接 source 的前提下加入 E0 temporal fusion，并检查它是否改善强回波边缘
   与小尺度移动。
4. 对新候选先运行 teacher-forced predicted-flow versus zero-flow gate；只有强回波
   CSI/FSS/centroid 超过 zero-flow，才进入完整 20-step 训练。

不建议直接增加到 20/30 epoch。当前 epoch 6 后 CSI 已平台化，val loss 从 epoch 0
开始即未改善，相同调度的额外训练缺少依据。

## 10. Artifacts

- Training log: `work_dirs/bth_temporal_unet_motion_10ep_seed0/train_20260806_151001.log`
- Training metrics: `lightning_logs/version_50/metrics.csv`
- Selected checkpoint: `work_dirs/bth_temporal_unet_motion_10ep_seed0/checkpoints/val-csi-epoch=06-val_csi_score=0.439920.ckpt`
- Full validation output: `work_dirs/bth_temporal_unet_motion_10ep_seed0_val_diagnostics/saved/precipitation_evaluation/`
- Evaluation summary: `work_dirs/bth_temporal_unet_motion_10ep_seed0_val_diagnostics/saved/precipitation_evaluation/summary.json`
- Per-lead metrics: `work_dirs/bth_temporal_unet_motion_10ep_seed0_val_diagnostics/saved/precipitation_evaluation/per_lead_metrics.csv`
- FSS/object metrics: `work_dirs/bth_temporal_unet_motion_10ep_seed0_val_diagnostics/saved/precipitation_evaluation/per_window_metrics.csv`
- Normalized-dBZ ConvLSTM comparison: `.research/history/r4b_motion_pre0788_5ep_analysis.md`
- Operational R4-b baseline: `.research/baselines/r4b_0640662_val/baseline_report.md`
