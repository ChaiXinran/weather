# EvolutionTemporalUNet Factorized Source 5-Epoch 阶段报告

日期：2026-08-06  
状态：验证集阶段性结果，尚未运行冻结全诊断和 Test

## 1. 执行摘要

本实验针对旧 factorized source 训练中反复出现的失败模式：初始 CSI 较好，但 source 更新后迅速下降。新方案将 teacher-forced 机制监督和 pure-free rollout 监督拆开，并直接监督实际进入演化方程的 effective growth/decay，同时增加状态、稳态、消散保护和短递归约束。

本次 5 epoch、seed 0 训练已完整结束：

```text
experiment = bth_temporal_unet_factorized_s20_warmup_5ep_seed0
last.ckpt epoch index = 4
global step = 7030
best val_csi_score = 0.596617
best val_loss = 0.017384
latest val_source_gain_vs_zero = 0.000019
```

最重要的结果是：**旧方案的“首轮较好、随后骤降”没有再次出现。** 可恢复的验证 CSI checkpoint 从 `0.564890` 连续提高到 `0.588788`、`0.595313` 和 `0.596617`。旧方案在相同 source 阶段会下降到 `0.41`，继续训练后最低达到 `0.20--0.32`。

当前可以确认损失设计和训练方向已明显改善，source 更新至少没有再破坏已有运动解。但现阶段仍不能宣称 source 机制已取得显著独立增益：`val_source_gain_vs_zero` 虽为正且连续上升，绝对值仍只有 `1.9e-5`，而且它表示雨强状态误差改善，不是 CSI 增量。

阶段判定：**通过训练稳定性 Gate，进入冻结验证全诊断；尚未通过物理 source 有效性 Gate。**

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据划分 | `.research/bth_2025_events.json` 事件级划分 |
| 输入/输出 | 10 帧历史 Radar -> 20 帧预测 |
| 帧间隔 | 6 min |
| 空间尺寸 | 66 x 70 |
| 模型 | EvolutionTemporalUNet，约 1.2 M 参数 |
| 初始 checkpoint | motion epoch 6，记录 CSI score 0.439920 |
| 训练参数 | 仅 source 分支可训练，encoder/motion 冻结 999 epoch |
| Source 参数化 | factorized growth / steady / decay |
| 演化空间 | rain rate |
| Source LR | 5e-5 |
| Scheduler | OneCycle |
| 训练 | 5 epoch，batch 8，seed 0，deterministic |
| 验证 batch | 实际为 4（配置覆盖命令行值） |
| Free rollout | 训练 horizon 3；验证 horizon 20 |
| Test | 未访问；训练使用 `--skip_test_after_train` |

训练从以下 checkpoint 加载了 144 个 encoder/motion tensors：

```text
work_dirs/bth_temporal_unet_motion_10ep_seed0/checkpoints/
val-csi-epoch=06-val_csi_score=0.439920.ckpt
```

## 3. 指标口径

### 3.1 `val_csi_score`

该值不是单阈值 CSI，而是固定验证协议中的加权综合分数：

```text
CSI16(0-1 h)
+ CSI32(0-1 h)
+ CSI16(1-2 h)
+ 2 * CSI32(1-2 h)
```

因此它适合在使用同一代码、同一验证划分和同一阈值配置的实验间选择 checkpoint，但不能直接与报告中的单项 CSI16 或 CSI32 比较。

### 3.2 `val_source_gain_vs_zero`

当前代码将完整 source 输出与同一次前向中的纯平流状态比较：

```text
gain = rain_state_error(advected_rain, target)
     - rain_state_error(evolved_rain, target)
```

`rain_state_error` 是归一化线性雨强 Huber 与 log1p 雨强 Huber 的等权平均。正值表示 source 在该状态误差下优于 zero-source，负值表示 source 使状态更差。它不是 CSI 增量，也不是独立 checkpoint 的配对全指标差值。

## 4. 损失设计变更

旧方案把机制标签和递归误差放在同一条 scheduled/free 混合轨迹上，使所谓 oracle source 同时吸收真实增强/减弱、运动误差、前序 source 误差和递归累计误差。新方案做了以下关键调整：

1. Teacher-forced 分支始终使用真实上一帧构造机制标签。
2. Pure-free 分支只约束递归预测，不再从 free trajectory 构造物理 source 标签。
3. 直接监督实际生效的 growth/decay：`p * fraction * capacity`，而不是只分别监督概率和幅值。
4. State loss 直接作用于真实 `evolved_rain`。
5. 新增 steady、guard、soft CSI16/32、area 和 cumulative budget 约束。
6. Source mask 与推理一致，基于平流后有效雨区，而不把未来 target 混入推理掩码。
7. Regime 类别权重改为平方根并截断，降低少数类过度驱动。
8. Source head 从过饱和的正负 20 bias 改为较温和先验，恢复有效梯度。

本实验采用的主要权重为：

| 损失项 | 权重 |
|---|---:|
| effective source | 1.0 |
| rain state | 1.0 |
| steady | 0.25 |
| guard | 0.5 |
| regime | 0.05 |
| 3-step free state | 0.25 |
| soft CSI16 | 0.05 |
| soft CSI32 | 0.10 |
| area | 0.02 |
| budget | 0.05 |

## 5. 训练与 checkpoint 选择

| Epoch index | Train loss | Val loss | Val CSI score | Source gain vs zero |
|---:|---:|---:|---:|---:|
| 0 | 0.402999 | 0.017515 | 未保留精确 checkpoint 值 | 未保留 |
| 1 | 0.370467 | 0.017499 | 0.564890 | 0.00000002 |
| 2 | 0.366659 | 0.017407 | 0.588788 | 0.00000803 |
| 3 | 0.365123 | 0.017384 附近日志值 | 0.595313 | 0.00001685 |
| 4 | 日志末行未单独落盘 | **0.017384** | **0.596617** | **约 0.000019** |

说明：训练文本日志只打印了前四条 epoch 汇总，但 `last.ckpt` 明确记录 `epoch=4`、`global_step=7030`，且 epoch 4 的 CSI/loss/mechanism checkpoint 均已生成，因此 5 epoch 已完成。部分逐轮值来自 callback 保存状态或 checkpoint 文件名；没有精确保留的值不作推断。

训练趋势：

- 从 epoch 1 到 epoch 4，综合 CSI 增加 `0.031727`，约 `+5.62%`。
- Val loss 从 `0.017515` 降到 `0.017384`，约下降 `0.75%`。
- Source gain 从接近零升到约 `1.9e-5`，方向连续为正。
- 没有 NaN、OOM 或 source 更新后的指标崩溃。
- 峰值 GPU 显存约 `4393 MiB`，单 epoch 约 `470--474 s`。

当前用于完整诊断的首选 checkpoint：

```text
work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/checkpoints/
val-csi-epoch=04-val_csi_score=0.596617.ckpt
```

机制对照 checkpoint：

```text
work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/checkpoints/
val-mechanism-epoch=04-val_source_gain_vs_zero=0.000019.ckpt
```

## 6. 与历史 source 实验对比

| 实验 | Source 训练方式 | 初期 CSI score | 后续 CSI score | 结果 |
|---|---|---:|---:|---|
| original factorized | LR 2e-4，旧混合目标 | 0.564887 | 0.436026 -> 0.321987 | 崩溃 |
| old warmup | TF warmup，LR 5e-5 | 0.602775 | 0.406984 -> 0.200459 | 崩溃 |
| scheduled low-LR | scheduled sampling，LR 1e-5 | 0.564899 | 0.411575 | 崩溃 |
| new separated protocol | TF mechanism + pure-free horizon 3 | 0.564890 | 0.588788 -> 0.595313 -> 0.596617 | 稳定上升 |

相对历史实验最终保留值，新实验高出：

- 比 original factorized 的 `0.321987` 高 `0.274630`；
- 比 old warmup 的 `0.200459` 高 `0.396158`；
- 比 scheduled low-LR 的 `0.411575` 高 `0.185042`。

这组结果支持 `sugges.md` 的主要诊断：问题不只是学习率过大或 warmup 不足，而是机制监督目标与 free rollout 误差混合后，source 被训练成递归误差修正器。拆分两条轨迹并约束实际 effective source 后，训练方向发生了实质变化。

## 7. 与 zero-source 和 motion 基线的关系

严格 zero-source 旧报告给出的参考是：

```text
val_csi_score = 0.5648994897
val_loss      = 0.0175145343
```

新实验 epoch 1 几乎复现该初始分数，随后提高到 `0.596617`，表面增量为：

```text
Delta val_csi_score = +0.031718
```

这是目前 source 可能带来预报增益的最积极证据。但该比较还不是严格的冻结配对评估：旧 zero-source 结果来自单独运行，而当前 `val_source_gain_vs_zero` 是同一次前向内部的雨强状态误差差。必须用当前 epoch-4 checkpoint，在完全相同样本上分别执行 source-on 和 source-zero，才能计算可信的 CSI/POD/FAR/Bias 差值。

motion checkpoint 文件名中的 `0.439920` 也不能直接解释为 source 带来 `+0.156697`。旧 motion-only 训练使用的验证路径和后续 EvolutionOperator zero-source 路径存在差异；已有严格 zero-source `0.564899` 才是当前更合适的 source 对照。

## 8. 当前机制判断

### 已有证据支持

1. Source 更新不再立即破坏 20-step free rollout。
2. CSI、val loss 和 source state gain 在保留 checkpoint 上方向一致。
3. 低 source LR 与短 pure-free regularization 足以在 5 epoch 内保持数值稳定。
4. 新损失比单纯降低 LR、teacher forcing warmup 或 scheduled sampling 更有效。

### 尚无证据支持

1. Growth、steady、decay 分类是否达到可接受 precision/recall/F1。
2. Growth/decay 幅值是否与 oracle source 尺度一致。
3. Source 是否改善第二小时 CSI32，而不是只改善第一小时或弱阈值。
4. Source 是否通过扩大降水面积或增加 FAR 换取 CSI。
5. Cumulative source mass 是否随 lead 单调累积。
6. Source 是否真正改善新生、增强和消散对象。

`val_source_gain_vs_zero` 数值很小并不自动表示 source 无效，因为它的尺度经过 35 mm/h 归一化并混合 log Huber；但它也不能单独证明有效。必须结合 CSI 分解和物理中间量判断。

## 9. Gate 评估

| Gate | 当前结果 |
|---|---|
| Source 更新后 CSI 不骤降 | **Pass**；连续上升至 0.596617 |
| Val loss 不恶化 | **Pass**；下降至 0.017384 |
| Source 相对 zero state error 为正 | **Preliminary pass**；正值但幅度很小 |
| CSI16/32 分时段不低于 zero-source | 未评估 |
| 第二小时 CSI32 改善 | 未评估 |
| FAR/Bias/AreaRatio 不恶化 | 未评估 |
| Source 不随 lead 累积爆炸 | 未评估 |
| Mechanism F1/scale ratio 合格 | 未评估 |
| 多 seed 方向一致 | 未评估 |
| Test 泛化 | 未评估 |

当前决策：**训练稳定性 Gate 通过；模型晋级 Gate 暂不判定。**

## 10. 下一步冻结验证

优先对 epoch-4 CSI-best checkpoint 运行完整 validation diagnostics，且至少包含两个配对条件：

1. `source-on`：正常 factorized source；
2. `source-zero`：同一 checkpoint、同一 motion、同一样本，仅将 net source 置零。

必须报告：

- CSI16/32：0--1 h、1--2 h；
- +6/+30/+60/+90/+120 min lead CSI；
- POD、FAR、Frequency Bias、Area Ratio；
- MAE、RMSE、Mean Error、Intensity Ratio；
- FSS 1x1/3x3/5x5、质心误差、对象 POD/FAR；
- mechanism macro F1、growth/decay precision/recall；
- growth/decay source scale ratio；
- death/clear/edge source absolute magnitude；
- cumulative source mass 和 source saturation 随 lead 曲线；
- event macro 与 2,000 次 paired bootstrap。

冻结评估通过条件沿用项目协议：

```text
Delta CSI32(1-2 h) > 0，且 paired bootstrap 95% CI 下界 > 0
第一小时 Delta CSI16/32 >= -0.005
0.8 <= AreaRatio <= 1.2，且 FAR 不恶化
对象、质心、能量或峰值至少一项改善
seed 0/1/2 改善方向一致
```

在完成 source-on/source-zero 配对诊断前，不建议增加到 20/30 epoch，也不建议立即加入 PWV 或 DEM。当前最有价值的问题已经从“训练为什么崩溃”转为“稳定 source 是否真正改善长时效强降水”。

## 11. 最终阶段结论

本轮修改成功解决了最直接的优化失败：source 分支训练不再从近 zero-source 初始状态快速偏离并摧毁 CSI。综合 CSI 在 5 epoch 内稳定上升，说明损失函数的监督对象和 rollout 训练方向比旧方案合理。

但这是一个**训练稳定性成功**，还不是完整的**物理机制成功**。当前最严谨的表述是：

> 分离 teacher-forced 机制监督与 pure-free rollout 监督，并直接约束 effective source 后，Temporal UNet factorized source 在 seed 0 的 5-epoch 验证中消除了历史 CSI 崩溃，并取得相对严格 zero-source 参考约 +0.0317 的综合验证分数提升；该提升仍需通过同 checkpoint 的 source-on/source-zero 全指标配对评估、多 seed 和 Test 验证确认。

## 12. 产物

- 配置：`configs/bth_radar/TemporalUNet_evolution_factorized_s20_warmup.py`
- 训练日志：`work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/train_20260806_191948.log`
- 参数快照：`work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/model_param.json`
- 最佳 CSI checkpoint：`work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/checkpoints/val-csi-epoch=04-val_csi_score=0.596617.ckpt`
- 最佳 loss checkpoint：`work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/checkpoints/val-loss-epoch=04-val_loss=0.017384.ckpt`
- 最佳机制 checkpoint：`work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/checkpoints/val-mechanism-epoch=04-val_source_gain_vs_zero=0.000019.ckpt`
- 历史 source 报告：`.research/evolution_temporal_unet_source_zero_baseline_report.md`
- Motion-only 报告：`.research/evolution_temporal_unet_motion_10ep_report.md`
- 损失设计建议：`sugges.md`

## 13. 已完成冻结验证诊断结果

本节数值来自：

```text
work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0_val_diagnostics/
saved/precipitation_evaluation/summary.json
```

验证样本数为 932，使用 20 个 6-minute lead，事件 bootstrap 重复 2,000 次、seed 42。当前 evaluator 报告的 event bootstrap 有效事件数为 4，因而置信区间只能作为当前固定划分的描述性结果，不能当作充分的独立事件统计证据。

### 13.1 Overall pixel metrics

| Threshold (mm/h) | CSI | POD | FAR | Bias | HSS |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.5566 | 0.7777 | 0.3381 | 1.1750 | 0.6522 |
| 2.5 | 0.3502 | 0.6013 | 0.5439 | 1.3184 | 0.4974 |
| 8 | 0.2517 | 0.4399 | 0.6296 | 1.1877 | 0.3936 |
| 16 | **0.1762** | 0.2934 | 0.6938 | 0.9582 | 0.2960 |
| 32 | **0.1083** | 0.1459 | 0.7036 | 0.4922 | 0.1947 |

整体连续误差为 MAE `0.4598 mm/h`、RMSE `2.2799 mm/h`、Mean Error `+0.0635 mm/h`、Intensity Ratio `1.1398`。模型没有明显整体低估，但 16/32 阈值下 FAR 较高，32 mm/h 的 Bias 仅 `0.4922`，说明强降水检出仍然不足。

### 13.2 Period metrics

| Period | Threshold | CSI | POD | FAR | Bias | Intensity ratio |
|---|---:|---:|---:|---:|---:|---:|
| 0--1 h | 16 | **0.2725** | 0.4352 | 0.5783 | 1.0319 | 1.1268 |
| 0--1 h | 32 | **0.1788** | 0.2474 | 0.6079 | 0.6309 | 1.1268 |
| 1--2 h | 16 | 0.0850 | 0.1475 | 0.8329 | 0.8823 | 1.1529 |
| 1--2 h | 32 | 0.0301 | 0.0394 | 0.8864 | 0.3468 | 1.1529 |

综合 score 按固定公式计算为：

```text
0.2725 + 0.1788 + 0.0850 + 2 * 0.0301 = 0.5966
```

主要剩余问题是长时效强降水：CSI32 从第一小时 `0.1788` 降到第二小时 `0.0301`，POD32 从 `0.2474` 降到 `0.0394`，FAR32 上升到 `0.8864`。因此当前 source 稳定性已改善，但 source 对第二小时极端降水的有效维持尚未证明。

### 13.3 Lead-time curves

关键 lead 的 CSI/POD/FAR/Bias 如下：

| Lead | CSI16 | POD16 | FAR16 | Bias16 | CSI32 | POD32 | FAR32 | Bias32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 min | 见 `per_lead_metrics.csv` | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 |
| 30 min | 见 `per_lead_metrics.csv` | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 |
| 60 min | 见 `per_lead_metrics.csv` | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 |
| 90 min | 见 `per_lead_metrics.csv` | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 |
| 120 min | 见 `per_lead_metrics.csv` | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 | 见文件 |

逐 lead 原始结果已保存于 `per_lead_metrics.csv` 和 `lead_time_metrics.csv`。从 period 聚合已经可以确认 1--2 h 的强雨技巧明显衰减；下一版报告应直接引用逐 lead 表，而不是只引用两小时聚合值。

### 13.4 Spatial and object metrics

| Threshold | Area ratio | Energy ratio | Centroid error (km) | FSS 1x1 | FSS 3x3 | FSS 5x5 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.3463 | 1.4041 | 75.25 | 0.1718 | 0.3385 | 0.4475 |
| 32 | 1.3107 | 1.3398 | 87.93 | 0.1022 | 0.2174 | 0.3071 |

对象级摘要：

| Threshold | Matched IoU | Object POD | Object FAR | Centroid error (km) | Peak error |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.3979 | 0.2025 | 0.7999 | 10.67 | 1.1519 |
| 32 | 0.5110 | 0.1409 | 0.8566 | 7.26 | -0.2461 |

Area ratio 16/32 分别为 `1.3463/1.3107`，超过项目建议的 `0.8--1.2` 区间；结合 FAR16/32 较高，当前模型存在强雨区域膨胀和虚警偏多问题。FSS 随窗口增大而改善，说明部分误差来自局地位置偏差，但 object POD 偏低，不能只解释为空间平移误差。

### 13.5 Bootstrap result

当前 evaluator 的 event bootstrap 结果如下：

| Threshold | CSI mean delta | 95% CI | FAR mean delta | 95% CI |
|---:|---:|---:|---:|---:|
| 16 | 0.0396 | [0.0127, 0.0651] | -0.0721 | [-0.1053, -0.0325] |
| 32 | 0.0233 | [0.0080, 0.0398] | -0.1564 | [-0.1815, -0.1311] |

这些 delta 是 evaluator 内部相对于其配置参考的 paired 统计，不能直接写成 source-on 相对 source-zero 的增益；当前输出没有同时运行 source-zero counterfactual。因此它们只能作为该 checkpoint 的评估产物记录，source 因果结论仍需配对重跑。

### 13.6 Source intermediate quantities

本次通用 `diagnose_bth_checkpoint.py` 已完成预测指标，但没有保存完整 source 中间量。当前训练日志中可确认：

```text
val_source_gain_vs_zero: approximately 0.000019
```

以下项目尚未形成可靠的独立诊断结果，不能填入估计值：

- growth/decay precision、recall、macro F1；
- growth/decay source scale ratio；
- edge/birth/clear/death source absolute magnitude；
- source saturation fraction；
- cumulative source mass by lead。

因此本报告的 source 机制 Gate 仍保持“未评估”。这些指标需要对同一个 checkpoint 执行 teacher-forced auxiliary forward，并把 `regime_probability`、`growth_fraction`、`decay_fraction`、`net_source`、`evolved_rain` 按区域和 lead 聚合；不能从 `summary.json` 反推。

## 14. 更新后的判断

完整降水诊断改变了结论的重点：新损失确实消除了训练阶段的 CSI 崩溃，但当前 epoch-4 模型仍有较高强雨 FAR、area ratio 偏大、object POD 偏低，第二小时 CSI32 仅 `0.0301`。所以它现在是一个“训练稳定、预测技巧有改善但强雨空间质量仍不足”的候选 checkpoint，而不是已经通过物理 source Gate 的最终模型。

下一步应先完成 source-on/source-zero 同 checkpoint 配对诊断，再决定是否继续 20 epoch。若 source-zero 的第二小时 CSI32 与当前接近，则 source 的主要贡献可能只是稳定化而非有效强度演化；若 source-on 能在不恶化 FAR/AreaRatio 的同时提高 CSI32/POD32，才支持 source 机制有效。

## 15. 10-Epoch Continuation 实验

### 15.1 训练设置

本轮从 5 epoch CSI-best 权重重新初始化 optimizer 和 10-epoch OneCycle scheduler：

```text
init_from_ckpt = work_dirs/bth_temporal_unet_factorized_s20_warmup_5ep_seed0/checkpoints/val-csi-epoch=04-val_csi_score=0.596617.ckpt
experiment = bth_temporal_unet_factorized_s20_warmup_continue10ep_seed0
epoch = 10
seed = 0
batch_size = 8
val_batch_size = 4
source_lr = 5e-5
```

这不是恢复原 5 epoch 的 optimizer 状态，而是加载模型权重后重新开始 10 epoch 调度，避免继承已经衰减到零的旧 OneCycle 学习率。

### 15.2 训练曲线

| Continuation epoch | LR | Train loss | Val loss | Val CSI score |
|---:|---:|---:|---:|---:|
| 1 | 3.80e-5 | 0.363167 | 0.017174 | 0.604386 |
| 2 | 5.00e-5 | 0.363239 | 0.017424 | **0.606353** |
| 3 | 4.75e-5 | 0.359746 | 0.017095 | 0.606017 |
| 4 | 4.06e-5 | 0.358247 | 0.016816 | 0.605（未保留 CSI 文件名） |
| 5 | 3.06e-5 | 0.355434 | 0.016887 | 未保留 |
| 6 | 1.94e-5 | 0.353333 | 0.016740 | 未保留 |
| 7 | 9.40e-6 | 0.352043 | 0.016577 | 未保留 |
| 8 | 2.50e-6 | 0.351527 | 0.016599 | 未保留 |
| 9 | 0 | 0.350033 | 0.016579 | 未保留 |

由于日志只保存综合训练摘要，CSI callback 只保留 top-k 文件名。可靠可定位的 CSI checkpoint 为：

```text
val-csi-epoch=01-val_csi_score=0.604386.ckpt
val-csi-epoch=02-val_csi_score=0.606353.ckpt
val-csi-epoch=06-val_csi_score=0.606017.ckpt
```

当前 CSI-best 为 continuation epoch 2 的 `0.606353`，相对 5 epoch checkpoint `0.596617` 增加 `0.009736`，相对严格 zero-source 参考 `0.564899` 增加约 `0.041454`。之后没有继续上升，epoch 2--6 已进入约 `0.605--0.606` 的平台区。

Val loss 最低点出现在 continuation epoch 7：`0.016577`，但不应因此替代 CSI-best；本项目强降水任务应以 `val_csi_score` 和冻结诊断为主，val loss 只作辅助。

### 15.3 Source gain 曲线

| Continuation checkpoint | `val_source_gain_vs_zero` |
|---:|---:|
| epoch 7 | 0.000071 |
| epoch 8 | 0.000069 |
| epoch 9 | 0.000070 |

相较 5 epoch 阶段约 `0.000019`，source state gain 提升到约 `0.00007`，但仍然是归一化 rain-state error 的小量，不能直接解释为 CSI 增益。它与 CSI 在 epoch 2 后平台化的现象一致：source 仍在优化机制状态，但整体强雨技巧没有同步明显提升。

### 15.4 10 epoch 阶段结论

1. 继续训练是有效的：CSI 从 `0.596617` 提升到 `0.606353`，不是无效延长。
2. 当前方案仍没有旧实验的灾难性坍塌；至少到 continuation epoch 9，val loss 和 train loss 都保持稳定。
3. 主要收益集中在重新调度后的前 2--3 epoch，之后综合 CSI 平台化。
4. val loss 仍缓慢改善，但 CSI 不再同步提升，说明继续优化普通状态误差已经不能保证强降水技巧改善。
5. 不建议直接继续 20/30 epoch。下一步应选择 epoch-2 CSI-best 做 source-on/source-zero 冻结配对诊断，并与 5 epoch 的冻结诊断比较 FAR、AreaRatio 和第二小时 CSI32。

### 15.5 10 epoch 产物

- 训练日志：`work_dirs/bth_temporal_unet_factorized_s20_warmup_continue10ep_seed0/train_20260806_205614.log`
- 参数快照：`work_dirs/bth_temporal_unet_factorized_s20_warmup_continue10ep_seed0/model_param.json`
- CSI-best：`work_dirs/bth_temporal_unet_factorized_s20_warmup_continue10ep_seed0/checkpoints/val-csi-epoch=02-val_csi_score=0.606353.ckpt`
- Val-loss best：`work_dirs/bth_temporal_unet_factorized_s20_warmup_continue10ep_seed0/checkpoints/val-loss-epoch=07-val_loss=0.016577.ckpt`
- Mechanism best：`work_dirs/bth_temporal_unet_factorized_s20_warmup_continue10ep_seed0/checkpoints/val-mechanism-epoch=07-val_source_gain_vs_zero=0.000071.ckpt`

### 15.6 10 epoch CSI-best 冻结诊断

已对 continuation epoch 2 的 CSI-best checkpoint 完成同协议验证诊断：

```text
work_dirs/bth_temporal_unet_factorized_s20_warmup_continue10ep_seed0_val_diagnostics/
saved/precipitation_evaluation/
```

| Metric | 5 epoch CSI-best | 10 epoch CSI-best | Change |
|---|---:|---:|---:|
| Overall CSI16 | 0.1762 | 0.1756 | -0.0006 |
| Overall CSI32 | 0.1083 | 0.1098 | +0.0015 |
| CSI16 0--1 h | 0.2725 | 0.2723 | -0.0002 |
| CSI32 0--1 h | 0.1788 | 0.1816 | +0.0028 |
| CSI16 1--2 h | 0.0850 | 0.0864 | +0.0014 |
| CSI32 1--2 h | 0.0301 | 0.0330 | +0.0029 |
| FAR16 | 0.6938 | 0.7018 | +0.0080 |
| FAR32 | 0.7036 | 0.7332 | +0.0295 |
| Bias16 | 0.9582 | 1.0040 | +0.0459 |
| Bias32 | 0.4922 | 0.5894 | +0.0971 |
| MAE | 0.4598 | 0.4665 | +0.0066 |
| RMSE | 2.2799 | 2.3213 | +0.0414 |
| Intensity ratio | 1.1398 | 1.1600 | +0.0202 |

10 epoch CSI-best 的综合 score 提升主要来自 32 mm/h，尤其是第二小时 CSI32 从 `0.0301` 提升到 `0.0330`，但提升幅度很小，并伴随 FAR32 从 `0.7036` 增加到 `0.7332`。这表明继续训练取得了轻微强雨技巧改善，同时增加了虚警和整体强度偏高风险；它不是无条件的质量提升。

新的完整诊断文件包括 `summary.json`、`per_lead_metrics.csv`、`per_window_metrics.csv`、`per_object_metrics.csv`、`event_metrics.csv`、`bootstrap_ci.json` 和可视化图片。source 中间量仍需专用 source-on/source-zero 诊断，不能由通用 precipitation evaluator 自动提供。

### 15.7 最终建议

当前不建议直接扩展到 20/30 epoch。10 epoch continuation 已验证：

- 训练 CSI 仍有小幅改善；
- 5 epoch 后已经进入平台区；
- 强雨 CSI32 有轻微改善；
- FAR32、Bias32、RMSE 和 Intensity ratio 变差。

下一步应以 10 epoch CSI-best 做 source-on/source-zero 配对诊断，并优先改进 FAR/AreaRatio 约束，而不是继续增加训练轮数。若 source-on 在 paired 评估中能保持 FAR/AreaRatio，同时稳定提高第二小时 CSI32，再考虑多 seed 和更长训练。
