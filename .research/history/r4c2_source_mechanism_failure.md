# R4-c2 源汇机制验证失败报告

日期：2026-08-05  
结论：Gate 0 工程检查通过；R4-c2a 与 R4-c2b 均未通过 Gate 1；停止训练，不进入 R4-c2c、scheduled sampling、多 seed 或 test。

## 1. 研究问题

本阶段检验：在冻结且已校准的 R4-b Radar 运动场下，逐步状态条件化、物理有界
的 signed tendency 模块，能否预测可辨识的非平流强度变化，并在不依赖边缘位移
补偿、不破坏运动分支空间优势的前提下改善 16/32 mm/h 降水的增强、维持、减弱、
新生和消散。

这不是图像锐化实验。source 可以恢复部分强核，但不能只通过扩大强回波、累积
正增量或修补 flow 误差获得表面上的 POD 改善。

## 2. 实现与固定协议

- R4-b 初始化：
  `work_dirs/bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0/checkpoints/val-csi-epoch=01-val_csi_score=0.640662.ckpt`。
- encoder、motion head 和 0.5 flow 标定冻结；optimizer 仅包含 source decoder 与
  lead embedding。
- 10 帧输入、20 帧输出、6 min/帧；事件级 train/val/test 划分不变。
- 训练和本报告只使用 train/val，未访问 test。
- source 与 state loss 均在 rain-rate 空间计算，固定 `Z=200R^1.6`。
- R4-c2a 输入：`F, A_t, v_t, e_t`。
- R4-c2b 仅增加 `R0-R-1` 与 `R0-R-5`。
- 两者均训练 3 epoch、batch 8、seed 0、teacher forcing；验证预报为完整 20-step
  rollout。

source 参数化为：

```text
u_t = tanh(z_t)
C_t+ = min(35, max(R_max - A_t, 0))
S_t = C_t+ * u_t,  u_t >= 0
S_t = A_t * u_t,   u_t < 0
R_t = A_t + S_t
```

其中 `R_max ~= 48.6 mm/h` 对应 50 dBZ。该实现同时限制 sink 和正 source，不依赖
结果端 clamp 隐藏超限预测。

## 3. Gate 0 结果

17 项定向检查全部通过，结果保存在：

- `work_dirs/r4c2_gate0/gate0_checks.json`
- `work_dirs/r4c2_gate0/c2a_overfit.json`
- `work_dirs/r4c2_gate0/c2b_overfit.json`

已确认：

- 零初始化 c2 source 与 R4-b 逐像素一致；
- sink 不超过 `A_t`，正 source 不超过剩余可表示容量；
- `0 <= evolved_rain <= R_max`；
- 正负分支都有有限非零梯度；
- teacher-forced 时间索引正确；
- c2 decoder 逐 lead 共享，checkpoint 加载与配置完整；
- source-only optimizer 仅包含约 8 万个新增参数。

真实固定 batch 上，c2a/c2b 总 loss 分别下降 87.44%/87.54%，证明训练链路和
decoder 容量不是完全失效。但两者均出现约 98--99% source 像素为负、growth
符号准确率约 2--3% 的偏置；这是后续正式实验失败的早期信号。固定 batch 中只有
7 个 32 mm/h 像素，强区损失主导了总 loss，active loss 反而上升，因此“总 loss
下降”没有被当作机制通过证据。

## 4. 正式验证结果

### 4.1 按 epoch 的综合结果

| 实验 | Epoch | val CSI score | val loss | 第一小时强度比 | 第二小时强度比 |
|---|---:|---:|---:|---:|---:|
| c2a | 0 | **0.267329** | **0.050259** | 3.027 | 7.789 |
| c2a | 1 | 0.219643 | 0.082078 | 3.362 | 12.970 |
| c2a | 2 | 0.207709 | 0.089826 | 3.497 | 14.494 |
| c2b | 0 | 0.166444 | 0.126563 | 4.528 | 18.786 |
| c2b | 1 | **0.278036** | **0.055084** | 2.563 | 8.889 |
| c2b | 2 | 0.243601 | 0.068181 | 2.966 | 11.865 |

R4-b 固定 checkpoint 的 `val_csi_score=0.640662`。c2a/c2b 的最佳值仅为
0.267329/0.278036，且继续训练时 source 累积和指标恶化。历史 tendency 只带来
很小的最佳分数变化，没有改变失败性质。

### 4.2 最佳 epoch 的 CSI/POD/FAR

| 实验 | 时段 | 阈值 | CSI | POD | FAR |
|---|---|---:|---:|---:|---:|
| c2a e0 | 0--1 h | 16 | 0.1316 | 0.8276 | 0.8646 |
| c2a e0 | 0--1 h | 32 | 0.0618 | 0.7288 | 0.9368 |
| c2a e0 | 1--2 h | 16 | 0.0429 | 0.7235 | 0.9563 |
| c2a e0 | 1--2 h | 32 | 0.0155 | 0.6235 | 0.9844 |
| c2b e1 | 0--1 h | 16 | 0.1421 | 0.7801 | 0.8520 |
| c2b e1 | 0--1 h | 32 | 0.0702 | 0.6571 | 0.9271 |
| c2b e1 | 1--2 h | 16 | 0.0383 | 0.7557 | 0.9612 |
| c2b e1 | 1--2 h | 32 | 0.0137 | 0.6529 | 0.9862 |

POD 并未塌缩；CSI 低主要是 FAR 极高。模型通过制造过多强回波维持了较高 POD，
而不是准确恢复目标强核。

### 4.3 c2a 完整 val-only 评估

c2a 最佳 checkpoint 已完成独立 val-only 评估，产物位于：

`work_dirs/bth_r4c2a_bounded_state_tf_seed0/validation_best_csi/`

整体结果：

| MAE | RMSE | Intensity ratio | CSI16 | POD16 | FAR16 | CSI32 | POD32 | FAR32 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.2576 | 8.6882 | 5.3905 | 0.0675 | 0.7763 | 0.9311 | 0.0264 | 0.6774 | 0.9733 |

窗口级 FSS 均值：

| 阈值 | FSS 1x1 | FSS 3x3 | FSS 5x5 |
|---:|---:|---:|---:|
| 16 mm/h | 0.1840 | 0.2273 | 0.2380 |
| 32 mm/h | 0.1140 | 0.1400 | 0.1470 |

以下文件已经留存：

- `precipitation_evaluation/summary.json`
- `precipitation_evaluation/per_lead_metrics.csv`
- `precipitation_evaluation/per_event_metrics.csv`
- `precipitation_evaluation/per_window_metrics.csv`（含 FSS 1/3/5）
- `precipitation_evaluation/confusion_counts.csv`
- `precipitation_evaluation/bootstrap_ci.json`
- `precipitation_evaluation/lead_time_curves.png`
- `precipitation_evaluation/psd_comparison.png`
- `source_diagnostics.json`

c2b 的完整 FSS 评估按用户指令中止，未形成完整评估目录；其 CSI/POD/FAR、面积、
强度和 teacher-forced source 指标已完整保存在
`lightning_logs/version_46/metrics.csv`。不再为失败模型追加计算。

## 5. Source 机制诊断

### 5.1 量级不再塌缩，但符号不可辨识

| 实验/epoch | active abs | 16 abs | 32 abs | growth sign | decay sign | 正像素比例 |
|---|---:|---:|---:|---:|---:|---:|
| c2a e0 | 0.806 | 5.366 | 4.858 | 0.276 | 0.695 | 0.0431 |
| c2a e2 | 0.837 | 6.040 | 5.704 | 0.265 | 0.748 | 0.0330 |
| c2b e1 | 0.759 | 5.519 | 5.297 | 0.247 | 0.762 | 0.0279 |
| c2b e2 | 0.830 | 5.958 | 5.647 | 0.269 | 0.748 | 0.0309 |

R4-c1 的 source 约为 `0.003--0.006 mm/h`；R4-c2 已成功把 source 提升到有意义
量级，因此这次失败不是“source 仍为零”。问题变成了量级与符号/空间分布失衡：
decoder 主要预测微弱负值，同时在少量强区施加较大正值。

### 5.2 c2a 空间分区

c2a 最佳 checkpoint 的 val teacher-forced source：

| 区域 | abs mean mm/h | signed mean mm/h | 正比例 | 负比例 |
|---|---:|---:|---:|---:|
| persistent interior | 0.9430 | +0.7015 | 0.2479 | 0.7521 |
| object edge band | 0.2862 | +0.2231 | 0.0651 | 0.9349 |
| newborn | 0.0929 | +0.0697 | 0.0290 | 0.9710 |
| dissipated | 0.2593 | +0.2098 | 0.0668 | 0.9332 |
| clear background | 0.0193 | +0.0103 | 0.0157 | 0.9843 |
| growth | 1.0446 | +0.8220 | 0.2523 | 0.7477 |
| decay | 1.1347 | +0.9168 | 0.2707 | 0.7293 |
| >=16 mm/h | 4.7997 | +4.7597 | 0.9576 | 0.0423 |
| >=32 mm/h | 4.5412 | +4.5222 | 0.9854 | 0.0144 |

这里“多数像素为负但 signed mean 为正”说明负值数量多但幅度很小，少数正值幅度
很大。特别是 16/32 区几乎统一加雨，而 growth 与 decay 的正负比例很接近，证明
decoder 更像“强区增强器”，没有可靠地区分真实增长和衰减。newborn 区也有 97.1%
像素为负，不支持它已学到新生机制。

物理边界本身工作正常：86,116,800 个 val teacher-forced 像素中，`R>R_max`、
source 正上限饱和和 sink 清空计数均为 0；normalized dBZ 达到上界仅 27 个
（`3.14e-7`）。失败不是由隐藏 clamp 或上下界饱和造成。

## 6. CSI 异常的直接因果链

```text
teacher forcing 始终提供真实上一帧
        ↓
decoder 只在真实状态分布上学习单步修正
        ↓
损失鼓励少量 16/32 强区施加较大正 source
        ↓
自由 rollout 中正增量被平流保留，并在后续 lead 再次增强
        ↓
强回波面积和总强度逐步膨胀
        ↓
POD 较高，但 false alarms 激增
        ↓
FAR 接近 1，CSI 显著下降
```

因此 CSI 异常是模型行为，不是指标实现错误。CSI、POD、FAR 的关系在混淆计数和
逐 lead 曲线中一致，且 c2a/c2b 两次独立训练方向相同。

## 7. 可能原因

按证据强弱排序：

1. **teacher-forced 状态分布与 rollout 不一致。** decoder 从未在自身膨胀后的
   `A_t` 上学习停止增强，单步合理增量在 20 步递归中累积。
2. **区域损失被稀少强像素主导。** `L_active + 0.5 L16 + 0.5 L32` 对每个区域分别
   求均值；少量 32 mm/h 像素获得与整个 active 区同量级的梯度贡献，容易形成
   “只要强就加”的策略。
3. **growth/decay 不可辨识。** c2a 和 c2b 的 growth/decay 符号准确率均明显不对称；
   tendency 输入没有带来实质改进，说明当前冻结特征、输入尺度或 decoder 结构没有
   提供足够条件区分真实增强和衰减。
4. **signed 分段在零点的梯度尺度不对称。** 零初始化时正分支梯度受正容量控制，
   负分支受 `A_t` 控制；弱雨区两者差异大，可能促成“少量大正值、多数小负值”。
5. **oracle residual 混合多种误差。** 尽管它没有作为主回归标签，growth/decay
   诊断仍包含 flow 剩余误差、形态变化和插值误差，Radar-only 输入未必足以预测。
6. **source decoder 可能利用强度捷径。** 强区几乎统一正 source，而 newborn 和
   growth 没有正确符号，说明 `A_t` 强度比历史变化信息更容易被利用。

## 8. 已排除或暂不支持的原因

- 不是 source 参数未进入 optimizer：真实 batch 可过拟合，正式 source 量级显著。
- 不是 encoder/motion 被联合更新：source-only optimizer 且两者冻结。
- 不是 R4-b checkpoint 加载失败：24 个 encoder/motion tensor 完整匹配。
- 不是下界 clamp 隐藏 sink：c2 bounded 路径不依赖结果端 clamp。
- 不是上界裁剪主导：超 `R_max` 和 source 饱和计数为 0。
- 不是简单缺少历史 tendency：c2b 未改变失败方向。
- 当前没有证据支持继续增加 epoch；两组实验在后期 rollout 指标均恶化。

## 9. Gate 决策

R4-c2a 和 R4-c2b 均未满足以下 Gate 1 条件：

- growth 区以正 source 为主、decay 区以负 source 为主；
- source 不通过统一强区增雨恢复 POD；
- active/16/32 改善不伴随严重 FAR、面积与强度膨胀；
- 自由 rollout 保持 R4-b 的空间优势。

决定：**No-go**。

- 不运行 R4-c2c scheduled sampling。
- 不解冻 motion。
- 不加入 PWV/DEM。
- 不访问 test，不运行多 seed。
- 不继续增加 c2a/c2b epoch 或学习率。

## 10. 后续若重启该方向

下一轮不应直接把当前 c2b 接入 rollout。应先设计更小的机制实验：

- 重新平衡 state loss，使 active、16、32 的贡献按像素支持度或有效样本数受控，
  避免极少强像素与整个 active 区等权；
- 在 train-only 数据上预注册 growth/decay 强区采样，检查零点第一次更新的方向；
- 对正增量加入与状态容量、持续时间或对象内部一致性有关的约束，而不是 L1/TV；
- 单独验证 decoder 是否能区分 matched persistent interior 的 growth 与 decay，
  暂时排除 edge/newborn/flow 误差混合；
- 只有 teacher-forced growth/decay 两侧都可辨识后，才讨论短 horizon 的离散
  Bernoulli rollout；不能用 scheduled sampling 掩盖当前单步机制失败。

这些是下一轮研究假设，不是本次已验证修复。本阶段的有效结论是：**逐步状态条件化
和有界参数化解决了 R4-c1 的零 source 问题，但当前损失与 Radar-only 条件使模型
退化为强区增雨/弱背景减雨策略，导致自由 rollout 强度与面积爆炸，Gate 1 失败。**
