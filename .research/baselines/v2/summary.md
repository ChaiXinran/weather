目前 V2 指的是：

```text
DirectPhysicsHybrid V2 clean_manifest
```

对应配置：

[DirectPhysicsHybrid_r2d_no_deep_convlstm.py](D:/_Search/AIforScience/Rewritten/origin/OpenSTL/configs/bth_radar/DirectPhysicsHybrid_r2d_no_deep_convlstm.py)

完整报告：

[satge1.md](D:/_Search/AIforScience/Rewritten/origin/OpenSTL/.research/mixed/satge1.md)

## V2 架构

V2 由两部分组成：

```text
历史10帧Radar
   ├─→ 冻结的完整ConvLSTM → 未来20帧direct forecast
   │
   └─→ Temporal U-Net/FPN → motion/source/gate
                              ↓
                       对direct进行局部修正
```

总参数：

```text
总参数       4,809,641
冻结参数     3,745,024  # ConvLSTM
可训练参数   1,064,617  # U-Net物理修正分支
```

与最早7.5M版本不同，V2取消了修正分支最深层的 bottleneck ConvLSTM：

```python
hybrid_temporal_mix_scales = [0, 1, 2, 3]
hybrid_convlstm_scales = []
```

所以 U-Net 修正分支从约3.7M降到约1.06M。

## V2 的预测方式

ConvLSTM 先生成：

\[
D_t
\]

U-Net 根据历史特征、当前时效的 \(D_t\) 和 lead embedding，预测：

- residual flow；
- signed source；
- motion gate；
- source gate。

然后在 rain-rate 空间融合：

\[
\hat R_t
=
D_t
+
g_t^m\left[
W(D_t,\Delta U_t)-D_t
\right]
+
g_t^s S_t
\]

其中门控上限为：

```python
hybrid_motion_alpha_max = 0.5
hybrid_source_alpha_max = 0.25
```

所以它不是完整替换 ConvLSTM，而是：

- motion 最多使用50%的位移候选；
- source 最多使用25%的源汇候选；
- 其余仍由 ConvLSTM direct prediction 决定。

## 当前效果

```text
Weighted validation CSI = 0.937194
```

关键时效：

| Lead | CSI16 | CSI32 | FAR16 | FAR32 |
|---|---:|---:|---:|---:|
| +60 min | 0.21377 | 0.13006 | 0.67048 | 0.82603 |
| +120 min | 0.10706 | 0.04980 | 0.84663 | 0.93726 |

Branch attribution：

```text
Direct ConvLSTM   ≈ 0.93267
Motion-only       ≈ 0.93694
Source-only       ≈ 0.93294
Full V2           = 0.93719
```

因此 V2 的真实结论是：

> 冻结 ConvLSTM 提供绝大多数预测能力；U-Net motion 带来小幅稳定提升；source 几乎没有有效贡献。

## V2 与计划中的 V3a 区别

| 项目 | V2 | V3a |
|---|---|---|
| ConvLSTM | 冻结direct prior | 仍作为冻结prior |
| Motion | residual flow + 最大50% gate | 独立motion candidate，可完全接管 |
| Source | signed growth/decay混合 | 删除 |
| Decay | 隐含在signed source中 | 独立decay expert |
| Growth | 与decay混在source里 | 暂不加入 |
| Routing | motion/source分别独立开关 | preserve/motion/decay三分类路由 |
| 融合 | 两个residual相加 | 三个候选soft选择 |
| U-Net作用 | 小幅校正 | 错误区域强修正 |

一句话概括：

> V2 是“ConvLSTM 主预测 + U-Net 小幅 motion/source 修正”；V3a 将变成“ConvLSTM 提供 prior，U-Net 判断错误类型并选择 preserve、motion 或 decay 强力修正”。