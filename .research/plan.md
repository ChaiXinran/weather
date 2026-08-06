# 现阶段的核心目标

目前先不要追求“完整 20 帧预测的 CSI 立刻提高”，而应先回答三个机制问题：

1. **已有降水内部的增强与减弱能否被 Radar 历史辨识；**
2. **边缘形变究竟应由运动项还是源汇项承担；**
3. **单步正确的源汇机制能否在递归过程中保持稳定。**

当前 R4-b 运动基线已经证明平流骨架可用，但第二小时明显丢失强度：强度比从第一小时的 0.8915 降到第二小时的 0.7376，CSI@32 在 1–2 h 只有 0.0092。因此确实需要非平流强度演化机制，但它必须提高 POD 和强度保持，同时避免再次制造大面积虚警。

R4-c2 的问题也已经很明确：不是 source 太小，而是模型学成了“强区统一加雨”。16/32 mm/h 区域几乎全部得到正 source，growth 和 decay 的符号分布却非常接近，完整 rollout 强度比膨胀到 5 倍以上，FAR 接近 1。

因此，下一阶段不应该继续训练现有 signed source，而应该进入一个新的、严格分阶段的 **Radar-only 机制识别阶段**。

---

# 一、固定实验协议

后续所有机制实验固定：

| 项目       | 固定设置                         |
| -------- | ---------------------------- |
| 数据       | 仅 Radar                      |
| 输入输出     | 10→20，6 min/帧                |
| 数据划分     | 当前事件级 Train/Val/Test         |
| 开发阶段     | 只使用 Train/Val，不访问 Test       |
| 运动骨架     | 冻结 R4-b `.640662` checkpoint |
| 演化空间     | rain-rate 空间                 |
| Z–R      | 固定 (Z=200R^{1.6})            |
| 第一阶段预测长度 | 只预测 1 帧                      |
| 优化参数     | 只训练新增机制分支                    |
| 判断依据     | 中间机制指标优先于总体 CSI              |

当前代码已经支持 rain-rate 平流、正负有界 source、零 source 与 R4-b 完全兼容，因此不需要重写整个 evolution operator，只需要新增更合理的 source 参数化。

仓库里的 `ConvLSTM_evolution_source_s1_pixel_weighted.py` 已经冻结 encoder 和 motion，只执行第一步预测，是下一轮实验最合适的配置起点。

---

# 二、R4-d0：先重新构造“机制标签”

## 1. 计算 teacher-forced 物理残差

保持 R4-b 运动场不变：

[
A_t=\mathcal W(R_{t-1}^{true},v_t^{R4b})
]

其中：

* (R_{t-1}^{true})：真实上一帧；
* (v_t^{R4b})：冻结的运动场；
* (A_t)：只考虑平流后的降水场。

定义 oracle source：

[
S_t^*=R_t^{true}-A_t
]

但这次不能把全部 (S_t^*) 当作同一种 source 标签，因为它混合了：

* 降水内部增强；
* 降水内部减弱；
* 对象边缘形变；
* 对流新生；
* 对流消散；
* 剩余运动误差；
* 双线性插值误差。

## 2. 对空间区域进行物理分区

令：

[
M_A=A_t\geq 0.1
]

[
M_Y=R_t^{true}\geq 0.1
]

构造五个互斥区域。

### Persistent interior：持续存在对象内部

[
M_{\text{interior}}
===================

\operatorname{Erode}(M_A\cap M_Y,1)
]

这是第一阶段最重要的区域。对象内部受边缘位移误差影响较小，比较适合检验真正的强度增强和减弱。

### Edge band：对象边缘

[
M_{\text{edge}}
===============

## \operatorname{Dilate}(M_A\cup M_Y,1)

M_{\text{interior}}
]

这里暂时不训练 source。

### Newborn：新生区域

[
M_{\text{birth}}=(\neg M_A)\cap M_Y
]

### Dissipated：消散区域

[
M_{\text{death}}=M_A\cap(\neg M_Y)
]

### Clear background：持续无降水

[
M_{\text{clear}}=(\neg M_A)\cap(\neg M_Y)
]

## 3. 只在 persistent interior 中定义三分类

建议主阈值先设为：

[
\delta=0.5\ \mathrm{mm/h}
]

定义：

[
y=
\begin{cases}
\text{growth},&S_t^*>\delta\
\text{steady},&|S_t^*|\leq\delta\
\text{decay},&S_t^*<-\delta
\end{cases}
]

同时使用 (\delta=0.25) 和 (1.0) mm/h 做敏感性分析，但不因此重复完整训练。

选择 0.5 而不是当前代码中的 0.1，是因为太靠近零点时，插值误差、Z–R 非线性和微小运动误差都会被错误地标记为增强或减弱。

## 4. R4-d0 应输出的报告

新增：

```text
tools/diagnostics/r4d0_partition_oracle_source.py
```

输出：

```text
.research/r4d0_source_partition/
├── region_counts.csv
├── regime_counts.csv
├── source_by_region.csv
├── source_by_intensity.csv
├── source_by_lead.csv
├── source_histograms.png
├── region_examples/
└── summary.md
```

至少统计：

* growth、steady、decay 的样本比例；
* 各区域 oracle source 的均值、绝对值均值、P50/P90/P95/P99；
* 按 0.1–8、8–16、16–32、≥32 mm/h 分层；
* persistent interior 中正负 source 的空间连续性；
* edge 区 oracle residual 是否显著大于 interior；
* newborn 是否主要集中在已有对象附近。

这一阶段不训练模型。

---

# 三、R4-d1：已有降水的“增强—维持—减弱”分解

这是当前最应该优先实现的机制。

## 1. 不再输出一个 signed tendency

当前实现是：

[
u_t=\tanh(z_t)
]

[
S_t=
\begin{cases}
C_t^+u_t,&u_t\geq0\
A_tu_t,&u_t<0
\end{cases}
]

虽然物理上下界正确，但正负分支在零点附近的尺度不同，而且一个连续变量同时承担“判断方向”和“预测幅值”，很容易退化成当前的强区增雨策略。当前报告也已经将这种零点梯度不对称列为可能原因。

建议拆成：

### 方向分类

[
[p_t^+,p_t^0,p_t^-]
===================

\operatorname{Softmax}(z_t^+,z_t^0,z_t^-)
]

对应：

* (p_t^+)：增强概率；
* (p_t^0)：维持概率；
* (p_t^-)：减弱概率。

### 增强和减弱幅度

[
\alpha_t^+=\sigma(g_t),\qquad
\alpha_t^-=\sigma(d_t)
]

构造三个候选状态：

[
R_t^{growth}
============

A_t+\alpha_t^+C_t^+
]

[
R_t^{steady}=A_t
]

[
R_t^{decay}
===========

(1-\alpha_t^-)A_t
]

最终状态：

[
\hat R_t
========

p_t^+R_t^{growth}
+
p_t^0R_t^{steady}
+
p_t^-R_t^{decay}
]

净 source 为：

[
\hat S_t=\hat R_t-A_t
]

这种写法的优势是：

* 增强、维持、减弱显式竞争；
* source 方向有明确概率解释；
* 减弱永远不会使降水变成负数；
* 即使分类不确定，输出仍是三个物理候选状态的凸组合；
* 可以分别评价方向和幅值，不再只有一个模糊的 source loss。

## 2. 增强上限不能继续直接使用全局 35 mm/h

当前 `source_max_rain=35` 对单个 6 min 步长过于宽松，使模型很容易在强区施加大幅正 source。

建议从 Train oracle source 中按 advected intensity 分箱：

| (A_t) | 正 source 上限         |
| ----- | ------------------- |
| 0.1–8 | 该分箱正 source P95/P99 |
| 8–16  | 该分箱正 source P95/P99 |
| 16–32 | 该分箱正 source P95/P99 |
| ≥32   | 该分箱正 source P95/P99 |

定义：

[
C_t^+
=====

\min
\left(
Q^+*{\operatorname{bin}(A_t)},
R*{\max}-A_t
\right)
]

其中 (Q^+) 只从 Train 统计，Val 仅验证。

这样仍允许强降水出现大 source，但模型不能把所有像素都推向全局最大值。

## 3. source 只作用于 persistent interior

第一轮：

[
\hat S_t=
M_{\text{interior}}\hat S_t
]

在 edge、newborn、clear background 中强制：

[
\hat S_t=0
]

这是必要的机制隔离，不是最终模型设计。

当前 R4-c2 直接在所有 active 区域学习 state loss，导致 edge、birth 和运动误差混在一起；失败报告已经明确建议先单独检验 matched persistent interior 的 growth/decay。

## 4. 模型输入

R4-d1a 只使用：

[
[F,\ A_t,\ v_t,\ |\nabla A_t|]
]

其中：

* (F)：历史 Radar 的 ConvLSTM 特征；
* (A_t)：平流后状态；
* (v_t)：冻结运动场；
* (|\nabla A_t|)：帮助模型识别对象内部和边缘。

暂时不要再加入：

[
R_0-R_{-1},\qquad R_0-R_{-5}
]

因为 c2b 已经证明原始差分没有改变失败性质。原始差分包含大量位置移动，不是纯强度 tendency。

## 5. 损失函数

### 方向分类损失

只在 persistent interior：

[
L_{\text{regime}}
=================

L_{\text{balanced-CE}}
(\hat y,y)
]

三类采用逆频率或 effective-number 权重。

每个 batch 中，growth、steady、decay 像素可以等量抽样计算分类损失，避免 steady 或 decay 主导。

### 条件幅值损失

只在真实 growth 像素监督 (\alpha^+)：

[
\alpha_t^{+,*}
==============

\frac{S_t^*}{C_t^+}
]

只在真实 decay 像素监督 (\alpha^-)：

[
\alpha_t^{-,*}
==============

\frac{-S_t^*}{A_t}
]

[
L_{\text{magnitude}}
====================

L_{\text{Huber}}(\alpha_t^+,\alpha_t^{+,*})
+
L_{\text{Huber}}(\alpha_t^-,\alpha_t^{-,*})
]

### 状态重建损失

[
L_{\text{state}}
================

L_{\text{Huber}}(\hat R_t,R_t^{true})
]

继续使用当前 1/2/3 的像素权重：

* active：1；
* ≥16 mm/h：2；
* ≥32 mm/h：3。

当前代码已经实现了这种 capped pixel-weighted state loss，可以直接复用。

### 总损失

初始建议：

[
L
=

L_{\text{regime}}
+
L_{\text{magnitude}}
+
0.2L_{\text{state}}
]

第一轮不要加：

* source L1；
* source TV；
* soft CSI；
* 完整 20-step forecast loss。

因为当前首先要验证的是方向可辨识性，不是通过正则把 source 压小或压平。

## 6. R4-d1 的通过条件

这些是建议预注册的项目 Gate，不是领域通用标准：

| 指标                   |                       通过标准 |
| -------------------- | -------------------------: |
| Growth sign accuracy |                      ≥0.60 |
| Decay sign accuracy  |                      ≥0.60 |
| Regime macro-F1      |              比多数类基线高 ≥0.10 |
| Growth scale ratio   |                    0.5–1.5 |
| Decay scale ratio    |                    0.5–1.5 |
| Interior state MAE   |       比 zero-source 降低 ≥5% |
| ≥16/32 区统一正 source   |                    不允许再次出现 |
| Val loss             | 应和 mechanism metrics 同方向改善 |

如果 growth 仍约 0.25、decay 约 0.75，即使 state loss 降低，也判定失败。

---

# 四、R4-d2：边缘形变与强度源汇分离

R4-d1 通过后，下一步不是立刻加入 newborn，而是处理对象边缘。

## 1. 为什么边缘应单独处理

对象边缘上的 oracle source 可能只是：

* flow 偏移半个像素；
* 对象形状伸展；
* 双线性平流造成的平滑；
* 真实强度生消。

如果让 intensity source 修复所有边缘误差，source 会自然学成“哪里位置没对齐，就加雨或减雨”。

## 2. 加入 bounded local deformation

保留 R4-b 粗运动：

[
v_t^{base}
]

新增一个只在 edge band 生效的局地运动残差：

[
\delta v_t
==========

0.25\tanh(f_{\text{edge}}(F,A_t,\nabla A_t))
]

[
v_t^{final}
===========

v_t^{base}
+
M_{\text{edge}}\delta v_t
]

每个 6 min 步长，局地残差限制为约 (\pm0.25) 像素，避免它取代原有运动场。

新的平流状态：

[
A_t'
====

\mathcal W(R_{t-1},v_t^{final})
]

## 3. 训练方式

仍然只做单步：

* encoder 冻结；
* base motion 冻结；
* source 分支关闭；
* 只训练 edge deformation head。

损失：

[
L_{\text{edge}}
===============

L_{\text{Huber}}(A_t',R_t^{true};M_{\text{edge}})
+
\lambda_g L_{\text{gradient}}
+
\lambda_s L_{\text{smooth}}
]

其中：

[
L_{\text{smooth}}
=================

|\nabla\delta v_t|_1
]

## 4. R4-d2 Gate

| 指标                      |     建议条件 |   |                |
| ----------------------- | -------: | - | -------------- |
| Edge-band transport MAE |   降低 ≥5% |   |                |
| Edge-band oracle (      |      S^* | ) | 降低 ≥10%        |
| 对象质心误差                  |      不恶化 |   |                |
| 对象面积比                   |  0.9–1.1 |   |                |
| Persistent interior MAE | 不恶化超过 1% |   |                |
| (                       | \delta v | ) | 不长期饱和在 0.25 px |

如果边缘 residual 没有明显下降，就不保留这个分支。

---

# 五、R4-d3：短时自由递归稳定性

只有 R4-d1 单步源汇通过后，才进入递归。

## 1. 不直接从 1 步跳到 20 步

依次进行：

[
1\rightarrow3\rightarrow5\rightarrow10\rightarrow20
]

其中：

* 1 步：纯 teacher-forced 机制识别；
* 3 步：全部自由 rollout；
* 5 步：全部自由 rollout；
* 10/20 步：只有短 rollout 通过后才运行。

3-step 训练时：

[
\hat R_1=f(R_0)
]

[
\hat R_2=f(\hat R_1)
]

[
\hat R_3=f(\hat R_2)
]

不能再让每一步都看到真实上一帧。

当前失败的直接因果链就是：teacher forcing 中学习单步强区加雨，进入自由 rollout 后正 source 被持续平流并反复增强，最终造成面积和强度爆炸。

## 2. 历史强度包络机制

如果 factorized source 单步正确，但 3-step 仍持续累积，可以加入 Radar-only 的历史强度包络。

构造：

[
H_0
===

\max_{\tau=-4,\dots,0}
\operatorname{MaxPool}*{3\times3}(R*\tau)
]

用当前运动场把该包络平流至未来：

[
H_t=\mathcal W(H_{t-1},v_t)
]

增长候选状态不再直接趋向全局 (R_{\max})，而是：

[
C_t
===

\min(R_{\max},H_t+\Delta_t)
]

[
R_t^{growth}
============

A_t+\alpha_t^+(C_t-A_t)_+
]

其中 (\Delta_t) 为小范围可学习的额外增长容量。

物理含义是：

> 历史雷达已经观测到的局地强度结构给出一个移动的强度背景，source 允许在此基础上增强，但不能每一步无条件向全局最大雨强推进。

## 3. 递归 Gate

每个阶段都检查：

| 指标                |            建议条件 |
| ----------------- | --------------: |
| Intensity ratio   |       0.85–1.15 |
| 强降水面积比            |         0.8–1.2 |
| FAR@16/32 增量      |           ≤0.03 |
| CSI@16/32         |     不低于同长度 R4-b |
| Growth/decay sign | 不因 rollout 明显失效 |
| 连续正 source        |    不随 lead 单调积累 |
| source saturation |            接近 0 |

3-step 不通过，就不进入 5-step，更不能使用 scheduled sampling 掩盖问题。

---

# 六、R4-d4：雷达可见的新生机制

这一部分放在已有对象的增强/减弱通过之后。

## 1. 不尝试预测完全无雷达前兆的对流新生

只使用 Radar 时，对完全晴空中突然产生的对流缺乏环境信息。因此应把研究对象限定为：

> **Radar-visible initiation：历史弱回波、局地回波增长或已有对象附近的新生。**

候选区域可定义为：

[
M_{\text{candidate}}
====================

(M_{\text{history-max}}\ge0.1)
\cup
\operatorname{Dilate}(M_A,2)
]

其中：

[
M_{\text{history-max}}
======================

\max_{\tau=-9,\dots,0}R_\tau
]

## 2. 单独的新生门控

[
p_t^{birth}
===========

\sigma(f_{\text{birth}}(F,A_t))
]

[
I_t^{birth}
===========

\alpha_t^{birth}I_{\max}
]

[
R_t
===

R_t^{existing}
+
(1-M_A)
M_{\text{candidate}}
p_t^{birth}
I_t^{birth}
]

不允许 existing growth head 在无降水背景产生新回波。

## 3. 评价指标

新生属于稀有事件，不能只看准确率，应报告：

* PR-AUC/AP；
* newborn POD；
* newborn FAR；
* newborn FSS；
* 新生对象质心误差；
* 新生面积比；
* clear-background false birth 面积。

如果 AP 没有显著超过事件 prevalence，就说明 Radar-only 中缺乏足够的新生信息，不应强行保留这个模块。

---

# 七、建议的具体实验矩阵

| 编号     | 实验                             |  预测长度 | 冻结部分                           | 目的                 |
| ------ | ------------------------------ | ----: | ------------------------------ | ------------------ |
| R4-d0  | Oracle source 空间分区             |   无训练 | 全部                             | 清理机制标签             |
| R4-d1a | Factorized growth/steady/decay |     1 | encoder+motion                 | 验证方向可辨识性           |
| R4-d1b | 分箱 source capacity             |     1 | encoder+motion                 | 防止全局强区增雨捷径         |
| R4-d2  | Edge residual flow             |     1 | encoder+base motion            | 分离形变与强度            |
| R4-d3a | Existing-cell source rollout   |     3 | encoder+motion                 | 检验递归稳定性            |
| R4-d3b | Historical intensity envelope  |   3/5 | encoder+motion                 | 抑制正 source 累积      |
| R4-d4  | Radar-visible initiation       |     1 | encoder+motion+existing source | 检验新生机制             |
| R4-d5  | 完整组合                           | 10/20 | 先冻结，后谨慎联合                      | 最终 Radar-only 机制模型 |

当前真正应该执行的只有：

[
\boxed{
R4\text{-}d0
\rightarrow
R4\text{-}d1a
\rightarrow
R4\text{-}d1b
}
]

R4-d1 不通过时，不做后续实验。

---

# 八、代码层面的修改方案

## 1. `openstl/modules/evolution_operator.py`

新增：

```python
def evolve_factorized_step(
    field,
    flow,
    regime_logits,
    growth_fraction,
    decay_fraction,
    positive_capacity,
):
    ...
```

返回：

```python
{
    "prediction": prediction,
    "advected_rain": advected_rain,
    "regime_probability": regime_probability,
    "growth_state": growth_state,
    "steady_state": advected_rain,
    "decay_state": decay_state,
    "growth_source": growth_source,
    "sink": sink,
    "net_source": net_source,
    "evolved_rain": evolved_rain,
}
```

必须保留现有：

* source-free；
* legacy signed；
* bounded-state

三个路径，用于回归测试。

## 2. `openstl/models/evolution_convlstm_model.py`

新增参数：

```python
evolution_source_parameterization = "factorized_regime"
```

新增三个 head：

```text
regime_head       -> 3 channels
growth_head       -> 1 channel
decay_head        -> 1 channel
```

后续再增加：

```text
edge_flow_head
birth_gate_head
birth_intensity_head
```

不要一次全部实现。

## 3. `openstl/methods/evolution_convlstm.py`

新增：

```python
_build_physical_region_masks()
_build_regime_labels()
_factorized_source_terms()
_balanced_regime_loss()
```

记录：

```text
growth_precision
growth_recall
decay_precision
decay_recall
regime_macro_f1
growth_source_scale_ratio
decay_source_scale_ratio
interior_state_mae
edge_source_abs
birth_source_abs
clear_source_abs
```

当前方法已经记录 source sign accuracy、scale ratio、正负像素比例和物理上界诊断，可以直接扩展，而不是另写一套评估系统。

## 4. 配置文件

建议新增：

```text
configs/bth_radar/
├── ConvLSTM_evolution_factorized_s1.py
├── ConvLSTM_evolution_factorized_capacity_s1.py
├── ConvLSTM_evolution_edge_flow_s1.py
├── ConvLSTM_evolution_factorized_s3.py
├── ConvLSTM_evolution_factorized_envelope_s5.py
└── ConvLSTM_evolution_birth_s1.py
```

不要覆盖原有 c1/c2 配置。

## 5. 定向测试

至少增加：

```text
test_factorized_zero_initialization_matches_r4b
test_growth_state_is_not_below_advected_state
test_decay_state_is_not_above_advected_state
test_decay_state_is_nonnegative
test_regime_probabilities_sum_to_one
test_source_is_zero_outside_interior_mask
test_growth_and_decay_heads_receive_gradients
test_edge_flow_is_zero_outside_edge_mask
test_edge_flow_respects_displacement_bound
test_three_step_free_rollout_uses_own_previous_prediction
```

---

# 九、现阶段明确不要做的事情

暂时不要：

* 继续训练 c2a/c2b；
* 单纯增加 epoch 或学习率；
* 直接解冻 motion；
* 将 source loss 换成更多 L1/TV；
* 直接加入 scheduled sampling；
* 用完整场 MSE 掩盖 growth/decay 不可辨识；
* 让 source 同时解释 interior、edge、newborn 和 clear background；
* 直接进入 20-step；
* 访问 Test；
* 运行多 seed。

仓库现有失败报告已经明确给出了同样的 No-go 判断：当前应重新平衡损失、隔离 persistent interior、先验证 growth/decay 两侧是否都可辨识，而不是继续 rollout 或联合训练。

**最优先的一次训练实验，应当是：冻结 R4-b、只预测第 1 帧、只在 persistent interior 上训练三分类的增强/维持/减弱分支，并分别监督正负幅值。**这一步通过以后，源汇项才真正从“有物理名字的残差”变成“可验证的物理机制”。
