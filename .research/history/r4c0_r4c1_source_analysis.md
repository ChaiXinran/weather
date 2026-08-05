# R4-c0 oracle-source audit and R4-c1 source-only analysis

## Physical operator correction

The evolution operator now adds a signed source in rain-rate space:

```text
advected_rain = Warp(previous_rain, flow)
evolved_rain = clamp_min(advected_rain + source_rain, 0)
```

`source_rain` is an increment in mm/h per six-minute evolution step. The
operator returns normalized-dBZ predictions plus `advected_rain`,
`source_rain`, and `evolved_rain` diagnostics. With no source, the validated
R4-b path is unchanged. A zero-initialized source head also reproduces the
motion-only prediction. Ten targeted WSL OpenSTL checks passed; pytest itself
is not installed in that environment.

## R4-c0 protocol

- Frozen motion checkpoint: `.640662`.
- Oracle definition:
  `true_rain_t - warp(true_rain_t-1, predicted_flow_t)`.
- Teacher-forced true previous frame at every lead; no model training.
- Full training split: 11,245 windows, used to select source bounds.
- Frozen validation split: 932 windows, used only as an independent mechanism
  check, not to select the bound.

Artifacts:

- `work_dirs/bth_r4c0_oracle_source_train_scale05_0640662/`
- `work_dirs/bth_r4c0_oracle_source_scale05_0640662/`

## Oracle-source distribution

| Region | Train mean abs | Train abs P95 | Train abs P99 | Val abs P99 |
|---|---:|---:|---:|---:|
| All pixels | .130 | .46 | 2.78 | 2.95 |
| Active union >=.1 mm/h | .582 | 2.56 | 8.26 | 9.69 |
| 16-mm/h union | 6.458 | 18.36 | 27.32 | 27.23 |
| 32-mm/h union | 9.607 | 24.89 | 33.35 | 32.30 |
| Existing interior | .820 | 3.60 | 9.80 | 11.72 |
| Previous edge band | .187 | .60 | 3.20 | 3.86 |
| Newborn >=.1 mm/h | .126 | .38 | 1.34 | .98 |
| True growth | 1.102 | 4.80 | 12.90 | 14.00 |
| True decay | .753 | 3.18 | 8.33 | 9.30 |

The signs are physically ordered. In true-growth pixels, 72.6% of oracle
source values are positive; in true-decay pixels, 76.3% are negative. In the
32-mm/h union, 74.5% are positive and mean signed source is +6.23 mm/h. Source
is therefore both necessary and physically interpretable.

The generic active-area P99 (8.26) is too small for the primary 32-mm/h goal.
The training 32-mm/h absolute P99 is 33.35 and positive P99 is 34.41; validation
independently gives 32.30/33.23. The controlled symmetric bound is therefore:

```text
source_rain = 35 mm/h * tanh(raw_source)
```

## R4-c1 implementation

- Encoder and motion head loaded from `.640662` and frozen for all three
  epochs.
- New shallow source head: 3x3 Conv, GroupNorm+SiLU, 3x3 Conv+SiLU, 1x1 Conv,
  20 signed maps.
- Final layer zero initialized.
- Loss: recursive precipitation R2 + weighted oracle-source Huber + light L1
  sparsity + spatial/temporal TV.
- No learned motion gate and no source/motion joint fine-tuning.

## R4-c1 results

The first weighting gave every background pixel weight one:

| Epoch | CSI score | CSI16 0--1 h | CSI32 0--1 h | CSI16 1--2 h | CSI32 1--2 h | Mean abs source |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .640814 | .322073 | .187893 | .112277 | .009285 | .00362 |
| 1 | .640852 | .322084 | .187920 | .112307 | .009270 | .00572 |
| 2 | .641011 | .322138 | .188015 | .112289 | .009285 | .00320 |

Because background dominated, a controlled rerun restricted direct source
supervision to the active union (`rain >= .1` or `|oracle source| >= .1`) while
retaining full-field sparsity:

| Epoch | CSI score | CSI16 0--1 h | CSI32 0--1 h | CSI16 1--2 h | CSI32 1--2 h | Mean abs source |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .640982 | .322139 | .187948 | .112296 | .009299 | .00385 |
| 1 | **.641226** | **.322168** | **.188128** | **.112333** | .009298 | .00562 |
| 2 | .640984 | .322142 | .188017 | .112283 | .009271 | .00338 |

The active source-supervision loss does not decrease (`.03059 -> .03066`) and
the predicted source remains two orders of magnitude below strong-rain oracle
residuals. POD32 and Bias32 remain effectively unchanged. The tiny CSI gain
(.000564 over R4-b) is not a source-mechanism pass.

## Decision

R4-c0 passes: the missing signed residual is large, sign-consistent, stable
between train and validation, and especially important at 16/32 mm/h.

R4-c1 as currently parameterized fails the mechanism gate. Do not enter R4-c2
or unfreeze motion. More identical epochs are not justified. The next isolated
test should remove the recursive forecast-gradient conflict entirely and
pretrain the source head only on active oracle residuals, with explicit
positive/negative and 16/32-mm/h diagnostics. If direct-only loss still cannot
fall, the final history feature is insufficient and the source input needs an
explicit history-change cue or temporal decoder.

Selected diagnostic checkpoint (not promoted as a new baseline):
`work_dirs/bth_r4c1_source_active_3ep_seed0/checkpoints/`
`val-csi-epoch=01-val_csi_score=0.641226.ckpt`.


你的直觉是有道理的。重新拆开看，当前版本的问题不是“源汇项在演化方程里加错了”，而是：

> **演化算子层面引入正确，但 source 分支的输入、监督目标和训练方式不够合理。**

它目前更接近“用一个浅层头回归未来20帧的 oracle 残差”，还不是一个真正根据当前演化状态逐步判断增强、衰减、新生和消散的 source/sink 模块。

# 为什么提出过拟合检查

单 batch 过拟合检查不是为了提高最终指标，也不是为了证明模型科学合理，它只是一个代码级单元检查。

它回答的是：

> 在只给一个小批次、反复训练很多步的情况下，source head 有没有能力把这个批次的 source target 记住？

如果连一个 batch 都记不住，通常意味着：

* source参数没有进入optimizer；
* 梯度被意外截断；
* target和预测单位不一致；
* mask或归一化实现有问题；
* source输出经过某处被压得太小；
* 正则项压倒了source监督。

但即使单batch能过拟合，也只能证明：

> 代码路径和优化过程能工作。

不能证明：

* oracle source定义合理；
* source输入足够；
* 训练和推理一致；
* 模型真正学到了物理生消。

所以在你已经怀疑**生消项设计本身不对**的情况下，不应该把过拟合检查作为主线实验。它最多是在重新设计后花很短时间做一次代码检查。

---

# 上一版 R4-b 和这一版 R4-c1 的具体区别

## 上一版 R4-b：只有运动

上一版模型是：

[
F=Encoder(X_{-9:0})
]

[
v_{1:20}=MotionHead(F)
]

[
\hat R_t
========

Warp(\hat R_{t-1},v_t)
]

它只有一个职责：

> 将已有降水回波从上一时刻移动到下一时刻。

正式配置中使用rain-rate空间、固定0.5位移校准，不使用gate，也不使用source。

因此它的问题非常明确：

* 能移动已有回波；
* 不能增强强核；
* 不能产生新生回波；
* 不能维持快速增强的32 mm/h区域；
* 不能解释消散。

## 当前 R4-c1：运动后再加一个预测残差

当前算子变成：

[
A_t=Warp(R_{t-1},v_t)
]

[
R_t=\max(A_t+S_t,0)
]

这里的物理单位和加法位置是正确的：

* (A_t)：平流后的雨强；
* (S_t)：每6分钟的雨强增量，单位mm/h；
* 正值表示增强；
* 负值表示减弱。

这一层改动本身没有问题。原来的算子已经预留了 `source` 接口，但旧实现是在转换回归一化dBZ后相加；现在改成在rain-rate内部状态上相加才是合理做法。旧算子的执行逻辑可以在仓库版本中看到。

真正的区别在source如何得到。

当前：

[
S_{1:20}=SourceHead(F)
]

即：

> 只根据历史10帧经过ConvLSTM得到的最后隐藏特征，一次性输出未来20张source图。

然后使用一个显式标签：

[
S_t^*
=====

## R_t^{true}

Warp(R_{t-1}^{true},v_t)
]

直接监督预测source接近这个oracle residual。

所以两版差异可以概括为：

| 部分       | R4-b         | 当前R4-c1                  |
| -------- | ------------ | ------------------------ |
| 历史编码器    | 冻结ConvLSTM   | 同一冻结ConvLSTM             |
| Motion   | 冻结的0.5校准flow | 完全相同                     |
| Source   | 无            | 浅层source head            |
| Source输入 | 无            | 只有最终历史隐藏特征               |
| Source输出 | 无            | 一次输出未来20帧                |
| 演化       | 只有warp       | warp后在rain-rate空间加source |
| 新增监督     | 无            | oracle-source Huber      |
| 额外约束     | 无source约束    | L1、空间TV、时间TV             |
| 最终训练目标   | 递归预报         | 递归预报+source标签回归          |

---

# 当前source引入中最值得质疑的地方

## 1. Source head没有看到“当前平流后的状态”

这是最重要的问题。

真实的source应该回答：

> 当前已经平流到这里的降水场，还需要在哪里增强、减弱、新生或消散？

但当前source head只看到：

[
F=Encoder(X_{-9:0})
]

它没有看到：

* 当前时刻的 (A_t=Warp(R_{t-1},v_t))；
* 当前对象被平流到了哪里；
* 当前强度还剩多少；
* 当前flow的大小和方向；
* 上一步source已经修改了什么；
* 现在是第6分钟还是第120分钟。

也就是说，source head在生成第20帧source时，并不知道前19步的实际演化状态。

虽然不同输出通道可以隐式代表不同lead，但这仍然是一种**开环预测**：

[
F\rightarrow S_1,S_2,\ldots,S_{20}
]

而不是逐步的：

[
(F,A_t,t)\rightarrow S_t
]

这很可能是source几乎保持零的主要结构原因之一。

---

## 2. Oracle source不是纯粹的物理生消项

当前oracle：

[
S_t^*
=====

## R_t^{true}

Warp(R_{t-1}^{true},v_t)
]

其中不仅包含真实的增强和减弱，还混入了：

* flow方向误差；
* flow尺度误差；
* 雷达对象形变；
* 分裂和合并；
* 插值误差；
* 对象匹配错误；
* 对流随机性。

例如模型flow向右偏了一个像素，oracle source会表现为：

* 旧的错误位置需要负source；
* 真正位置需要正source。

这会让source head被迫“擦除错误对象，再在正确位置重画对象”。

因此它更准确的名字是：

> **平流后离散强度残差代理项**

而不是干净的“物理生消真值”。

R4-c0中符号统计具有物理一致性，说明它有价值；但不能把它当成严格标签强制拟合。

---

## 3. 训练时和推理时的source目标不一致

直接source监督使用的是：

[
Warp(R_{t-1}^{true},v_t)
]

即真实上一帧。

但实际20步推理使用的是：

[
Warp(\hat R_{t-1},v_t)
]

即模型自己的上一帧。

训练标签描述的是：

> 从真实上一帧出发需要补多少。

实际需要的却是：

> 从已经包含历史预测误差的模型场出发需要补多少。

第二小时两者可能差别很大。

所以当前source标签存在明显的 teacher-forced / rollout mismatch。

这也解释了为什么source即使勉强拟合oracle residual，也不一定能改善递归预报。

---

## 4. Source同时受到相反目标的拉扯

当前一方面要求：

[
S_t\approx S_t^*
]

另一方面：

* forecast loss担心正source扩大降水面积、增加FAR；
* L1要求source尽量小；
* TV要求source尽量平滑；
* source最后一层从零初始化。

对于当前低FAR、低POD的R4-b，最安全的局部解就是：

[
S_t\approx0
]

因为一旦source开始产生强降水：

* POD可能上升；
* 但FAR和空事件惩罚也可能迅速上升。

所以“source输出接近零”不一定只是学习率不足，而可能是损失目标本身存在冲突。

---

# 这和NowcastNet的source方式有什么区别

NowcastNet也是：

[
x_t^{adv}=Advect(x_{t-1},v_t)
]

[
x_t=x_t^{adv}+s_t
]

但它的 evolution network同时预测：

* motion fields；
* intensity residuals；
* precipitation fields。

训练时主要通过：

* 演化后最终场与真值的距离；
* 仅平流场与真值的距离；
* motion regularization；

联合优化演化过程。论文将这两部分距离组合为 accumulation loss，并没有把

[
x_t^{true}-Advect(x_{t-1}^{true},v_t)
]

作为唯一、严格的source标签直接回归。

所以当前版本更像：

> Oracle residual regression。

而NowcastNet更像：

> State-conditioned evolution learning。

这就是两者最关键的区别。

---

# 更合理的source引入方式

建议保留现在已经修正正确的rain-rate算子，但重写source decoder的输入和监督。

## 新结构

历史特征：

[
F=Encoder(X_{-9:0})
]

每一步：

[
A_t=Warp(R_{t-1},v_t)
]

然后source根据当前演化状态预测：

[
S_t
===

D_s(
F,,
A_t,,
v_t,,
\Delta X_{history},,
e_t
)
]

最终：

[
R_t=\max(A_t+S_t,0)
]

其中：

* (F)：历史上下文；
* (A_t)：当前平流后的雨强场；
* (v_t)：当前运动状态；
* (\Delta X_{history})：近期增强或减弱趋势；
* (e_t)：lead-time embedding。

最小输入可以先用：

[
[F,\ A_t,\ R_0-R_{-1},\ R_0-R_{-5},\ |v_t|,\ e_t]
]

不需要一次性上复杂网络。

---

# Source最好逐步预测，不要一次输出20张

第一版可以使用共享的小source decoder：

```text
history feature
+ 当前advected rain
+ 当前flow magnitude
+ lead embedding
        ↓
shared source decoder
        ↓
当前一步source
```

20个lead共享参数，但每步输入的：

* advected field不同；
* flow不同；
* lead embedding不同。

这样它才能知道：

> 当前这一步已经演化成什么状态，现在应该增还是减。

而不是仅凭一个静态历史特征提前画好未来20张source map。

---

# 训练目标也应修改

## 第一阶段：teacher-forced逐步最终状态训练

使用真实上一帧：

[
A_t^{TF}=Warp(R_{t-1}^{true},v_t)
]

source预测：

[
S_t=D_s(F,A_t^{TF},v_t,e_t)
]

优化的是：

[
L_{\mathrm{evolved}}
====================

\ell(
A_t^{TF}+S_t,
R_t^{true}
)
]

而不是一开始强制：

[
S_t=S_t^*
]

Oracle residual可以继续用于：

* 分析；
* 符号监督；
* 弱辅助loss；
* source上限统计。

但不建议作为主loss。

## 第二阶段：混合teacher forcing和rollout

逐渐将真实上一帧替换成预测上一帧：

[
R_{t-1}^{input}
===============

mR_{t-1}^{true}
+
(1-m)\hat R_{t-1}
]

让source逐步适应真实推理中的误差状态。

## 第三阶段：完整rollout

最后才优化20步最终forecast。

---

# 是否还要做单batch过拟合

在当前设计不变的情况下，优先级不高。

因为即使它能过拟合，也不能解决：

* source没看到advected field；
* teacher-forced标签与rollout不一致；
* oracle residual混合运动误差；
* 一次输出20帧的开环问题。

更合理的顺序是：

1. 先重构source输入和逐步解码；
2. 再用一个batch跑几十到几百步，确认梯度链正常；
3. 然后进入正式机制实验。

这时过拟合检查只是短暂的工程验证，不是下一阶段研究结论。

# 最终判断

你的“生消项引入不对”可以更精确地表达为：

[
\boxed{
\text{Source在物理算子中加对了，但Source predictor和监督方式设计得不够对。}
}
]

当前版本的问题不是：

[
Advect+Source
]

这个公式错误，而是：

[
S_{1:20}=SourceHead(F_{\mathrm{last}})
]

以及：

[
S_t\leftarrow R_t^{true}-Warp(R_{t-1}^{true},v_t)
]

被当成了主要学习方式。

下一版建议改成：

[
\boxed{
S_t
===

SourceDecoder(
history\ feature,,
advected\ field,,
flow,,
history\ tendency,,
lead
)
}
]

并首先优化：

[
\boxed{
Advected_t+S_t
\rightarrow
R_t^{true}
}
]

而不是强制source精确复制oracle residual。这样才更接近真正的运动—源汇演化模型。

`max(A_t+S_t,0)` 的目的不是阻止“消散”，而是阻止模型产生**负降雨率**。

这里：

[
A_t=\operatorname{Warp}(R_{t-1},v_t)\ge 0
]

如果：

[
S_t<0
]

就会削弱已有降水。例如：

[
A_t=20,\quad S_t=-8
\Rightarrow R_t=12
]

表示衰减；若：

[
A_t=20,\quad S_t=-20
\Rightarrow R_t=0
]

表示完全消散。

但如果网络预测：

[
S_t=-30
]

直接相加会得到 (-10\ \mathrm{mm/h})，这在物理上没有意义，所以才截断为0。

## 但你的疑问是对的：`max`比较粗糙

当前形式：

[
R_t=\max(A_t+S_t,0)
]

虽然数值上安全，但存在两个问题。

第一，它允许模型先预测一个过大的负source，再依靠 `max`截断。这样：

* 无法判断sink到底预测得是否合理；
* (S_t=-20) 和 (S_t=-100) 最终都可能得到0；
* 小于0的区域经过 `clamp` 后梯度可能消失，不利于训练；
* source的物理解释会变弱。

第二，正source和负source的物理意义不同：

* 正source可以在无雨区产生新生；
* 负source只能消耗已经存在的降水，理论上不应该超过 (A_t)。

因此正式版本最好不要依赖 `max`修复不合理输出，而是从参数化上保证结果非负。

# 更合理的源汇表达

推荐把source和sink明确拆开：

[
R_t=A_t-D_t+B_t
]

其中：

* (D_t)：消散量；
* (B_t)：生成或增强量。

约束为：

[
0\le D_t\le A_t
]

[
0\le B_t\le S_{\max}
]

可以写成：

[
D_t=A_tg_t^{-}
]

[
B_t=S_{\max}g_t^{+}
]

其中：

[
g_t^{-},g_t^{+}\in[0,1]
]

最终：

[
\boxed{
R_t=(1-g_t^{-})A_t+S_{\max}g_t^{+}
}
]

这时天然有：

[
R_t\ge0
]

不需要再取 `max`。

它的解释也更清楚：

* (g_t^-=0)：不消散；
* (g_t^-=1)：已有降水完全消失；
* (g_t^+>0)：发生增强或新生。

为了保持R4-b的零初始化起点，可以让两个输出头初始接近0，或者采用零点严格为0的激活组合。

也可以保留一个signed head，但把负值限制在 (-A_t) 内：

[
u_t=\tanh(z_t)
]

[
S_t=
S_{\max}\operatorname{ReLU}(u_t)
--------------------------------

A_t\operatorname{ReLU}(-u_t)
]

然后：

[
R_t=A_t+S_t
]

这样：

* 正source最多增加 (S_{\max})；
* 负source最多消耗全部 (A_t)；
* (z_t=0) 时source严格为0；
* 不需要 `clamp`。

我更推荐显式的 source/sink 双分支，因为后面加入PWV时，可以只让PWV主要影响正source：

[
B_t^{PWV}
]

而消散分支仍主要由雷达历史控制。

---

# 逐步预测确实会累积误差

是的。若每一步使用上一步的预测：

[
\hat R_t=
\operatorname{Evolve}(\hat R_{t-1})
]

那么前一步的：

* 位置误差；
* 强度误差；
* source误差；
* 虚假新生；

都会进入下一步，第二小时可能被持续放大。

例如第一步多生成了一个虚假对象，下一步motion会继续移动它，source还可能继续增强它，于是错误可能越来越严重。

但这不代表不应该逐步预测。因为source本来应该根据**当前演化状态**判断下一步生消：

> 当前场已经移动到哪里、还剩多强、哪些对象正在增强或减弱。

若一次性从历史特征直接输出未来20张source：

[
F\rightarrow S_1,\ldots,S_{20}
]

虽然没有显式反馈误差，但它也看不到演化过程，属于开环预测。当前R4-c1正是这种方式，而这很可能是它没有学起来的原因之一。

所以两种方案各有问题：

| 方式                         | 优点           | 问题        |
| -------------------------- | ------------ | --------- |
| 一次输出20帧source              | 没有source递归反馈 | 看不到当前演化状态 |
| 逐步state-conditioned source | 更符合演化过程      | 会累积误差     |

最科学的做法不是二选一，而是逐步训练并控制误差累积。

# 推荐训练方式

## 阶段一：Teacher-forced单步训练

训练source机制时，每一步都使用真实上一帧：

[
A_t^{TF}
========

Warp(R_{t-1}^{true},v_t)
]

然后：

[
S_t=
SourceDecoder(F,A_t^{TF},v_t,e_t)
]

最终：

[
\hat R_t^{TF}=A_t^{TF}+S_t
]

这样每一步彼此独立，不会积累前面预测误差，先回答：

> 给定正确的上一状态，source能否学会增强和衰减？

这才是source机制是否有效的第一道门。

## 阶段二：混合真实状态与预测状态

随后逐渐引入模型自己的上一帧：

[
R_{t-1}^{input}
===============

m_tR_{t-1}^{true}
+
(1-m_t)\hat R_{t-1}
]

训练初期 (m_t) 较大，后期逐渐减小。这就是 scheduled sampling。

这样source逐步适应推理时会遇到的预测误差，不会突然从完全teacher forcing切换到20步自由递归。

## 阶段三：完整rollout微调

最后才使用：

[
R_{t-1}^{input}=\hat R_{t-1}
]

并优化完整20步预报指标。

这时还应保留单步机制损失，防止模型为了长期CSI又把source压回零或变成无物理意义的图像生成器。

---

# 还可以采用双轨迹设计

为了进一步控制误差，可以同时维护两条路径：

### Teacher-forced机制路径

用于监督source是否正确：

[
R_{t-1}^{true}
\rightarrow A_t^{TF}
\rightarrow S_t^{TF}
]

### Free-rollout业务路径

用于训练实际推理：

[
\hat R_{t-1}
\rightarrow A_t^{roll}
\rightarrow S_t^{roll}
]

总损失：

[
L=
\lambda_{TF}L_{TF}
+
\lambda_{roll}L_{roll}
]

其中训练前期：

[
\lambda_{TF}>\lambda_{roll}
]

后期逐渐提高rollout权重。

这样既不会忽略误差累积，也不会让第二小时的复杂梯度一开始就破坏source机制。

# 对当前R4-c的具体建议

当前版本：

[
S_{1:20}=SourceHead(F_{\text{last}})
]

一次输出20帧，而且使用：

[
R_t=\max(A_t+S_t,0)
]

我建议下一版同时做两项结构修正：

1. 将 `clamp` 改为物理有界的source/sink参数化：

[
R_t=(1-g_t^-)A_t+S_{\max}g_t^+
]

2. 将source改成逐步状态条件化：

[
(g_t^+,g_t^-)
=============

D_s(
F,,
A_t,,
v_t,,
\Delta R_{history},,
e_t
)
]

但训练时先用teacher forcing，不直接20步自由递归。

所以你的两个担心都成立：

* `max`只是一种数值保护，不是最理想的生消建模；
* 逐步预测确实会误差累积，但应通过分阶段训练解决，而不是退回完全开环的一次性source预测。

下一版最合理的公式是：

[
\boxed{
R_t=(1-g_t^-)A_t+S_{\max}g_t^+
}
]

它比：

[
\max(A_t+S_t,0)
]

更物理、更可解释，也更适合后续让PWV调制正source。
