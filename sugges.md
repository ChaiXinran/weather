有，而且结合最新代码看，**当前 source 训练确实存在结构性的损失错配**。但要先区分两个问题：

1. 新的 Temporal U-Net motion-only 本身没有通过 Gate，最佳综合 CSI 只有 0.4399，弱于 ConvLSTM 对照 0.6225，报告本身已经给出“不要直接进入 source 阶段”的结论。
2. 在这个较弱的 motion checkpoint 上加入 source 后，CSI 又从初始较高值骤降，说明 **source loss 还在进一步破坏已有运动解**。

因此重新设计损失函数的合理目标不是马上让 Temporal U-Net 超过 ConvLSTM，而是先保证：

[
\boxed{\text{训练后的 source 至少不劣于 zero-source}}
]

然后再考虑 source 是否能够产生正增益。

# 一、最新代码中最关键的损失问题

## 1. 物理标签和 free rollout 使用了同一条混合轨迹

当前 `_factorized_source_training_step()` 只执行一次模型前向：

```python
result = self.model(
    batch_x,
    return_aux=True,
    teacher_forcing=batch_y if use_teacher_forcing else None,
    teacher_forcing_ratio=teacher_forcing_ratio,
)
```

随后：

* `regime_loss`
* `magnitude_loss`
* `state_loss`
* `rollout_loss`

全部基于同一个 `result`。

这会产生一个根本问题。

在 teacher forcing 下：

[
A_t=\mathcal W(Y_{t-1},v_t)
]

[
S_t^*=Y_t-A_t
]

还可以近似理解为“给定运动后的物理源汇”。

但在 scheduled/free rollout 下：

[
A_t=\mathcal W(\hat Y_{t-1},v_t)
]

此时：

[
S_t^*=Y_t-A_t
]

混合了：

* 真实增强与减弱；
* 前几步 source 误差；
* 前几步运动误差；
* 插值误差；
* 位置偏差；
* 强度累积误差。

它已经不再是干净的物理 source 标签。

于是当前训练实际上要求 source-head：

> 一边学习真实增长/衰减，一边负责补偿自身递归产生的全部误差。

这很容易形成正反馈。

---

## 2. 当前监督的不是实际进入方程的 source

当前 operator 中实际生效的 correction 是：

[
\hat S
======

## p_g\alpha_g C^+

p_d\alpha_d A
]

其中：

* (p_g,p_d)：growth/decay 概率；
* (\alpha_g,\alpha_d)：growth/decay fraction；
* (C^+)：正 source capacity；
* (A)：advected rain。

但是当前损失分别监督：

[
L_{\mathrm{regime}}(p_g,p_s,p_d)
]

和：

[
L_{\mathrm{magnitude}}(\alpha_g,\alpha_d)
]

而 magnitude 只在真实 growth 或 decay 区域计算：

```python
growth_mask = labels == 0
decay_mask = labels == 2
```

steady 像素不直接约束 `growth_fraction` 和 `decay_fraction`。

这意味着：

* 单独的类别可能预测得还可以；
* 单独的幅度可能也接近标签；
* 但二者乘积形成的实际 source 仍可能错误；
* steady 区域的小概率 growth/decay 会在20步中不断累积。

真正应该监督的是：

[
\hat G=p_g\alpha_gC^+
]

[
\hat D=p_d\alpha_dA
]

而不是只监督 (p) 和 (\alpha) 各自。

---

## 3. state loss看不到许多实际产生的有害source

当前代码先执行：

```python
masked_source = result['net_source'] * interior
masked_evolved_rain = advected_rain + masked_source
```

然后用 `masked_evolved_rain` 计算 state loss。

也就是说：

> state loss只评估 persistent interior 中的source。

但在 mixed/free rollout 中，模型实际使用的 source mask 是：

```python
Erode(advected_rain >= threshold)
```

它允许 source 作用于所有平流后仍有降水的内部区域，包括未来即将消散的区域。

因此会出现：

```text
实际 rollout：
source 在 death 区域发生了作用

state loss：
把 death 区域 source 乘成了0，因此没有直接惩罚
```

这些有害 correction 只能由全场 MSE 间接发现，但全场 MSE 又容易被大量弱雨与背景主导。

---

## 4. death区域没有被直接监督

当前 regime 标签只在：

[
\operatorname{Erode}
[(A\geq0.1)\cap(Y\geq0.1)]
]

即 persistent interior 内定义。

因此：

[
A\geq0.1,\quad Y<0.1
]

这种真正的消散区域被排除在 growth/steady/decay 标签之外。

但推理时，只要 advected rain 仍存在，source-head就会在这里工作。

这会导致明显的训练—推理不一致：

* 训练时模型不知道怎样完全消散一个对象；
* 推理时它却必须对这些区域作出 source 决策。

---

## 5. inverse-frequency regime loss过度提高少数类的重要性

当前 `_balanced_regime_loss()` 使用：

[
w_c=\frac{N}{3N_c}
]

进行完全反频率加权。

如果真实标签中 steady 占绝大多数，完整平衡会人为让：

```text
growth
steady
decay
```

三类对梯度的贡献近似相等。

这对普通分类可能有助于少数类 recall，但对 source 递归很危险：

* 错误预测 steady：不修正；
* 错误预测 growth：持续加雨；
* 错误预测 decay：持续减雨；
* 后两类错误会进入下一步继续放大。

因此分类错误的物理代价并不对称，不能简单追求三类完全平衡。

---

## 6. source初始化过度饱和

当前 source head 初始化为：

```python
regime bias = [0, 20, 0]
growth bias = -20
decay bias = -20
```

这能保证初始输出接近 zero-source，但存在严重梯度问题：

[
\sigma(-20)\approx2\times10^{-9}
]

growth/decay fraction几乎饱和在0，梯度也极小。

与此同时，regime CE在growth/decay标签上的梯度并不小，因此训练初期容易出现：

```text
regime概率先快速改变
magnitude分支仍处于饱和状态
```

三条分支学习速度严重不同。

这会让模型先学会“在哪里改变”，但还没有稳定学会“改多少”。

---

## 7. rollout loss就是归一化dBZ MSE

当前：

```python
rollout_loss = self.validation_criterion(
    result['prediction'], target
)
```

而 `validation_criterion` 固定为 `nn.MSELoss()`。

模型输出又是归一化dBZ，因此它优化的是：

[
\operatorname{MSE}(\widehat{\mathrm{dBZ}},\mathrm{dBZ})
]

而最终关心的是：

* rain-rate；
* CSI@16/32；
* 强降水面积；
* 强度保持；
* long-lead free rollout。

所以：

[
L_{\mathrm{MSE}}\downarrow
]

并不保证：

[
CSI_{16/32}\uparrow
]

当前配置也明确将 `loss_type='mse'`，而 source loss 权重为1/1/0.2/1。

# 二、建议把训练拆成两条独立分支

不要再用一条 scheduled trajectory 同时生成物理标签和rollout损失。

## 分支A：teacher-forced机制监督

始终使用真实上一帧：

[
Y_{t-1}^{true}
]

计算：

[
A_t^{TF}=\mathcal W(Y_{t-1}^{true},v_t)
]

[
S_t^*=Y_t-A_t^{TF}
]

这一分支只负责学习：

* growth；
* steady；
* decay；
* source幅度；
* 单步状态修正。

这里的标签才具有相对清楚的物理含义。

## 分支B：pure-free rollout监督

始终使用模型上一帧：

[
\hat Y_{t-1}
]

得到：

[
\hat Y_t
]

这一分支不再构造所谓的“oracle physical source”，只计算：

* free-rollout rain-rate state loss；
* soft CSI；
* intensity/bias约束；
* source累积约束。

也就是：

```python
mechanism_result = self.model(
    batch_x,
    return_aux=True,
    teacher_forcing=batch_y,
    teacher_forcing_ratio=1.0,
)

free_result = self.model(
    batch_x,
    return_aux=True,
    teacher_forcing=None,
    teacher_forcing_ratio=0.0,
)
```

不要使用75%/25%的混合轨迹来构造机制标签。

# 三、重新设计核心source损失

定义teacher-forced平流场：

[
A=A_t^{TF}
]

真实目标：

[
Y=Y_t
]

oracle source：

[
S^*=Y-A
]

预测的实际增长和衰减贡献：

[
\hat G=p_g\alpha_gC^+
]

[
\hat D=p_d\alpha_dA
]

预测net source：

[
\hat S=\hat G-\hat D
]

目标贡献：

[
G^*=\operatorname{clip}(\max(S^*,0),0,C^+)
]

[
D^*=\operatorname{clip}(\max(-S^*,0),0,A)
]

## 1. Effective source loss

直接监督真正进入物理方程的贡献：

[
L_{\mathrm{eff}}
================

\operatorname{Huber}
\left(
\frac{\hat G}{C^++\epsilon},
\frac{G^*}{C^++\epsilon}
\right)
+
\operatorname{Huber}
\left(
\frac{\hat D}{A+\epsilon},
\frac{D^*}{A+\epsilon}
\right)
]

对应代码：

```python
pred_growth = (
    result["regime_probability"][:, :, 0:1]
    * result["growth_fraction"]
    * result["positive_capacity"]
)

pred_decay = (
    result["regime_probability"][:, :, 2:3]
    * result["decay_fraction"]
    * result["advected_rain"]
)

target_growth = oracle_source.clamp_min(0.0)
target_growth = torch.minimum(
    target_growth,
    result["positive_capacity"].detach(),
)

target_decay = (-oracle_source).clamp_min(0.0)
target_decay = torch.minimum(
    target_decay,
    result["advected_rain"].detach(),
)
```

这一项应该取代当前主要的 `magnitude_loss`。

---

## 2. Rain-state loss

直接使用模型真实输出：

```python
result["evolved_rain"]
```

而不是重新构造：

```python
advected + net_source * interior
```

推荐同时使用线性雨强和log雨强：

[
L_{\mathrm{state}}
==================

0.5,\operatorname{Huber}
\left(
\frac{\hat R}{R_{\max}},
\frac{Y}{R_{\max}}
\right)
+
0.5,\operatorname{Huber}
\left(
\frac{\log(1+\hat R)}{\log(1+R_{\max})},
\frac{\log(1+Y)}{\log(1+R_{\max})}
\right)
]

线性项保护强降水，log项避免弱中雨完全失去作用。

---

## 3. Steady/abstention loss

对于：

[
|S^*|\leq\delta
]

要求source尽量不修改motion结果：

[
L_{\mathrm{steady}}
===================

\frac{|\hat G|+|\hat D|}
{A+C^++\epsilon}
]

或者：

[
L_{\mathrm{steady}}=|\hat S|
]

这是非常重要的保守约束：

> 没有明确增长或衰减证据时，优先相信motion-only。

---

## 4. Do-no-harm loss

明确约束source不能比平流结果更差：

[
e_{\mathrm{motion}}=|A-Y|
]

[
e_{\mathrm{source}}=|\hat R-Y|
]

[
L_{\mathrm{guard}}
==================

\max
\left(
e_{\mathrm{source}}-e_{\mathrm{motion}},
0
\right)
]

这相当于把zero-source结果设为局部安全基线。

当前现象就是source一更新便离开稳定motion解，因此这项非常适合你们。

---

## 5. Regime loss只保留为辅助项

不建议彻底删除三分类，因为它仍然提供解释性。

但应改成：

[
\lambda_{\mathrm{regime}}=0.05\sim0.1
]

并取消完全反频率平衡。

可使用：

[
w_c=
\min
\left[
\left(\frac{N}{N_c}\right)^{1/2},
2
\right]
]

即：

* 使用平方根平衡；
* 最大权重不超过2；
* 不让growth/decay梯度无限放大。

# 四、重新定义训练mask

建议source训练作用域与推理一致：

[
M_{\mathrm{action}}
===================

\operatorname{Erode}(A\geq0.1,1)
]

然后再划分：

### Persistent interior

[
A\geq0.1,\quad Y\geq0.1
]

用于growth/steady/decay。

### Death

[
A\geq0.1,\quad Y<0.1
]

应作为强decay目标：

[
D^*\approx A
]

### Edge

边缘仍可暂时排除，因为残差可能主要来自运动误差。

### Birth

[
A<0.1,\quad Y\geq0.1
]

当前source mask本来不允许在无雨区域生成新生，因此第一阶段仍然不训练birth。

所以可靠监督mask可以是：

[
M_{\mathrm{train}}
==================

M_{\mathrm{persistent\ interior}}
\cup
M_{\mathrm{death}}
]

而不是只用persistent interior。

# 五、free-rollout损失

先不要直接20步。

采用：

```text
1 step
→ 3 step pure-free
→ 5 step pure-free
→ 10 step
→ 20 step
```

而不是：

```text
20步 + scheduled sampling比例逐渐下降
```

## 1. Free rain-state loss

[
L_{\mathrm{roll,state}}
=======================

\sum_{t=1}^{K}
w_t L_{\mathrm{state}}(\hat R_t,Y_t)
]

例如3步：

[
w=[1.0,1.25,1.5]
]

稍微提高后续时效权重。

## 2. Soft CSI loss

对16和32 mm/h：

[
P_\tau
======

\sigma
\left(
\frac{\hat R-\tau}{T}
\right)
]

[
L_{\mathrm{CSI},\tau}
=====================

1-
\frac{
\sum P_\tau Y_\tau+\epsilon
}{
\sum P_\tau+\sum Y_\tau-\sum P_\tau Y_\tau+\epsilon
}
]

建议：

[
\lambda_{16}=0.05
]

[
\lambda_{32}=0.10
]

不要让soft CSI成为主损失，只作为高阈值方向校正。

## 3. Area/bias loss

防止靠扩大或压缩雨区获得虚假改善：

[
L_{\mathrm{area},\tau}
======================

\left|
\log
\frac{
\sum P_\tau+\epsilon
}{
\sum Y_\tau+\epsilon
}
\right|
]

分别用于16和32 mm/h。

## 4. Source budget loss

当前每一步最大capacity为4/6/8/10 mm/h，连续20步仍可能产生较大累计修正。

可以约束累计净source：

[
B_t=\sum_{\tau=1}^{t}\sum_{x,y}\hat S_\tau(x,y)
]

与teacher-forced oracle累计source保持同量级：

[
L_{\mathrm{budget}}
===================

\frac{
|B_t-B_t^*|
}{
\sum Y_t+\epsilon
}
]

先以较小权重：

[
\lambda_{\mathrm{budget}}=0.05
]

使用。

# 六、推荐的总损失

## 阶段S1：单步机制识别

[
\boxed{
L_{S1}
======

1.0L_{\mathrm{eff}}
+
1.0L_{\mathrm{state}}
+
0.25L_{\mathrm{steady}}
+
0.5L_{\mathrm{guard}}
+
0.05L_{\mathrm{regime}}
}
]

暂时关闭：

```python
evolution_magnitude_loss_weight = 0.0
evolution_rollout_loss_weight = 0.0
```

预测长度：

```python
evolution_forecast_steps = 1
```

## 阶段S3：三步pure-free

[
\boxed{
L_{S3}
======

L_{S1}
+
0.5L_{\mathrm{roll,state}}
+
0.05L_{\mathrm{softCSI16}}
+
0.10L_{\mathrm{softCSI32}}
+
0.02L_{\mathrm{area}}
+
0.05L_{\mathrm{budget}}
}
]

不使用scheduled sampling：

```text
teacher forcing mechanism branch = 100%
free rollout branch = 0%
```

两条branch各自承担不同职责。

# 七、source初始化也要同步修改

不要继续使用±20。

建议使用数据先验初始化regime：

```python
prior = torch.tensor([0.05, 0.90, 0.05])
regime_head.bias.copy_(prior.log())
```

即大约：

```text
growth: -3.00
steady: -0.105
decay:  -3.00
```

幅度初始化为1%左右：

```python
growth_head.bias = logit(0.01) ≈ -4.60
decay_head.bias  = logit(0.01) ≈ -4.60
```

这样初始有效source仍然极小：

[
0.05\times0.01\times C
]

但不会像 (-20) 那样几乎完全失去梯度。

# 八、代码应该怎样改

主要修改：

```text
openstl/methods/evolution_convlstm.py
```

新增：

```python
_teacher_forced_effective_source_terms()
_free_rollout_source_terms()
_soft_csi_loss()
_source_guard_loss()
_source_budget_loss()
```

重写：

```python
_factorized_source_training_step()
```

结构建议：

```python
def _factorized_source_training_step(self, batch_x, batch_y):
    mechanism_result = self.model(
        batch_x,
        return_aux=True,
        teacher_forcing=batch_y,
        teacher_forcing_ratio=1.0,
    )

    mechanism_terms = self._teacher_forced_effective_source_terms(
        mechanism_result,
        batch_y,
    )

    if self.hparams.evolution_rollout_horizon > 1:
        free_result = self.model(
            batch_x,
            return_aux=True,
            teacher_forcing=None,
            teacher_forcing_ratio=0.0,
        )

        rollout_terms = self._free_rollout_source_terms(
            free_result,
            batch_y,
        )
    else:
        rollout_terms = {}

    loss = (
        mechanism_terms["effective_loss"]
        + mechanism_terms["state_loss"]
        + 0.25 * mechanism_terms["steady_loss"]
        + 0.5 * mechanism_terms["guard_loss"]
        + 0.05 * mechanism_terms["regime_loss"]
    )

    if rollout_terms:
        loss = loss + (
            0.5 * rollout_terms["state_loss"]
            + 0.05 * rollout_terms["soft_csi_16"]
            + 0.10 * rollout_terms["soft_csi_32"]
            + 0.02 * rollout_terms["area_loss"]
            + 0.05 * rollout_terms["budget_loss"]
        )

    return loss
```

同时修改：

```text
openstl/modules/temporal_unet_modules.py
```

调整source head bias初始化。当前代码中的±20初始化位置就在 `UNetFactorizedSourceHead.reset_parameters()`。

# 九、validation和checkpoint也要改变

目前 `val_loss`固定是归一化dBZ MSE。

建议额外记录：

```text
val_free_rain_state_loss
val_source_gain_vs_zero
val_source_harm_fraction
val_effective_growth_loss
val_effective_decay_loss
val_cumulative_source_mass
val_intensity_ratio_0_1h
val_intensity_ratio_1_2h
```

其中：

[
\text{source gain}
==================

## L_{\mathrm{motion-only}}

L_{\mathrm{with-source}}
]

还要记录：

[
\text{harm fraction}
====================

P
\left(
|\hat R-Y|>|A-Y|
\right)
]

checkpoint建议：

```text
best_val_csi.ckpt
→ 最终主checkpoint

best_val_mechanism.ckpt
→ val_free_rain_state_loss 或 val_source_gain_vs_zero

best_val_loss.ckpt
→ 保留，但不作为source主结论
```

当前 `best_val_mechanism`仍监控 `val_loss`，实际上与 `best_val_loss`重复。

# 十、建议的实验顺序

## E0：zero-source

使用当前已有zero-source配置，在相同：

* motion checkpoint；
* rain-rate operator；
* displacement；
* 20步free rollout；

下建立基准。

## E1：新损失单步

```text
effective + state + steady + guard
forecast_steps=1
3–5 epochs
source_lr=1e-5
```

通过标准：

* 单步state error优于advected baseline；
* source harm fraction < 50%；
* growth/decay scale ratio合理；
* 20步free CSI相对zero-source下降不超过0.01。

## E2：加入低权重regime

```text
regime weight=0.05
```

检查它是否改善解释性而不破坏CSI。

## E3：三步pure-free

加入free rollout rain-state和soft CSI。

## E4：5/10步

只有三步不出现：

* intensity爆炸；
  -过度清空；
  -面积膨胀；
  -CSI骤降；

才继续。

# 最重要的判断

当前问题最可能不是简单的：

```text
state_loss 0.2太小
rollout_loss 1.0太大
```

而是：

[
\boxed{
\text{物理source标签}
\quad\text{和}\quad
\text{free-rollout误差修正}
}
]

被塞进了同一个scheduled trajectory里。

重新设计时要把两者彻底拆开：

[
\boxed{
\text{Teacher-forced branch学习物理source}
+
\text{Pure-free branch学习递归稳定性}
}
]

同时把监督对象从“分类概率和幅度各自正确”改成：

[
\boxed{
p_g\alpha_gC^+
--------------

p_d\alpha_dA
}
]

这个真正进入演化方程的有效source。
