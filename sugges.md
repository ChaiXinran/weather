你说得对。**“优化目标可以和业务指标不同”只是数学层面的事实，不能成为当前现象的合理解释。**

CSI 的硬阈值统计不可直接反向传播，所以训练时必须用可微代理损失；但代理损失的职责就是让验证 CSI 同方向改善。现在出现：

$$
\text{训练/验证损失下降或持平}
\qquad
\text{CSI持续下降}
$$

对于你们明确以 **CSI@16、CSI@32 和强降水结构保持**为核心的任务，这就说明：

> **当前可微优化目标没有正确代理业务目标，现有训练方向确实有问题。**

而且你们的重点正是解决 MSE、L1 导致的平滑和强回波衰减，所以不能期待“多训练几十轮以后自己恢复”。当前配置不建议继续跑 50 轮。

---

# 一、仓库代码中确实存在明显的目标错配

我扫描了 `ChaiXinran/weather` 当前配置和训练代码。

## 1. 名义上预测20步，训练却只约束3步自由 rollout

全开放配置继承了 `factorized_s20_warmup`：

```python
evolution_free_rollout_training = True
evolution_rollout_horizon = 3
evolution_rollout_state_loss_weight = 0.25
```

也就是说，模型验证和部署时递推 20 步，但真正的 free-rollout 损失只看前 3 步。

训练目标主要回答的是：

> 前18分钟是否合理？

业务指标回答的是：

> 未来两小时、尤其第二小时强降水是否存活？

这两者天然不一致。

---

## 2. 绝大多数训练信号是 teacher-forced

`_factorized_source_training_step()` 首先运行：

```python
teacher_forcing=batch_y
teacher_forcing_ratio=1.0
```

然后计算 factorized source 的主要机制损失；只有附加的 rollout 分支使用真正的自由递推，而且只递推3步。

模型内部 teacher forcing 的实现是：

```python
if teacher_forcing is not None and step > 0:
    current = teacher_forcing[:, step - 1]
```

因此每一步都从真实上一帧重新开始，而不是从模型自己的上一帧开始。

当前训练实际上更接近：

$$
R_{t-1}^{true}
\rightarrow
\hat R_t
$$

而部署是：

$$
\hat R_{t-1}
\rightarrow
\hat R_t
$$

所以模型可以把“真实上一帧条件下的单步残差”学得很好，却没有被充分要求处理自身预测误差、插值衰减和强降水递归消失。

---

## 3. CSI损失权重明显太弱

当前配置中：

```python
evolution_effective_loss_weight = 1.0
evolution_state_loss_weight = 1.0
evolution_steady_loss_weight = 0.25
evolution_guard_loss_weight = 0.5

evolution_rollout_state_loss_weight = 0.25
evolution_soft_csi_16_loss_weight = 0.05
evolution_soft_csi_32_loss_weight = 0.10
evolution_area_loss_weight = 0.02
evolution_budget_loss_weight = 0.05
```

CSI@16 和 CSI@32 的代理损失不仅权重低，而且只作用于3步 rollout。

于是训练梯度主要来自：

* source magnitude；
* state Huber；
* growth/steady/decay 分类；
* teacher-forced 单步状态误差。

而真正的业务目标：

* 20步 CSI16；
* 20步 CSI32；
* 第二小时强降水存活；
* 强核心和边缘保持；

在总梯度中占比很小。

---

## 4. 当前状态损失仍然会鼓励条件均值和平滑

当前 `_rain_state_error()` 是线性雨强 Huber 与 `log1p` 雨强 Huber 的平均：

$$
L_{\mathrm{state}}
==================

\frac12L_{\mathrm{Huber}}
\left(
\frac{\hat R}{R_{\max}},
\frac{R}{R_{\max}}
\right)
+
\frac12L_{\mathrm{Huber}}
\left(
\frac{\log(1+\hat R)}{\log(1+R_{\max})},
\frac{\log(1+R)}{\log(1+R_{\max})}
\right)
$$

这比普通 MSE 好，但仍然是逐像素回归目标。

在位置不确定时，逐像素回归最容易采取的策略仍是：

* 降低峰值；
* 扩散边缘；
* 预测多个可能位置的平均；
* 把 35 mm/h 压到 25–30 mm/h。

这样 Huber 可能下降，但 CSI32 会直接变成漏报。

---

## 5. 强降水像素加权也偏弱

目前强雨权重设置为：

```python
evolution_pixel_16_increment = 1.0
evolution_pixel_32_increment = 1.0
evolution_pixel_max_weight = 3.0
```

即使达到 32 mm/h，像素权重最多也只有普通活跃像素的约3倍。

但在数据中，32 mm/h 像素远少于弱雨和背景像素。三倍权重通常不足以抵消样本数量差异。

---

## 6. `val_loss` 本身就是 MSE

配置最后仍然是：

```python
loss_type = 'mse'
```

而 factorized 模型验证阶段明确使用：

```python
loss = self.validation_criterion(result['prediction'], target)
```

所以日志里的 `val_loss` 本质上仍然主要是 MSE，而不是 CSI 导向损失。

因此：

```text
val_loss 下降
CSI 下降
```

不是神秘现象，而是当前验证目标本来就允许模型通过平滑降低误差。

---

# 二、当前 full-open 并不是真正的“20步业务目标联合训练”

虽然参数全部开放了，但从优化目标看，它实际上是：

> **全参数参与的 teacher-forced source 机制训练
> ＋低权重、短3步的 free-rollout 正则化。**

并不是：

> **以20步 CSI16/32 和强降水结构保持为中心的联合训练。**

所以这次失败不应该归结为：

* 参数量太小；
* scratch 随机性太强；
* 训练轮数不够。

更核心的问题是：

$$
\boxed{
\text{训练时效、训练状态分布、损失权重}
\neq
\text{实际业务目标}
}
$$

降低学习率只能减慢错误方向，不能改变目标方向。

---

# 三、应该怎样重新设计优化目标

不建议再单纯增加一个损失模块，而应把目标明确分成三个职责。

## 1. 运动项：确保平流本身正确

运动分支需要单独的 teacher-forced transport loss：

$$
L_{\mathrm{motion}}
===================

\sum_t
w(R_t)
,
\rho
\left(
\operatorname{Warp}(R_{t-1}^{true},v_t)-R_t
\right)
+
\lambda_{\mathrm{smooth}}L_{\mathrm{flow\ smooth}}
$$

其中强降水权重可以改为：

$$
w(R)=
1
+2\mathbf1(R\ge16)
+5\mathbf1(R\ge32)
$$

上限可以从当前3提高到6–8，但需要监控梯度。

这个损失专门回答：

> 不考虑 source，运动能否把上一帧搬到正确位置？

---

## 2. Source项：只学习运动无法解释的生消残差

保留现有 factorized source 机制：

* growth；
* steady；
* decay；
* effective source；
* guard；
* source budget。

这部分机制设计本身是合理的。

但 source 的 oracle 应继续基于：

$$
s_t^{oracle}
============

R_t-
\operatorname{Warp}(R_{t-1}^{true},v_t)
$$

并对运动背景停止梯度，避免 source loss 反过来要求 motion 通过错误移动制造一个更容易拟合的残差。

---

## 3. 真正的20步自由 rollout 目标

核心必须变成：

$$
L_{\mathrm{rollout}}
====================

\sum_{t=1}^{20}
\gamma_t
\left[
\lambda_I L_{\mathrm{intensity},t}
+
\lambda_{16}L_{\mathrm{CSI16},t}
+
\lambda_{32}L_{\mathrm{CSI32},t}
+
\lambda_G L_{\mathrm{gradient},t}
+
\lambda_A L_{\mathrm{area},t}
\right]
$$

其中后期权重应不低于前期，例如：

$$
\gamma_t
========

1+\frac{t-1}{19}
$$

第20步约为第1步的两倍，防止模型只优化前几帧。

---

# 四、预防平滑最关键的四项损失

不需要堆十几个模块，先保留四个核心目标。

## 1. 强雨加权的稳健雨强损失

不要用普通 MSE，使用加权 Charbonnier、Huber 或 log-Huber：

$$
L_{\mathrm{intensity}}
======================

\frac{
\sum w(R)
,
\rho\bigl(\log(1+\hat R)-\log(1+R)\bigr)
}{
\sum w(R)
}
$$

它保证数值基本正确，但不让背景主导。

---

## 2. 直接的 Soft CSI16 和 Soft CSI32

当前代码已经有 soft CSI：

$$
\operatorname{SoftCSI}_\tau
===========================

\frac{
\sum p_\tau y_\tau
}{
\sum p_\tau+\sum y_\tau-\sum p_\tau y_\tau
}
$$

问题不是没有，而是：

* 权重过小；
* 只作用3步；
* 温度固定；
* 被其他损失淹没。

第一版可以将权重提高到类似：

```python
soft_csi_16_weight = 0.3
soft_csi_32_weight = 0.6
```

但更重要的是记录每个**加权后损失值和梯度范数**，确保 CSI 项至少贡献总梯度的约 30%–50%，而不是只看配置数字。

温度可以退火：

$$
T:2.0\rightarrow0.75
$$

早期梯度平滑，后期逐渐接近硬阈值。

---

## 3. 梯度/边缘保持损失

用于直接惩罚边缘变钝：

$$
L_{\mathrm{grad}}
=================

\left|
\nabla\hat R-\nabla R
\right|_1
$$

最好只在降水邻域和强降水附近计算，避免背景噪声。

这比单纯加大 MSE 更直接地解决：

* 雨带边缘模糊；
* 强核心扩散；
* 对流结构被抹平。

---

## 4. 面积和强度保持

仅优化 CSI 可能通过扩大预测面积提高 POD，因此仍需要面积约束：

$$
L_{\mathrm{area}}
=================

\sum_{\tau\in{16,32}}
\left|
\log
\frac{
A_\tau(\hat R)+\epsilon
}{
A_\tau(R)+\epsilon
}
\right|
$$

再增加一个强雨峰值或分位数保持项：

$$
L_{\mathrm{peak}}
=================

\left|
Q_{0.95}(\hat R)-Q_{0.95}(R)
\right|
$$

这可以直接防止强雨峰值被压低。

---

# 五、建议的最小可行总损失

第一版不要过于复杂，可以使用：

$$
\boxed{
L=
L_{\mathrm{motion}}
+
L_{\mathrm{source}}
+
0.5L_{\mathrm{rollout\ intensity}}
+
0.3L_{\mathrm{CSI16}}
+
0.6L_{\mathrm{CSI32}}
+
0.1L_{\mathrm{grad}}
+
0.05L_{\mathrm{area}}
}
$$

这些权重只是初始值。实际应对各项做 EMA 归一化：

$$
\widetilde L_i
==============

\frac{L_i}
{\operatorname{stopgrad}(\operatorname{EMA}(L_i))}
$$

然后再加权，避免某个损失仅仅因为数值尺度大而支配训练。

---

# 六、20步 rollout 可以使用课程式训练，但最终必须训练到20步

显存或训练稳定性有限时，可以：

```text
Epoch 0–2:  rollout horizon = 3
Epoch 3–5:  rollout horizon = 6
Epoch 6–9:  rollout horizon = 10
Epoch 10+:  rollout horizon = 20
```

这是训练课程，不是硬冻结分阶段。

最终模型必须在训练阶段实际经历：

$$
\hat R_0
\rightarrow
\hat R_1
\rightarrow
\cdots
\rightarrow
\hat R_{20}
$$

否则无法真正优化第二小时强降水存活。

---

# 七、这次实验应该怎样处理

当前 run 停止是正确的，不建议恢复并跑满50轮。

这次实验已经给出足够明确的否定结果：

> 在当前 teacher-forced 主导、3步短 rollout、弱 CSI 权重的联合目标下，全参数更新降低了业务 CSI。

下一次不应只修改学习率，而应优先修改：

1. rollout 从3步逐渐扩展到20步；
2. CSI16/32 代理损失进入核心目标；
3. 提高强雨像素权重；
4. 加入边缘/梯度保持；
5. 将 checkpoint 与 early stopping 直接绑定业务 CSI 组合指标；
6. 分别记录 motion-only、source gain 和 full rollout，避免模块互相掩盖。

一个可用于选模型的综合指标是：

$$
S=
0.2,CSI_{16}^{0-1h}
+
0.3,CSI_{32}^{0-1h}
+
0.2,CSI_{16}^{1-2h}
+
0.3,CSI_{32}^{1-2h}
$$

同时设置 Bias 和 FAR 保护条件，防止单纯扩大降水面积骗取 CSI。

**结论就是：不是“loss 和业务指标偶尔不一致也正常”，而是当前代理目标没有完成它的职责。你们接下来真正应解决的核心问题，正是让优化目标在20步自由演化、强降水阈值和结构清晰度上与业务指标一致。**
