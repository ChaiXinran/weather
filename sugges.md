结合你现在 GitHub `ChaiXinran/weather` 的最新代码看，这次结果可以解释得更具体了。最新 `mix` 提交里，`DirectPhysicsHybrid` 的实现已经不是“ConvLSTM 和 Temporal U-Net 简单拼接”，而是一个非常明确的 **frozen direct forecast + bounded residual physics adapter**。

而且我认为现在已经能找到为什么它只能从 **0.7880 → 0.7904**，却没有明显推动 `CSI16@0–1h` 的几个结构性原因。它们不只是“参数还没调好”。

## 一、先把当前代码真正做的事情写清楚

当前模型实际上是：

[
Y_t^{dir}=F_{\text{ConvLSTM}}(X)
]

ConvLSTM 直接分支完整加载 0.788 checkpoint，并被冻结。

与此同时：

[
H=\operatorname{FPN}(\operatorname{TemporalUNet}(X))
]

Temporal U-Net 的时间建模是：

* scale 0/1/2：TemporalWeightedFusion；
* scale 3：ConvLSTM；
* 最后经过 FPN；
* correction head 实际拿的是最终 `fine` feature。

每一个未来 lead，head 输入：

[
[H_{\rm fine},Y_t^{dir},E_t]
]

其中 (E_t) 是 lead embedding。

然后产生：

[
v_t=2\tanh(f_v)
]

和

[
S_t=12\tanh(f_s)
]

代码随后先把 **ConvLSTM 已经预测出来的未来帧**进行 residual warp：

[
Y_t^{warp}
==========

W(Y_t^{dir},v_t)
]

并且 warp 确实是在 **rain-rate space** 做的，这一点是正确的，它避免了之前 normalized-dBZ bilinear warp 对强回波的严重侵蚀。

再加 source：

[
R_t^{phys}
==========

R(Y_t^{warp})+S_t
]

最后转换回 normalized dBZ：

[
Y_t^{phys}
==========

Z^{-1}(R_t^{phys})
]

真正输出则是：

[
\boxed{
Y_t
===

Y_t^{dir}
+
\alpha_t(Y_t^{phys}-Y_t^{dir})
}
]

对应代码完全如此。

所以这次 0.7904 其实验证的是：

> **Temporal U-Net 能否产生一个对 0.788 ConvLSTM 有益的物理残差候选，并通过一个小系数注入最终预测。**

这个设计目前是成功的，但有几个非常明显的瓶颈。

---

# 二、我认为现在最大的瓶颈其实是 `alpha`：它不是空间 gate

这是代码里非常关键的一点。

现在：

```python
self.blend_logit = nn.Parameter(
    torch.zeros(configs.aft_seq_length)
)
```

也就是说只有：

[
20
]

个参数。

最终：

```python
learned_alpha =
    alpha_max * torch.tanh(self.blend_logit)
```

所以：

[
\alpha_t
]

**一个 lead 只有一个全局 scalar。**

它不是：

[
\alpha_t(x,y)
]

。

这意味着假设同一张图里：

```text
区域A：
ConvLSTM漏报
physics可以正确增强

区域B：
ConvLSTM已经预测正确
physics反而会破坏

区域C：
ConvLSTM虚警
physics可以正确减弱
```

模型根本不能分别处理。

它只能选择：

> “第30分钟这一整张图到底用多少physics？”

所以优化以后最安全的答案自然就是：

[
\boxed{\alpha_t\rightarrow\text{很小}}
]

因为只要 physics correction 有好有坏，全局平均下来，gate 就不敢放大。

这与结果：

[
+0.0024
]

非常一致。

### 甚至当前 `alpha` 严格来说都不是“blend gate”

因为：

[
\alpha_t
========

0.08\tanh(z_t)
]

所以：

[
-0.08\le\alpha_t\le0.08
]

它允许负数。

也就是说最终可能：

[
Y^{final}
=========

## Y^{direct}

0.05(Y^{phys}-Y^{direct})
]

即**向 physics candidate 的反方向修正**。

从优化角度没问题，但如果论文说：

> “模型根据物理可信度决定是否采用物理结果”

这个解释就不严格了。

它现在实际上更像：

> **signed per-lead residual scaling coefficient**

而不是物理置信 gate。

这是我会优先改的第一件事。

---

# 三、第二个非常重要的问题：现在的“growth / decay”实际上并没有真正分解

报告里把 architecture 描述成：

```text
motion / growth / decay correction
```

但从代码看，当前 `DirectPhysicsHybrid` 并没有真正的：

```text
growth_head
steady_head
decay_head
```

只有：

```python
self.source_head = nn.Conv2d(...)
```

然后：

```python
source_rain = 12 * tanh(source_logit)

growth = source_rain.clamp_min(0)
decay  = (-source_rain).clamp_min(0)
```

也就是说：

[
S>0
\Rightarrow growth
]

[
S<0
\Rightarrow decay
]

只是**输出以后按符号拆开**。

这和之前 factorized source：

[
p_g,\ p_s,\ p_d,\ \alpha_g,\ \alpha_d
]

不是一回事。

所以现在严格来说，你证明的是：

[
\boxed{
\text{residual motion}
+
\text{signed rain-rate residual}
}
]

有效。

还不能特别强地说：

> 网络显式识别了增长、稳定和衰减机制。

这点如果以后想成为论文的“物理可解释性”核心，是需要继续完善的。

---

# 四、第三个问题可能直接解释“为什么MSE涨得多，CSI16几乎不涨”

当前 loss 继承的是 `ConvLSTM_r2d.py`。

你的 R2 loss 是：

[
L=L_{\rm weightedHuber}+L_{\rm softCSI}
]

其中强降水像素 Huber 权重：

```text
>=16 : +2
>=32 : +3
```

但 soft CSI 权重非常小：

```text
第一小时：
CSI16  0.00180
CSI32  0.00090

第二小时：
CSI16  0.00216
CSI32  0.00108
```

而实际实现中就是：

```python
total = huber + soft_csi
```

没有额外归一到二者相同量级。

所以模型优化的核心依然是：

[
\boxed{weighted\ Huber}
]

而不是 CSI。

这就完美对应你的结果：

```text
MSE：
-2.86%

Aggregate CSI：
+0.30%

CSI16 0–1h：
+0.00052
```



模型确实在学习，而且不是没学到东西。

只是它主要在学：

> “怎样把大量像素的连续数值修得更准。”

不是：

> “怎样让 14 mm/h → 17 mm/h，跨过16阈值，从miss变成hit。”

而你的最终目标恰恰是后一件事。

---

# 五、还有一个更深的目标错配：physics auxiliary 训练的不是“修正”

现在 warm-up：

```python
aux = self.criterion(
    result['physics_prediction'],
    batch_y
)
```

也就是说前三轮实际上要求：

[
Y^{phys}\rightarrow Y
]

让 physics branch 自己尽量成为一个完整 forecast。

但是你的架构真正想要的并不是：

> physics branch 再训练一个新的天气预报器。

而是：

> physics branch 找出 0.788 ConvLSTM 的错误，然后只修这些错误。

理论上应该更接近：

[
\Delta^*
========

Y-Y^{direct}
]

然后学习：

[
\Delta^{phys}\approx\Delta^*
]

特别是：

[
\Delta^*
]

在强降水 miss / false alarm / displacement 区域的结构。

现在的 aux objective 会诱导 Temporal U-Net：

> “尽量重建整个未来场。”

所以它很容易又学成一个比较平滑的替代预测器。

最后全局 alpha 再取其中8%以内。

这很容易得到现在这种：

[
MSE\downarrow
]

但：

[
CSI\approx\text{不动}
]

的状态。

---

# 六、现在还有一个 blanket anchor 在明确阻止模型做大修正

训练里面：

```python
anchor = smooth_l1(
    prediction,
    direct_prediction.detach()
)
```

并且：

```text
weight = 0.10
```

同时还有：

```text
alpha regularization = 0.01
```

它们的作用是：

[
\boxed{\text{不要离0.788 baseline太远}}
]

在第一阶段是非常正确的——所以没有再发生 collapse。

但是现在你的目标已经不是：

> 保持0.788。

而是：

[
CSI16_{0-1h}:
0.335\rightarrow0.45\sim0.50
]

这意味着必须在一些区域发生**相当明显的改变**。

现在 anchor 对所有像素一视同仁：

```text
baseline已经正确 → 不让改
baseline明显错误 → 还是不让改
```

这就不合理了。

真正需要的是：

[
L_{\rm anchor}
]

只强约束 baseline 正确区域。

baseline miss / false alarm 区域反而应该：

[
w_{\rm anchor}\approx0
]

允许 physics 大改。

---

# 七、最值得注意的另一个细节：最终 blend 又回到了 dBZ 空间

physics 内部做得很好：

[
dBZ
\rightarrow rain
\rightarrow warp
\rightarrow source
\rightarrow dBZ
]

但是最后：

```python
prediction =
    direct
    + alpha * (physics - direct)
```

这里：

```text
direct
physics
prediction
```

全部都是 normalized dBZ。

也就是说最终混合实际上发生在：

[
dBZ
]

这个对数空间。

这意味着：

[
(1-\alpha)dBZ_{dir}+\alpha dBZ_{phys}
]

并不等价于：

[
(1-\alpha)R_{dir}+\alpha R_{phys}
]

对于你特别关心的16/32 mm/h crossing，这个区别并不小。

尤其现在：

[
\alpha_{\max}=0.08
]

本身就很小，再在对数空间做8%的 interpolation，真正落实到 rain-rate 上的强度修正可能更弱。

我认为下一版值得直接比较：

### 当前

[
dBZ_{final}
===========

dBZ_{dir}
+
\alpha(dBZ_{phys}-dBZ_{dir})
]

### Physical-space residual

[
R_{final}
=========

R_{dir}
+
\alpha(R_{phys}-R_{dir})
]

最后：

[
dBZ_{final}=Z^{-1}(R_{final})
]

这在物理意义上也更统一。

---

# 八、Temporal U-Net 本身现在在干什么，也值得注意

它不是在递归预测20帧。

它只从10帧 history 提取一次：

[
H_{\rm fine}
]

然后：

```python
fine = pyramid['fine'][:, None].expand(..., t, ...)
```

也就是说：

[
H_{\rm fine}
]

20个 lead 全部共享。

每个 lead 的区别主要来自：

[
Y_t^{direct}
]

和：

[
E_t
]

lead embedding。

所以当前 correction branch 很容易学成：

[
\Delta_t
========

f(Y_t^{direct},t,H)
]

而不是严格意义上的：

[
\text{explicit evolving physical state}
]

这其实很好地解释为什么目前表现像一个：

> **ConvLSTM forecast calibration/refinement network**

而不是非常强的：

> motion-growth-decay simulator。

---

# 九、结合代码以后，我对当前 +0.0024 的解释变了

报告本身说：

> hybrid correction mainly improves first-hour 32 mm/h CSI and reduces second-hour over-coverage. 

结合代码，我觉得可以进一步解释成：

[
\boxed{
\text{Temporal U-Net 已经学会了一些 ConvLSTM 条件校准}
}
]

特别是：

[
\text{过预测 suppression}
]

但是模型结构现在天然不利于 **miss recovery**：

| 当前机制                  | 对 miss recovery 是否友好 |
| --------------------- | -------------------- |
| 全局 per-lead alpha     | ❌                    |
| alpha 最大0.08          | ❌偏保守                 |
| blanket direct anchor | ❌                    |
| alpha regularization  | ❌                    |
| Huber主导loss           | ❌                    |
| physics完整预测aux        | ❌                    |
| source可正可负            | ✅                    |
| rain-space warp       | ✅                    |
| Temporal U-Net空间特征    | ✅                    |
| direct forecast作为条件   | ✅                    |

所以我现在**不认为0.7904说明Temporal U-Net没有潜力**。

反而更像：

> 当前 correction framework 被设计得过于“安全”，以至于它只敢做全局的小校准。

---

# 十、下一版我会怎么改：不要大改网络，先改“修正机制”

按照当前代码，我会按这个优先级做，而且一次只改一项：

1. **先加完全不影响训练的 attribution diagnostic。** 对16/32 mm/h分别统计 `miss→hit`、`hit→miss`、`FA→correct negative`、`correct negative→FA`，同时输出每个 lead 的 `alpha`、flow abs、positive source、negative source，以及 standalone physics candidate CSI。现在报告只有最终指标，无法知道 +0.0024 是哪些像素贡献的。

2. **把全局 (\alpha_t) 改成 spatial gate (\alpha_t(x,y))。** correction head 已经有 `hidden[B*T,C,H,W]`，直接多接一个 `gate_head = Conv2d(...,1,1)` 就可以。初始仍然保持zero-start。这个改动我认为是当前潜在收益最大、又最符合已有架构的修改。

3. **把 fused correction 放到 rain-rate space。** 不再先把 physics 转成 dBZ 再blend，而直接计算
   [
   R_{final}=R_{direct}+g(x,y,t)(R_{physics}-R_{direct})
   ]
   然后一次性转换回dBZ。

4. **把 blanket anchor 改成 baseline-aware anchor。** baseline正确区域强anchor；baseline miss/FA区域弱anchor。比如16 mm/h miss：
   [
   R_{dir}<16,\quad R_{true}\ge16
   ]
   就不要再要求 prediction 接近 direct。

5. **把 physics auxiliary 从“完整预测真值”改成“解释 baseline residual”。**
   [
   \Delta^*=R_{true}-R_{direct}
   ]
   source/motion branch应该主要学习这个，而不是重新承担全部forecast。

6. **最后才考虑把 alpha_max 0.08 提到0.12/0.15。** 现在直接提高我不建议，因为当前 correction 明显偏 suppression，会先进一步降低 intensity ratio。

---

# 十一、我现在最看好的具体形式

如果稍微进一步整理，我会把当前：

[
Y_{final}
=========

Y_{direct}
+
\alpha_t(Y_{phys}-Y_{direct})
]

改成：

[
\boxed{
R_{final}
=========

R_{direct}
+
g_m(x,y,t)\Delta R_{motion}
+
g_s(x,y,t)S(x,y,t)
}
]

其中：

[
\Delta R_{motion}
=================

W(R_{direct},v)-R_{direct}
]

而：

[
S
]

就是 source。

这样 motion 与 source 不再先混成一个 `physics_prediction` 再共享一个 alpha。

于是可以直接分析：

[
\Delta CSI_{\rm motion}
]

和：

[
\Delta CSI_{\rm source}
]

这对于论文特别重要：

> 强基线出现位置偏差时，motion gate激活；
> 出现局地增强/减弱错误时，source gate激活。

这就比当前“一个physics candidate + 一个全局alpha”真正有解释性得多。

---

# 十二、还有一个实验协议问题建议现在就钉死

历史报告里的 ConvLSTM 最佳值是：

```text
0.788316
```

而当前 hybrid 报告实际复现出来的是：

```text
0.788006964
```



差：

[
0.000309
]

本身很小。

但你现在宣称的提升才：

[
0.002397
]

所以这个差值已经相当于当前增益的大约13%。

当前代码载入的是：

```text
best_val_csi.ckpt
```

而不是把历史记录中的：

```text
val-csi-epoch=02-val_csi_score=0.788316.ckpt
```

硬编码进去。

代码已经有 checkpoint 完整性 audit，会验证 `cell_list + conv_last` 是否完整加载以及 zero-start 是否完全复现 direct output，这是好的。

但下一步最好再固定：

> **checkpoint SHA / exact filename + exact evaluator version**

否则当你的增益处于0.002量级时，这种0.0003级协议差异就不能忽略。

---

## 最后的判断

结合代码，我现在不会把结果总结成：

> “Temporal U-Net只能给ConvLSTM增加0.0024。”

更准确的是：

> **当前实现只允许 Temporal U-Net 通过一个全局、每lead、最大±8%的 dBZ-space residual coefficient 去修正冻结ConvLSTM，同时受到全局anchor和以Huber为主的loss限制。在这个非常保守的条件下，它仍然从0.7880稳定提高到了0.7904。**

这其实是好消息。

现在最值得突破的不是继续扩大网络，也不是继续训练，而是把：

[
\boxed{\alpha_t}
]

升级成：

[
\boxed{\alpha_t(x,y)}
]

再把 correction supervision 从“重新预测天气”改成：

[
\boxed{\text{专门纠正0.788 ConvLSTM的miss / FA / displacement}}
]

我认为这是当前代码里最清晰、最可能把 **0.335 的 CSI16@0–1h 真正往上推** 的下一步，而不是再围绕 `alpha_max=0.08` 做小范围调参。
