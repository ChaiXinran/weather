# V3a 效果优先实施方案：Error-aware Routed Motion + Decay

更新日期：2026-08-08

## 1. 目标与开发原则

当前工程基线是 `DirectPhysicsHybrid V2 clean_manifest`，统一 Validation
weighted CSI score 为 `0.937194`。V3a 不预设硬性晋级阈值，直接观察完整结果，
重点关注 +60/+120 min 的 CSI16、CSI32 与 FAR16、FAR32。

开发原则：

1. 效果优先，允许重写融合方式，不要求兼容 V2 内部 source/gate 设计；
2. ConvLSTM 保留为稳定 prior，但 U-Net 必须能在局部强力覆盖其错误；
3. V3a 只做 `preserve / motion / decay`，不让不可靠的 growth 混入；
4. 先判断错误类型，再路由到互斥候选，禁止多个 residual 无条件相加；
5. 少做形式化验证：机制跑通前只用 seed 0 和 Validation，不碰 Test；
6. 保留 V2 原模型、配置和 checkpoint，V3a 新建独立实现，便于随时回退。

## 2. V3a 模型定义

### 2.1 ConvLSTM prior

完整加载并冻结当前纯 ConvLSTM checkpoint，生成未来 20 帧 direct prior：

\[
D_{1:20}=F_{ConvLSTM}(X_{1:10}).
\]

V3a 第一轮不联合微调 ConvLSTM。只有 routed U-Net 已经稳定产生正作用后，才
允许在后续单独实验中以极低学习率解冻 `conv_last` 或最后一层 cell。

### 2.2 U-Net 输入与时效上下文

当前 V2 的 U-Net 主要编码历史 Radar，并在 head 中逐时效拼接单帧 `D_t`。
V3a 应显著增强 prior-error context。每个 lead 至少输入：

```text
history U-Net/FPN feature
D(t-1), D(t), D(t+1)
D(t)-D(t-1)
|spatial_gradient(D(t))|
lead embedding
```

边界 lead 复制最近可用帧。这样 router 能看到 direct 的持续、扩张、衰减和
移动趋势，而不是只根据一张预测图猜测它是否错误。

U-Net/FPN 是 V3a 的主要可训练容量。可以继续使用 V2 的轻量通道配置作为
初始化，但不再用全局 alpha 或小比例上限限制它的局部作用。

### 2.3 三个候选专家

所有候选在 rain-rate 空间构造：

\[
R^{preserve}_t=D_t,
\]

\[
R^{motion}_t=\mathcal W(D_t,\Delta U_t),
\]

\[
R^{decay}_t=D_t(1-Q^{decay}_t),
\qquad Q^{decay}_t\in[0,1].
\]

- residual flow 最大位移首版保持 2 px；
- decay 可以在局部将 direct 回波削弱到 0，不设置 0.08/0.25 的全局混合上限；
- 删除 V2 signed source、source gate 和 growth 输出，避免旧 source 干扰。

### 2.4 Soft router

Router 输出逐像素、逐 lead 的三个 logits：

\[
p_t=\operatorname{Softmax}(z_t/\tau),
\qquad p_t\in\mathbb R^{3\times H\times W}.
\]

最终输出：

\[
\hat R_t=p_t^pR_t^{preserve}
+p_t^mR_t^{motion}
+p_t^dR_t^{decay}.
\]

初始化以 preserve 为主，但不要使用接近永久关闭的 `bias=-8`。建议初始概率
约为 `preserve=0.8, motion=0.15, decay=0.05`，让 motion/decay 从第一轮就有
梯度和可观察作用。训练后允许任一错误像素由 motion 或 decay 完全接管。

## 3. Routing truth 与缓存

标签协议以 `.research/mixed/v3a_routing_protocol.md` 为准：

```text
16 mm/h object footprint
+ 32 mm/h core refinement
+ r=2 px
+ w16=1.0 / w32=1.5 soft labels
+ ambiguous ignore
```

对象匹配需要 frozen ConvLSTM direct 与 training target，成本较高，不应在每个
DataLoader batch 内重复运行。建议一次性生成 routing cache：

```text
V3A_ROUTING_CACHE/
├── labels.npy       # uint8 encoded 16/32 route classes or compact soft labels
├── confidence.npy   # uint8 confidence/ignore mask
├── manifest.json    # sample identity, direct checkpoint hash, protocol params
└── summary.json     # preserve/motion/decay/ignore counts by lead/threshold
```

缓存必须绑定：

- `.research/bth_2025_events.json` 的样本顺序；
- frozen direct checkpoint 路径和 SHA/hash；
- r、IoU、面积比、16/32 权重；
- train/val split；
- 生成代码版本。

首版直接使用 `r=2`。`r=1/3` 只在后续确有必要时运行 oracle，不阻塞主实验。

## 4. 分阶段训练

### Stage A：Motion + Decay 专家预训练

初始化：

- 从 V2 best checkpoint 加载 `features / decoder / head / flow_head`；
- 丢弃 V2 `source_head / motion_gate_head / source_gate_head`；
- 新建 `decay_head` 和 `router_head`；
- ConvLSTM frozen；router frozen。

训练目标：

```text
motion expert: 仅在高置信 motion 区域优化 warped candidate
decay expert:  仅在高置信 false-alarm/decay 区域优化衰减比例
preserve:      不需要参数
```

Decay 连续目标可取：

\[
Q^*=\operatorname{clip}((D-Y)/(D+\epsilon),0,1).
\]

Motion 使用候选场相对 target 的区域 R2d/Huber，并保留轻量 flow smoothness。
不对 ignore 区域计算专家分类监督。

建议先跑 3 epoch。这里只看 candidate attribution，不因单轮波动重做调参。

### Stage B：Router 预训练

冻结 ConvLSTM、motion expert 和 decay expert，只训练 router 2–3 epoch。

损失：

\[
L_{route}=-\sum_k y_k\log p_k
\]

只在 confidence mask 内计算，并按 route 类别频率做温和重加权，防止全部
塌缩到 preserve。ignore 区域不参加 routing loss，但仍可参加最终场损失。

Router 预训练期间同时记录 learned route 和 oracle route 的候选组合结果，
但不要求先达到某个门槛才能进入下一阶段。

### Stage C：联合强修正

解冻 U-Net、motion、decay 和 router，ConvLSTM 仍冻结，联合训练 5–10 epoch。

建议总损失：

\[
L=L_{R2d}(\hat R,Y)
+\lambda_rL_{route}
+\lambda_mL_{motion}
+\lambda_dL_{decay}
+\lambda_{keep}L_{preserve}.
\]

其中 `L_preserve` 仅在 direct 已正确且 routing truth 为 preserve 的区域约束，
不能再对整幅 fused output 做 direct-anchor。错误区域允许强修正。

如果远期 FAR 仍然很高，再增加第二小时 decay/FA 权重；不要同时引入 growth、
改 backbone 和改 loss，以免无法知道哪一项产生作用。

### Stage D：可选联合微调 ConvLSTM

只有 Stage C 的 learned routed 已经优于 preserve-only 时才尝试：

- 优先只解冻 `conv_last`；
- 或只解冻最后一层 ConvLSTM cell；
- 学习率约为 U-Net 的 1/20–1/50；
- 完整 ConvLSTM 解冻不是默认方案。

## 5. 代码改造地图

为了不破坏当前最好基线，建议新增而不是重写：

```text
openstl/models/direct_physics_routed_model.py
openstl/methods/direct_physics_routed.py
openstl/modules/v3a_routing.py
openstl/datasets/v3a_routing_cache.py
configs/bth_radar/DirectPhysicsRouted_v3a.py
configs/bth_radar/DirectPhysicsRouted_v3a_router.py
configs/bth_radar/DirectPhysicsRouted_v3a_joint.py
tools/build_bth_v3a_routing_cache.py
tools/evaluate_v3a_attribution.py
```

注册位置：

```text
openstl/models/__init__.py
openstl/methods/__init__.py
openstl/utils/parser.py
openstl/api/exp.py
```

建议 model 的 `return_aux=True` 固定返回：

```text
prediction
direct_prediction
preserve_prediction
motion_prediction
decay_prediction
residual_flow
decay_fraction
route_probability
route_logits
```

这样现有自动报告工具可以扩展，而不需要从内部 hook 抓变量。

## 6. 最少验证集合

机制探索阶段只保留三项：

1. **routing cache 抽查**：随机显示少量对象匹配和 ignore mask，防止标签方向
   或样本索引错误；
2. **初始化检查**：确认 checkpoint 映射完整、各候选形状/范围正确；
3. **一次完整 Validation 报告**：每个阶段最佳 checkpoint 用统一报告工具产生
   +60/+120 CSI/FAR 和 candidate attribution。

暂不做：

- seed 1/2；
- Test；
- bootstrap 显著性门槛；
- r=1/2/3 三套训练；
- 大规模超参数搜索；
- growth/PWV/DEM。

## 7. 建议实验顺序

```text
E0  V2 baseline/report reuse，不重训
E1  V3a expert pretrain（motion + decay，3 epoch）
E2  V3a router pretrain（2–3 epoch）
E3  V3a joint（5–10 epoch）
E4  仅在需要时：late-lead decay weighting
E5  仅在 V3a 有效后：加入 growth + PWV
```

每次实验只改一个主要机制，但无需等待完整统计验证后才开始下一个结构尝试。

## 8. 报告重点

每份 V3a 报告至少列出：

```text
Preserve/direct
Motion candidate
Decay candidate
Oracle routed
Learned routed
```

以及：

```text
weighted validation CSI score
CSI16/32 and FAR16/32 at +60 min
CSI16/32 and FAR16/32 at +120 min
0–1 h / 1–2 h CSI, POD, FAR, Bias
route probability/count by lead
Miss→Hit, Hit→Miss, FA→Correct, Correct→FA
```

最终判断以实际效果排序，不设置预先淘汰线。`0.937194` 和 Stage 1 endpoint
只作为同协议参考坐标。

## 9. 已实现状态（2026-08-08）

已完成 routing packed-cache builder、sample-aligned mmap reader、独立 V3a
model/method、expert/router/joint 三套配置、V3a attribution，以及 +60/+120
CSI16/32 训练日志。实际命令以 `.research/detail.md` 第 18.8 节为准。
