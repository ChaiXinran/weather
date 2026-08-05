# R4-c2 源汇机制验证计划

状态：待实施  
阶段：R4 - 运动—源汇分解  
更新日期：2026-08-05

## 1. 阶段结论

下一步停止继续训练 R4-c1，不直接解冻 encoder 或 motion，也不引入 PWV、DEM
或新的时序主干。启动新的 R4-c2 机制实验：固定已经验证的 R4-b 运输骨架，
把 source 重构为逐 lead、状态条件化、物理有界的 signed tendency，并先在
teacher-forced 路径上验证它是否真的学会非平流的降水增强和衰减。

本阶段的核心问题是：**在冻结且已校准的 Radar 运动场下，逐步状态条件化、物理
有界的 signed tendency 模块，能否从 Radar 历史中预测可辨识的非平流强度变化，
在不依赖边缘位移补偿、不破坏运动分支空间优势的条件下，改善 16/32 mm/h 降水
的增强、维持、减弱、新生和消散？**

source 不是锐化器或双线性插值的专用反模糊模块。数值 warp 扩散和真实非平流
生消是两个不同来源；本阶段允许 source 顺带恢复部分强核，但必须辨别它主要在
表达对象内部强度变化与真实新生/消散，还是在对象边缘形成正负偶极以修补 flow
和插值误差。后者不能单独作为源汇机制成立的证据。

R4-c0 已给出继续研究 source 的依据：强降水缺失残差显著，增强区以正残差为主、
衰减区以负残差为主，训练与验证分布一致，且 32 mm/h 区域的 oracle source
绝对值 P99 约为 33--35 mm/h。R4-c1 的失败不视为 source 机制无效，而视为当前
参数化、条件输入和训练路径不足。

已观测的 oracle residual 量级也不是可忽略的插值噪声：active 区绝对均值约
0.582 mm/h，16 mm/h 区约 6.458 mm/h，32 mm/h 区约 9.607 mm/h；growth 区
72.6% 为正、decay 区 76.3% 为负。但这些残差仍混有 flow 剩余误差、形态变化、
分裂合并、插值误差和不可预测变化，不能被称为纯物理 source 真值。

## 2. 固定边界

以下内容在 R4-c2 Gate 1 结束前保持不变：

- 使用 R4-b checkpoint：
  `work_dirs/bth_r4b_motion_rainrate_scale05_ft5ep_from0633323_seed0/checkpoints/val-csi-epoch=01-val_csi_score=0.640662.ckpt`。
- encoder、motion head 和已选择的 0.5 flow 标定全部冻结。
- 任务仍为 10 帧历史预测 20 帧未来，每帧间隔 6 分钟，网格为 66 x 70。
- source 和演化计算在 rain-rate 空间进行；Z-R 固定为 `Z=200R^1.6`。
- 使用现有 `RADAR_CACHE_UINT8` 和 `.research/bth_2025_events.json`。
- 保持事件级 train/val/test 划分；不得将重叠滑窗随机拆分到不同集合。
- 只用 train 选择结构、权重和阈值，val 用于机制确认，暂不访问 test。
- Gate 1 不加入 PWV、DEM、双 source/sink head 或新时序 backbone。
- Gate 1 不用 source L1 和时间 TV；oracle residual 只作诊断或弱辅助。

R4-b 是固定基线，R4-c1 是已知失败对照。不得覆盖两者的 checkpoint、配置或
报告目录。当前未提交的
`.research/history/r4c0_r4c1_source_analysis.md` 也不在本计划的修改范围内。

## 3. 目标参数化

对每个 lead `t`，先得到当前平流后的雨强 `A_t`，source decoder 输出 logit
`z_t`。当前 50 dBZ 表示上限在固定 `Z=200R^1.6` 下对应
`R_max = ((10^(50/10))/200)^(1/1.6) ~= 48.6 mm/h`，因此正 source 的可用容量为：

```text
u_t = tanh(z_t)
C_t+ = min(S_max, max(R_max - A_t, 0))

S_t = C_t+ * u_t,   u_t >= 0
S_t = A_t * u_t,    u_t < 0

R_t = A_t + S_t
```

其中 `S_max = 35 mm/h`。实现应使用张量运算保持分段两侧可求导，且不再在结果
上调用 `clamp_min(0)`。该表达必须满足：

- `z_t = 0` 时 `S_t = 0`，模型逐像素复现 R4-b；
- `0 <= S_t <= min(35 mm/h, R_max-A_t)`（正分支）；
- `-A_t <= S_t <= 0`（负分支）；
- `0 <= R_t <= R_max`，sink 和超上限 source 均无法被额外 clamp 隐藏；
- 只有一个 signed 自由度，不允许 source 和 sink 同时增大后相互抵消。

`rain_to_normalized_dbz` 当前会把输出裁剪到 `[0,1]`。bounded source 必须在进入
该转换前满足上下界，使转换函数的 clamp 只保留为数值防护，而不是模型机制的一
部分。使用 `torch.where(u >= 0, ...)` 时零点选择正分支，左右梯度尺度分别受正容量
和 `A_t` 控制；该零点不光滑性作为 Gate 0 的显式检查项，不能假定零初始化自然
无偏。

共享的逐步 decoder 定义为：

```text
z_t = D_s(F, A_t, v_t, delta_recent, e_t)
```

`F` 是冻结 encoder 的最后历史特征；`A_t` 是本 lead 的平流雨强；`v_t` 是固定
motion head 给出的二维 flow；`e_t` 是 lead embedding；`delta_recent` 在 c2b
才启用，包括 `R_0-R_-1` 和 `R_0-R_-5`。

拼接和主要卷积在 patch 分辨率完成，输入归一化固定如下：

- `A_t` 用 `A_t/R_max` 作为首选基线；若改用 `log1p(A_t)/log1p(R_max)`，必须作为
  独立配置消融，不得静默替换。
- 雨强和 tendency 使用 area pooling 对齐 patch 网格。
- 全分辨率像素单位的 flow 用 bilinear 降采样；`patch_size=2` 时位移同步除以 2，
  再除以 patch 网格最大位移得到无量纲 flow。
- `R_0-R_-1` 与 `R_0-R_-5` 先在 rain-rate 空间计算，再以 train split 分位数或
  `S_max` 归一化并截断；采用的尺度写入配置快照。
- lead embedding 扩展到 patch 网格；decoder 输出 logit 再上采样到原始网格，
  最后应用物理有界变换。

所有缩放常数和插值模式都必须进入配置或固定代码注释，确保实验可复现。

## 4. 代码改动范围

### 4.1 演化算子

文件：`openstl/modules/evolution_operator.py`

- 增加由 `source_logit`（或等价的 `u_t`）和 `A_t` 计算 bounded signed source
  的单步函数，集中维护物理约束。
- source 路径返回 `advected_rain`、`source_rain`、`evolved_rain` 和必要的诊断量。
- 有界路径在 rain-rate 到 normalized dBZ 转换前断言或统计 `R_t` 的上下界，避免
  转换函数的 `[0,1]` clamp 静默吸收超限值。
- 保留无 source 路径的现有行为，确保旧配置和 R4-b 不回归。
- 有界 source 路径移除结果端 `clamp_min(0)`；旧 R4-c1 路径如需保留，应由显式
  配置区分，避免静默改变历史实验语义。

### 4.2 模型与逐步 decoder

文件：`openstl/models/evolution_convlstm_model.py`

- 用共享的小型 per-step source decoder 替代“一次从 `F` 输出未来 20 帧 source”。
- motion head 仍一次生成固定的未来 flow；source decoder 在每个 lead 接收本步
  `A_t` 和 `v_t`，因此 source 预测必须位于演化循环内部。
- 增加 lead embedding，长度至少覆盖配置中的 `aft_seq_length`。
- c2a 输入：`F, A_t, v_t, e_t`。
- c2b 在 c2a 基础上加入 `R_0-R_-1` 和 `R_0-R_-5`。
- decoder 最后一层零初始化；加载 R4-b 后首次前向必须严格产生零 source。
- `forward` 应能明确选择 rollout 与 teacher-forced previous-state 路径，并返回
  两条路径需要的辅助张量，避免训练方法在模型外复制演化逻辑。
- 保留旧 checkpoint 的非严格加载能力，并记录实际加载、缺失和新增参数。

### 4.3 训练方法

文件：`openstl/methods/evolution_convlstm.py`

- Gate 1 的主路径按 lead 使用真实上一帧：
  `A_t^TF = Warp(R_(t-1)^true, v_t)`，其中第一个 lead 的 previous state 为最后一帧
  历史观测。
- 主损失直接比较 `R_hat_t^TF` 与 `R_t^true`，不把 oracle source 当作纯净标签。
- Gate 1 主损失是在 rain-rate 空间直接计算的区域 Huber（或单独配置的
  Charbonnier）状态损失，不直接复用包含 soft CSI、空事件惩罚和分时段权重的
  完整 R2 loss。对区域 mask `M` 定义：

  ```text
  L_M = sum(M * rho(R_hat_t^TF - R_t^true)) / max(sum(M), 1)
  L_state = L_active + lambda_16*L_16 + lambda_32*L_32
  ```

- 同时独立记录 `L_all`、`L_active`、`L_16`、`L_32`；各 mask、Huber beta 和
  `lambda_16/lambda_32` 必须在 train-only smoke 后预注册，不能看完 val 后调整。
- soft CSI 只作诊断；若以后加入优化目标，作为独立消融。
- oracle residual 保留为 detached 诊断；若后续试验弱辅助，必须另建可消融配置，
  不与 c2a/c2b 主结果混写。
- Gate 1 中 source sparse L1 和 temporal TV 权重为 0；空间平滑默认为 0，只有出现
  明确棋盘伪影时才允许以独立实验加入极小权重。
- optimizer 明确包含 source decoder 和 lead embedding，且只允许这些参数更新。
- 增加 source 正负比例、绝对值均值/P50/P90/P99、按 lead 和强度区间分布，以及
  `evolved_rain>R_max`、转换后 dBZ 等于 1、正 source 容量触发、sink 清空到 0 的
  比例等日志。

### 4.4 配置

新增配置，不修改 R4-c1 原配置：

- `configs/bth_radar/ConvLSTM_evolution_source_c2a_tf.py`
- `configs/bth_radar/ConvLSTM_evolution_source_c2b_tf.py`
- Gate 2 通过批准后再新增
  `configs/bth_radar/ConvLSTM_evolution_source_c2c_mixed.py`。

配置项应显式表达 source 参数化版本、decoder 输入、lead embedding 维度、训练
路径、冻结策略、state loss 权重和 scheduled-sampling 策略，禁止依赖无法从运行
产物恢复的隐式默认值。

### 4.5 测试

主要文件：

- `tests/test_modules/test_evolution_operator.py`
- `tests/test_models/test_evolution_convlstm.py`
- 视职责新增 `tests/test_methods/test_evolution_convlstm.py`
- `tests/test_configs/test_bth_evolution_convlstm_config.py`

现有 R4-b 无 source 和 R4-c1 测试继续保留，新测试只补充 c2 行为。

## 5. Gate 0：工程检查

Gate 0 是几十到几百步的工程验证，不得作为研究结论。

### 5.1 单元与集成检查

- 零初始化 c2 source 与 R4-b prediction、flow 逐像素一致。
- 对随机和边界输入验证 `source_rain >= -A_t`、正 source 不超过
  `min(35, R_max-A_t)`、`0 <= evolved_rain <= R_max`，并确认有界路径未依赖
  结果端 clamp。
- 人工构造 `z_t > 0` 与 `z_t < 0`，两侧 decoder 参数梯度均非零且有限。
- 检查零初始化后的第一次更新：全场及 growth/decay 子集的梯度方向、source
  正负比例和零附近 logit 分布；不得出现所有像素由分段零点统一推向正 source。
- 每个 lead 的 decoder 输入和输出 shape 正确，同一共享 decoder 被重复调用。
- 改变 `A_t` 或 lead id 时 source 输出可响应；固定输入时结果可复现。
- teacher-forced 第一个 previous state 是 `R_0`，后续严格对应真实 `R_(t-1)`，
  不存在一帧错位或未来信息进入 encoder。
- source decoder 与 lead embedding 参数在 optimizer 中；encoder/motion 参数不在
  optimizer 或 `requires_grad=False`。
- R4-b checkpoint 加载后，对新增参数、缺失键和冻结参数进行断言。
- 固定 batch 必须同时包含 growth、decay 和 16/32 mm/h 强回波；过拟合几十到几百
  步后，teacher-forced state loss 至少下降 50%，growth/decay 符号准确率均相对
  初始值明确提高，正负 source 都被激活，且 source 不大面积触及上下界。
- 无 source、旧 source 配置的既有测试均通过。

### 5.2 Gate 0 退出条件

所有上述断言通过；没有 NaN/Inf；固定 batch loss 达到上述工程标准；显存可支持当前
`batch_size=8`，否则只调整 batch size，不改变模型机制。任一物理约束、时间索引
或冻结检查失败时停止，不启动正式训练。

Gate 0 产物放入独立 smoke 目录，至少保存配置快照、Git 状态、checkpoint 加载
摘要、首末 loss、梯度统计和一个 batch 的 source 分布。

## 6. Gate 1：Teacher-forced 机制实验

### 6.1 实验顺序

1. **R4-c2a**：`F, A_t, v_t, e_t`，纯 teacher forcing。
2. **R4-c2b**：在 c2a 上仅增加 `R_0-R_-1` 与 `R_0-R_-5`。
3. 只有 c2a/c2b 的基础机制有效后，才允许做小型 decoder 容量调整；每次只改变
   一个因素并保留独立配置与输出目录。

先做短程 train-only smoke，再进行正式 train/val。结构与权重只能根据 train
行为选择；val 只用于确认预先定义的机制判据，不用于反复搜索超参数。

### 6.2 主损失与诊断

主目标为 rain-rate 空间的 teacher-forced evolved-state loss：

```text
L_state = L_active + lambda_16*L_16 + lambda_32*L_32
```

第一轮使用第 4.3 节定义的加权 Huber/Charbonnier，不混入完整 R2 loss、soft CSI、
空事件惩罚、source L1 或 temporal TV。oracle source 定义为
`R_t^true - A_t^TF`，只用于下列诊断：

- growth (`oracle_source > threshold`) 中预测 source 的正值比例；
- decay (`oracle_source < -threshold`) 中预测 source 的负值比例；
- source 与 oracle residual 的符号一致率、相关性和分位数尺度；
- 按 lead、事件和真值强度区间统计 source 分布。
- 分别在 active、16、32、growth、decay 区计算
  `E(|S_pred|)/(E(|S_oracle|)+epsilon)`，不以全场单一绝对均值判断量级。

oracle residual 包含形变和 motion 误差，报告中必须继续标注它不是纯 source 真值。

另外建立只用于机制诊断的空间分区，阈值、连通域和形态学宽度在 train-only smoke
后预注册：

| 空间分区 | 诊断目的 |
|---|---|
| persistent-object interior | 对象内部增强、维持与减弱 |
| object edge band | 扩张、收缩、形态变化及可能的位移补偿 |
| newborn | 真实上一帧/平流场无对象、目标帧出现对象 |
| dissipated | 真实上一帧/平流场有对象、目标帧对象消失 |
| clear background | source 应接近零 |

分区由 truth 和固定 transport 结果构造，仅用于 detached 诊断，不作为 decoder 输入。
额外计算 source 沿 flow 方向的正负邻接/偶极指标，并配合 source 空间图判断它是否
主要在“旧位置减、新位置加”。

### 6.3 预先定义的通过条件

与零 source 的同一路径基线比较，Gate 1 至少要求：

- train 和 val 的 active-region state loss 均有明确下降，且不是仅由背景改善造成；
- 16/32 mm/h 区域 MAE 明显下降；
- growth 区预测 source 主要为正，decay 区主要为负；
- active 区 source 不再接近零，16/32、growth、decay 区的预测/oracle 绝对量级比
  达到 train-only smoke 预注册的合理非零范围；不得以全场统一增雨达到该条件；
- POD16/POD32 出现实质变化，不能只依靠 FAR 降低获得很小的 CSI 改善；
- source 随 lead、强度和事件的分布合理，无全域恒正、恒负、边界饱和或 35 mm/h
  大面积饱和；
- source 不能主要集中在对象边缘形成沿 flow 的位移补偿偶极；对象内部
  growth/decay 以及 newborn/dissipated 区必须呈现可分辨的独立贡献；
- 全域误差、FAR、面积比和强度比没有出现足以否定机制的恶化。

“明显/实质变化”在正式运行前用 train-only smoke 的方差确定数值容差，并写入
运行配置或报告，不能看完 val 结果后修改标准。Gate 1 是机制门，不要求此时完整
20 步 rollout CSI 已优于 R4-b。

### 6.4 失败分流

- c2a source 仍接近零：先查损失尺度、梯度、active mask 和 decoder 容量。
- c2a 有量级但符号混乱：运行 c2b，检验历史 tendency 是否提供必要可辨识信息。
- train 改善、val 不改善：检查事件过拟合和 decoder 容量，不解冻 motion。
- source 大量触边：检查单位、归一化、插值与 state loss 权重，不提高 `S_max`。
- c2b 仍不能改善 active/16/32 区域：停止进入 rollout，形成失败报告，再决定是否
  需要更强历史表征或重新审视固定 motion 误差。

## 7. Gate 2：混合状态训练

仅在 Gate 1 通过并选定 c2a 或 c2b 后启动 R4-c2c。保留两条显式路径：

```text
L = lambda_TF * L_TF + lambda_roll * L_roll
```

训练初期以 teacher-forced 路径为主，随后按写入配置的 schedule 提高 rollout
权重；最后才允许完整 20 步自由递归微调。scheduled sampling 不对真值和预测场
做连续加权混合，而是按 sample 和 lead（不按像素）进行 Bernoulli 选择：

```text
b_t ~ Bernoulli(p)
R_(t-1)^input = R_(t-1)^true if b_t=1 else R_hat_(t-1)
```

`p` 随训练逐步下降，其 schedule 和随机状态必须可由 checkpoint 恢复并写入日志；
验证和推理始终使用确定性的完整 rollout。

Gate 2 需要同时检查 teacher-forced 机制是否保留，以及误差累积时 source 是否
开始补偿模型自身伪影。若 rollout 改善来自 source 全域增雨、FAR 或面积比明显恶化，
则不通过。

## 8. 对照矩阵

| 实验 | 参数化 | Source 输入 | 训练路径 | 状态 |
|---|---|---|---|---|
| R4-b | 无 source | 无 | rollout | 固定基线 |
| R4-c1 | signed + clamp | 仅 `F` | 旧联合损失 | 已有失败对照 |
| R4-c2a | 有界 signed | `F,A_t,v_t,e_t` | teacher forcing | Gate 1 |
| R4-c2b | 有界 signed | c2a + 历史 tendency | teacher forcing | Gate 1 |
| R4-c2c | 最佳 c2 | 与胜出版本相同 | TF + rollout | Gate 2 |

“只替换 clamp、仍一次输出 20 帧”的版本仅可用于单元检查，不分配完整训练预算，
因为它没有解决 source 对逐步状态开环的问题。

R4-c2a 相对 R4-c1 同时改变参数化、输入、逐步解码、主损失和训练路径；若成功，
本阶段只能得出“R4-c2 整体机制设计有效”，不能把收益归因给其中单一改动。当前
机制筛选不为每项改动分配完整实验预算；进入论文级结论前，c2a（无 tendency）与
c2b（有 tendency）是最低限度必须保留的简化消融。

## 9. 运行与记录规范

每次运行应使用新的 `work_dirs`，名称至少包含 `r4c2a/r4c2b/r4c2c`、训练路径、
seed 和关键变体。运行前保存：

- 完整配置快照、命令、Git commit/status 和随机种子；
- checkpoint 路径及校验信息；
- train/val 事件和样本计数；
- trainable/frozen 参数名、数量及 optimizer param groups；
- source 单位、`S_max`、active/growth/decay 阈值和 flow 尺度。

运行后至少保存：

- all/active/16/32 state loss 与 MAE；
- CSI/POD/FAR/Frequency Bias、面积比、强度比，按 lead 和 0--1 h/1--2 h 分段；
- source 正负比例、绝对值分位数、区域预测/oracle 量级比和饱和比例，按
  lead/强度/事件分组；
- growth/decay 符号一致率与 oracle residual 诊断；
- interior/edge/newborn/dissipated/background 分区贡献及沿 flow 偶极诊断；
- `evolved_rain>R_max`、normalized dBZ 等于 1、正上限触发和 sink 清空比例；
- 典型成功和失败事件的 `truth / advected / source / evolved` 并排图；
- 与 R4-b、R4-c1 使用相同口径的比较表。

Gate 阶段不触碰 test，不做三 seed 正式结论。R4-c2c 通过后再冻结方案，随后按
seed 0/1/2 运行并进行事件级配对 bootstrap；只有这之后才讨论 PWV 对正 source
的调制。

## 10. 执行清单

- [ ] 为 bounded signed tendency 添加独立算子与边界/梯度测试。
- [ ] 固定 `R_max`、输入归一化、pooling、flow 单位和零点梯度检查。
- [ ] 实现共享 per-step source decoder、lead embedding 和 c2a 输入。
- [ ] 实现模型内 teacher-forced 演化路径及时间索引测试。
- [ ] 确认仅 source decoder/embedding 进入 optimizer，R4-b encoder/motion 冻结。
- [ ] 完成 Gate 0 固定 batch 过拟合与全套回归测试。
- [ ] 新增并运行 R4-c2a train-only smoke，冻结正式判据。
- [ ] 运行 R4-c2a train/val 并形成机制报告。
- [ ] 仅增加历史 tendency，运行 R4-c2b 并与 c2a 配对比较。
- [ ] 完成 interior/edge/newborn/dissipated/background 与 flow 偶极诊断。
- [ ] 根据 Gate 1 判据作出 go/no-go 决策。
- [ ] Gate 1 通过后实现 R4-c2c mixed-state/scheduled sampling。
- [ ] Gate 2 通过后再制定多 seed 与 PWV source modulation 计划。

## 11. 完成定义

本阶段完成不是“训练跑完”，而是得到一个可复核的机制结论：

- **Go**：有界、逐步状态条件化 source 在冻结 motion 下通过 Gate 1，并在 Gate 2
  中保持机制方向且改善自由 rollout；随后冻结结构进入多 seed 验证。
- **No-go**：c2a/c2b 在工程检查无误后仍不能改善 active/16/32 状态误差或不能
  学到正确生消符号；停止 rollout 和多模态扩展，记录失败证据并重新评估历史表征、
  固定 motion 误差或 source 可辨识性。

在任一结论之前，不以总 CSI 的微小变化替代机制证据，也不通过解冻 motion、增加
训练 epoch 或加入 PWV 来掩盖 source decoder 本身是否有效的问题。
