# 京津冀区域多源物理可解释短临降水预测研究方案

> **文档版本**：v0.1  
> **更新时间**：2026-07-29  
> **当前阶段**：OpenSTL 环境与官方示例已跑通，准备接入本地 Radar 数据并建立 V0 Radar-only 基线。  
> **文档范围**：整理当前研究目的、数据与任务协议、研究要求、技术路线、已讨论的核心文献及其可借鉴内容、当前进展与后续计划。

---

## 1. 研究背景与目的

### 1.1 研究区域

研究区为 **Beijing–Tianjin–Hebei（BTH，京津冀）region**，主要面向京津冀汛期短时强降水预报。

该区域同时包含：

- 华北平原；
- 燕山—太行山山地及山前过渡带；
- 北京、天津及河北城市群；
- 复杂地形、季风水汽输送、城市热岛与局地对流共同影响的强降水环境。

因此，京津冀既适合研究单地区专用模型，也适合分析 **雷达运动信息、水汽环境信息与静态地形信息之间的互补关系**。

### 1.2 总体目标

构建一个面向京津冀地区的、参数量相对较小的短临降水预测模型，在有限的单地区汛期数据下实现：

1. **可靠的 Radar-only 短临外推基线**；
2. 评估 GNSS-PWV 和 DEM 是否提供超出历史雷达的稳定增量信息；
3. 将不同数据源放入具有明确物理角色的模块，而不是仅做无约束通道拼接；
4. 将降水演化尽可能拆分为可检查的过程，例如：
   - 降水系统的平流与位置变化；
   - 降水增强、减弱、新生和消散；
   - 水汽条件对源汇演化的调制；
   - 地形对局地降水发生和增强的静态调制；
5. 在单地区条件下首先验证 **跨月份、跨事件、跨降水类型的泛化**，并为未来跨地区迁移保留模块化接口；
6. 在提升 CSI@16、CSI@32 的同时，控制虚警、降水面积膨胀、位置偏差和长时效强度衰减。

### 1.3 核心研究问题

本项目不只回答“加入 PWV 后指标是否提高”，而重点回答：

1. 历史 Radar 是否已经足够预测主要平流运动？
2. PWV 的信息增益主要体现在哪里：
   - 已有强降水的维持；
   - 降水增强；
   - 对流新生；
   - 降水消散；
   - 位置修正；
3. DEM 是否能稳定改善山地—平原过渡带的强降水预测，而不是仅让模型记住固定位置的降水气候分布？
4. 显式的运动—源汇分解，是否比直接预测整幅未来 Radar 更稳定、更可解释？
5. 在一个季度级别的有限数据上，轻量地区专用模型能否比大参数通用模型具有更好的稳定性和数据效率？
6. 模型提升是否具有可信度：
   - 多随机种子是否一致；
   - 独立降水事件是否一致；
   - 提升是否由 POD 增长带来，还是以 FAR 和面积膨胀为代价；
   - 中间变量是否真的承担了声称的物理作用。

### 1.4 当前工作假设

- **H1：Radar 主导运动。** 历史雷达序列主要负责描述已有降水对象的位置、形态和运动。
- **H2：PWV 主导环境调制。** PWV 及其变化率更适合调制降水源汇，而不是与 Radar 等价地控制运动场。
- **H3：DEM 是静态空间条件。** DEM 更适合通过静态适配器影响局地源汇或地形抬升条件，不应被简单复制为普通时间序列通道。
- **H4：显式角色优于无约束拼接。** 当数据较少时，限制不同模态的作用范围有望减少过拟合和虚假相关。
- **H5：可解释性必须可验证。** Attention、gate 或带有物理名称的分支本身不足以证明物理可解释性，需要消融、置乱、屏蔽和反事实实验。

---

## 2. 数据集情况
存放位置：D:\_Search\AIforScience\Rewritten\capsule-3935105\data\DATA_2025_S
包括：RADAR,PWV,RAIN
### 2.1 时间范围

- **2025 年 5 月 1 日至 2025 年 8 月 31 日**
- 覆盖京津冀主汛期
- 当前仅有单地区、一个汛期共四个月数据，属于小样本地区专用建模条件

### 2.2 数据源

| 数据源 | 数量/覆盖 | 原始时间分辨率 | 统一后时间分辨率 | 统一空间分辨率 | 主要物理意义 |
|---|---:|---:|---:|---:|---|
| GNSS-PWV | 京津冀 250 个 CORS 站 | 需以最终产品为准 | 6 min | 0.1° × 0.1° | 柱积分水汽与降水前环境条件 |
| Radar | 京津冀组合反射率拼图 | 约 6 min/帧 | 6 min | 0.1° × 0.1° | 已有水凝物、降水位置、形态与运动 |
| 地面降雨观测 | 250 个地面气象站 | 逐小时 | 原始观测仍为逐小时 | 站点数据 | Z–R 拟合与独立站点评价 |
| DEM | 京津冀区域静态地形 | 静态 | 静态 | 0.1° × 0.1° | 海拔、山地—平原过渡与地形抬升条件 |

### 2.3 文件格式与灰度编码

当前数据均为 PNG 格式，且采用 **反向灰度编码**：像素值越小，物理量越大。

| 数据 | PNG 像素范围 | 对应物理范围 |
|---|---:|---:|
| PWV | 255 → 0 | 0 → 80 mm |
| Radar | 255 → 0 | 0 → 50 dBZ |
| Rain | 255 → 0 | 0 → 35 mm/h |

对于像素值 \(p\in[0,255]\)，推荐统一解码为：

\[
x_{\text{norm}}=\frac{255-p}{255}
\]

\[
PWV=80x_{\text{norm}}\;\mathrm{mm}
\]

\[
dBZ=50x_{\text{norm}}
\]

\[
Rain=35x_{\text{norm}}\;\mathrm{mm/h}
\]

### 2.4 内部数据表示要求

模型内部不建议继续使用“强度越大、数值越小”的反向灰度语义。推荐：

- 读入 PNG 后立即转为浮点数；
- 统一变成 **物理量越大，模型输入数值越大**；
- Radar 主训练变量可使用：
  - 归一化反射率 \(dBZ/50\)，或
  - 经固定变换后的标准化浮点值；
- 评估时恢复为真实 dBZ 和 mm/h；
- PNG 仅作为磁盘存储与可视化格式，不直接作为评价量纲；
- 无效区、缺测区和有效零值必须使用独立 mask 区分。

特别需要注意：

> Rain PNG 的上限为 35 mm/h，而主指标包含 CSI@32。32 mm/h 已非常接近 PNG 饱和上限，因此不应将被截断到 35 mm/h 的 Rain PNG 作为唯一的极端降水主评价真值。主评价应优先使用未来真实 Radar 经冻结 Z–R 关系转换得到的浮点降雨率；地面站原始逐小时降雨作为独立验证。

---

## 3. 固定任务定义

### 3.1 当前已经确定的任务协议

| 项目 | 当前定义 |
|---|---|
| 输入帧数 | 10 帧 |
| 输出帧数 | 20 帧 |
| 时间间隔 | 6 min/帧 |
| 历史时长 | 60 min |
| 预报时长 | 120 min |
| 动态输入 | 历史 Radar；后续可加入 PWV |
| 静态输入 | 后续可加入 DEM |
| 预测目标 | 未来 Radar 组合反射率 |
| 后处理 | 固定、本地化 Z–R 转换 |
| 主要评价量 | 降雨率 |
| 主指标 | CSI@16 mm/h、CSI@32 mm/h |
| 研究区域 | 京津冀 BTH |
| 统一空间分辨率 | 0.1° × 0.1° |
| 统一时间分辨率 | 6 min |

总体流程暂定为：

\[
\text{历史 Radar（可加 PWV/DEM）}
\rightarrow
\text{未来 Radar 反射率}
\rightarrow
\text{冻结的本地 Z-R 转换}
\rightarrow
\text{降雨率评价}
\]

### 3.2 10→20 的实现要求

10 帧输入、20 帧输出意味着：

- 输入过去 1 h；
- 预测未来 2 h；
- 需要分别报告：
  - 0–1 h；
  - 1–2 h；
  - 每个 6 min lead time；
  - 0–2 h 总体结果。

需要在正式训练前确认 OpenSTL 对 SimVP 的 10→20 处理方式：

- 是直接多时效输出；
- 还是以 10 帧为一段进行两段递归预测。

一旦确定，应在所有主模型与基线中统一或明确区分，避免某些模型直接预测 20 帧、另一些模型递归两次而造成不公平比较。

### 3.3 输出 Radar 而不是直接输出 Rain 的原因

当前优先预测未来 Radar，主要因为：

1. Radar 的时间分辨率与任务一致，为 6 min；
2. 历史输入和未来标签来自同一类连续观测，序列结构更自然；
3. 地面站降雨为逐小时点观测，无法直接提供同分辨率的密集 6 min 降雨场真值；
4. 所有模型统一经过同一个冻结 Z–R 转换，可避免不同模型分别学习不同的反射率—降水映射；
5. 可以同时进行：
   - Radar 空间场评价；
   - Z–R 后降雨率评价；
   - 地面站逐小时独立验证。

---

## 4. 数据划分与防泄漏要求

### 4.1 原论文划分

原论文使用：

- 5—7 月：训练集；
- 8 月：验证集和测试集；
- 8 月内部约 20% 验证、80% 测试。

### 4.2 本项目建议

原则上保留“5—7 月训练、8 月验证与测试”的时间外推设定，但必须注意：

> 不能在生成重叠滑窗后，将 8 月样本随机按 20%/80% 划分。

相邻滑窗共享大量帧，随机划分会导致验证与测试之间高度重叠。正确顺序应为：

1. 先按完整日期、连续天气过程或独立降水事件划分；
2. 确保一个事件只属于 Train、Val、Test 中的一个集合；
3. 再在各集合内部生成 10→20 滑窗；
4. 保存固定的事件清单、日期清单和样本索引；
5. 后续任何模型不得修改划分。

### 4.3 推荐的划分单位

优先级如下：

1. **独立降水事件划分**；
2. 连续日期块划分；
3. 最后才考虑简单月份内时间切分。

8 月验证集和测试集应尽量兼顾：

- 降水事件数量；
- 强降水像素比例；
- 山区、平原和城市群事件；
- 层状降水与对流降水；
- CSI@16、CSI@32 的有效样本数量。

---

## 5. 本地化 Z–R 关系方案

### 5.1 基本形式

采用经典幂律关系：

\[
Z=aR^b
\]

其中：

\[
Z=10^{dBZ/10}
\]

\[
R=\left(\frac{Z}{a}\right)^{1/b}
\]

- \(Z\)：线性雷达反射率因子；
- \(R\)：降雨率；
- \(a,b\)：京津冀本地化拟合参数。

### 5.2 防泄漏流程

固定流程为：

1. **先划分日期/事件**；
2. 仅使用 Train 中的 Radar 与地面站降水拟合 \(a,b\)；
3. 不使用 Val 和 Test 参与参数拟合；
4. 在 Train 内通过交叉验证或事件重采样确定拟合策略；
5. 拟合完成后冻结 Z–R；
6. 所有模型、所有基线、所有数据集统一使用同一组 \(a,b\)。

### 5.3 时间尺度匹配

地面站为逐小时降雨，而 Radar 为 6 min/帧。推荐在站点—小时层面拟合：

1. 对每个气象站匹配最近 Radar 网格，或预先固定局地邻域汇聚规则；
2. 对每个小时内的 10 帧 Radar 分别通过候选 Z–R 转为瞬时雨强；
3. 按每帧 0.1 h 将 10 帧雨强积分为小时累计雨量；
4. 与该站该小时的实测累计降雨比较；
5. 在 Train 中优化 \(a,b\)。

即：

\[
\widehat P_h(a,b)=\sum_{k=1}^{10}
\left(\frac{10^{dBZ_{h,k}/10}}{a}\right)^{1/b}\times0.1
\]

其中 \(\widehat P_h\) 的单位为 mm。

这种做法比“先平均 dBZ 再回归”更符合 Z–R 的非线性关系。

### 5.4 拟合与评价要求

- 优先考虑非线性最小二乘、加权最小二乘或鲁棒优化；
- 不能只在对数空间排除全部零降水样本后拟合，否则可能产生条件偏差；
- 需要报告：
  - 拟合得到的 \(a,b\)；
  - Train 内交叉验证误差；
  - 不同事件下参数稳定性；
  - 站点匹配方式；
  - 是否使用邻域均值/中位数；
  - 对弱降水与强降水的误差差异；
- 建议保留经典 Z–R 关系作为对照，但主实验统一使用冻结的本地关系。

### 5.5 两套降雨评价

建议最终同时保留：

**A. 网格主评价**

- 预测 Radar 和真实未来 Radar；
- 通过相同冻结 Z–R 转为雨强；
- 在完整网格上计算 CSI@16、CSI@32 等指标。

**B. 地面站独立评价**

- 从预测雨强场抽取 250 个站点位置；
- 将未来 10 个 6 min 雨强积分为逐小时累计雨量；
- 与原始逐小时站点降雨比较；
- 用于评价 Z–R 后的业务意义和局地偏差。

---

## 6. 研究要求

### 6.1 轻量化与稳定性

- 主模型应优先采用小参数量配置；
- 暂不将参数上限写死，首版可将约 1–5M 参数作为工程起点；
- 所有模型必须报告：
  - 参数量；
  - FLOPs 或近似计算量；
  - 单轮训练时间；
  - 推理时间；
  - 峰值显存；
- 参数规模不能成为唯一目标，多随机种子的稳定性比单次最好结果更重要。

### 6.2 物理可解释性

本项目中的“物理可解释性”至少包含三层：

1. **输入变量有物理意义**
   - Radar：已有降水对象；
   - PWV：柱积分水汽；
   - DEM：静态地形。

2. **模块职责有物理意义**
   - 运动分支；
   - 平流演化；
   - 源汇分支；
   - PWV 调制；
   - DEM 调制。

3. **解释可以被验证**
   - 去掉 PWV 是否只影响源汇相关指标；
   - 置乱 PWV 时间序列是否破坏增益；
   - 使用静态 PWV 气候态是否与动态 PWV 等效；
   - 错误时间偏移的 PWV 是否仍然获得相同增益；
   - DEM 旋转或置乱后是否仍然改善；
   - 运动头是否真的对应位移，而不是被源汇头补偿；
   - gate、motion、source 的可视化与对象级误差是否一致。

### 6.3 可推广性

由于当前只有 BTH 单地区数据，现阶段不能直接声称“跨地区泛化”。本项目优先验证：

- 跨月份泛化；
- 跨独立降水事件泛化；
- 不同降水类型泛化；
- 山区、平原、城市群子区域泛化；
- 冻结 Radar 主干、只调整 PWV/DEM adapter 的迁移能力。

未来获得新区域数据后，再进行真正的跨区域测试。

### 6.4 公平比较

所有模型必须统一：

- 数据划分；
- 10→20 任务；
- 输入和标签；
- Z–R 参数；
- 有效区 mask；
- 训练 epoch 或计算预算；
- early stopping 规则；
- 指标实现；
- 评估事件；
- lead-time 分组；
- 随机种子集合。

### 6.5 统计可信度

至少使用：

- 多随机种子均值与标准差；
- 按独立事件聚类的 paired bootstrap；
- CSI 增量的置信区间；
- POD、FAR、Bias 分解；
- 关键强降水事件逐例分析。

### 6.6 主指标与辅助指标

**主指标**

- CSI@16 mm/h；
- CSI@32 mm/h；
- 分 0–1 h、1–2 h 报告。

**必要辅助指标**

- POD；
- FAR；
- Bias；
- MAE；
- RMSE；
- 每个 lead time 的曲线。

**建议扩展指标**

- 邻域 CSI；
- FSS；
- 强降水面积偏差；
- 对象质心距离；
- 峰值强度误差；
- 强降水能量保持；
- CRA 位移、形态、体量误差；
- PSD 或频谱保持；
- 站点小时累计降水误差。

---

## 7. 总体技术路线

### 7.1 工程骨架

以 **OpenSTL** 为统一工程骨架，负责：

- Dataset/DataLoader；
- 配置管理；
- 训练与验证流程；
- checkpoint；
- 日志与可视化；
- 统一模型接口；
- 统一评估；
- 多模型公平比较。

### 7.2 主骨架选择

当前主骨架建议为：

> **OpenSTL + Tiny-SimVP-gSTA**

原因：

- 非循环结构，训练并行度高；
- 代码结构清晰；
- 容易缩小参数；
- 容易建立 Radar-only 稳定基线；
- 容易添加独立 PWV/DEM adapter；
- 容易在输出端增加 motion/source heads；
- 适合小数据下反复做消融和多种子实验。

### 7.3 Earthformer 的定位

Earthformer 不作为首个主骨架，而作为 **Transformer 强基线**：

- 先完成 Tiny-SimVP Radar-only；
- 再以同一数据、同一划分、同一 Z–R、同一指标运行 Tiny-Earthformer；
- 优先比较参数规模相近的版本；
- 资源允许时，可增加接近官方配置的 Earthformer 作为性能上界参考；
- Earthformer 官方代码不在 OpenSTL 原生模型列表中，可采用两种方式：
  1. 在官方仓库中运行，但调用本项目固定的数据与评估协议；
  2. 将 Earthformer 包装为 OpenSTL 的 model/method，以统一训练入口。

Earthformer 的作用是回答：

> 在相同小数据条件下，更强的 Cuboid Attention 是否比轻量卷积主干更有效，还是更容易过拟合和产生 seed 波动？

### 7.4 分阶段模型路线

#### V0：Radar-only 最小基线

- Persistence；
- 可选 pySTEPS/光流；
- ConvLSTM；
- Tiny-SimVP-gSTA；
- 后续加入 Tiny-Earthformer。

目的：固定数据、训练、推理和评价流程。

#### V1：简单多源融合基线

- Radar + PWV 直接通道拼接；
- Radar + DEM；
- Radar + PWV + DEM。

目的：回答“额外数据源本身是否具有可利用增量”，不将其作为最终可解释模型。

#### V2：Radar-only 运动—源汇分解

基本形式：

\[
\hat R_t=
\operatorname{Warp}(\hat R_{t-1},v_t)+s_t
\]

- \(v_t\)：运动场；
- \(s_t\)：增强、减弱、新生和消散；
- 首先只用 Radar，证明分解本身可学习。

#### V3：PWV 调制源汇

推荐形式：

\[
s_t=s_t^{radar}+g_t^{pwv}\odot\Delta s_t^{pwv}
\]

PWV 分支可输入：

- 当前 PWV；
- PWV anomaly；
- 30 min 变化；
- 60 min 变化；
- 插值置信度或站点距离。

PWV 默认不直接重写全部运动场；是否允许其修正运动需要单独实验和证据。

#### V4：DEM 静态适配

推荐形式：

\[
s_t=s_t^{radar}
+g_t^{pwv}\odot\Delta s_t^{pwv}
+g_t^{dem}\odot\Delta s_t^{dem}
\]

DEM adapter 可使用：

- 高程；
- 坡度；
- 坡向的周期编码；
- 山地—平原过渡特征；
- 静态空间置信度。

#### V5：可推广性实验

- 跨月份；
- 留一事件；
- 山区/平原/城市子区域；
- 冻结 Radar backbone，仅训练 PWV/DEM adapter；
- 未来跨地区迁移。

#### V6：可选高级扩展

只有确定性模型稳定后再考虑：

- FACL；
- AlphaPre 风格的幅相分解；
- DiffCast 残差扩散；
- CasCast 级联生成；
- PreDiff 概率预报；
- 对象级单体图和分裂/合并建模。

---

## 8. 参考文献与可借鉴内容

下面收录当前项目中已经明确讨论、且与现阶段路线直接相关的核心文献。代码状态以 2026-07-29 的公开页面为准。

---

### 8.1 工程骨架与确定性时空预测

#### 1. OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning

- **类别**：统一时空预测框架与基准
- **发表**：NeurIPS 2023 Datasets and Benchmarks
- **可借鉴**：
  - 统一 Dataset、Method、Model、Config、Metric；
  - 公平比较循环与非循环模型；
  - 作为本项目长期工程骨架；
  - 支持 SimVP、ConvLSTM、PhyDNet、TAU、SwinLSTM 等。
- **开源代码**：有，官方仓库  
  [OpenSTL GitHub](https://github.com/chengtan9907/OpenSTL)

#### 2. SimVP: Simpler Yet Better Video Prediction

- **类别**：轻量确定性主干
- **发表**：CVPR 2022
- **可借鉴**：
  - Spatial Encoder—Temporal Translator—Spatial Decoder；
  - 结构简洁、并行度高；
  - 适合改造成多模态 adapter 和物理多头结构；
  - 当前推荐的 V0 主骨架。
- **开源代码**：OpenSTL 中含实现
- **本项目定位**：主骨架

#### 3. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

- **类别**：经典循环基线
- **发表**：NeurIPS 2015
- **可借鉴**：
  - 最经典的雷达时空序列基线；
  - 与非循环 SimVP 对比；
  - 检验小数据下循环归纳偏置是否更稳定。
- **开源代码**：OpenSTL 中含实现
- **本项目定位**：必做经典基线

#### 4. Deep Learning for Precipitation Nowcasting: A Benchmark and a New Model（TrajGRU）

- **类别**：可学习运动连接的循环模型
- **发表**：NeurIPS 2017
- **可借鉴**：
  - 对非刚性降水运动建模；
  - 是 GNSS-PWV 融合论文所采用的重要主干；
  - 可作为运动建模参考。
- **开源代码**：当前未确认维护良好的作者官方仓库；可参考公开复现
- **本项目定位**：方法参考，不作为首个骨架

#### 5. Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning（TAU）

- **类别**：高效非循环预测
- **发表**：CVPR 2023
- **可借鉴**：
  - 作为 SimVP 之外的高效确定性强基线；
  - 评估时间注意力是否在小数据上有稳定收益。
- **开源代码**：OpenSTL 中含实现
- **本项目定位**：后续强基线

#### 6. Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction（PhyDNet）

- **类别**：物理动态与未知残差分解
- **发表**：CVPR 2020
- **可借鉴**：
  - “可解释动态 + 数据驱动残差”的通用思路；
  - 与本项目运动—源汇分解进行概念对照。
- **开源代码**：OpenSTL 中含实现
- **本项目定位**：物理结构对照，不直接照搬

#### 7. Earthformer: Exploring Space-Time Transformers for Earth System Forecasting

- **类别**：时空 Transformer 强基线
- **发表**：NeurIPS 2022
- **可借鉴**：
  - Cuboid Attention；
  - 局地块与全局向量联合建模；
  - 分层 Encoder–Decoder；
  - 检验长距离依赖对 BTH 强降水的作用。
- **主要风险**：
  - 参数量较大；
  - 两个季度数据下可能过拟合；
  - Attention 不等于物理解释。
- **开源代码**：有，官方仓库  
  [Earthformer GitHub](https://github.com/amazon-science/earth-forecasting-transformer)
- **本项目定位**：Transformer 强基线，优先使用 Tiny-Earthformer

---

### 8.2 GNSS-PWV、Radar 与区域多模态融合

#### 8. A Deep Learning-Based Precipitation Nowcasting Model Fusing GNSS-PWV and Radar Echo Observations

- **类别**：Radar + GNSS-PWV 直接融合
- **发表**：IEEE TGRS 2025
- **DOI**：10.1109/TGRS.2025.3554745
- **可借鉴**：
  - 过去 10 帧 Radar+PWV 输入；
  - 未来 Radar 输出；
  - PWV 插值到 Radar 网格；
  - TrajGRU/ConvLSTM 与时间注意力；
  - 与本项目最接近的简单多模态基线。
- **局限**：
  - 主要采用输入通道拼接；
  - 物理角色未显式分解；
  - 预测时长较短；
  - 不能直接说明 PWV 改善了哪种过程。
- **开源代码**：截至当前未检索到作者官方公开仓库
- **本项目定位**：V1 简单 PWV 融合基线参考

#### 9. Synergistic Fusion of GNSS-PWV and Radar for Precipitation Nowcasting: An AI-Empowered Spatio-Temporal Attention Network（STEA-Swin）

- **类别**：BTH 区域前期工作
- **发表**：Remote Sensing 2026
- **DOI**：10.3390/rs18121929
- **可借鉴**：
  - 与当前 BTH 数据源和研究区高度一致；
  - 250 个 CORS 站；
  - PWV、Radar 与地区强降水任务；
  - Swin U-Net、时空注意力、边缘损失；
  - 数据处理、区域描述、指标和业务背景。
- **局限**：
  - 同时包含较多模块，难以归因；
  - Transformer 容量对小数据存在风险；
  - 多模态物理作用仍主要依赖网络隐式学习。
- **开源代码**：截至当前未检索到作者官方公开仓库
- **本项目定位**：同数据强基线与实验协议参考

#### 10. 利用北斗/GNSS观测数据分析“21·7”河南极端暴雨过程

- **类别**：PWV 物理机制与事件分析
- **发表**：地球物理学报 2022
- **DOI**：10.6038/cjg2022P0706
- **可借鉴**：
  - 极端降水前 1–3 h PWV 快速上升；
  - PWV 高值与强降水空间分布关系；
  - PWV 质量验证；
  - 支持 PWV tendency 与前兆时间窗设计。
- **局限**：
  - 相关性不等于因果；
  - 高 PWV 不是降水充分条件。
- **开源代码**：不属于模型代码论文
- **本项目定位**：物理动机和时间尺度依据

#### 11. A Fusion Framework for Producing an Accurate PWV Map With Spatiotemporal Continuity Based on GNSS, ERA5, and MODIS Data

- **类别**：PWV 产品构建与质量控制
- **发表**：IEEE TGRS 2024
- **DOI**：10.1109/TGRS.2024.3447832
- **可借鉴**：
  - 稀疏 GNSS、连续 ERA5、精细 MODIS 的互补关系；
  - PWV 场不能只看空间完整性，还要验证动态真实性；
  - 插值/融合误差和置信度应进入模型分析。
- **局限**：
  - 主要是日尺度产品；
  - 与 6 min 临近预报时间尺度不同。
- **开源代码**：截至当前未检索到作者官方公开仓库
- **本项目定位**：PWV 数据质量、置信度与插值敏感性参考

#### 12. Pointwise is Pointless? A Multimodal Ablation Study for Precipitation Nowcasting with Graph Neural Networks

- **类别**：稀疏站点、多模态消融与评价
- **发表**：arXiv 2026
- **可借鉴**：
  - 区分站点局地增益和完整 Radar 场增益；
  - 不同数据源可能改善不同目标；
  - 站点评价、降水发生评价、位移和幅度评价应分开；
  - 对本项目 250 个地面站和 250 个 CORS 站尤其重要。
- **开源代码**：当前未确认官方代码
- **本项目定位**：多模态消融和站点—网格双评价参考

#### 13. Integrating GNSS-Derived Zenith Wet Delay into a Weather Foundation Model Improves Precipitation Forecasting

- **类别**：GNSS 水汽与天气基础模型
- **发表**：arXiv 2026
- **可借鉴**：
  - 将 GNSS-ZWD 同时作为辅助输入与辅助预测目标；
  - 水汽观测对高分位降水的增益可能随强度增加；
  - 为未来跨尺度、跨地区预训练提供方向。
- **局限**：
  - 属于中期天气基础模型；
  - 与本项目 6 min Radar 外推任务尺度不同。
- **开源代码**：论文独立实现当前未确认
- **本项目定位**：远期扩展，不进入首版骨架

---

### 8.3 显式物理演化与多模态物理引导

#### 14. Skilful Nowcasting of Extreme Precipitation with NowcastNet

- **类别**：物理嵌入式生成模型
- **发表**：Nature 2023
- **DOI**：10.1038/s41586-023-06184-4
- **核心公式**：

\[
\frac{\partial R}{\partial t}+(\mathbf v\cdot\nabla)R=s
\]

\[
\hat R_t=\operatorname{Advect}(\hat R_{t-1},v_t)+s_t
\]

- **可借鉴**：
  - 运动场与强度残差显式分解；
  - 可微平流算子；
  - 对整个预测时域端到端优化；
  - motion regularization；
  - 物理中间场可视化和验证。
- **不建议直接照搬**：
  - 完整 GAN 和生成网络；
  - 大规模训练设置；
  - 复杂两阶段训练。
- **开源代码**：有，作者在 Code Ocean 发布代码与预训练权重  
  [NowcastNet Code Ocean](https://codeocean.com/capsule/3935105/tree/v1)
- **本项目定位**：V2 运动—源汇结构的首要参考

#### 15. PiMMNet: Introducing Multi-Modal Precipitation Nowcasting via a Physics-Informed Perspective

- **类别**：多模态、物理引导、生成式修正
- **发表**：ACM Multimedia 2025
- **DOI**：10.1145/3746027.3755436
- **可借鉴**：
  - 多模态不应只做像素拼接；
  - 通过运动空间对齐不同观测；
  - 确定性平流与随机源项分解；
  - 多模态物理先验与生成修正结合。
- **适配本项目时的修改**：
  - 原模态为 Radar+Satellite；
  - PWV 不直接观测云体运动，因此本项目应让 Radar 主导运动，PWV 主要进入源汇。
- **开源代码**：有，官方仓库  
  [PiMMNet GitHub](https://github.com/DeminYu98/PiMMNet)
- **本项目定位**：多模态物理融合机制参考

#### 16. Integrating Multi-Source Data for Long Sequence Precipitation Forecasting

- **类别**：长序列多源预测
- **发表**：AAAI 2025
- **DOI**：10.1609/aaai.v39i27.35077
- **可借鉴**：
  - 多分支编码器—解码器；
  - 跨模态 attention；
  - temporal adaptive layer；
  - 长时效初步预测与分布修正分开。
- **局限**：
  - 结构较复杂；
  - 原始多源数据与 PWV/DEM 不完全相同；
  - 流模型不适合首版小样本路线。
- **开源代码**：截至当前未检索到独立作者官方仓库
- **本项目定位**：后续多模态对齐和长时效修正参考

#### 17. MoCast: Learning Turbulent Motions Under Physical Guidance for Precipitation Nowcasting

- **类别**：物理运动分解
- **发表**：AAAI 2026
- **可借鉴**：
  - 平流和源汇；
  - Reynolds 分解；
  - Helmholtz 分解；
  - 小波多尺度运动；
  - 运动结构如何调制降水生消。
- **局限**：
  - 分支多、结构复杂；
  - 两个季度数据下各分支可识别性不足；
  - 易出现模块间补偿。
- **开源代码**：截至当前未检索到作者官方仓库
- **本项目定位**：物理问题定义参考，不完整复刻

---

### 8.4 确定性—随机性分解与生成式预测

#### 18. DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting

- **类别**：确定性趋势 + 随机残差扩散
- **发表**：CVPR 2024
- **可借鉴**：
  - 将全局确定性运动与局地随机变化分开；
  - 可包裹 SimVP、Earthformer、ConvGRU、PhyDNet 等主干；
  - 可作为确定性主模型稳定后的第二阶段扩展。
- **局限**：
  - 扩散训练与推理成本高；
  - 残差不一定对应明确物理过程；
  - 当前公开仓库说明为部分训练与推理代码。
- **开源代码**：有，官方仓库  
  [DiffCast GitHub](https://github.com/DeminYu98/DiffCast)
- **本项目定位**：后期概率/细节修正，不进入 V0–V4

#### 19. CasCast: Skillful High-Resolution Precipitation Nowcasting via Cascaded Modelling

- **类别**：中尺度确定性 + 小尺度概率生成
- **发表**：ICML 2024
- **可借鉴**：
  - 将整体分布和极端局地细节分开；
  - 高分辨率确定性预测；
  - 潜空间扩散减少生成成本。
- **局限**：
  - 两阶段级联；
  - 前级误差会传递；
  - 可解释性主要是尺度分解，不是物理过程分解。
- **开源代码**：有，官方仓库  
  [CasCast GitHub](https://github.com/OpenEarthLab/CasCast)
- **本项目定位**：后期极端细节扩展

#### 20. PreDiff: Precipitation Nowcasting with Latent Diffusion Models

- **类别**：概率预报与知识对齐
- **发表**：NeurIPS 2023
- **可借鉴**：
  - 条件潜空间扩散；
  - 在去噪过程中施加领域知识约束；
  - 生成多个可能未来和不确定性。
- **局限**：
  - 潜空间难以物理解释；
  - 小样本难以学习可靠分布；
  - 评价体系更复杂。
- **开源代码**：有，官方仓库  
  [PreDiff GitHub](https://github.com/gaozhihan/PreDiff)
- **本项目定位**：未来不确定性研究

#### 21. Skilful Precipitation Nowcasting Using Deep Generative Models of Radar（DGMR）

- **类别**：深度生成雷达临近预报
- **发表**：Nature 2021
- **可借鉴**：
  - 生成式预测的空间清晰度；
  - 专家评价；
  - 概率与多样性。
- **局限**：
  - 物理过程不显式；
  - 极端降水可能出现位置和强度问题；
  - 训练数据和计算量大。
- **开源代码**：有，DeepMind 官方仓库  
  [DGMR GitHub](https://github.com/deepmind/deepmind-research/tree/master/nowcasting)
- **本项目定位**：生成式历史基线与评价参考

---

### 8.5 频域、损失函数与强降水细节

#### 22. Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss for Skillful Precipitation Nowcasting（FACL）

- **类别**：频域损失
- **发表**：NeurIPS 2024
- **可借鉴**：
  - FAL 约束 Fourier amplitude；
  - FCL 补充相关性/相位信息；
  - 解决 MSE 导致的模糊和极值衰减；
  - 参数无关，可插入多种主干。
- **局限**：
  - 是损失函数而不是架构；
  - 频谱改善不保证位置和物理机制改善；
  - 可能牺牲像素误差。
- **开源代码**：有，官方仓库  
  [FACL GitHub](https://github.com/argenycw/FACL)
- **本项目定位**：V4 后单独消融，不与主结构同时加入

#### 23. AlphaPre: Amplitude-Phase Disentanglement Model for Precipitation Nowcasting

- **类别**：幅值—相位解耦
- **发表**：CVPR 2025
- **可借鉴**：
  - Phase network 建模位置变化；
  - Amplitude network 建模强度变化；
  - 与“运动—强度”解耦相呼应；
  - 可用于诊断位置误差和强度误差。
- **局限**：
  - 幅值/相位是信号解释，不等于严格气象物理；
  - FFT 边界和局地对象存在解释风险；
  - 当前项目早期频域实验尚未显示稳定收益。
- **开源代码**：有，官方仓库；公开页面说明为部分训练与推理代码  
  [AlphaPre GitHub](https://github.com/linkenghong/AlphaPre)
- **本项目定位**：频域对照与机制诊断

---

### 8.6 评价、对象级与扩展方向

#### 24. Hybrid Physics-AI Outperforms Numerical Weather Prediction for Extreme Precipitation Nowcasting

- **类别**：独立应用评价
- **发表**：npj Climate and Atmospheric Science 2024
- **DOI**：10.1038/s41612-024-00834-8
- **可借鉴**：
  - NowcastNet、HRRR、平流和 persistence 的比较；
  - CSI@16 等极端阈值；
  - CRA 将误差拆为位移、形态和体量；
  - 中位数、四分位数和事件分布；
  - 提醒长时效可能出现空间总量高估和弱降水面积扩大。
- **开源代码**：无独立新模型代码；依赖 NowcastNet 与公开基线
- **本项目定位**：评价体系核心参考

#### 25. pySTEPS: An Open-Source Python Library for Probabilistic Precipitation Nowcasting

- **类别**：传统平流与概率临近预报
- **发表**：Geoscientific Model Development 2019
- **可借鉴**：
  - 光流/平流基线；
  - 业务化指标；
  - 雷达外推的非神经网络比较。
- **开源代码**：有，官方仓库  
  [pySTEPS GitHub](https://github.com/pySTEPS/pysteps)
- **本项目定位**：推荐加入的传统基线

#### 26. TITAN: Thunderstorm Identification, Tracking, Analysis, and Nowcasting—A Radar-Based Methodology

- **类别**：对象级风暴识别与跟踪
- **发表**：Journal of Atmospheric and Oceanic Technology 1993
- **可借鉴**：
  - 以雷达阈值定义风暴对象；
  - 跟踪质心、面积、强度和轨迹；
  - 对象级指标与可解释中间变量；
  - split/merge 是标准风暴演化现象。
- **开源代码**：原始经典系统无统一现代官方仓库
- **本项目定位**：对象级评价和未来对象分支参考

#### 27. Forecasting Convective Storms Trajectory and Intensity by Neural Networks

- **类别**：单体轨迹与强度预测
- **发表**：Forecasting 2024
- **可借鉴**：
  - 以单体质心、面积、平均/最大反射率预测未来轨迹和强度；
  - 支持将整场网格预测与对象级预测结合。
- **开源代码**：截至当前未确认官方仓库
- **本项目定位**：未来对象级辅助任务

#### 28. Analysis of Convective Cell Evolution with Split and Merge Events Using a Graph-Based Methodology

- **类别**：分裂/合并事件图建模
- **发表**：Atmospheric Measurement Techniques 2026
- **可借鉴**：
  - 将单体演化表示为时空有向图；
  - 显式记录 predecessor、successor、split、merge；
  - 为未来对象级解释和图预测提供物理事件定义。
- **开源资源**：论文提供数据资产；当前不是端到端深度预测代码
- **本项目定位**：远期对象级解释扩展

#### 29. Bayesian Deep Learning for Convective Initiation Nowcasting Uncertainty Estimation

- **类别**：对流新生与不确定性
- **发表**：Artificial Intelligence for the Earth Systems 2026
- **可借鉴**：
  - 对流新生的概率预测；
  - ensemble、MC dropout 和校准；
  - 评估不确定性是否能区分高误差和低误差样本。
- **开源代码**：截至当前未确认官方仓库
- **本项目定位**：未来新生预报与不确定性评价

---

## 9. 当前进展

### 9.1 已完成

- 已完成物理可解释性、多模态融合、生成式预测、频域方法和对象级方法的初步文献梳理；
- 已确定以 OpenSTL 作为工程骨架；
- 已完成 OpenSTL 环境安装；
- 已克隆仓库；
- 官方示例已经成功运行；
- 已初步确定主骨架为 Tiny-SimVP-gSTA；
- 已确定 Earthformer 作为后续 Transformer 强基线；
- 已确定任务为 10 帧输入、20 帧输出、6 min/帧；
- 已确定输出未来 Radar，再统一通过冻结 Z–R 转为降雨率；
- 已确定 CSI@16、CSI@32 为主指标；
- 已明确 BTH 研究区和四个月数据范围；
- 已明确 Radar、PWV、Rain PNG 的反向灰度范围；
- 已确定 Z–R 必须只用 Train 拟合并冻结；
- 已形成“Radar 主运动、PWV/DEM 主源汇调制”的初步物理假设。

### 9.2 尚未完成

- 数据完整性审计；
- 独立事件识别与最终 Train/Val/Test 清单；
- OpenSTL 自定义 Radar Dataset；
- 反向灰度统一解码；
- 有效区 mask；
- 10→20 输出机制确认；
- Persistence、pySTEPS 和 ConvLSTM 基线；
- Tiny-SimVP Radar-only V0；
- 本地 Z–R 参数拟合；
- 站点—Radar 网格匹配规则；
- 多随机种子稳定性测试；
- Earthformer 基线适配；
- PWV、DEM 与物理分解模块。

---

## 10. 当前实施计划

### 阶段 A：冻结协议与数据审计

需要完成：

1. 固定研究区域网格；
2. 核对 Radar、PWV、Rain、DEM 的尺寸、方向、经纬度范围；
3. 确认 PNG 反向灰度；
4. 检查缺帧、重复帧、异常值和时间错位；
5. 建立有效区 mask；
6. 统计 5—8 月强降水事件；
7. 先按日期/事件划分，再生成滑窗；
8. 输出固定 `train/val/test manifest`；
9. 决定 8 月验证与测试的事件分配；
10. 明确 10→20 在 OpenSTL 中是直接输出还是递归输出。

**完成标准**：任意样本都可由 manifest 追溯到原始时间、事件和文件。

### 阶段 B：Radar-only 数据接入

需要完成：

1. 编写 OpenSTL 自定义 Dataset/DataLoader；
2. 输入形状统一为 `[B, 10, 1, H, W]`；
3. 标签形状统一为 `[B, 20, 1, H, W]`；
4. PNG 解码后使用正向强度语义；
5. 可视化随机样本的 10 帧输入和 20 帧标签；
6. 人工核查时间顺序和空间方向；
7. 用少量样本做过拟合测试。

**完成标准**：模型能在小样本上明显拟合，输出时序和空间位置正确。

### 阶段 C：建立 V0 Radar-only 基线

按顺序运行：

1. Persistence；
2. 可选 pySTEPS；
3. ConvLSTM；
4. Tiny-SimVP-gSTA；
5. 多随机种子；
6. 固定 V0 checkpoint、配置和指标。

**完成标准**：

- Tiny-SimVP 在关键指标上至少能与 Persistence 形成有意义差异；
- 训练无异常；
- 预测不全零、不全均值、不反相；
- 指标与可视化一致；
- 不同 seed 波动可接受。

### 阶段 D：拟合并冻结本地 Z–R

1. 仅用 Train；
2. Radar 6 min 与站点小时降雨匹配；
3. 对每小时 10 帧积分；
4. 在 Train 内确定 \(a,b\)；
5. 冻结参数；
6. 建立网格评价与站点评价；
7. 所有模型统一使用。

**完成标准**：Z–R 参数、拟合样本、匹配规则和误差可复现。

### 阶段 E：Earthformer 强基线

1. 使用相同 Radar-only 数据；
2. 构建 Tiny-Earthformer；
3. 尽量匹配 Tiny-SimVP 参数规模；
4. 统一训练预算和评价；
5. 比较性能、稳定性、显存与训练时间；
6. 资源允许时增加官方规模配置。

**目标**：判断 Transformer 是否真的适合当前单地区小样本，而不是默认其更先进。

### 阶段 F：简单多源基线

依次增加：

1. Radar + PWV 直接拼接；
2. Radar + DEM；
3. Radar + PWV + DEM。

这些版本只用于回答数据增量，不作为最终创新。

### 阶段 G：Radar-only 物理分解

1. motion head；
2. differentiable warp；
3. source/sink head；
4. 运动平滑与中间变量评价；
5. 与直接预测 V0 比较。

### 阶段 H：PWV 与 DEM 物理接入

1. PWV adapter；
2. PWV tendency；
3. PWV 置信度；
4. PWV 仅调制 source；
5. DEM static adapter；
6. 山区/平原分区评价；
7. 置乱、静态气候态和错误时间偏移对照。

### 阶段 I：高级模块

仅在前面稳定后单独测试：

- FACL；
- AlphaPre 式幅相分解；
- 对象级辅助损失；
- DiffCast/CasCast/PreDiff；
- 不确定性与集合预报。

---

## 11. 首轮实验矩阵

| 编号 | 输入 | 主干/结构 | 目的 |
|---|---|---|---|
| B0 | 最后一帧 Radar | Persistence | 最低基线 |
| B1 | Radar | pySTEPS/光流 | 平流基线 |
| B2 | Radar | ConvLSTM | 经典循环基线 |
| B3 | Radar | Tiny-SimVP-gSTA | 主骨架 V0 |
| B4 | Radar | Tiny-Earthformer | Transformer 强基线 |
| M1 | Radar+PWV | 简单拼接 SimVP | PWV 信息增量 |
| M2 | Radar+DEM | 静态融合 SimVP | DEM 信息增量 |
| M3 | Radar+PWV+DEM | 简单融合 SimVP | 多源上限基线 |
| P1 | Radar | Motion+Source | 物理分解本身 |
| P2 | Radar+PWV | PWV 调制 Source | 核心研究模型 |
| P3 | Radar+PWV+DEM | PWV/DEM 调制 Source | 完整地区专用模型 |

---

## 12. 当前关键风险

### 12.1 数据量与样本独立性

四个月数据可以生成大量滑窗，但独立天气事件数量仍然有限。不能将滑窗数量等同于独立样本数量。

### 12.2 8 月内部泄漏

若在滑窗级随机分割 Val/Test，相邻样本会共享绝大多数帧，测试结果将失真。

### 12.3 PWV 插值过平滑

250 个 CORS 站插值至 0.1° 网格后，可能生成过度平滑或虚假的局地梯度。需要站点距离、插值置信度或留站验证。

### 12.4 DEM 位置记忆

单地区固定 DEM 容易让模型记住“哪些位置通常下雨”，而不是学习地形机制。必须加入分区评价、置乱对照和 adapter 设计。

### 12.5 Rain PNG 饱和

Rain PNG 最大仅 35 mm/h，CSI@32 接近上限。主评价不应依赖截断后的 PNG。

### 12.6 Z–R 不确定性

Z–R 关系受地区、季节、降水类型和雷达误差影响。需要报告拟合不确定性，并确保所有模型使用同一冻结关系。

### 12.7 复杂模型不稳定

Earthformer、扩散和多分支物理模型可能在小数据上产生较大 seed 方差。必须先建立轻量稳定基线。

### 12.8 可解释性名不副实

分支被命名为 motion、source、PWV gate，并不证明其学习了对应过程。需要中间变量监督、诊断和干预。

---

## 13. 近期最优先事项

当前不应立即加入 PWV、DEM 或物理模块。下一步优先完成：

1. 数据审计；
2. 固定事件/日期划分；
3. PNG 正向解码；
4. Radar-only OpenSTL Dataset；
5. 10→20 输出机制确认；
6. 小样本过拟合；
7. Persistence；
8. ConvLSTM；
9. Tiny-SimVP；
10. 统一 CSI@16/32、POD、FAR、Bias、MAE、RMSE；
11. 多随机种子；
12. 冻结 V0。

随后再进行：

> Earthformer 强基线 → 本地 Z–R → 简单 PWV/DEM 融合 → Radar-only 运动—源汇 → PWV/DEM 物理调制。

---

## 14. 阶段性成功标准

### V0 成功

- 数据无明显错位和泄漏；
- 小样本可过拟合；
- Tiny-SimVP 稳定训练；
- 多 seed 可重复；
- 与 Persistence、ConvLSTM 公平比较；
- CSI@16/32 和可视化一致。

### 物理分解成功

- Motion head 对位移有可测贡献；
- Source head 对强度变化有可测贡献；
- 关闭任一分支会产生符合预期的退化；
- 中间变量不能被其他分支完全补偿。

### PWV 成功

- PWV 相对 Radar-only 在独立事件上有可信增益；
- 增益集中在预设过程，例如增强、新生或维持；
- 不以 FAR、Bias 或面积膨胀显著恶化为代价；
- 动态 PWV 优于静态气候态、置乱 PWV 和错误时间偏移。

### DEM 成功

- 山区—平原过渡带收益高于不相关区域；
- 置乱 DEM 后收益消失；
- 不只是整体位置记忆；
- 对独立事件仍稳定。

### 可推广性成功

- 在未见月份和未见事件上稳定；
- 参数较少且 seed 方差可控；
- 冻结主干、调整 adapter 仍能保持较高性能；
- 在没有跨地区数据前，不做超出证据的跨区域结论。

---

## 15. 建议的项目目录逻辑

不限定具体代码实现，但建议研究资产按以下逻辑隔离：

- `data_manifest/`：固定样本、事件和划分；
- `configs/`：每个实验独立配置；
- `datasets/`：Radar-only 与多模态 Dataset；
- `models/`：SimVP、Earthformer、物理分解模型；
- `adapters/`：PWV、DEM；
- `physics/`：warp、motion、source、Z–R；
- `metrics/`：网格、对象、站点和 bootstrap；
- `reports/`：每次实验自动报告；
- `visualizations/`：预测、真值、中间变量；
- `checkpoints/`：按协议和 seed 保存；
- `docs/`：任务协议、数据说明和变更记录。

原则是：

> 数据划分、Z–R、评价代码和 V0 一旦冻结，后续模型实验不得暗中修改。

---

## 16. 冻结评估协议 V1（2026-07-30）

后续实验统一采用“数据质量—像素评分—空间结构—对象演化—统计可信度”
五层体系。主任务固定为过去 10 帧（60 min）Radar/PWV/DEM 输入，预测未来
20 帧 Radar（120 min），帧间隔 6 min。主真值为雷达反射率经同一套冻结
Z–R 关系转换得到的二维降雨率；地面站用于训练集 Z–R 标定和独立验证，
不能替代二维真值场。

### 16.1 不可变口径

- 必须先按连续时段或独立天气事件切分，再生成滑窗；
- 缺测值使用独立 valid mask，不能作为 0 降雨；
- 主分类评分是严格 1×1 格点，FSS 3×3/5×5 仅作空间容差诊断；
- 雨强阈值固定为 0.1、2.5、8、16、32 mm/h；
- 核心指标固定为 CSI@16 与 CSI@32，重点比较 0–1 h 和 1–2 h；
- 同时保存 +6…+120 min 曲线及 +30/+60/+90/+120 min 快照；
- 无事件且分母为 0 时，CSI/POD/FAR/HSS 记为 NaN，macro 时跳过；
- 所有模型必须使用完全相同的 Z–R 参数和测试事件清单。

当前配置中的 `Z=200R^1.6` 只是 Marshall–Palmer 通用关系的工程占位基线，
不是已完成的本地化标定。正式论文实验前，必须只用训练集站点—雷达配对
数据拟合本地 `(a,b)`，保存拟合范围、样本数、R²、MAE、RMSE、Mean Bias
及 16/32 mm/h 分类评分，然后冻结参数。验证集与测试集严禁重新拟合。

### 16.2 五层输出

1. **数据质量**：PWV 的 Pearson R、RMSE、Mean Bias（整体、逐站、地形分区、
   非降水/强降水期）；Z–R 的参数、R²、连续和分类误差。
2. **像素评分**：全有效格点及真值雨区分别计算 MAE、RMSE、Mean Error、
   Intensity Ratio；每个阈值计算 TP/FP/FN/TN、POD、FAR、CSI、HSS 和
   Frequency Bias。
3. **空间结构**：FSS 1×1/3×3/5×5、质心误差（km）、Area Ratio、
   Energy Ratio、Peak/P95/P99 Error、PSD 和高频能量保持率。
4. **对象演化**：16/32 mm/h 连通对象的 Object POD/FAR、匹配 IoU、位置、
   面积和峰值误差；生命周期、新生/增强/消散及 split/merge 必须在真正
   独立事件轨迹冻结后计算。
5. **统计可信度**：同时报告 micro/global 与 event macro
   mean/median/Q25/Q75；以独立事件为单位做 2000 次配对 bootstrap；
   正式实验至少使用 seed={0,1,2} 并报告 mean±std。

当前 Radar 数据集的 `event_id` 仍是日期代理。由它生成的 event macro 与
bootstrap 只用于代码联调，不得作为论文置信区间。需要先建立独立天气事件
manifest，避免重叠滑窗被当成独立样本。

### 16.3 模型晋级条件

新模型相对同参数量 Radar-only 应满足：

- `ΔCSI@32_(1–2h) > 0`，且配对事件 bootstrap 95% CI 下界大于 0；
- 第一小时 `ΔCSI@16` 和 `ΔCSI@32` 均不低于 −0.005；
- 不能依赖雨区膨胀，建议 `0.8 ≤ AreaRatio ≤ 1.2` 且 FAR 不恶化；
- 质心、能量、峰值或对象 POD 至少一项改善；
- 三个随机种子的改善方向一致。

PWV 的有效性必须通过 Radar+PWV 与 Radar-only、Static-PWV、
Shuffled-PWV（必要时 Wrong-time/Zero-PWV）的成组比较证明，不能只以
“优于 Persistence”作为物理信息有效的证据。

### 16.4 当前代码状态

`openstl/core/precipitation_metrics.py` 已实现流式主评分、冻结 Z–R 转换、
HSS/TN、雨区连续误差、逐时效/逐事件统计、FSS、质心/面积/能量/峰值、
逐帧连通对象匹配、PSD 诊断及事件配对 bootstrap。输出包括：

- `summary.json` / `metrics.json`；
- `per_lead_metrics.csv`；
- `per_window_metrics.csv`；
- `per_event_metrics.csv`；
- `per_object_metrics.csv`；
- `confusion_counts.csv`；
- `bootstrap_ci.json`；
- 时效曲线、PSD、成功/失败案例图。

尚未宣称完成的部分是 PWV 产品质检、本地 Z–R 拟合、真正事件轨迹、
对象生命周期与 split/merge，以及多随机种子汇总。这些功能依赖后续数据
和事件 manifest，不应以日期代理结果替代。

---

## 17. Radar 无损训练缓存（2026-07-30）

为避免每个 epoch 重复扫描和解压 32,457 个 PNG，Radar 原始灰度帧已一次性
打包为 `uint8` NPY：

```text
DATA_2025_S/RADAR_CACHE_UINT8/
├── frames.npy
└── manifest.json
```

- 数组形状：`[32457, 66, 70]`；
- 数据类型：`uint8`；
- 有效像素数据约 149.95 MB；
- `manifest.json` 保存逐帧 ISO 时间戳、形状、格式版本和解码公式；
- 训练时通过只读 `numpy.memmap` 按需取帧；
- 反色与归一化仍在取样时执行：`(255-pixel)/255`；
- 未提前保存 float32，因此缓存无损且空间开销较小；
- 每个 DataLoader worker 延迟打开自己的 mmap 句柄，操作系统共享页缓存；
- 删除配置中的 `radar_cache_path` 即可无缝回退到原 PNG loader。

真实数据抽查了首帧、中间帧和末帧，缓存与 PNG 逐像素完全一致。单进程顺序
读取 100 个 10→20 样本的本机基准中，PNG 用时约 11.73 s，mmap 缓存约
0.069 s，约 169× 加速。该数字是热缓存、小批量读取基准，不等同于端到端
训练加速；GPU 计算时间不会因此减少，但数据等待会显著降低。

缓存构建命令：

```bash
python tools/cache_bth_radar.py \
  --source /path/to/DATA_2025_S/RADAR_2025_S \
  --output /path/to/DATA_2025_S/RADAR_CACHE_UINT8
```

缓存只改变存储与读取方式，不改变样本窗口、时间戳、事件划分、像素值、
归一化或评估协议。原 PNG 仍是可追溯源数据，不应删除。

> **项目续接入口（2026-07-31）**：新对话或新执行环境应先阅读
> [`.research/context_index.md`](context_index.md)。机器可读的项目、数据和实验状态分别在
> `project_manifest.yml`、`data_dictionary.yml`、`experiment_matrix.yml`；冻结决策和待解决问题分别在
> `decisions.md`、`open_questions.md`。本文件继续作为研究方案和评估协议的详细正文。

---

## 18. 2026-08-08 项目续接快照（优先于前文过时状态）

### 18.1 当前最好工程基线

当前同一 Validation 协议下效果最好的已报告模型是：

```text
bth_direct_physics_hybrid_v2_clean_manifest_10ep_seed0
config: configs/bth_radar/DirectPhysicsHybrid_r2d_no_deep_convlstm.py
report: .research/mixed/satge1.md
baseline summary: .research/baselines/v2/summary.md
weighted validation CSI score: 0.937194
parameters: 4,809,641 total / 1,064,617 trainable / 3,745,024 frozen
```

该模型由冻结的完整 ConvLSTM direct prior 和轻量 U-Net motion/source 修正组成。
Branch attribution 表明增益主要来自 motion；source-only 贡献接近零。+60/+120
分钟关键参考值：

| Lead | CSI16 | CSI32 | FAR16 | FAR32 |
|---:|---:|---:|---:|---:|
| +60 min | 0.213774 | 0.130061 | 0.670481 | 0.826032 |
| +120 min | 0.107056 | 0.049803 | 0.846631 | 0.937255 |

`0.937194` 是当前 radar-derived Validation 工程参考，不是论文最终真值结论。
目前开发策略改为效果优先：少做形式化验证，多尝试新机制；机制有效前只跑
seed 0/Validation，不默认跑 Test、多 seed 或 bootstrap。

### 18.2 当前主线 V3a

下一模型固定为 error-aware routed `preserve + motion + decay`，暂不包含
growth。详细协议和改造计划：

- `.research/mixed/v3a_routing_protocol.md`
- `.research/mixed/v3a_implementation_plan.md`
- `.research/design_brief.md`

核心形式：

\[
\hat R=p^pD+p^m\mathcal W(D,\Delta U)+p^dD(1-Q^{decay}).
\]

16 mm/h 连通对象定义 storm footprint，32 mm/h core 作内部细化；首版搜索
半径 2 grid（约 20 km），16/32 路由软标签权重为 1.0/1.5，模糊情况 ignore。

### 18.3 本地 WSL 环境与命令

```text
Windows repo: D:\_Search\AIforScience\Rewritten\origin\OpenSTL
WSL repo:     /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL
Python:       /home/ranye/miniconda3/envs/OpenSTL/bin/python
Data root:    /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S
GPU:          RTX 4060 Laptop 8 GB
```

训练模板：

```bash
cd /mnt/d/_Search/AIforScience/Rewritten/origin/OpenSTL
PYTHONPATH=. /home/ranye/miniconda3/envs/OpenSTL/bin/python tools/train.py \
  --dataname bth_radar \
  --method METHOD_NAME \
  --config_file configs/bth_radar/CONFIG.py \
  --data_root /mnt/d/_Search/AIforScience/Rewritten/capsule-3935105/data/DATA_2025_S \
  --ex_name EXPERIMENT_NAME \
  --epoch EPOCHS \
  --batch_size 4 \
  --val_batch_size 4 \
  --seed 0 \
  --deterministic \
  --no_display_method_info \
  --skip_test_after_train
```

### 18.4 服务器环境与命令

服务器仓库与工作目录通常为：

```text
repo/work root: /root/weather
data root:      /root/weather/data（在仓库内可写作 data）
work dirs:      /root/weather/work_dirs
```

服务器 `data/` 下应直接包含数据目录和缓存，例如：

```text
data/
├── RADAR_CACHE_UINT8/
├── PWV_CACHE_UINT8/
├── RAIN_CACHE_UINT8/
├── RADAR_2025_S/       # 若保留原 PNG
├── PWV_2025_S/
└── RAIN_2025_S/
```

在已激活 OpenSTL Python/Conda 环境后使用：

```bash
cd /root/weather
export PYTHONPATH=.
python tools/train.py \
  --dataname bth_radar \
  --method METHOD_NAME \
  --config_file configs/bth_radar/CONFIG.py \
  --data_root data \
  --ex_name EXPERIMENT_NAME \
  --epoch EPOCHS \
  --batch_size 4 \
  --val_batch_size 4 \
  --seed 0 \
  --deterministic \
  --no_display_method_info \
  --skip_test_after_train
```

服务器 Python 可执行文件以实际激活环境为准；不要把本地
`/home/ranye/miniconda3/...` 路径复制到服务器。

### 18.5 缓存约定

本地三个 lossless mmap 缓存位于本地 `DATA_2025_S` 下；服务器位于 `data/`
下。配置使用相对路径：

```python
radar_cache_path = 'RADAR_CACHE_UINT8'
pwv_cache_path = 'PWV_CACHE_UINT8'       # PWV loader 接入后使用
rain_cache_path = 'RAIN_CACHE_UINT8'
```

PWV/RAIN 缓存构建与验证见 `docs/bth_multisource_cache.md`：

```bash
python tools/cache_bth_png.py --variable pwv --data-root DATA_ROOT
python tools/cache_bth_png.py --variable rain --data-root DATA_ROOT
python tools/verify_bth_png_cache.py --cache DATA_ROOT/PWV_CACHE_UINT8
python tools/verify_bth_png_cache.py --cache DATA_ROOT/RAIN_CACHE_UINT8
```

缓存只改变 I/O，不改变样本、时间戳、归一化或评价协议。

### 18.6 Validation 与自动报告命令

显式 checkpoint Validation：

```bash
PYTHONPATH=. python tools/validate.py \
  --dataname bth_radar \
  --method METHOD_NAME \
  --config_file configs/bth_radar/CONFIG.py \
  --data_root DATA_ROOT \
  --ckpt_path work_dirs/EXPERIMENT/checkpoints/best_val_csi.ckpt \
  --val_batch_size 4 \
  --no_display_method_info
```

单个 work directory 自动生成完整 Validation 报告：

```bash
PYTHONPATH=. python tools/report_bth_workdir.py \
  --work_dir work_dirs/EXPERIMENT \
  --data_root DATA_ROOT \
  --val_batch_size 4
```

默认输出 `work_dirs/EXPERIMENT/validation_report.md`，自动读取
`model_param.json` 和 `best_val_csi.ckpt`。这是 Validation-only，不使用 Test。

批量扫描并生成比较表：

```bash
PYTHONPATH=. python tools/report_bth_all_workdirs.py \
  --work_root work_dirs \
  --data_root DATA_ROOT \
  --output_dir work_dirs/evaluation_summary \
  --batch_size 4 \
  --val_batch_size 4 \
  --reuse_reports
```

输出 `bth_model_comparison.csv/.md`、失败清单和扫描 inventory。

当前 DirectPhysicsHybrid 分支 attribution：

```bash
PYTHONPATH=. python tools/evaluate_direct_physics_attribution.py \
  --work_dir work_dirs/EXPERIMENT \
  --data_root DATA_ROOT \
  --val_batch_size 4
```

输出 `branch_attribution.json` 和 `branch_attribution.md`。V3a 后续应新增
`preserve/motion/decay/oracle-routed/learned-routed` attribution 工具。

本地命令将 `python` 替换为
`/home/ranye/miniconda3/envs/OpenSTL/bin/python`，`DATA_ROOT` 替换为本地绝对
路径；服务器保持已激活环境的 `python`，`DATA_ROOT` 使用 `data`。

### 18.7 后续 AI 的读取顺序

仓库根目录 `AGENTS.md` 是最短入口。任何新 AI/会话在修改数据、训练配置、
模型或报告工具前，应按顺序读取：

1. `.research/detail.md` 本节；
2. `.research/context_index.md`；
3. `.research/project_manifest.yml`、`data_dictionary.yml`；
4. 与当前模型相关的 `.research/mixed/` 方案和最新报告；
5. 仅在需要时读取历史实验，不要重新扫描整个 `work_dirs/`。

### 18.8 V3a 已实现入口与运行顺序（2026-08-08）

V3a 当前代码入口：

```text
model:   openstl/models/direct_physics_routed_model.py
method:  openstl/methods/direct_physics_routed.py
routing: openstl/modules/v3a_routing.py
cache:   openstl/datasets/v3a_routing_cache.py
builder: tools/build_bth_v3a_routing_cache.py
audit:   tools/audit_v3a_init.py
attrib:  tools/evaluate_v3a_attribution.py
```

第一步，一次性生成 train/val routing cache：

```bash
PYTHONPATH=. python tools/build_bth_v3a_routing_cache.py \
  --config_file configs/bth_radar/DirectPhysicsRouted_v3a.py \
  --data_root DATA_ROOT \
  --output DATA_ROOT/V3A_ROUTING_CACHE \
  --splits train val \
  --batch_size 4 \
  --num_workers 4
```

服务器用 `DATA_ROOT=data`；本地使用本节前述绝对路径，并把 `python` 替换为
本地 OpenSTL Python。标签保存为 packed uint8，不保存庞大的 float soft mask；
训练读取时再按 `w16=1.0,w32=1.5` 解码。

第二步，专家预训练：

```bash
PYTHONPATH=. python tools/train.py \
  --dataname bth_radar \
  --method DirectPhysicsRouted \
  --config_file configs/bth_radar/DirectPhysicsRouted_v3a.py \
  --data_root DATA_ROOT \
  --ex_name bth_v3a_expert_seed0 \
  --epoch 3 --batch_size 4 --val_batch_size 4 \
  --seed 0 --deterministic --no_display_method_info \
  --skip_test_after_train
```

本地首次运行可将配置换成
`configs/bth_radar/DirectPhysicsRouted_v3a_local.py`，它使用本地已有 V2
checkpoint；服务器主配置指向服务器的 V2 clean-manifest checkpoint。

第三步，router 预训练：

```bash
PYTHONPATH=. python tools/train.py \
  --dataname bth_radar \
  --method DirectPhysicsRouted \
  --config_file configs/bth_radar/DirectPhysicsRouted_v3a_router.py \
  --data_root DATA_ROOT \
  --init_from_ckpt work_dirs/bth_v3a_expert_seed0/checkpoints/best_val_csi.ckpt \
  --ex_name bth_v3a_router_seed0 \
  --epoch 3 --batch_size 4 --val_batch_size 4 \
  --seed 0 --deterministic --no_display_method_info \
  --skip_test_after_train
```

第四步，联合强修正：

```bash
PYTHONPATH=. python tools/train.py \
  --dataname bth_radar \
  --method DirectPhysicsRouted \
  --config_file configs/bth_radar/DirectPhysicsRouted_v3a_joint.py \
  --data_root DATA_ROOT \
  --init_from_ckpt work_dirs/bth_v3a_router_seed0/checkpoints/best_val_csi.ckpt \
  --ex_name bth_v3a_joint_seed0 \
  --epoch 8 --batch_size 4 --val_batch_size 4 \
  --seed 0 --deterministic --no_display_method_info \
  --skip_test_after_train
```

V3a attribution：

```bash
PYTHONPATH=. python tools/evaluate_v3a_attribution.py \
  --work_dir work_dirs/bth_v3a_joint_seed0 \
  --data_root DATA_ROOT \
  --val_batch_size 4
```

训练进度条现在额外显示：

```text
val_csi16_t60  val_csi32_t60
val_csi16_t120 val_csi32_t120
```

完整长名称仍保留在 Lightning metrics/checkpoint 中。
