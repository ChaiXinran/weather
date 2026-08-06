可以直接在你们现有 `weather/OpenSTL` 上替换，而且**不应从零新建工程，也不应直接把 `EvolutionConvLSTM_Model` 改得面目全非**。

最稳妥的方案是：

[
\boxed{
\text{保留现有数据、评估、EvolutionOperator和物理监督}
+
\text{新增一条Temporal U-Net物理模型分支}
}
]

也就是让下面两套模型并行存在：

```text
EvolutionConvLSTM        当前基线，永久保留
EvolutionTemporalUNet    新主干，逐阶段验证
```

这样既能复用前面的全部工作，也能公平判断“换成 U-Net/FPN 多尺度主干是否真的有效”。

---

# 一、当前代码实际是什么结构

目前 `EvolutionConvLSTM_Model` 可以概括为：

```text
历史 Radar 10帧
    ↓
pixel_unshuffle
    ↓
4层 ConvLSTM
    ↓
最后一层、最后时刻 feature
    ├── motion_head → 20步流场
    └── source trunk
          ├── growth / steady / decay
          ├── growth magnitude
          ├── decay magnitude
          └── 可选 edge residual flow
    ↓
EvolutionOperator
    ↓
未来场
```

当前历史编码器只返回最后一层 ConvLSTM 的最终隐藏状态；运动头从这一个特征一次性输出全部未来时效的二维流场。

目前的 factorized source 已经实现了：

* growth、steady、decay 三分类；
* growth、decay 独立幅度；
* 分段 source capacity；
* persistent interior mask；
* 可选边缘残差流；
* 自回归物理演化。

`EvolutionOperator` 也已经独立实现：

* rain-rate 空间平流；
* dBZ 与雨强转换；
* 有界 signed source；
* factorized growth/steady/decay；
* source mask；
* source-free 兼容路径。

训练方法中已有：

* persistent interior、edge、birth、death、clear 区域划分；
* growth/steady/decay 标签；
* 类别平衡交叉熵；
* 条件幅度损失；
* 强降水像素加权状态损失；
* source 中间指标诊断。

因此，本次替换的重点不是重写物理部分，而是替换：

[
\boxed{
\texttt{encode_history()}
+
\texttt{motion_head}
+
\text{提供给source的历史特征}
}
]

---

# 二、建议的新主干：Evolution Temporal U-Net

建议模型名称使用：

```text
EvolutionTemporalUNet
```

或者论文内部暂称：

```text
PTU-Net
Physics-guided Temporal U-Net
```

不要直接叫 U-Net，因为它不是普通的“输入历史、直接回归未来图像”的 U-Net，而是：

> U-shaped 多尺度网络负责提取历史条件，物理算子负责真正生成未来。

整体结构：

```text
Radar history [B,10,1,66,70]
              │
              ▼
      Shared frame encoder
              │
     ┌────────┼────────┬────────┐
     ▼        ▼        ▼        ▼
   E0全分辨率 E1半分辨率 E2 1/4   E3 1/8
     │        │        │        │
     └────────┴────────┴────────┘
              │
              ▼
       Temporal fusion
              │
              ▼
       U-Net/FPN decoder
              │
       ┌──────┴───────┐
       ▼              ▼
 motion feature    source feature
       │              │
       ▼              ▼
 20步 flow      growth/steady/decay
       │              │
       └──────┬───────┘
              ▼
      EvolutionOperator
              ▼
     Future Radar / rain rate
```

---

# 三、具体张量架构

输入固定为：

[
X\in\mathbb R^{B\times10\times1\times66\times70}
]

## 1. 共享逐帧空间编码器

将每一帧通过同一个 encoder：

```python
flat = history.reshape(B * T, 1, 66, 70)
features = encoder(flat)
```

推荐四个空间尺度：

| 层级 |  通道 |  空间大小 |
| -- | --: | ----: |
| E0 |  32 | 66×70 |
| E1 |  64 | 33×35 |
| E2 | 128 | 17×18 |
| E3 | 192 |   9×9 |

恢复时间维：

```text
E0: [B,10,32,66,70]
E1: [B,10,64,33,35]
E2: [B,10,128,17,18]
E3: [B,10,192,9,9]
```

这里不需要强行把输入 pad 成 64、72 或其他尺寸。下采样使用：

```python
nn.Conv2d(..., kernel_size=3, stride=2, padding=1)
```

上采样时始终：

```python
F.interpolate(x, size=skip.shape[-2:])
```

这样 66×70 的奇偶尺寸可以自然恢复，不会出现 SimVP 中对 `N_S` 的尺寸限制。

---

# 四、空间块怎么实现

不要照搬 2015 年原始双卷积 U-Net，也不要导入完整 ConvNeXt 网络。

建议每个 stage 使用轻量残差块：

```text
Depthwise 5×5 Conv
    ↓
GroupNorm
    ↓
1×1 Conv，通道扩张2倍
    ↓
GELU
    ↓
1×1 Conv，恢复通道
    ↓
Residual
```

可以优先复用仓库现有的：

```python
ConvNeXtSubBlock
```

当前 `openstl/modules/__init__.py` 已经导出了：

* `ConvNeXtSubBlock`；
* `TAUSubBlock`；
* `SwinSubBlock`；
* `GASubBlock`；
* `MogaSubBlock` 等。

但第一版建议只使用：

```text
ConvNeXtSubBlock
```

或者自己写一个约 30–40 行的 `DWResidualBlock`。不要第一版同时比较 ConvNeXt、Swin、Moga、TAU。

推荐每级：

```text
Stem:
Conv 3×3, 1→32

E0:
2 × DWResidualBlock(32)

Down1:
Conv 3×3 stride 2, 32→64
2 × DWResidualBlock(64)

Down2:
Conv 3×3 stride 2, 64→128
2 × DWResidualBlock(128)

Down3:
Conv 3×3 stride 2, 128→192
2 × DWResidualBlock(192)
```

---

# 五、时间融合怎么做

这里不要把 `TAUSubBlock` 误认为可以直接接收：

```text
[B,T,C,H,W]
```

你们仓库中的 `TAUSubBlock` 实际接收的是普通二维特征：

```text
[B,C,H,W]
```

其内部是大核深度卷积、通道门控和残差结构。

在 SimVP 中，时间维先被折叠进通道维，再交给 MetaFormer/TAU block。因此有两种合理实现。

## 第一版：显式 TemporalConv，最稳妥

在每个尺度上使用：

```python
class TemporalMixer(nn.Module):
    def forward(self, x):
        # x: [B,T,C,H,W]
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]

        x = depthwise_conv3d(
            x,
            kernel_size=(3, 1, 1),
            groups=C,
        )

        weights = temporal_attention(x)
        return (x * weights).sum(dim=2)
```

输出：

```text
F0: [B,32,66,70]
F1: [B,64,33,35]
F2: [B,128,17,18]
F3: [B,192,9,9]
```

优点：

* 真正沿时间维卷积；
* 参数量低；
* 时间顺序明确；
* 不需要递归；
* 不会把所有历史压进最后一个 hidden state。

## 第二版：在粗尺度加入 TAU

在得到 `F2/F3` 后，可增加：

```python
F2 = TAUSubBlock(128)(F2)
F3 = TAUSubBlock(192)(F3)
```

它在这里负责：

* 扩大空间感受野；
* 识别大范围雨带组织；
* 调整不同通道中的历史模式。

建议实验顺序：

```text
TemporalConv only
→ TemporalConv + TAU at E3
→ TemporalConv + TAU at E2/E3
```

不要一开始就在四个尺度全放 TAU。

---

# 六、U-Net/FPN Decoder

建议不是标准 U-Net 的单纯 concat，而是使用 FPN 风格的横向投影。

```python
D3 = block(F3)

D2 = block(
    lateral2(F2)
    + interpolate(D3, size=F2.shape[-2:])
)

D1 = block(
    lateral1(F1)
    + interpolate(D2, size=F1.shape[-2:])
)

D0 = block(
    lateral0(F0)
    + interpolate(D1, size=F0.shape[-2:])
)
```

建议输出：

```text
D3: [B,192,9,9]
D2: [B,128,17,18]
D1: [B,64,33,35]
D0: [B,32,66,70]
```

相比 concat，FPN 加法融合：

* 参数更少；
* 显存更低；
* 尺度职责更清晰；
* 更适合 8 GB 显存。

可以额外输出：

```python
{
    "fine": D0,
    "middle": D1,
    "coarse": D2,
    "bottleneck": D3,
}
```

---

# 七、运动头怎么替换

当前 motion head 使用 ConvLSTM 最终 feature，一次性预测：

```text
[B,20×2,h,w]
```

然后上采样并 reshape 为：

```text
[B,20,2,66,70]
```

新模型先保持完全相同的输出协议，以便公平比较。

## 推荐 motion feature

使用：

```python
motion_feature = torch.cat([
    D2,
    interpolate(D3, size=D2.shape[-2:])
], dim=1)
```

形状约：

```text
[B,320,17,18]
```

经过投影：

```python
motion_feature = Conv1x1(320, 128)
```

motion head：

```python
self.motion_head = nn.Sequential(
    DWResidualBlock(128),
    DWResidualBlock(128),
    nn.Conv2d(128, 20 * 2, kernel_size=1),
)
```

最后一层零初始化：

```python
nn.init.zeros_(self.motion_head[-1].weight)
nn.init.zeros_(self.motion_head[-1].bias)
```

预测：

```python
raw_flow = self.motion_head(motion_feature)
raw_flow = F.interpolate(
    raw_flow,
    size=(66, 70),
    mode="bilinear",
    align_corners=False,
)
raw_flow = raw_flow.view(B, 20, 2, 66, 70)
flow = max_displacement * torch.tanh(raw_flow)
```

这样初始模型仍接近 persistence，与当前 R4-b 的初始化策略一致。当前 ConvLSTM motion head 本身也是零初始化并通过 `tanh` 限制最大位移。

---

# 八、第一阶段只做 motion-only

新模型第一版只执行：

[
R_t=\mathcal W(R_{t-1},v_t)
]

不要同时迁移 source。

调用仍然是：

```python
result = self.operator(
    history[:, -1],
    flow,
    source=None,
)
```

因此输出接口与现有模型保持一致：

```python
{
    "prediction": ...,
    "advected": ...,
    "flow": ...,
    "source": None,
}
```

这一步的目标是明确回答：

> U-Net/FPN 多尺度历史表示是否比 ConvLSTM 最终隐藏状态更适合预测运动场？

如果运动基线本身不行，就不要把 source 接上去。

---

# 九、source/sink如何迁移

motion-only 通过以后，source 使用高分辨率 decoder feature：

```text
source_feature = D0
[B,32,66,70]
```

每个未来时刻构造：

```python
source_input = torch.cat([
    source_feature,                # 32
    advected_rain / max_rain,      # 1
    flow / max_displacement,       # 2
    gradient_magnitude,            # 1
], dim=1)
```

总通道：

```text
36 channels
```

source trunk：

```python
self.factorized_trunk = nn.Sequential(
    nn.Conv2d(36, 64, 3, padding=1),
    nn.GroupNorm(8, 64),
    nn.SiLU(),
    DWResidualBlock(64),
)
```

三个输出头：

```python
regime_head: 64 → 3
growth_head: 64 → 1
decay_head:  64 → 1
```

然后直接调用现有：

```python
self.operator.evolve_factorized_step(...)
```

因此物理公式、mask、capacity 和损失不变。

## 第一版不要添加 lead embedding

当前 factorized ConvLSTM source 输入中没有 lead embedding，而是使用：

* 历史 feature；
* 当前 advected rain；
* 当前 flow；
* 当前梯度。

为了判断收益是否来自主干，Temporal U-Net 第一版也应保持一致。

后续再单独消融：

```text
+ lead-time embedding
```

否则主干和 source 条件同时变化，无法归因。

---

# 十、代码文件的具体修改方案

## 新增模块

```text
openstl/modules/temporal_unet_modules.py
```

建议包含：

```python
class DWResidualBlock(nn.Module):
    ...

class SharedFrameEncoder(nn.Module):
    ...

class TemporalMixer(nn.Module):
    ...

class TemporalFeaturePyramid(nn.Module):
    ...

class FPNDecoder(nn.Module):
    ...

class UNetMotionHead(nn.Module):
    ...

class UNetFactorizedSourceHead(nn.Module):
    ...
```

并在：

```text
openstl/modules/__init__.py
```

导出。

---

## 新增模型

```text
openstl/models/evolution_temporal_unet_model.py
```

主体接口：

```python
class EvolutionTemporalUNet_Model(nn.Module):

    def __init__(self, configs, **kwargs):
        super().__init__()

        self.backbone = TemporalFeaturePyramid(configs)
        self.decoder = FPNDecoder(configs)
        self.motion_head = UNetMotionHead(configs)
        self.operator = EvolutionOperator(...)

        self.use_source = ...
        if self.use_source:
            self.source_head = UNetFactorizedSourceHead(configs)

    def encode_history(self, history):
        pyramid = self.backbone(history)
        decoded = self.decoder(pyramid)

        return {
            "motion_feature": decoded["coarse"],
            "source_feature": decoded["fine"],
            "pyramid": decoded,
        }

    def forward(
        self,
        history,
        return_aux=False,
        teacher_forcing=None,
    ):
        features = self.encode_history(history)
        flow = self.predict_flow(features)

        if not self.use_source:
            result = self.operator(
                history[:, -1],
                flow,
                source=None,
            )
        else:
            result = self.factorized_rollout(
                history,
                features["source_feature"],
                flow,
                teacher_forcing,
            )

        return result if return_aux else result["prediction"]
```

在：

```text
openstl/models/__init__.py
```

新增：

```python
from .evolution_temporal_unet_model import EvolutionTemporalUNet_Model
```

---

## 新增 Method

```text
openstl/methods/evolution_temporal_unet.py
```

短期可以：

```python
class EvolutionTemporalUNet(EvolutionConvLSTM):
    ...
```

复用现有：

* `_build_physical_region_masks`；
* `_build_regime_labels`；
* `_balanced_regime_loss`；
* `_factorized_source_terms`；
* validation/test评估；
* source诊断。

只覆盖：

```python
_build_model()
on_train_epoch_start()
configure_optimizers()
```

因为当前 `EvolutionConvLSTM` 的冻结和 optimizer 逻辑明确写死了：

```python
self.model.cell_list
```

而新模型应改成：

```python
self.model.backbone
self.model.decoder
self.model.motion_head
self.model.source_head
```

当前 method 的训练物理逻辑主要通过标准输出字典与 `self.model.operator` 工作，因此可以被新模型复用；真正绑定 `cell_list` 的主要是冻结和优化器配置部分。

长期等新模型通过后，再抽出：

```text
openstl/methods/evolution_physics_base.py
```

让：

```text
EvolutionConvLSTM
EvolutionTemporalUNet
```

共同继承。不要在新模型尚未验证前就大规模重构当前成功代码。

---

## 注册 Method

当前 `method_maps` 已经集中管理算法。

新增：

```python
from .evolution_temporal_unet import EvolutionTemporalUNet

method_maps = {
    ...
    "evolutiontemporalunet": EvolutionTemporalUNet,
    "evolution_temporal_unet": EvolutionTemporalUNet,
}
```

---

# 十一、配置文件

建议新增，不覆盖任何旧配置：

```text
configs/bth_radar/
├── TemporalUNet_evolution_motion_p0.py
├── TemporalUNet_evolution_motion_p1.py
├── TemporalUNet_evolution_factorized_s1.py
├── TemporalUNet_evolution_factorized_s3.py
└── TemporalUNet_evolution_factorized_s20.py
```

P0 建议：

```python
method = "EvolutionTemporalUNet"

temporal_unet_channels = [32, 64, 128, 192]
temporal_unet_blocks = [2, 2, 2, 2]
temporal_unet_temporal_kernel = 3
temporal_unet_use_tau = False

evolution_use_source = False
evolution_max_displacement = 1.0
evolution_forecast_steps = 20
evolution_field_space = "rain_rate"

batch_size = 4
val_batch_size = 4
lr = 2e-4
epoch = 5
```

先用 batch 4 做显存验证，再判断能否升到 8。

---

# 十二、checkpoint迁移问题

当前 R4-b checkpoint 不能直接加载进 Temporal U-Net。

现有 `load_pretrained_motion()` 明确要求 checkpoint 中存在：

```text
cell_list.*
motion_head.*
```

而新模型没有 `cell_list`，motion head 的输入通道也不同。

因此：

## 可以复用

* 数据；
* operator；
* loss；
  -机制标签；
  -评估；
  -训练协议；
* Gate；
* R4-b结果作为基准。

## 不能直接复用

* ConvLSTM encoder权重；
* ConvLSTM motion head权重；
* optimizer state。

Temporal U-Net motion-only 应从头训练。

后续可以考虑用 R4-b flow 作为软教师：

[
L_{\text{distill}}
==================

|v_{\text{UNet}}-v_{\text{R4b}}|_1
]

但不能在第一轮使用，否则无法判断新主干本身的能力。

---

# 十三、测试文件

新增：

```text
tests/test_models/test_evolution_temporal_unet.py
tests/test_methods/test_evolution_temporal_unet.py
tests/test_configs/test_bth_temporal_unet_config.py
```

最低测试：

```text
test_input_output_shape_66x70
test_multiscale_feature_shapes
test_odd_size_decoder_alignment
test_zero_motion_matches_persistence
test_motion_head_has_finite_gradients
test_motion_output_respects_bound
test_source_free_operator_matches_existing_operator
test_factorized_source_is_bounded
test_factorized_zero_initialization_matches_motion_only
test_checkpoint_round_trip
test_existing_evolution_convlstm_unchanged
```

最后一项很重要：新增主干不能破坏当前 ConvLSTM 的位级或指标兼容性。

---

# 十四、推荐参考的开源代码

## 1. OpenSTL：主要工程参考

这是最重要的参考。

它本身将算法拆成：

* `modules`；
* `models`；
* `methods`；
* `api`；
* `configs`。

官方也明确将这种模块化设计作为方便添加新网络和训练策略的核心能力，并采用 Apache-2.0 许可证。([GitHub][1])

主要参考：

```text
openstl/models/simvp_model.py
openstl/modules/simvp_modules.py
openstl/methods/simvp.py
openstl/methods/tau.py
```

用途：

* frame-shared encoder写法；
  -时序维组织；
* ConvNeXt/TAU block；
* method注册；
  -训练接口。

---

## 2. segmentation_models.pytorch

这个仓库包含：

* U-Net；
* U-Net++；
* FPN；
* UPerNet；
* DeepLab；
* SegFormer等。

主要参考它的：

```text
encoder返回多尺度feature列表
decoder按尺度恢复
FPN lateral projection
任意输入分辨率处理
```

它的当前版本支持多种 encoder-decoder 架构，项目主体为 MIT，但部分文件有独立许可声明。([GitHub][2])

建议：

* 阅读结构；
  -借鉴接口；
  -不要直接把它安装成核心依赖；
  -不要直接使用 ImageNet encoder。

你们只需要三级或四级轻量 U-Net，自己实现更可控。

---

## 3. SmaAt-UNet

SmaAt-UNet 是专门面向降水临近预报的轻量 U-Net，使用：

* depthwise-separable convolution；
* attention module；
* 小参数量设计。

官方代码公开，论文报告其在保持预测性能的同时显著压缩参数量。([Awesome Ecosystem][3])

适合参考：

```text
轻量卷积块
注意力门控skip
小模型通道配置
```

不建议直接照搬：

* 它主要是直接预测未来场；
  -没有显式运动—源汇；
  -时间建模不是你们需要的核心；
  -需要自行核查具体文件许可证后再复制代码。

---

## 4. RainNet

RainNet 是 U-Net/SegNet 风格的雷达降水临近预报模型，代码公开且采用 MIT 许可证。([GMD][4])

适合参考：

* 雷达场进入 U-Net 的方式；
  -输出层；
  -递归短时预报流程；
  -降水任务的卷积结构。

但 RainNet 同时报告了明显空间平滑，这是你们不能采用“U-Net直接MSE回归未来场”的重要反例。([GMD][5])

所以它用于参考空间骨架，不用于复制最终预测方式。

---

## 5. WF-UNet

WF-UNet 是基于 3D U-Net 的多时效降水预测开源实现。([GitHub][6])

适合参考：

* `[B,C,T,H,W]` 张量处理；
* 3D卷积时间融合；
  -多时效输出；
* precipitation-specific训练脚本。

不建议直接采用完整 3D U-Net，因为：

* 8 GB显存压力较大；
  -逐步物理递推不自然；
  -你们输入时间只有10帧；
  -新主干主要应增强历史编码，而不是直接生成20帧。

但它的 temporal convolution 可以作为 `TemporalMixer` 的实现参考。

---

## 6. RainAI

RainAI 的开源代码使用 2D U-Net 进行降水预测，并研究了：

* lead-time conditioning；
* 2D U-Net与3D U-Net效率比较；
  -分类损失与MSE；
  -不同上采样方式。([GitHub][7])

它最适合参考后续的：

```text
lead embedding
不同未来时效条件化
强降水分类监督
```

这些不要在 P0 motion-only 阶段引入。

---

## 7. ConvNeXt官方代码

ConvNeXt官方仓库采用 MIT许可证，但已归档。([GitHub][8])

它适合核对：

* depthwise convolution；
* LayerNorm；
  -通道扩张；
  -残差缩放。

不过你们 OpenSTL 已经包含 `ConvNeXtSubBlock`，没有必要再复制官方整网。

---

# 十五、推荐的实施顺序

## P0：结构验证

```text
Temporal U-Net
→ 20步flow
→ EvolutionOperator
```

完成：

* shape测试；
* odd-size恢复；
  -显存测试；
  -单batch过拟合；
* zero-flow兼容。

不跑完整训练。

## P1：正式motion-only

与 R4-b 同协议比较：

```text
EvolutionConvLSTM motion-only
vs
EvolutionTemporalUNet motion-only
```

Gate建议：

* 第一小时 CSI@16/32 下降不超过0.02；
* FSS 3×3/5×5不明显下降；
  -质心误差不恶化超过10%；
* teacher-forced transport MAE不高于R4-b；
  -不通过面积膨胀制造CSI；
  -训练显存可控。

## P2：单步factorized source

复用当前：

* interior mask；
  -三分类标签；
  -平衡CE；
  -幅度监督；
  -状态损失。

只换历史 feature。

## P3：3步递归

检查：

* source累积；
  -强度比；
  -面积比；
  -FAR；
  -growth/decay可辨识性。

## P4：5、10、20步

只有短递归稳定后再推进。

---

# 最终方案

具体来说，不要做：

```text
删除 EvolutionConvLSTM
→ 下载一个U-Net仓库
→ 从零接数据和训练
```

而应做：

```text
weather/OpenSTL
│
├── 原数据与评估                       保留
├── EvolutionOperator                 保留
├── EvolutionConvLSTM                 保留为基线
│
└── EvolutionTemporalUNet             新增
    ├── SharedFrameEncoder
    ├── TemporalMixer
    ├── U-Net/FPN Decoder
    ├── MotionHead
    ├── FactorizedSourceHead
    └── 复用EvolutionOperator
```

最推荐的第一版结构是：

[
\boxed{
\text{4级轻量ConvNeXt式Encoder}
+
\text{显式TemporalConv融合}
+
\text{FPN式Decoder}
+
\text{现有motion/源汇物理协议}
}
]

TAU先作为粗尺度增强消融，而不是第一版的必要组件。这样每次变化都能清楚归因，也最适合交给 Codex 分阶段实现。

[1]: https://github.com/chengtan9907/OpenSTL?utm_source=chatgpt.com "GitHub - chengtan9907/OpenSTL: OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning · GitHub"
[2]: https://github.com/qubvel-org/segmentation_models.pytorch?utm_source=chatgpt.com "GitHub - qubvel-org/segmentation_models.pytorch: Semantic segmentation models with 500+ pretrained convolutional and transformer-based backbones. · GitHub"
[3]: https://awesome.ecosyste.ms/projects/github.com%2Fhansbambel%2Fsmaat-unet?utm_source=chatgpt.com "https://github.com/hansbambel/smaat-unet | Ecosyste.ms: Awesome"
[4]: https://gmd.copernicus.org/articles/13/2631/2020/index.html?utm_source=chatgpt.com "GMD - RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting"
[5]: https://gmd.copernicus.org/articles/13/2631/2020/gmd-13-2631-2020-relations.html?utm_source=chatgpt.com "GMD - Relations - RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting"
[6]: https://github.com/christoskaparakis/WF-UNet?utm_source=chatgpt.com "GitHub - christoskaparakis/WF-UNet · GitHub"
[7]: https://github.com/rafapablos/w4c23-rainai?utm_source=chatgpt.com "GitHub - rafapablos/w4c23-rainai: Weather4Cast 2023 NeurIPS Competition - RainAI · GitHub"
[8]: https://github.com/facebookresearch/ConvNeXt?utm_source=chatgpt.com "GitHub - facebookresearch/ConvNeXt: Code release for ConvNeXt model · GitHub"

**总体可行，而且“新增并行模型”比直接改 `EvolutionConvLSTM` 安全得多。**
它不会天然造成“层层叠叠”，因为 OpenSTL 运行时只会根据 `method_maps` 选择一个方法实例，并不会同时把两个模型套在一起。真正可能导致混乱的不是多一个分支，而是：

1. `EvolutionTemporalUNet` 长期继承 `EvolutionConvLSTM`，形成隐藏耦合；
2. P0、P1、P2、P3 每个实验都新建一套类；
3. 过早同时加入 TemporalConv、TAU、source、lead embedding；
4. 配置和训练逻辑大量复制；
5. Git 分支也按每个小实验无限分叉。

你整理的方案整体可以保留，但需要修正几个关键点。

# 一、我对当前方案的总体评价

| 部分                              | 判断   | 说明                  |
| ------------------------------- | ---- | ------------------- |
| 新增独立 `EvolutionTemporalUNet`    | 正确   | 不破坏现有 R4-b/R4-d     |
| 复用 `EvolutionOperator`          | 正确   | 当前 operator 已经充分独立  |
| 先做 motion-only                  | 正确   | 能隔离主干变化             |
| 共享逐帧 encoder                    | 正确   | 适合固定 10 帧历史         |
| TemporalConv 后再考虑 TAU           | 正确   | 避免一次改变过多            |
| FPN 而非大规模 concat                | 合理   | 更节省显存               |
| 暂时继承 `EvolutionConvLSTM` method | 短期可行 | 长期需要解除继承            |
| 当前 FPN 伪代码                      | 需要修正 | 通道无法直接相加            |
| P1 配置                           | 需要修正 | 和 R4-b 不完全公平        |
| 一次新建 5 个配置                      | 不建议  | 现在只建 P0/P1          |
| 时间注意力定义                         | 需要补全 | 必须明确沿 T 做 softmax   |
| 零初始化梯度测试                        | 需要调整 | 第一轮 backbone 梯度可能为零 |

---

# 二、它会不会让项目结构变得特别混乱

## 模型结构本身不会

理想状态下，最终只新增：

```text
openstl/
├── modules/
│   └── temporal_unet_modules.py
├── models/
│   └── evolution_temporal_unet_model.py
└── methods/
    └── evolution_temporal_unet.py
```

运行时仍然是二选一：

```text
EvolutionConvLSTM
或
EvolutionTemporalUNet
```

`method_maps` 本来就是通过名称选取对应方法，因此新增一个方法符合现有 OpenSTL 的组织方式。

模型内部也是组合，不是层层继承：

```text
EvolutionTemporalUNet_Model
├── backbone
├── temporal_mixers
├── decoder
├── motion_head
├── source_head（P2以后才有）
└── operator
```

这很清楚。

## 真正容易混乱的是 method 继承

你提出：

```python
class EvolutionTemporalUNet(EvolutionConvLSTM):
    ...
```

P0/P1 阶段可以临时这么做，因为现有 `EvolutionConvLSTM` method 已经封装了：

* motion loss；
* teacher-forced transport；
  -验证降水指标；
  -物理区域标签；
  -source机制损失和诊断。

但当前 method 中已经直接引用：

```python
self.model.cell_list
self.model.motion_head
self.model.source_parameters()
```

而 Temporal U-Net 没有 `cell_list`。因此你至少需要覆盖：

```python
_build_model()
on_train_epoch_start()
configure_optimizers()
```

这和你的方案一致。当前这些 ConvLSTM 专用访问确实存在于 method 中。

问题在于，以后如果再加入：

```text
EvolutionTemporalUNetSource
EvolutionTemporalUNetTAU
EvolutionTemporalUNetPWV
```

并继续层层继承，就会变成：

```text
Base_method
  └── EvolutionConvLSTM
       └── EvolutionTemporalUNet
            └── EvolutionTemporalUNetSource
                 └── EvolutionTemporalUNetPWV
```

这个才会真正失控。

## 推荐的处理方式

### P0/P1阶段

允许：

```python
class EvolutionTemporalUNet(EvolutionConvLSTM):
```

但是明确标注为临时复用训练逻辑，只存在这一层继承。

### P1通过之后、P2接source之前

再抽取：

```text
openstl/methods/evolution_physics_base.py
```

结构变为：

```text
Base_method
    └── EvolutionPhysicsBase
          ├── EvolutionConvLSTM
          └── EvolutionTemporalUNet
```

`EvolutionPhysicsBase` 放：

* 物理区域 mask；
* regime label；
* source损失；
  -运动损失；
  -统一 validation；
  -机制诊断。

两个具体 method 只实现：

```python
_build_model()
backbone_parameters()
motion_parameters()
source_parameters()
```

这样不会层层叠叠，而是两个平行实现。

不过不建议现在立刻进行这个重构。先让 P0/P1 跑通，避免“换主干”和“大规模重构训练方法”同时发生。

---

# 三、当前方案中最大的技术问题：FPN通道对不上

你写的是：

```python
D3 = block(F3)

D2 = block(
    lateral2(F2)
    + interpolate(D3, size=F2.shape[-2:])
)
```

但按照前面的通道：

```text
F2: 128 channels
D3: 192 channels
```

两者不能直接相加。

后面的：

```text
F1: 64
D2: 128
```

也同样不能相加。

## 推荐使用统一 FPN 宽度

例如：

```python
fpn_channels = 96
```

所有尺度先横向投影到 96 通道：

```python
P3 = lateral3(F3)  # 192 → 96

P2 = lateral2(F2) + interpolate(P3)  # 128 → 96
P2 = refine2(P2)

P1 = lateral1(F1) + interpolate(P2)  # 64 → 96
P1 = refine1(P1)

P0 = lateral0(F0) + interpolate(P1)  # 32 → 96
P0 = refine0(P0)
```

输出统一为：

```text
P0: [B,96,66,70]
P1: [B,96,33,35]
P2: [B,96,17,18]
P3: [B,96, 9, 9]
```

然后分别投影：

```python
source_feature = source_projection(P0)  # 96 → 32
motion_feature = motion_projection(
    torch.cat([P2, interpolate(P3)], dim=1)
)  # 192 → 128
```

这样更规整，也不会把每一级 decoder 的通道变化隐藏在 `block()` 里。

## 另一种方案

使用逐级上采样投影：

```text
D3: 192
up3: 192→128
D2: 128

up2: 128→64
D1: 64

up1: 64→32
D0: 32
```

这更接近标准 U-Net。

对你们的需求，我更推荐**统一宽度 FPN**，因为 motion 和 source 本身就是两个不同输出头，统一特征维度更方便，也更省事。

---

# 四、TemporalMixer需要进一步明确

你当前写的是：

```python
x = depthwise_conv3d(x)
weights = temporal_attention(x)
out = (x * weights).sum(dim=2)
```

方向正确，但 `temporal_attention` 必须明确输出什么。

推荐：

```python
class TemporalMixer(nn.Module):
    def __init__(self, channels, time_steps=10, kernel_size=3):
        super().__init__()
        self.temporal_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.logit_head = nn.Conv3d(
            channels, 1, kernel_size=1
        )
        self.output_proj = nn.Conv2d(
            channels, channels, kernel_size=1
        )

    def forward(self, x):
        # x: [B,T,C,H,W]
        latest = x[:, -1]

        x = x.permute(0, 2, 1, 3, 4)
        mixed = self.temporal_conv(x)

        logits = self.logit_head(mixed)
        weights = torch.softmax(logits, dim=2)

        fused = (mixed * weights).sum(dim=2)
        return latest + self.output_proj(fused)
```

重点有三个。

## 1. 权重沿时间维归一化

必须是：

```python
torch.softmax(logits, dim=2)
```

否则不同时间帧的贡献没有明确尺度。

## 2. 保留最后一帧残差

```python
return latest + ...
```

这样可以保留最新回波的位置、边缘和强核，同时让时间分支只学习历史修正。

这对你们很重要，因为已有 SimVP 的主要问题之一就是强回波容易被时序压缩和平滑。

## 3. 不要把它叫复杂的“temporal attention”

第一版更准确的名字是：

```text
TemporalWeightedFusion
```

因为它只是时间卷积加权融合，还不是完整的时空自注意力。

---

# 五、是否每个尺度都做TemporalMixer

你现在计划 E0–E3 全部进行时序融合，理论上可行，但第一版略重。

全分辨率 E0：

```text
[B,10,32,66,70]
```

在 batch 4、反向传播时会保存较多中间激活。对于 RTX 4060 Laptop 8 GB，不一定超显存，但会增加 P0 调试成本。

## 更稳妥的P0

```text
E0：只保留最后一帧特征
E1：TemporalMixer
E2：TemporalMixer
E3：TemporalMixer
```

即：

```python
F0 = E0[:, -1]
F1 = temporal_mixer1(E1)
F2 = temporal_mixer2(E2)
F3 = temporal_mixer3(E3)
```

原因是：

* motion主要依赖中低分辨率；
* P0/P1阶段还没有source；
  -高分辨率历史变化对source更重要，对coarse motion不是首要；
  -减少显存和变量数量。

等 P2 接 source 时，再做消融：

```text
+ E0 TemporalMixer
```

判断高分辨率时序信息是否改善 growth/decay，而不是一开始默认加入。

---

# 六、P1配置必须与R4-b保持公平

这是当前方案中另一个需要修正的地方。

你写的是：

```python
evolution_max_displacement = 1.0
evolution_field_space = "rain_rate"
```

但冻结的 R4-b motion 配置是：

```python
evolution_max_displacement = 2.0
evolution_field_space = "normalized_dbz"
evolution_tf_weight = 0.5
evolution_spatial_weight = 1e-3
evolution_temporal_weight = 1e-3
```

并且 R4-b 使用每6分钟最大2像素位移。

因此，如果 P1 同时改为：

* 新主干；
  -最大位移从2改为1；
  -平流空间从normalized dBZ改为rain rate；

最终就无法判断差异来自哪个变化。

## 严格主干对照P1

应固定：

```python
evolution_max_displacement = 2.0
evolution_field_space = "normalized_dbz"
evolution_align_corners = True
evolution_padding_mode = "zeros"
evolution_stop_gradient = False

evolution_tf_weight = 0.5
evolution_spatial_weight = 1e-3
evolution_temporal_weight = 1e-3
```

这样唯一主要变化是：

```text
ConvLSTM encoder
→ Temporal U-Net/FPN encoder
```

## rain-rate motion另做P1b

如果 P1通过，再做：

```text
P1a：normalized_dbz，与R4-b严格比较
P1b：rain_rate，只改变平流变量空间
```

虽然 source 必须工作在 rain-rate 空间，但 motion-only 的主干公平比较不能顺便改变 operator 设置。当前 `EvolutionOperator` 本身已经同时支持 normalized dBZ、linear Z 和 rain rate。

---

# 七、零初始化的一个隐藏问题

你提出最后一层完全零初始化：

```python
nn.init.zeros_(self.motion_head[-1].weight)
nn.init.zeros_(self.motion_head[-1].bias)
```

这是合理的，能保证初始状态是 persistence，也与当前 R4-b motion head 一致。当前 ConvLSTM motion head同样采用零初始化和有界 `tanh`。

但有一个测试细节：

> 最后一层权重为零时，第一次反向传播中，梯度可以到达最后一层，但不会继续传到前面的 motion feature 和 backbone。

因为反向传播到上一层的梯度会乘上零权重。

所以：

```text
test_motion_head_has_finite_gradients
```

第一步应该只要求：

-最后一层梯度有限；

* loss有限；
  -无NaN。

不要要求第一次 backward 后所有 backbone 参数梯度都非零。

如果要检查 backbone，可以：

1. 做一次 optimizer step；
2. 再执行第二次 forward/backward；
3. 检查 backbone 梯度。

或者使用极小权重初始化：

```python
nn.init.normal_(weight, mean=0.0, std=1e-4)
nn.init.zeros_(bias)
```

但为了和 R4-b 一致，我建议继续保持全零，测试改成两步。

---

# 八、source迁移总体正确，但初始化不能只写“zero initialization”

当前 factorized source 并不是所有输出都简单置零，而是初始化为近似 steady：

```text
regime bias = [0, 20, 0]
growth bias = -20
decay bias = -20
```

也就是说：

[
p^{steady}\approx1,\qquad
\alpha^{growth}\approx0,\qquad
\alpha^{decay}\approx0
]

这样新 source 模型初始状态才近似 motion-only。当前 `EvolutionConvLSTM_Model` 已经这样实现。

所以 P2 中应明确：

```python
nn.init.zeros_(regime_head.weight)
regime_head.bias = [0.0, 20.0, 0.0]

nn.init.zeros_(growth_head.weight)
nn.init.constant_(growth_head.bias, -20.0)

nn.init.zeros_(decay_head.weight)
nn.init.constant_(decay_head.bias, -20.0)
```

然后测试：

```text
test_factorized_initialization_approximately_matches_motion_only
```

这里最好写“approximately”，因为 softmax/sigmoid仍然不是数学上的绝对零，只是非常接近。

---

# 九、注册文件判断是正确的

你补充的三个位置确实需要考虑。

## `parser.py`

当前 `--method` 使用固定 `choices`，只包含 `EvolutionConvLSTM`，不加入新方法时，CLI会在建立实验前拒绝参数。

需要增加：

```python
"EvolutionTemporalUNet"
"evolution_temporal_unet"
```

## `methods/__init__.py`

增加：

```python
from .evolution_temporal_unet import EvolutionTemporalUNet

method_maps = {
    ...
    "evolutiontemporalunet": EvolutionTemporalUNet,
    "evolution_temporal_unet": EvolutionTemporalUNet,
}
```

由于 `BaseExperiment` 会先执行：

```python
self.args.method = self.args.method.lower()
```

所以注册小写形式是必要的。

## `api/exp.py`

`display_method_info()` 目前只有这些方法使用 `[B,T,C,H,W]` dummy input：

```python
simvp
tau
mmvp
wast
evolutionconvlstm
evolution_convlstm
```

需要加入：

```python
evolutiontemporalunet
evolution_temporal_unet
```

否则开启方法信息展示时会进入 `Invalid method name`。

因此你提出的：

```text
test_display_method_info_accepts_temporal_unet
```

很有必要。

---

# 十、配置文件不要一次建五个

当前提出：

```text
TemporalUNet_evolution_motion_p0.py
TemporalUNet_evolution_motion_p1.py
TemporalUNet_evolution_factorized_s1.py
TemporalUNet_evolution_factorized_s3.py
TemporalUNet_evolution_factorized_s20.py
```

从规划上没问题，但现在一次创建会让仓库看起来像已经完成了一整条实验线，而且后面的配置很可能在真正实现时变化。

第一轮只建：

```text
configs/bth_radar/
├── TemporalUNet_evolution_motion_smoke.py
└── TemporalUNet_evolution_motion.py
```

对应：

* `smoke`：shape、单batch过拟合、显存；
  -正式 motion：P1完整验证。

P1通过后再新增：

```text
TemporalUNet_evolution_factorized_s1.py
```

s1通过后再新增s3。

不要提前创建s20。

而且延续你们此前的可复现原则，配置最好保持自包含，不依赖容易失效的多层 `_base_` 继承。

---

# 十一、Git分支怎么管理才不会乱

建议区分“模型分支”和“Git分支”。

## Git层面

第一轮只创建一个功能分支：

```text
feature/evolution-temporal-unet-motion
```

不要再建：

```text
feature/PTU-P0
feature/PTU-P1
feature/PTU-FPN
feature/PTU-TemporalConv
```

P0和P1使用同一个Git分支，通过commit区分：

```text
commit 1: add temporal U-Net backbone modules
commit 2: add motion-only model and registration
commit 3: add model/config tests
commit 4: add smoke configuration
commit 5: add formal motion configuration and report template
```

P1通过并合入主线后，再创建：

```text
feature/evolution-temporal-unet-source
```

这样Git历史是按“大机制阶段”组织，不是按每一个小实验无限分叉。

## 代码层面

只保留一个模型类：

```python
EvolutionTemporalUNet_Model
```

P0、P1、P2、P3不要分别新建模型类，而是由配置控制：

```python
evolution_use_source = False
evolution_forecast_steps = 1 / 3 / 20
temporal_unet_use_tau = False / True
```

也就是说：

> 实验变化放配置，结构变化才放代码。

这是避免混乱最重要的原则。

---

# 十二、建议精简后的文件结构

第一阶段完成后应当只有：

```text
openstl/
├── modules/
│   ├── temporal_unet_modules.py
│   └── __init__.py
├── models/
│   ├── evolution_temporal_unet_model.py
│   └── __init__.py
├── methods/
│   ├── evolution_temporal_unet.py
│   └── __init__.py
├── api/
│   └── exp.py
└── utils/
    └── parser.py

configs/bth_radar/
├── TemporalUNet_evolution_motion_smoke.py
└── TemporalUNet_evolution_motion.py

tests/
├── test_models/test_evolution_temporal_unet.py
├── test_methods/test_evolution_temporal_unet.py
└── test_configs/test_bth_temporal_unet_config.py
```

不要第一阶段就加入：

```text
UNetFactorizedSourceHead
TAU配置
source配置
s3配置
s20配置
PWV
DEM
distillation
GAN
```

这样新增内容其实非常有限。

---

# 十三、建议修订后的P0架构

```text
History [B,10,1,66,70]
        │
        ▼
SharedFrameEncoder
        │
 ┌──────┼────────┬────────┐
 ▼      ▼        ▼        ▼
E0      E1       E2       E3
 │      │        │        │
last  Temporal Temporal Temporal
frame  Mixer    Mixer    Mixer
 │      │        │        │
 F0     F1       F2       F3
 └──────┴────────┴────────┘
        │
        ▼
统一宽度 FPN（96 channels）
        │
        ├── P0 fine
        ├── P1
        ├── P2 coarse
        └── P3 bottleneck
                │
                ▼
        Motion projection
         192 → 128
                │
                ▼
           MotionHead
                │
                ▼
       [B,20,2,66,70]
                │
                ▼
      Existing EvolutionOperator
```

建议首版参数：

```python
temporal_unet_channels = [32, 64, 128, 192]
temporal_unet_blocks = [2, 2, 2, 2]
temporal_unet_fpn_channels = 96

temporal_unet_mix_scales = [1, 2, 3]
temporal_unet_temporal_kernel = 3
temporal_unet_use_tau = False

evolution_use_source = False
evolution_max_displacement = 2.0
evolution_field_space = "normalized_dbz"
```

---

# 十四、P1比较Gate还应补充一条

你现在的Gate基本合理，但“CSI下降不超过0.02”只能作为初筛。

还应该加入：

```text
预测雨区面积比和Frequency Bias不得明显膨胀
```

因为新 U-Net/FPN 可能通过更平滑、更宽的流场扩大雨区，从而获得局部 CSI/FSS 改善。

完整P1 Gate建议是：

| 维度               | Gate                  |
| ---------------- | --------------------- |
| CSI@16/32        | 第一小时相对R4-b下降不超过0.02   |
| FSS 3×3/5×5      | 不显著下降                 |
| 质心误差             | 不恶化超过10%              |
| TF transport MAE | 不高于R4-b               |
| Frequency Bias   | 不因大面积膨胀明显提高           |
| Area Ratio       | 不出现系统性扩张              |
| Flow magnitude   | 不长期饱和在2像素上界           |
| Flow smoothness  | 不产生棋盘或尖峰流场            |
| 工程               | batch 4稳定，尽量支持batch 8 |

此外，P1只使用 Train/Val，不访问 Test，保持当前事件级划分和冻结评估协议。

# 最终判断

这个方案**可行，而且方向正确**。新增并行的 `EvolutionTemporalUNet` 不会天然导致结构混乱；它比直接替换现有 `EvolutionConvLSTM` 更安全，也更有利于公平比较。

但实施前应修改四个核心点：

1. **FPN所有尺度先投影到统一通道，否则当前加法无法执行；**
2. **P1保持R4-b的2像素位移和normalized-dBZ平流，避免主干之外的变量同时变化；**
3. **P0/P1可以临时继承`EvolutionConvLSTM` method，但P2前要抽出平行的物理method基类；**
4. **现在只实现motion-only及两个配置，不提前创建source、TAU和20步配置。**

所以第一轮真正应该交给Codex的范围就是：

```text
新增 Temporal U-Net/FPN backbone
新增 EvolutionTemporalUNet motion-only model/method
完成 parser/method_maps/display_method_info 注册
新增 smoke + formal motion 两个配置
新增单元测试
不修改 EvolutionConvLSTM
不实现 source
不实现 TAU
不运行 Test
```

这个范围足够独立、边界清楚，也不会让仓库变成层层叠叠的实验堆。
