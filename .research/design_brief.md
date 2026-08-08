---
project: "BTH physically interpretable precipitation nowcasting"
last_updated: "2026-08-08"
stage: design
status: draft
source: "project_manifest.yml + researcher clarification"
gap_verdict: ""
placeholder_segments: []
---

# Design brief

## 1. Research question

**Researcher's current formulation**:

> 我们现在最新的架构是纯ConvLSTM+Unet物理机制，基线是csi-score0.937，最终输出是谁产生都无所谓，重要的是效果要好，T+1h时刻，T+2h时刻的csi效果要好，同时far要得到有效的抑制。因为目前基线的贡献主要是纯ConvLSTM，我想让U-net物理机制这一部分发挥更大的作用，最好能够强力修正结果。

**Sharpened RQ** (one sentence, falsifiable):

_TODO: researcher to specify the numerical CSI/FAR improvement required at +60 and +120 min._

**Falsification condition** (what would you observe if FALSE):

_TODO: researcher to specify when a stronger U-Net correction should be judged ineffective rather than merely under-trained._

**Smallest answerable version** (1-week prototype scope):

_TODO: researcher to choose between an error-aware correction diagnostic and a full architecture run._

## 2. Expected mechanism

**Causal chain**:

> ConvLSTM prior → error identification → {motion, decay, growth, preserve}

> 最好不要让 motion/decay/growth 三个 head 无条件同时作用，而应该先判断“这是什么错误”，再路由到对应修正。

The first scoped model is:

> V3a = Error-aware routed Motion + Decay

with Growth deferred until routing, motion, and false-alarm suppression are
shown to work.

**Most uncertain step**:

> 最不可靠的不是 motion，而是 growth。

> 先把 error identification / routing 做对。

Current evidence ordering supplied by the researcher:

> motion > decay > growth

**First step you'd bet breaks**:

> candidate 会做对也会做错，决定“什么时候用”非常关键。再好的 motion candidate，如果在错误区域乱开，也会制造大量新的 FA。

The proposed routed candidates are recorded verbatim as:

> R_preserve = D_t

> R_motion = W(D_t, ΔU_t)

> R_decay = D_t(1-Q_t_decay)

> R_growth = D_t+S_t_growth

> 模型先判断 ConvLSTM 的局地误差类型，再由相应的物理修正专家负责处理。

## 3. Identifiability check

**Discriminating condition**:

> Oracle routed >> Learned routed 说明 candidate 足够好，routing 是瓶颈。

> Oracle routed ≈ Learned routed ≈ Stage1 说明当前 candidate 本身到顶。

Required attribution is `Preserve only / Motion candidate / Decay candidate /
Growth candidate (when introduced) / Oracle routed / Learned routed`.

**Confounders to rule out**:

- A displacement miss must not be mislabeled as growth or decay.
- Nearby but unrelated convective objects must not be treated as one storm.
- A sparse 32 mm/h core must not overwrite an otherwise correct 16 mm/h
  footprint label.
- Ambiguous one-to-many matches must not be converted into hard supervision.
- Routing truth generated from targets must never enter inference inputs.

**Missing-data plan**:

> truth有16对象、direct附近完全无对象 → 当前 V3a ignore，不要强迫 motion/decay解释。

Growth remains outside V3a and is deferred until PWV and/or more suitable
supervision is available.

**Operational routing definition**:

> 16 mm/h storm-object matching + 32 mm/h core refinement + 2-grid search radius + multi-threshold soft routing labels + ambiguous cases ignored.

Use 16 mm/h connected components as storm footprints and 32 mm/h cores only as
within-storm refinement. Set `r=2 pixels ≈20 km`, require an area ratio between
0.5 and 2.0, and combine threshold labels with `w16=1.0, w32=1.5`. Evaluate
`r∈{1,2,3}` with oracle routing only before fixing the training radius.

## 4. Validation plan

**Success metric**:
_TODO_

**Baseline being beaten**:
_TODO_

**Negative control**:
_TODO_

## 5. Risk register

| # | Risk | Early-warning signal | Mitigation |
|---|---|---|---|
| 1 | _TODO_ | _TODO_ | _TODO_ |
| 2 | _TODO_ | _TODO_ | _TODO_ |
| 3 | _TODO_ | _TODO_ | _TODO_ |

## Notes

The final output need not be owned exclusively by either branch. The desired
role split is a stable ConvLSTM temporal prior plus a U-Net physical mechanism
capable of strong, spatially selective corrections.
