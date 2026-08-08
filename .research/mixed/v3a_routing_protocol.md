# V3a Error-aware Motion–Decay Routing Protocol

Date: 2026-08-08

Status: draft — routing definition locked; confidence-margin threshold pending.

## 1. Scope

V3a uses a frozen ConvLSTM forecast as the prior and learns three routed
candidates:

\[
R^{preserve}=D_t,
\qquad
R^{motion}=\mathcal{W}(D_t,\Delta U_t),
\qquad
R^{decay}=D_t(1-Q_t^{decay}).
\]

The router predicts soft probabilities

\[
(p^{preserve},p^{motion},p^{decay})=\operatorname{Softmax}(z/\tau)
\]

and produces

\[
\hat R_t=
p^{preserve}R^{preserve}
+p^{motion}R^{motion}
+p^{decay}R^{decay}.
\]

Growth is excluded from V3a. Truth-only initiation that cannot be explained by
a nearby matched storm is ignored rather than forced into motion or decay.

## 2. Routing-truth principle

> 16 mm/h storm-object matching + 32 mm/h core refinement + 2-grid search
> radius + multi-threshold soft routing labels + ambiguous cases ignored.

Routing truth is generated from training targets only. Ground truth masks and
object matches must never be provided as inference inputs.

## 3. Four-level routing construction

### 3.1 Object level: 16 mm/h storm footprint

Detect connected components at 16 mm/h independently in the direct ConvLSTM
forecast and target. These objects define the strong-rain storm footprint.

For a predicted object \(i\) and target object \(j\), form a feasible match if

\[
\operatorname{IoU}_{ij}>\theta_{iou}
\quad\text{or}\quad
d_{c,ij}<r,
\]

subject to

\[
0.5 < A_i/A_j < 2.0.
\]

Rank feasible pairs by

\[
s_{ij}=\lambda_{iou}\operatorname{IoU}_{ij}
+\lambda_d\exp(-d_{c,ij}/r).
\]

Use one-to-one maximum-score matching. A match with insufficient confidence or
competing candidates without a clear score margin is ambiguous and receives
`ignore` supervision.

Initial search radius:

\[
r=2\text{ grid cells}\approx20\text{ km}.
\]

This equals the current maximum residual-motion range of 2 pixels.

Pending before implementation:

- lock \(\theta_{iou}\), \(\lambda_{iou}\), and \(\lambda_d\);
- define the minimum accepted match score;
- define the score-margin threshold for ambiguous multiple matches.

### 3.2 Core level: 32 mm/h refinement

Do not create an independent storm-object identity at 32 mm/h. Within each
matched 16 mm/h footprint, detect and compare 32 mm/h storm cores.

Routing interpretation:

| Condition | Routing supervision |
|---|---|
| 16-footprint and 32-core correctly aligned | preserve |
| 16-footprint matched but 32-core displaced within the feasible range | motion |
| 16-footprint matched but direct 32-core is too strong | decay |
| 16-footprint matched but the 32-core is missing from direct | ignore in V3a |
| Direct has a 16-object with no nearby truth match | decay |
| Truth has a 16-object and a nearby direct object can be matched | motion |
| Truth has a 16-object with no nearby direct object | ignore in V3a |

### 3.3 Pixel level: soft preserve/motion/decay masks

Generate threshold-specific routing targets

\[
y_k^{16},\quad y_k^{32},
\qquad k\in\{preserve,motion,decay\}.
\]

Combine them without a hard threshold priority:

\[
y_k=
\frac{w_{16}y_k^{16}+w_{32}y_k^{32}}
{w_{16}+w_{32}},
\qquad
w_{16}=1.0,\quad w_{32}=1.5.
\]

Thus a correct 16 mm/h footprint with a displaced 32 mm/h core may receive a
soft target such as `preserve=0.4, motion=0.6, decay=0`, rather than allowing
the sparse 32 mm/h label to overwrite the entire storm footprint.

### 3.4 Confidence level: ignore ambiguous cases

Exclude the following pixels/objects from routing classification loss:

- uncertain one-to-many or many-to-one object matches;
- truth-only initiation with no transportable direct object;
- missing 32 mm/h cores that V3a cannot distinguish from growth or intensity
  underprediction;
- boundary regions where object assignment changes under a one-pixel
  perturbation;
- any match rejected by area-ratio, distance, score, or score-margin checks.

Ignored regions may still contribute to the final forecast loss, but they must
not be assigned fabricated motion or decay routing labels.

## 4. Search-radius sensitivity analysis

Evaluate

```text
r = 1 px
r = 2 px
r = 3 px
```

using oracle routing only. Do not train three models. Select the radius using:

- oracle CSI at +60 and +120 min;
- oracle FAR at +60 and +120 min;
- accepted-match and ambiguous-match counts;
- object identity mismatch inspection.

Use `r=2 px` for the first implementation unless this oracle sensitivity check
shows a clearly better and physically credible alternative.

## 5. Required attribution

Report all of the following under the same validation protocol:

```text
Preserve only
Motion candidate
Decay candidate
Oracle routed
Learned routed
```

Interpretation:

- `Oracle routed >> Learned routed`: routing is the primary bottleneck.
- weak candidates and weak oracle: candidate experts are the bottleneck.
- `Oracle routed ≈ Learned routed ≈ Stage 1`: current candidate formulation is
  near its useful ceiling.
- learned routing approaches oracle while +60/+120 CSI rises and FAR falls:
  V3a passes and growth may be studied separately.

## 6. Non-regression and scientific endpoints

The Stage 1 weighted score of 0.937 is an engineering non-regression guard, not
the sole scientific endpoint. Model selection and reporting must additionally
include:

```text
CSI16@+60, CSI32@+60, FAR16@+60, FAR32@+60
CSI16@+120, CSI32@+120, FAR16@+120, FAR32@+120
```

Numerical improvement margins and the allowed non-regression tolerance remain
to be pre-committed in the validation section of the design brief.
