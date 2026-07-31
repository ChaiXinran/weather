# ConvLSTM R2d fine-tuning checkpoint evaluation

## Protocol and checkpoints

All results below use the frozen 2025 validation split (4 independent events,
932 windows), 10 -> 20 prediction, frozen Z-R conversion, and the same full
event/spatial evaluator used for the selected SimVP R3 checkpoint.

- ConvLSTM 0.788: `bth_convlstm_r2d_ft3ep_seed0`, cumulative epoch 8,
  `val-csi-epoch=02-val_csi_score=0.788316.ckpt`.
- ConvLSTM 0.775: `bth_convlstm_r2d_ft6ep_seed0`, cumulative epoch 11,
  `val-csi-epoch=02-val_csi_score=0.775037.ckpt`.
- SimVP R3 reference: `selected_csi_0.714247_epoch03.ckpt`.

The values 0.788316 and 0.775037 are full-validation checkpoint scores, not
sanity-check estimates.

## Period metrics

| Model | Period | CSI16 | CSI32 | FAR16 | FAR32 | Bias16 | Bias32 | Intensity ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SimVP R3 | 0-1 h | 0.3122 | 0.2258 | 0.5832 | 0.6852 | 1.3306 | 1.4104 | 1.1841 |
| ConvLSTM 0.788 | 0-1 h | **0.3347** | 0.2350 | 0.4768 | 0.6754 | 0.9205 | 1.4170 | 0.9075 |
| ConvLSTM 0.775 | 0-1 h | 0.3309 | **0.2369** | **0.4479** | **0.6567** | 0.8194 | **1.2616** | 0.8626 |
| SimVP R3 | 1-2 h | 0.1084 | 0.0339 | **0.7892** | **0.9052** | 0.8658 | **0.5284** | 1.0193 |
| ConvLSTM 0.788 | 1-2 h | **0.1098** | **0.0544** | 0.8213 | 0.9305 | 1.2400 | 2.8820 | 0.9580 |
| ConvLSTM 0.775 | 1-2 h | 0.1015 | 0.0529 | 0.8152 | 0.9287 | **0.9945** | 2.3869 | 0.8305 |

ConvLSTM's score gain comes mainly from the first hour and from recovering more
32 mm/h pixels in the second hour. The latter is not a clean localization gain:
second-hour Bias32 rises from 0.53 for SimVP to 2.88/2.39 and FAR32 remains above
0.92.

## Overall and event robustness

| Model | MAE | RMSE | Intensity ratio | CSI16 | CSI32 | Event-macro CSI16 | Event-macro CSI32 | Worst-event CSI16/32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SimVP R3 | 0.4142 | **2.2798** | 1.1023 | **0.2145** | **0.1457** | **0.1845** | **0.1342** | **0.1047 / 0.0655** |
| ConvLSTM 0.788 | 0.4096 | 2.8028 | 0.9326 | 0.2050 | 0.1191 | 0.1660 | 0.0965 | 0.0824 / 0.0395 |
| ConvLSTM 0.775 | **0.3892** | 2.6793 | 0.8467 | 0.2018 | 0.1216 | 0.1632 | 0.0969 | 0.0906 / 0.0482 |

Neither ConvLSTM checkpoint improves event-macro or worst-event performance.
Event 057 remains the hardest case and is worse than SimVP at both thresholds.
The 0.775 checkpoint has the best MAE, while SimVP retains a substantially lower
RMSE and higher overall/event-macro strong-rain CSI.

## Spatial and object diagnostics

| Model | Threshold | FSS 1x1 | FSS 3x3 | FSS 5x5 | Field centroid error | Matched IoU | Object FAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimVP R3 | 16 | 0.2408 | 0.3547 | 0.4062 | **75.23 km** | 0.3804 | **0.5060** |
| ConvLSTM 0.788 | 16 | **0.2722** | **0.4113** | **0.4778** | 79.43 km | 0.4135 | 0.5457 |
| ConvLSTM 0.775 | 16 | 0.2644 | 0.4022 | 0.4683 | 75.94 km | **0.4210** | 0.5274 |
| SimVP R3 | 32 | 0.1488 | 0.2274 | 0.2622 | **75.63 km** | 0.4181 | **0.6056** |
| ConvLSTM 0.788 | 32 | **0.1935** | 0.2991 | 0.3536 | 87.90 km | 0.4205 | 0.6814 |
| ConvLSTM 0.775 | 32 | 0.1932 | **0.3028** | **0.3607** | 84.05 km | **0.4297** | 0.6705 |

ConvLSTM improves FSS and matched-object IoU, showing better local neighborhood
overlap and shape matching. It does not reduce the approximately 75 km field
centroid error; at 32 mm/h the error increases to 84-88 km. Object FAR is also
higher. Thus sequential recurrence helps local overlap but does not solve the
long-horizon displacement/new-object problem.

## Decision

1. Retain ConvLSTM 0.788 as the maximum validation-score/retrieval candidate.
2. Retain ConvLSTM 0.775 as the more conservative Pareto candidate: lower MAE,
   lower FAR/Bias, slightly higher overall CSI32, better IoU, and lower centroid
   error than ConvLSTM 0.788.
3. Do not replace SimVP R3 unconditionally. SimVP remains stronger on overall
   CSI, event-macro robustness, RMSE, centroid error, and object FAR.
4. Stop extending ConvLSTM training. The baseline answers the structural
   question: changing from channel-mixed SimVP to a sequential recurrent model
   improves first-hour/neighborhood overlap but does not resolve second-hour
   motion and source/sink errors.
5. Proceed to an explicit motion-source formulation; use ConvLSTM 0.775 as the
   preferred recurrent backbone candidate when false alarms and calibration
   matter, and report 0.788 as the score-max checkpoint.

## Artifacts

- `work_dirs/bth_convlstm_eval_valcsi0788316/saved/precipitation_evaluation/`
- `work_dirs/bth_convlstm_eval_valcsi0775037/saved/precipitation_evaluation/`
- `work_dirs/bth_r3_epoch03_val_diagnostics/saved/precipitation_evaluation/`
