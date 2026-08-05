# R4-b source-free validation baseline

## Protocol

- Checkpoint: `val-csi-epoch=01-val_csi_score=0.640662.ckpt`
- Split: event-level validation only
- Samples: 932
- Input/output: 10 history frames / 20 rollout frames
- Interval: 6 minutes
- Field: rain rate, frozen `Z=200R^1.6`
- Source: disabled
- Thresholds: 0.1, 2.5, 8, 16, 32 mm/h
- FSS windows: 1x1, 3x3, 5x5

The checkpoint-name score `0.640662` is the training validation aggregate.
It is not a single-threshold CSI and must not be compared directly with the
16 or 32 mm/h CSI values below.

## Six-minute baseline

| Threshold | CSI | POD | FAR | Frequency bias |
|---:|---:|---:|---:|---:|
| 16 mm/h | 0.6673 | 0.7787 | 0.1765 | 0.9456 |
| 32 mm/h | 0.5803 | 0.6935 | 0.2194 | 0.8884 |

## Twenty-step aggregate

- MAE: 0.3264 mm/h
- RMSE: 1.9336 mm/h
- Mean error: -0.0839 mm/h
- Intensity ratio: 0.8151

| Threshold | CSI | POD | FAR | Frequency bias |
|---:|---:|---:|---:|---:|
| 0.1 mm/h | 0.6240 | 0.7455 | 0.2071 | 0.9402 |
| 2.5 mm/h | 0.4455 | 0.5827 | 0.3457 | 0.8905 |
| 8 mm/h | 0.3184 | 0.4355 | 0.4578 | 0.8032 |
| 16 mm/h | 0.2184 | 0.2982 | 0.5507 | 0.6637 |
| 32 mm/h | 0.1057 | 0.1187 | 0.5104 | 0.2425 |

## Periods

| Period | Threshold | CSI | POD | FAR | Frequency bias |
|---|---:|---:|---:|---:|---:|
| 0--1 h | 16 | 0.3221 | 0.4385 | 0.4519 | 0.8001 |
| 0--1 h | 32 | 0.1878 | 0.2226 | 0.4540 | 0.4077 |
| 1--2 h | 16 | 0.1123 | 0.1538 | 0.7061 | 0.5234 |
| 1--2 h | 32 | 0.0092 | 0.0098 | 0.8587 | 0.0692 |

First-hour MAE/RMSE/intensity ratio are 0.2681/1.6650/0.8915. Second-hour
values are 0.3848/2.1692/0.7376. The baseline therefore loses intensity and
strong-echo coverage with lead time; a source model must improve POD and bias
without reproducing the failed c2 false-alarm inflation.

## FSS

Mean FSS over validation sample-lead pairs:

| Threshold | FSS 1x1 | FSS 3x3 | FSS 5x5 |
|---:|---:|---:|---:|
| 16 mm/h | 0.2236 | 0.3233 | 0.3591 |
| 32 mm/h | 0.0994 | 0.1465 | 0.1643 |

All full-resolution outputs, per-lead/per-event tables, confusion counts,
object metrics, bootstrap intervals and figures are stored in
`precipitation_evaluation/` beside this report.
