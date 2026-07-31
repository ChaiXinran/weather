# BTH Radar V1 five-layer evaluation protocol

This protocol is used for every Radar-only baseline and later physics-informed
model. Changing it requires an explicit protocol version change.

## Units, Z-R relation, and thresholds

Model tensors are decoded from normalized reflectivity to dBZ with:

```text
dBZ = clip(model_value * 50, 0, 50)
```

The primary evaluation space is rain rate derived with one Z-R relation fitted
on training station-Radar pairs and then frozen for validation and test:

```text
Z = a R^b
rain thresholds = 0.1, 2.5, 8, 16, 32 mm/h
```

CSI at 16 and 32 mm/h, separately for leads 1-10 and 11-20, are the headline
categorical scores. The current config uses the universal `Z=200R^1.6` relation
only as a provisional engineering baseline. It must be replaced by a local
training-only fit before paper-level comparisons. The evaluator records the
corresponding dBZ threshold for every rain-rate threshold.

## Deterministic metrics

For every lead time and threshold:

```text
hits          = predicted event and observed event
false alarms  = predicted event and no observed event
misses        = no predicted event and observed event
correct negatives = no predicted event and no observed event

CSI  = hits / (hits + false alarms + misses)
POD  = hits / (hits + misses)
FAR  = false alarms / (hits + false alarms)
Bias = (hits + false alarms) / (hits + misses)
HSS  = Heidke skill score from all four contingency counts
```

MAE, RMSE, mean error, and intensity ratio are computed in physical units for
all valid pixels and separately for observed wet pixels. Overall
categorical metrics are computed from pooled contingency counts, not by
averaging per-frame ratios. A metric with a zero denominator is undefined and
is saved as `null` in JSON, rather than being reported as a perfect score.

Strict 1x1 categorical scores remain primary. FSS at 3x3 and 5x5, centroid
error, area/energy ratios, peak and percentile errors, per-frame connected
objects, and PSD/high-frequency energy are diagnostic spatial-structure
outputs and never replace strict CSI.

## Persistence baseline

The final input Radar frame is repeated for every future lead time. Model and
persistence use identical masks, thresholds, units, and aggregation.

## Event grouping and local calibration

Build the event-first split before fitting or generating sliding windows:

```bash
python tools/prepare_data/build_bth_protocol.py \
  --data-root /path/to/DATA_2025_S \
  --manifest data_manifest/bth_2025_events.json
```

An event is a Radar-active process (by default at least 1% of domain pixels at
or above 20 dBZ); two active periods are independent only when separated by
more than the configured dry gap (three hours by default).
Padding is added and overlaps are merged. Every window is wholly contained in
one event and every event belongs to exactly one split. The definition and its
parameters are stored in the manifest and must be frozen for reported runs.

The supplied `RAIN_2025_S` product has the same 70x66 grid and six-minute
timestamps as Radar. It can therefore calibrate the local Z--R directly:

```bash
python tools/prepare_data/build_bth_protocol.py \
  --data-root /path/to/DATA_2025_S \
  --manifest data_manifest/bth_2025_events.json \
  --fit-rain-png \
  --zr-output data_manifest/local_zr.json
```

The fitter uses every timestamp only once (overlapping model windows do not
duplicate calibration pairs), rejects validation/test events, robustly fits
the positive pixel pairs, and reports dry-pair counts separately. Rain PNG is
a quantized, 35 mm/h-capped gridded calibration product; it supports local
Z--R fitting but must not be described as independent station validation.

### Alignment and zero-aware calibration audit

`tools/prepare_data/calibrate_bth_zr_v2.py` performs the leakage-safe second
stage. Only 2025 train events select alignment or fit parameters. Five-fold
event cross-validation selected `RAIN(t+42 min)` for `Radar(t)` in every fold.
A spatial scan selected `(row=0, col=+1)`, although its gain over no shift was
small. The 2025 six-minute Rain fields were confirmed to be near-linear
interpolations of adjacent hourly anchors.

Two local alternatives were tested: a dry-aware censored power law and a
categorical-threshold fit. Neither beat `Z=200R^1.6` on the untouched 2025
validation events. Marshall--Palmer had the lowest MAE/RMSE and the best CSI
at 0.1, 8, 16, and 32 mm/h. It therefore remains the frozen operational
choice; the local fits are retained as rejected candidates rather than being
silently promoted. The complete decision evidence, including the external
2023 diagnostic, is stored in `.research/local_zr_v2.json`.

The 2023 Rain product has 2,952 unique hourly anchors. Twenty-four duplicate
files are recorded by path in the audit artifact and resolved deterministically
by timestamp without deleting source data. The 2023 results are external-year
diagnostics only and never influence parameter selection.

## Lossless Radar cache

Training configs use `radar_cache_path='RADAR_CACHE_UINT8'`, resolved relative
to `data_root`. The cache stores the original grayscale frames as one read-only
memory-mapped `uint8` NPY array plus a timestamp manifest. Normalization remains
`(255-pixel)/255`, so cached and PNG paths are numerically identical.

Build it once with:

```bash
python tools/cache_bth_radar.py \
  --source /path/to/DATA_2025_S/RADAR_2025_S \
  --output /path/to/DATA_2025_S/RADAR_CACHE_UINT8
```

The PNG files remain the traceable source data. Removing `radar_cache_path`
from a config restores PNG loading.

## Saved artifacts

`saved/precipitation_evaluation/` contains:

- `summary.json` / `metrics.json`: protocol, overall results, lead-time
  results, event macro summaries, spatial/object summaries and grouped
  event/date results, including pooled lead 1-10 and lead 11-20 periods;
- `per_lead_metrics.csv` / `lead_time_metrics.csv`;
- `per_window_metrics.csv`;
- `per_event_metrics.csv` / `event_metrics.csv`;
- `per_object_metrics.csv`;
- `confusion_counts.csv`;
- `bootstrap_ci.json`;
- `lead_time_curves.png`: MAE, RMSE, CSI, POD, FAR, and Bias curves;
- `psd_comparison.png`;
- `cases/cases.json`: ranked success and failure cases relative to persistence;
- `cases/*.png`: truth, model prediction, and persistence at representative
  lead times.

Case ranking first uses CSI improvement at the configured strong-reflectivity
threshold and uses RMSE improvement only as a tie breaker. Dry samples are not
eligible as typical success or failure cases.
