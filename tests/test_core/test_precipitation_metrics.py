import csv
import json

import numpy as np

from openstl.core.precipitation_metrics import PrecipitationEvaluator


def test_multithreshold_lead_metrics_and_persistence(tmp_path):
    inputs = np.zeros((2, 10, 1, 2, 2), dtype=np.float32)
    inputs[:, -1] = 0.4
    true = np.array([
        [[[[0.2, 0.8], [0.0, 1.0]]],
         [[[0.4, 0.8], [0.0, 1.0]]]],
        [[[[0.0, 0.0], [0.8, 0.8]]],
         [[[0.0, 0.4], [0.8, 1.0]]]],
    ], dtype=np.float32)
    pred = true.copy()
    pred[0, 0, 0, 0, 0] = 0.0

    evaluator = PrecipitationEvaluator(
        lead_count=2, thresholds=(10, 30), case_count=1)
    evaluator.update(
        pred, true, inputs, event_ids=['event-a', 'event-b'],
        sample_ids=[10, 11])
    report = evaluator.save(tmp_path)

    threshold = report['model']['lead_time']['thresholds']['10.0']
    assert np.allclose(report['model']['lead_time']['mae'], [1.25, 0.0])
    assert threshold['hits'][0] == 4
    assert threshold['misses'][0] == 1
    assert threshold['false_alarms'][0] == 0
    assert np.isclose(threshold['csi'][0], 4 / 5)
    assert np.isclose(threshold['pod'][0], 4 / 5)
    assert threshold['far'][0] == 0
    assert np.isclose(threshold['bias'][0], 4 / 5)
    assert threshold['correct_negatives'][0] == 3
    assert np.isfinite(threshold['hss'][0])
    assert set(report['events']) == {'event-a', 'event-b'}

    assert (tmp_path / 'metrics.json').is_file()
    assert (tmp_path / 'lead_time_metrics.csv').is_file()
    assert (tmp_path / 'event_metrics.csv').is_file()
    assert (tmp_path / 'per_window_metrics.csv').is_file()
    assert (tmp_path / 'per_object_metrics.csv').is_file()
    assert (tmp_path / 'bootstrap_ci.json').is_file()
    assert (tmp_path / 'confusion_counts.csv').is_file()
    assert (tmp_path / 'psd_comparison.png').is_file()
    assert (tmp_path / 'lead_time_curves.png').is_file()
    assert (tmp_path / 'cases' / 'cases.json').is_file()

    with (tmp_path / 'lead_time_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 2 * 2
    with (tmp_path / 'metrics.json').open() as stream:
        saved = json.load(stream)
    assert saved['protocol']['undefined_policy'].startswith('null')


def test_undefined_categorical_metrics_are_nan_in_memory():
    values = np.zeros((1, 1, 1, 2, 2), dtype=np.float32)
    evaluator = PrecipitationEvaluator(lead_count=1, thresholds=(40,))
    evaluator.update(values, values, values, event_ids=['dry'])
    metrics = evaluator.report()['model']['overall']['thresholds']['40.0']
    assert np.isnan(metrics['csi'])
    assert np.isnan(metrics['pod'])
    assert np.isnan(metrics['far'])
    assert np.isnan(metrics['bias'])


def test_frozen_zr_rain_thresholds_and_five_layer_outputs(tmp_path):
    # 30 dBZ -> Z=1000 -> R=(1000/200)^(1/1.6).
    normalized_dbz = 30.0 / 50.0
    inputs = np.full(
        (1, 2, 1, 4, 4), normalized_dbz, dtype=np.float32)
    true = np.full(
        (1, 1, 1, 4, 4), normalized_dbz, dtype=np.float32)
    pred = true.copy()
    evaluator = PrecipitationEvaluator(
        lead_count=1,
        thresholds=(0.1, 2.5),
        value_unit='mm/h',
        convert_dbz_to_rain=True,
        zr_a=200.0,
        zr_b=1.6,
        case_threshold=2.5,
        bootstrap_repetitions=20)
    evaluator.update(
        pred, true, inputs, event_ids=['event-a'], sample_ids=[1])
    report = evaluator.save(tmp_path)

    expected_rain = (1000.0 / 200.0) ** (1.0 / 1.6)
    assert np.isclose(
        report['model']['overall']['intensity_ratio'], 1.0)
    assert np.isclose(report['model']['overall']['mae'], 0.0)
    assert expected_rain > 2.5
    assert np.isclose(
        report['protocol']['frozen_zr']['threshold_dbz']['2.5'],
        10 * np.log10(200.0) + 16 * np.log10(2.5))
    assert np.isnan(
        report['model']['overall']['thresholds']['2.5']['hss'])
    spatial = report['spatial_object_summary']['model']['2.5']
    assert np.isclose(spatial['area_ratio']['mean'], 1.0)
    assert np.isclose(spatial['energy_ratio']['mean'], 1.0)
