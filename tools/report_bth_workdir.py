"""Validate a BTH work directory and write a complete Markdown report."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import torch

from openstl.api import BaseExperiment
from openstl.utils import default_parser


EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+): Lr: (?P<lr>[0-9.eE+-]+) \| "
    r"Train Loss: (?P<train>[0-9.eE+-]+) \| "
    r"Vali Loss: (?P<val>[0-9.eE+-]+) \| "
    r"Wall: (?P<wall>[0-9.eE+-]+)s \| Peak GPU: (?P<gpu>[0-9.eE+-]+) MiB")
CSI_RE = re.compile(
    r"epoch=(?P<epoch>\d+)-val_csi_score=(?P<value>[0-9.]+)\.ckpt")


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def metric(metrics, key):
    value = metrics.get(key)
    return None if value is None else scalar(value)


def fmt(value, digits=6):
    return "n/a" if value is None else f"{value:.{digits}f}"


def lead_metric(metrics, stem, lead_minutes):
    return metric(metrics, f"{stem}_lead_{lead_minutes:03d}m")


def evaluate_rain_truth(experiment, params):
    """Evaluate predictions against aligned RAIN PNG validation targets."""
    loader = experiment.data.valid_loader
    dataset = loader.dataset
    if getattr(dataset, "rain_frames", None) is None:
        return {}
    thresholds = tuple(float(value) for value in params.get(
        "val_precip_thresholds", [16.0, 32.0]))
    lead_count = int(params.get("aft_seq_length", 20))
    shape = (len(thresholds), lead_count)
    hits = np.zeros(shape, dtype=np.int64)
    false_alarms = np.zeros(shape, dtype=np.int64)
    misses = np.zeros(shape, dtype=np.int64)
    method = experiment.method
    method.eval()
    sample_offset = 0
    with torch.inference_mode():
        for batch in loader:
            batch_x, batch_y = batch[:2]
            batch_size = batch_x.shape[0]
            indices = range(sample_offset, sample_offset + batch_size)
            rain_true = dataset.rain_targets(indices) * 35.0
            batch_x = batch_x.to(method.device, non_blocking=True)
            batch_y = batch_y.to(method.device, non_blocking=True)
            rain_pred = method._to_precipitation(
                method(batch_x, batch_y)).detach().cpu().numpy()
            valid = np.isfinite(rain_true) & np.isfinite(rain_pred)
            for threshold_index, threshold in enumerate(thresholds):
                pred_event = (rain_pred >= threshold) & valid
                true_event = (rain_true >= threshold) & valid
                axes = (0, 2, 3, 4)
                hits[threshold_index] += (
                    pred_event & true_event).sum(axis=axes)
                false_alarms[threshold_index] += (
                    pred_event & ~true_event).sum(axis=axes)
                misses[threshold_index] += (
                    ~pred_event & true_event & valid).sum(axis=axes)
            sample_offset += batch_size

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")

    result = {}
    period_width = min(10, lead_count)
    for threshold_index, threshold in enumerate(thresholds):
        label = f"{threshold:g}"
        for lead_index in range(lead_count):
            lead = (lead_index + 1) * int(params.get("lead_minutes", 6))
            h = hits[threshold_index, lead_index]
            fa = false_alarms[threshold_index, lead_index]
            m = misses[threshold_index, lead_index]
            suffix = f"{label}_lead_{lead:03d}m"
            result[f"rain_truth_val_csi_{suffix}"] = ratio(h, h + fa + m)
            result[f"rain_truth_val_pod_{suffix}"] = ratio(h, h + m)
            result[f"rain_truth_val_far_{suffix}"] = ratio(fa, h + fa)
        for period_name, start, end in (
                ("0_1h", 0, period_width),
                ("1_2h", period_width, lead_count)):
            h = hits[threshold_index, start:end].sum()
            fa = false_alarms[threshold_index, start:end].sum()
            m = misses[threshold_index, start:end].sum()
            suffix = f"{label}_{period_name}"
            result[f"rain_truth_val_csi_{suffix}"] = ratio(h, h + fa + m)
            result[f"rain_truth_val_pod_{suffix}"] = ratio(h, h + m)
            result[f"rain_truth_val_far_{suffix}"] = ratio(fa, h + fa)
    if 16.0 in thresholds and 32.0 in thresholds and lead_count > period_width:
        result["rain_truth_val_csi_score"] = (
            result["rain_truth_val_csi_16_0_1h"]
            + result["rain_truth_val_csi_32_0_1h"]
            + result["rain_truth_val_csi_16_1_2h"]
            + 2.0 * result["rain_truth_val_csi_32_1_2h"])
    return result


def find_checkpoint(work_dir, explicit=None):
    if explicit:
        checkpoint = Path(explicit).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    checkpoint_dir = work_dir / "checkpoints"
    alias = checkpoint_dir / "best_val_csi.ckpt"
    if alias.is_file():
        return alias.resolve()
    candidates = []
    for path in checkpoint_dir.glob("val-csi-epoch=*-val_csi_score=*.ckpt"):
        match = CSI_RE.search(path.name)
        if match:
            candidates.append((float(match.group("value")), path))
    if not candidates:
        raise FileNotFoundError(f"No CSI checkpoint found in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[0])[1].resolve()


def training_curve(work_dir):
    rows = {}
    for log_path in sorted(work_dir.glob("train_*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for match in EPOCH_RE.finditer(text):
            epoch = int(match.group("epoch"))
            rows[epoch] = {
                "epoch": epoch,
                "lr": float(match.group("lr")),
                "train": float(match.group("train")),
                "val": float(match.group("val")),
                "wall": float(match.group("wall")),
                "gpu": float(match.group("gpu")),
            }
    for path in (work_dir / "checkpoints").glob(
            "val-csi-epoch=*-val_csi_score=*.ckpt"):
        match = CSI_RE.search(path.name)
        if match:
            epoch = int(match.group("epoch"))
            rows.setdefault(epoch, {"epoch": epoch})["csi"] = float(
                match.group("value"))
    return [rows[key] for key in sorted(rows)]


def checkpoint_inventory(work_dir):
    result = []
    for path in sorted((work_dir / "checkpoints").glob("*.ckpt")):
        result.append((path.name, path.stat().st_size / (1024 * 1024)))
    return result


def build_report(work_dir, checkpoint, params, metrics, experiment):
    model = experiment.method.model
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
    frozen = total - trainable
    score = metric(metrics, "val_csi_score")
    loss = metric(metrics, "val_loss_epoch")
    if loss is None:
        loss = metric(metrics, "val_loss")
    has_rain_truth = "rain_truth_val_csi_score" in metrics
    lines = [
        f"# {work_dir.name} Validation Report",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## 1. Executive Summary",
        "",
        f"- Work directory: `{work_dir}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Method: `{params.get('method')}`",
        f"- Configuration: `{params.get('config_file')}`",
        f"- Weighted validation CSI score: **{fmt(score)}**",
        f"- Validation MSE: **{fmt(loss, 8)}**",
        "- Evaluation split: Validation only; the Test split was not used.",
        "",
        "The weighted score is not a single CSI percentage. It is:",
        "",
        "```text",
        "CSI16(0-1h) + CSI32(0-1h) + CSI16(1-2h) + 2 * CSI32(1-2h)",
        "```",
        "",
        "## 2. Model and Protocol",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Total parameters | {total:,} |",
        f"| Trainable parameters | {trainable:,} |",
        f"| Frozen parameters | {frozen:,} |",
        f"| Batch size | {params.get('batch_size')} |",
        f"| Validation batch size | {params.get('val_batch_size')} |",
        f"| Seed | {params.get('seed')} |",
        f"| Epochs configured | {params.get('epoch')} |",
        f"| Loss type | {params.get('loss_type')} |",
        f"| Input/output frames | {params.get('pre_seq_length')} / {params.get('aft_seq_length')} |",
        "",
        "## 3. Training Curve",
        "",
        "| Epoch | LR | Train loss | Validation loss | Weighted CSI | Wall (s) | Peak GPU (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    curve = training_curve(work_dir)
    if curve:
        for row in curve:
            lines.append(
                f"| {row['epoch']} | {fmt(row.get('lr'), 7)} | "
                f"{fmt(row.get('train'))} | {fmt(row.get('val'))} | "
                f"{fmt(row.get('csi'))} | {fmt(row.get('wall'), 1)} | "
                f"{fmt(row.get('gpu'), 1)} |")
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend([
        "",
        "## 4. Period Metrics",
        "",
        "| Metric | First hour | Second hour |",
        "|---|---:|---:|",
    ])
    for label, stem in (
            ("CSI at 16 mm/h", "val_csi_16"),
            ("CSI at 32 mm/h", "val_csi_32"),
            ("POD at 16 mm/h", "val_pod_16"),
            ("POD at 32 mm/h", "val_pod_32"),
            ("FAR at 16 mm/h", "val_far_16"),
            ("FAR at 32 mm/h", "val_far_32"),
            ("Bias at 16 mm/h", "val_bias_16"),
            ("Bias at 32 mm/h", "val_bias_32"),
            ("Area ratio at 16 mm/h", "val_area_ratio_16"),
            ("Area ratio at 32 mm/h", "val_area_ratio_32")):
        lines.append(
            f"| {label} | {fmt(metric(metrics, stem + '_0_1h'))} | "
            f"{fmt(metric(metrics, stem + '_1_2h'))} |")
    lines.append(
        f"| Intensity ratio | {fmt(metric(metrics, 'val_intensity_ratio_0_1h'))} | "
        f"{fmt(metric(metrics, 'val_intensity_ratio_1_2h'))} |")

    if has_rain_truth:
        lines.extend([
            "",
            "### Rain-PNG Truth Period Metrics",
            "",
            f"Rain-PNG weighted CSI score: **{fmt(metric(metrics, 'rain_truth_val_csi_score'))}**",
            "",
            "| Metric | First hour | Second hour |",
            "|---|---:|---:|",
        ])
        for label, stem in (
                ("CSI at 16 mm/h", "rain_truth_val_csi_16"),
                ("CSI at 32 mm/h", "rain_truth_val_csi_32"),
                ("POD at 16 mm/h", "rain_truth_val_pod_16"),
                ("POD at 32 mm/h", "rain_truth_val_pod_32"),
                ("FAR at 16 mm/h", "rain_truth_val_far_16"),
                ("FAR at 32 mm/h", "rain_truth_val_far_32")):
            lines.append(
                f"| {label} | {fmt(metric(metrics, stem + '_0_1h'))} | "
                f"{fmt(metric(metrics, stem + '_1_2h'))} |")

    lines.extend([
        "",
        "## 5. Key Forecast-Time CSI",
        "",
        "These are single forecast frames, not averages over an hour.",
        "",
        "| Forecast time | CSI at 16 mm/h | CSI at 32 mm/h |",
        "|---:|---:|---:|",
        f"| T+1h (60 min) | {fmt(lead_metric(metrics, 'val_csi_16', 60))} | "
        f"{fmt(lead_metric(metrics, 'val_csi_32', 60))} |",
        f"| T+2h (120 min) | {fmt(lead_metric(metrics, 'val_csi_16', 120))} | "
        f"{fmt(lead_metric(metrics, 'val_csi_32', 120))} |",
    ])
    if has_rain_truth:
        lines.extend([
            "",
            "### Rain-PNG Truth",
            "",
            "| Forecast time | CSI at 16 mm/h | CSI at 32 mm/h |",
            "|---:|---:|---:|",
            f"| T+1h (60 min) | {fmt(metric(metrics, 'rain_truth_val_csi_16_lead_060m'))} | "
            f"{fmt(metric(metrics, 'rain_truth_val_csi_32_lead_060m'))} |",
            f"| T+2h (120 min) | {fmt(metric(metrics, 'rain_truth_val_csi_16_lead_120m'))} | "
            f"{fmt(metric(metrics, 'rain_truth_val_csi_32_lead_120m'))} |",
        ])

    lines.extend([
        "",
        "## 6. Lead-Time Metrics",
        "",
        "| Lead | CSI16 | CSI32 | POD16 | POD32 | FAR16 | FAR32 | Bias16 | Bias32 | Intensity ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lead_minutes = int(params.get("lead_minutes", 6))
    for index in range(int(params.get("aft_seq_length", 20))):
        lead = (index + 1) * lead_minutes
        suffix = f"lead_{lead:03d}m"
        values = [
            metric(metrics, f"val_csi_16_{suffix}"),
            metric(metrics, f"val_csi_32_{suffix}"),
            metric(metrics, f"val_pod_16_{suffix}"),
            metric(metrics, f"val_pod_32_{suffix}"),
            metric(metrics, f"val_far_16_{suffix}"),
            metric(metrics, f"val_far_32_{suffix}"),
            metric(metrics, f"val_bias_16_{suffix}"),
            metric(metrics, f"val_bias_32_{suffix}"),
            metric(metrics, f"val_intensity_ratio_{suffix}"),
        ]
        lines.append(
            f"| {lead} min | " + " | ".join(fmt(value) for value in values) + " |")

    if has_rain_truth:
        lines.extend([
            "",
            "### Rain-PNG Truth Lead-Time Metrics",
            "",
            "| Lead | CSI16 | CSI32 | POD16 | POD32 | FAR16 | FAR32 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for index in range(int(params.get("aft_seq_length", 20))):
            lead = (index + 1) * lead_minutes
            suffix = f"lead_{lead:03d}m"
            values = [
                metric(metrics, f"rain_truth_val_csi_16_{suffix}"),
                metric(metrics, f"rain_truth_val_csi_32_{suffix}"),
                metric(metrics, f"rain_truth_val_pod_16_{suffix}"),
                metric(metrics, f"rain_truth_val_pod_32_{suffix}"),
                metric(metrics, f"rain_truth_val_far_16_{suffix}"),
                metric(metrics, f"rain_truth_val_far_32_{suffix}"),
            ]
            lines.append(
                f"| {lead} min | "
                + " | ".join(fmt(value) for value in values) + " |")

    lines.extend([
        "",
        "## 7. All Validation Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ])
    for key in sorted(metrics):
        try:
            value = scalar(metrics[key])
        except (TypeError, ValueError):
            continue
        lines.append(f"| `{key}` | {fmt(value, 10)} |")

    lines.extend([
        "",
        "## 8. Checkpoint Inventory",
        "",
        "| Checkpoint | Size (MiB) |",
        "|---|---:|",
    ])
    for name, size in checkpoint_inventory(work_dir):
        lines.append(f"| `{name}` | {size:.2f} |")

    lines.extend([
        "",
        "## 9. Configuration Snapshot",
        "",
        "| Key | Value |",
        "|---|---|",
    ])
    for key in sorted(params):
        if key in {"test_mean", "test_std"}:
            continue
        value = params[key]
        if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
            lines.append(f"| `{key}` | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a BTH work directory and generate Markdown.")
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--report_path", default=None)
    parser.add_argument("--ex_name", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--compare_rain_truth", action="store_true")
    parser.add_argument("--rain_truth_lag_minutes", type=int, default=42)
    parser.add_argument("--rain_truth_row_shift", type=int, default=0)
    parser.add_argument("--rain_truth_col_shift", type=int, default=1)
    options = parser.parse_args()

    work_dir = Path(options.work_dir).expanduser().resolve()
    parameter_path = work_dir / "model_param.json"
    if not parameter_path.is_file():
        raise FileNotFoundError(
            f"Missing {parameter_path}; the work directory must come from tools/train.py")
    params = json.loads(parameter_path.read_text(encoding="utf-8"))
    for key, value in default_parser().items():
        params.setdefault(key, value)
        if params[key] is None:
            params[key] = value
    if options.data_root:
        params["data_root"] = options.data_root
    if options.batch_size:
        params["batch_size"] = options.batch_size
    if options.val_batch_size:
        params["val_batch_size"] = options.val_batch_size
    if options.num_workers is not None:
        params["num_workers"] = options.num_workers
    checkpoint = find_checkpoint(work_dir, options.ckpt_path)
    params["ckpt_path"] = str(checkpoint)
    params["init_from_ckpt"] = None
    params["test"] = False
    if options.compare_rain_truth:
        params["evaluation_truth"] = "rain_png"
        params["rain_truth_lag_minutes"] = options.rain_truth_lag_minutes
        params["rain_truth_row_shift"] = options.rain_truth_row_shift
        params["rain_truth_col_shift"] = options.rain_truth_col_shift
    else:
        params["evaluation_truth"] = "radar"
    params["no_display_method_info"] = True
    params["ex_name"] = options.ex_name or f"{work_dir.name}_auto_report_val"

    args = argparse.Namespace(**params)
    experiment = BaseExperiment(args)
    checkpoint_state = torch.load(
        checkpoint, map_location="cpu", weights_only=False)
    experiment.method.load_state_dict(
        checkpoint_state["state_dict"], strict=True)
    results = experiment.trainer.validate(
        experiment.method, experiment.data, ckpt_path=None)
    if not results:
        raise RuntimeError("Validation returned no metrics")
    metrics = results[0]
    if options.compare_rain_truth:
        metrics.update(evaluate_rain_truth(experiment, params))
    report = build_report(work_dir, checkpoint, params, metrics, experiment)
    report_path = Path(options.report_path) if options.report_path else (
        work_dir / "validation_report.md")
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
