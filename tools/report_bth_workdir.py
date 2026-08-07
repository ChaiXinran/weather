"""Validate a BTH work directory and write a complete Markdown report."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

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

    lines.extend([
        "",
        "## 5. Lead-Time Metrics",
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

    lines.extend([
        "",
        "## 6. All Validation Metrics",
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
        "## 7. Checkpoint Inventory",
        "",
        "| Checkpoint | Size (MiB) |",
        "|---|---:|",
    ])
    for name, size in checkpoint_inventory(work_dir):
        lines.append(f"| `{name}` | {size:.2f} |")

    lines.extend([
        "",
        "## 8. Configuration Snapshot",
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
    report = build_report(work_dir, checkpoint, params, metrics, experiment)
    report_path = Path(options.report_path) if options.report_path else (
        work_dir / "validation_report.md")
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
