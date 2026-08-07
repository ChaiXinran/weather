"""Generate validation reports and comparison tables for BTH work dirs."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


PERIOD_FIELDS = (
    "val_csi_16_0_1h", "val_csi_32_0_1h",
    "val_csi_16_1_2h", "val_csi_32_1_2h",
    "val_pod_16_0_1h", "val_pod_32_0_1h",
    "val_pod_16_1_2h", "val_pod_32_1_2h",
    "val_far_16_0_1h", "val_far_32_0_1h",
    "val_far_16_1_2h", "val_far_32_1_2h",
)
LEAD_FIELDS = tuple(
    f"val_{name}_{threshold}_lead_{minutes:03d}m"
    for minutes in (60, 120)
    for name in ("csi", "pod", "far")
    for threshold in (16, 32)
)
RAIN_PERIOD_FIELDS = tuple(f"rain_truth_{field}" for field in PERIOD_FIELDS)
RAIN_LEAD_FIELDS = tuple(f"rain_truth_{field}" for field in LEAD_FIELDS)
FIELDS = (("val_csi_score", "rain_truth_val_csi_score")
          + PERIOD_FIELDS + LEAD_FIELDS
          + RAIN_PERIOD_FIELDS + RAIN_LEAD_FIELDS)


def parse_report(path):
    metrics = {}
    in_metrics = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "## 7. All Validation Metrics":
            in_metrics = True
            continue
        if in_metrics and line.startswith("## "):
            break
        if not in_metrics or not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        key = cells[0].strip("`")
        try:
            metrics[key] = float(cells[1])
        except ValueError:
            pass
    return metrics


def fmt(value):
    return "n/a" if value is None else f"{value:.6f}"


def write_tables(output_dir, rows, failures, skipped,
                 include_rain_truth=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bth_model_comparison.csv"
    fields = (("val_csi_score",) + PERIOD_FIELDS + LEAD_FIELDS)
    if include_rain_truth:
        fields += (("rain_truth_val_csi_score",)
                   + RAIN_PERIOD_FIELDS + RAIN_LEAD_FIELDS)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("experiment",) + fields,
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    headings = [
        ("Radar-ZR Period CSI", PERIOD_FIELDS[:4]),
        ("Radar-ZR Period POD", PERIOD_FIELDS[4:8]),
        ("Radar-ZR Period FAR", PERIOD_FIELDS[8:12]),
        ("Radar-ZR Lead-Time CSI/POD/FAR", LEAD_FIELDS),
    ]
    if include_rain_truth:
        headings.extend([
            ("Rain-PNG Period CSI", RAIN_PERIOD_FIELDS[:4]),
            ("Rain-PNG Period POD", RAIN_PERIOD_FIELDS[4:8]),
            ("Rain-PNG Period FAR", RAIN_PERIOD_FIELDS[8:12]),
            ("Rain-PNG Lead-Time CSI/POD/FAR", RAIN_LEAD_FIELDS),
        ])
    lines = [
        "# BTH Model Validation Comparison", "",
        "All values are computed on the Validation split. T+1h and T+2h are "
        "single forecast frames at 60 and 120 minutes.", "",
    ]
    for title, fields in headings:
        lines.extend([
            f"## {title}", "",
            "| Experiment | " + " | ".join(fields) + " |",
            "|---|" + "---:|" * len(fields),
        ])
        for row in rows:
            lines.append(
                f"| {row['experiment']} | "
                + " | ".join(fmt(row.get(field)) for field in fields) + " |")
        lines.append("")
    score_header = (
        "| Experiment | Radar-ZR score | Rain-PNG score |"
        if include_rain_truth else "| Experiment | Radar-ZR score |")
    score_rule = "|---|---:|---:|" if include_rain_truth else "|---|---:|"
    lines.extend(["## Weighted Score", "", score_header, score_rule])
    for row in rows:
        if include_rain_truth:
            line = (
                f"| {row['experiment']} | {fmt(row.get('val_csi_score'))} | "
                f"{fmt(row.get('rain_truth_val_csi_score'))} |")
        else:
            line = f"| {row['experiment']} | {fmt(row.get('val_csi_score'))} |"
        lines.append(line)
    if failures:
        lines.extend(["", "## Failed Experiments", ""])
        lines.extend(f"- `{name}`: {reason}" for name, reason in failures)
    if skipped:
        lines.extend(["", "## Skipped Directories", ""])
        lines.extend(f"- `{name}`: {reason}" for name, reason in skipped)
    (output_dir / "bth_model_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "failures.json").write_text(
        json.dumps(dict(failures), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "scan_inventory.json").write_text(
        json.dumps({
            "evaluated": [row["experiment"] for row in rows],
            "failed": dict(failures),
            "skipped": dict(skipped),
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_empty_workdirs(work_root):
    removed = []
    for path in sorted(work_root.iterdir()):
        checkpoint_dir = path / "checkpoints"
        if (not path.is_dir() or not checkpoint_dir.is_dir()
                or any(checkpoint_dir.glob("*.ckpt"))):
            continue
        shutil.rmtree(path)
        removed.append(path.name)
        print(f"Removed empty checkpoint work dir: {path.name}", flush=True)
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_root", default="work_dirs")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--reuse_reports", action="store_true")
    parser.add_argument("--remove_empty_workdirs", action="store_true")
    parser.add_argument("--compare_rain_truth", action="store_true")
    parser.add_argument("--rain_truth_lag_minutes", type=int, default=42)
    parser.add_argument("--rain_truth_row_shift", type=int, default=0)
    parser.add_argument("--rain_truth_col_shift", type=int, default=1)
    options = parser.parse_args()

    work_root = Path(options.work_root).expanduser().resolve()
    output_dir = (Path(options.output_dir).expanduser().resolve()
                  if options.output_dir else work_root / "evaluation_summary")
    if options.remove_empty_workdirs:
        remove_empty_workdirs(work_root)
    reporter = Path(__file__).with_name("report_bth_workdir.py")
    rows, failures, skipped = [], [], []
    experiments = []
    for path in sorted(item for item in work_root.iterdir() if item.is_dir()):
        if path.resolve() == output_dir.resolve():
            continue
        checkpoint_dir = path / "checkpoints"
        if not checkpoint_dir.is_dir():
            skipped.append((path.name, "no checkpoints directory"))
            continue
        if not any(checkpoint_dir.glob("*.ckpt")):
            skipped.append((path.name, "checkpoints directory is empty"))
            continue
        if not (path / "model_param.json").is_file():
            skipped.append((path.name, "checkpoint exists but model_param.json is missing"))
            continue
        experiments.append(path)

    print(
        f"Scanned {len(experiments) + len(skipped)} directories: "
        f"{len(experiments)} evaluable, {len(skipped)} skipped.", flush=True)

    for index, experiment in enumerate(experiments, 1):
        report = experiment / "validation_report.md"
        print(f"[{index}/{len(experiments)}] {experiment.name}", flush=True)
        if not (options.reuse_reports and report.is_file()):
            command = [
                sys.executable, str(reporter), "--work_dir", str(experiment),
                "--batch_size", str(options.batch_size),
                "--val_batch_size", str(options.val_batch_size),
            ]
            if options.data_root:
                command.extend(["--data_root", options.data_root])
            if options.num_workers is not None:
                command.extend(["--num_workers", str(options.num_workers)])
            if options.compare_rain_truth:
                command.extend([
                    "--compare_rain_truth",
                    "--rain_truth_lag_minutes",
                    str(options.rain_truth_lag_minutes),
                    "--rain_truth_row_shift", str(options.rain_truth_row_shift),
                    "--rain_truth_col_shift", str(options.rain_truth_col_shift),
                ])
            result = subprocess.run(command, text=True)
            if result.returncode != 0:
                failures.append((experiment.name, f"report command exited {result.returncode}"))
                continue
        try:
            metrics = parse_report(report)
            rows.append({"experiment": experiment.name, **{
                field: metrics.get(field) for field in FIELDS}})
        except Exception as error:
            failures.append((experiment.name, str(error)))
        write_tables(
            output_dir, rows, failures, skipped,
            include_rain_truth=options.compare_rain_truth)

    if not experiments:
        write_tables(
            output_dir, rows, failures, skipped,
            include_rain_truth=options.compare_rain_truth)

    print(f"Summary written to {output_dir}")


if __name__ == "__main__":
    main()
