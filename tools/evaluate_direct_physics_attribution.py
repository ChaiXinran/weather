"""Branch-wise validation attribution for DirectPhysicsHybrid checkpoints."""

import argparse
import json
from pathlib import Path

import torch

from openstl.api import BaseExperiment
from openstl.utils import default_parser


BRANCHES = (
    "direct", "motion_only", "source_only", "fused")


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def empty_state():
    return {
        period: {
            threshold: {
                "hits": 0, "false_alarms": 0, "misses": 0,
                "pred_area": 0, "true_area": 0,
                "miss_to_hit": 0, "hit_to_miss": 0,
                "fa_to_correct": 0, "correct_to_fa": 0,
            }
            for threshold in (16.0, 32.0)
        }
        for period in ("0_1h", "1_2h")
    }


def update_state(states, predictions, target):
    direct = predictions["direct"]
    periods = (("0_1h", 0, 10), ("1_2h", 10, target.shape[1]))
    for period, start, end in periods:
        truth = target[:, start:end]
        direct_period = direct[:, start:end]
        for threshold in (16.0, 32.0):
            truth_event = truth >= threshold
            direct_event = direct_period >= threshold
            for branch, prediction in predictions.items():
                predicted_event = prediction[:, start:end] >= threshold
                values = states[branch][period][threshold]
                values["hits"] += int((predicted_event & truth_event).sum())
                values["false_alarms"] += int(
                    (predicted_event & ~truth_event).sum())
                values["misses"] += int((~predicted_event & truth_event).sum())
                values["pred_area"] += int(predicted_event.sum())
                values["true_area"] += int(truth_event.sum())
                if branch != "direct":
                    values["miss_to_hit"] += int(
                        (~direct_event & truth_event & predicted_event).sum())
                    values["hit_to_miss"] += int(
                        (direct_event & truth_event & ~predicted_event).sum())
                    values["fa_to_correct"] += int(
                        (direct_event & ~truth_event & ~predicted_event).sum())
                    values["correct_to_fa"] += int(
                        (~direct_event & ~truth_event & predicted_event).sum())


def summarize(states):
    document = {}
    for branch, branch_state in states.items():
        document[branch] = {}
        for period, period_state in branch_state.items():
            document[branch][period] = {}
            for threshold, values in period_state.items():
                hits = values["hits"]
                false_alarms = values["false_alarms"]
                misses = values["misses"]
                row = dict(values)
                row.update({
                    "csi": safe_ratio(hits, hits + false_alarms + misses),
                    "pod": safe_ratio(hits, hits + misses),
                    "far": safe_ratio(false_alarms, hits + false_alarms),
                    "bias": safe_ratio(
                        hits + false_alarms, hits + misses),
                })
                document[branch][period][str(int(threshold))] = row
    return document


def markdown(document, checkpoint):
    lines = [
        "# DirectPhysicsHybrid Branch Attribution",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "Validation only; the Test split was not used.",
        "",
        "## Branch Metrics",
        "",
        "| Branch | Period | Threshold | CSI | POD | FAR | Bias |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        for period in ("0_1h", "1_2h"):
            for threshold in ("16", "32"):
                row = document[branch][period][threshold]
                lines.append(
                    f"| {branch} | {period} | {threshold} | "
                    f"{row['csi']:.6f} | {row['pod']:.6f} | "
                    f"{row['far']:.6f} | {row['bias']:.6f} |")
    lines.extend([
        "",
        "## Event Transitions Relative to Direct",
        "",
        "| Branch | Period | Threshold | Miss->Hit | Hit->Miss | FA->Correct | Correct->FA |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for branch in BRANCHES[1:]:
        for period in ("0_1h", "1_2h"):
            for threshold in ("16", "32"):
                row = document[branch][period][threshold]
                lines.append(
                    f"| {branch} | {period} | {threshold} | "
                    f"{row['miss_to_hit']} | {row['hit_to_miss']} | "
                    f"{row['fa_to_correct']} | {row['correct_to_fa']} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    options = parser.parse_args()

    work_dir = Path(options.work_dir).expanduser().resolve()
    params = json.loads(
        (work_dir / "model_param.json").read_text(encoding="utf-8"))
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
    checkpoint = Path(options.ckpt_path).expanduser().resolve() \
        if options.ckpt_path else (work_dir / "checkpoints" / "best_val_csi.ckpt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    params.update({
        "ckpt_path": None,
        "init_from_ckpt": None,
        "test": False,
        "no_display_method_info": True,
        "ex_name": f"{work_dir.name}_attribution",
    })
    experiment = BaseExperiment(argparse.Namespace(**params))
    state = torch.load(checkpoint, map_location="cpu")
    experiment.method.load_state_dict(state["state_dict"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    method = experiment.method.to(device).eval()
    states = {branch: empty_state() for branch in BRANCHES}

    with torch.no_grad():
        for batch_x, batch_y in experiment.data.valid_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            result = method.model(batch_x, return_aux=True, blend_enabled=True)
            predictions = {
                "direct": method._to_precipitation(
                    result["direct_prediction"]),
                "motion_only": method._to_precipitation(
                    result["motion_fused_prediction"]),
                "source_only": method._to_precipitation(
                    result["source_fused_prediction"]),
                "fused": method._to_precipitation(result["prediction"]),
            }
            target = method._to_precipitation(batch_y)
            update_state(states, predictions, target)

    document = summarize(states)
    output_dir = Path(options.output_dir).expanduser().resolve() \
        if options.output_dir else work_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "branch_attribution.json"
    report_path = output_dir / "branch_attribution.md"
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    report_path.write_text(markdown(document, checkpoint), encoding="utf-8")
    print(f"Attribution JSON written to {json_path}")
    print(f"Attribution report written to {report_path}")


if __name__ == "__main__":
    main()
