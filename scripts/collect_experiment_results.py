from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect best metrics from CVS experiment history files."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/group/pmc085/hlionar/outputs/cvs-domain-ssl",
        help="Root directory containing experiment output folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="metadata/experiment_results_summary.csv",
        help="CSV path to write summary results.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="metadata/experiment_results_summary.md",
        help="Markdown path to write summary results.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def best_row(history: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    valid_rows = [
        row for row in history
        if metric in row and safe_float(row, metric) == safe_float(row, metric)
    ]

    if not valid_rows:
        return {}

    return max(valid_rows, key=lambda row: safe_float(row, metric))


def infer_experiment_info(exp_dir: Path) -> dict[str, Any]:
    config_path = exp_dir / "config.json"

    info = {
        "experiment_id": exp_dir.name,
        "dataset": "",
        "encoder": "",
        "variant": "",
        "loss": "",
        "epochs_config": "",
        "batch_size": "",
        "lr": "",
        "dropout": "",
    }

    if config_path.exists():
        config = load_json(config_path)
        info.update(
            {
                "dataset": config.get("dataset", ""),
                "encoder": config.get("encoder", ""),
                "variant": config.get("variant", ""),
                "loss": "weighted_bce" if config.get("use_pos_weight") else "normal_bce",
                "epochs_config": config.get("epochs", ""),
                "batch_size": config.get("batch_size", ""),
                "lr": config.get("lr", ""),
                "dropout": config.get("dropout", ""),
            }
        )
    else:
        # Fallback for older experiments before config.json was saved.
        name = exp_dir.name.lower()
        info["dataset"] = "sages" if "sages" in name else "endoscapes"
        info["encoder"] = "dinov2" if "dinov2" in name else ""
        info["variant"] = "base" if "_b_" in name or "dinov2_b" in name else ""
        info["loss"] = "weighted_bce" if "weighted" in name or "wbce" in name else "normal_bce"

    return info


def summarise_experiment(exp_dir: Path) -> dict[str, Any] | None:
    history_path = exp_dir / "history.json"
    if not history_path.exists():
        return None

    history = load_json(history_path)
    if not history:
        return None

    info = infer_experiment_info(exp_dir)

    best_map = best_row(history, "mAP")
    best_bacc = best_row(history, "mean_bacc")

    last = history[-1]

    row = {
        **info,
        "history_path": str(history_path),
        "epochs_completed": len(history),

        "best_map_epoch": best_map.get("epoch", ""),
        "best_map": safe_float(best_map, "mAP") if best_map else "",
        "best_map_mean_auc": safe_float(best_map, "mean_auc") if best_map else "",
        "best_map_mean_bacc": safe_float(best_map, "mean_bacc") if best_map else "",
        "best_map_c1_ap": safe_float(best_map, "c1_ap") if best_map else "",
        "best_map_c2_ap": safe_float(best_map, "c2_ap") if best_map else "",
        "best_map_c3_ap": safe_float(best_map, "c3_ap") if best_map else "",

        "best_bacc_epoch": best_bacc.get("epoch", ""),
        "best_bacc_map": safe_float(best_bacc, "mAP") if best_bacc else "",
        "best_bacc_mean_auc": safe_float(best_bacc, "mean_auc") if best_bacc else "",
        "best_bacc_mean_bacc": safe_float(best_bacc, "mean_bacc") if best_bacc else "",
        "best_bacc_c1_bacc": safe_float(best_bacc, "c1_bacc") if best_bacc else "",
        "best_bacc_c2_bacc": safe_float(best_bacc, "c2_bacc") if best_bacc else "",
        "best_bacc_c3_bacc": safe_float(best_bacc, "c3_bacc") if best_bacc else "",

        "last_epoch": last.get("epoch", ""),
        "last_map": safe_float(last, "mAP"),
        "last_mean_bacc": safe_float(last, "mean_bacc"),
    }

    return row


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if value != value:
            return ""
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "experiment_id",
        "dataset",
        "loss",
        "epochs_completed",
        "best_map_epoch",
        "best_map",
        "best_map_mean_auc",
        "best_map_mean_bacc",
        "best_bacc_epoch",
        "best_bacc_mean_bacc",
    ]

    lines = []
    lines.append("# CVS Experiment Results Summary")
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in rows:
        values = [format_value(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    rows = []
    for exp_dir in sorted(output_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        row = summarise_experiment(exp_dir)
        if row is not None:
            rows.append(row)

    rows = sorted(rows, key=lambda row: row["experiment_id"])

    write_csv(rows, output_csv)
    write_markdown(rows, output_md)

    print(f"Found experiments: {len(rows)}")
    print(f"Saved CSV:      {output_csv}")
    print(f"Saved Markdown: {output_md}")
    print()

    for row in rows:
        print(
            f"{row['experiment_id']}: "
            f"best mAP={format_value(row['best_map'])}, "
            f"best mean BAcc={format_value(row['best_bacc_mean_bacc'])}"
        )


if __name__ == "__main__":
    main()
