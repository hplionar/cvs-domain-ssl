from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarise CVS experiment history.")
    parser.add_argument("history_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.history_path)

    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise ValueError(f"No history found in {path}")

    best_map = max(history, key=lambda row: row["mAP"])
    best_bacc = max(history, key=lambda row: row["mean_bacc"])

    print(f"History file: {path}")
    print()
    print("Best by mAP")
    print(f"  epoch:      {best_map['epoch']}")
    print(f"  mAP:        {best_map['mAP']:.4f}")
    print(f"  mean_auc:   {best_map['mean_auc']:.4f}")
    print(f"  mean_bacc:  {best_map['mean_bacc']:.4f}")
    print(f"  c1_ap:      {best_map['c1_ap']:.4f}")
    print(f"  c2_ap:      {best_map['c2_ap']:.4f}")
    print(f"  c3_ap:      {best_map['c3_ap']:.4f}")
    print()
    print("Best by mean BAcc")
    print(f"  epoch:      {best_bacc['epoch']}")
    print(f"  mAP:        {best_bacc['mAP']:.4f}")
    print(f"  mean_auc:   {best_bacc['mean_auc']:.4f}")
    print(f"  mean_bacc:  {best_bacc['mean_bacc']:.4f}")
    print(f"  c1_bacc:    {best_bacc['c1_bacc']:.4f}")
    print(f"  c2_bacc:    {best_bacc['c2_bacc']:.4f}")
    print(f"  c3_bacc:    {best_bacc['c3_bacc']:.4f}")


if __name__ == "__main__":
    main()
