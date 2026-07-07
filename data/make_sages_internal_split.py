from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic internal video-level train/val/test split for SAGES."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="metadata/sages_frames.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="metadata/sages_frames_internal_split.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="metadata/sages_internal_split_summary.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def assign_group_split(
    video_ids: list[str],
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    video_ids = list(video_ids)
    rng.shuffle(video_ids)

    n = len(video_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Ensure total stays exactly n.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    split_map = {}

    for vid in video_ids[:n_train]:
        split_map[vid] = "train"

    for vid in video_ids[n_train:n_train + n_val]:
        split_map[vid] = "val"

    for vid in video_ids[n_train + n_val:]:
        split_map[vid] = "test"

    return split_map


def summarise(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "videos": int(df["video_id"].nunique()),
        "rows_by_split": {
            split: int(count)
            for split, count in df["internal_split"].value_counts().sort_index().items()
        },
        "videos_by_split": {
            split: int(count)
            for split, count in (
                df.drop_duplicates("video_id")["internal_split"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }

    frame_stats = {}
    for split, part in df.groupby("internal_split"):
        frame_stats[split] = {
            "frames": int(len(part)),
            "videos": int(part["video_id"].nunique()),
            "c1_positive_frames": int(part["c1_consensus"].sum()),
            "c2_positive_frames": int(part["c2_consensus"].sum()),
            "c3_positive_frames": int(part["c3_consensus"].sum()),
            "complete_cvs_positive_frames": int(part["cvs_consensus"].sum()),
            "c1_positive_frame_rate": float(part["c1_consensus"].mean()),
            "c2_positive_frame_rate": float(part["c2_consensus"].mean()),
            "c3_positive_frame_rate": float(part["c3_consensus"].mean()),
            "complete_cvs_positive_frame_rate": float(part["cvs_consensus"].mean()),
        }

    summary["frame_label_distribution"] = frame_stats

    video_df = df.drop_duplicates("video_id")
    video_stats = {}
    for split, part in video_df.groupby("internal_split"):
        video_stats[split] = {
            "videos": int(len(part)),
            "video_c1_positive": int(part["video_c1_consensus"].sum()),
            "video_c2_positive": int(part["video_c2_consensus"].sum()),
            "video_c3_positive": int(part["video_c3_consensus"].sum()),
            "video_complete_cvs_positive": int(part["video_cvs_consensus"].sum()),
            "video_c1_positive_rate": float(part["video_c1_consensus"].mean()),
            "video_c2_positive_rate": float(part["video_c2_consensus"].mean()),
            "video_c3_positive_rate": float(part["video_c3_consensus"].mean()),
            "video_complete_cvs_positive_rate": float(part["video_cvs_consensus"].mean()),
        }

    summary["video_label_distribution"] = video_stats

    leakage_check = df.groupby("video_id")["internal_split"].nunique()
    summary["videos_in_multiple_internal_splits"] = int((leakage_check > 1).sum())

    return summary


def main() -> None:
    args = parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    manifest_path = Path(args.manifest_path)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    df = pd.read_csv(manifest_path)

    required = [
        "video_id",
        "video_c1_consensus",
        "video_c2_consensus",
        "video_c3_consensus",
        "video_cvs_consensus",
        "c1_consensus",
        "c2_consensus",
        "c3_consensus",
        "cvs_consensus",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    video_df = (
        df[
            [
                "video_id",
                "video_c1_consensus",
                "video_c2_consensus",
                "video_c3_consensus",
                "video_cvs_consensus",
            ]
        ]
        .drop_duplicates("video_id")
        .copy()
    )

    # Stratify approximately by video-level CVS label combination.
    video_df["label_combo"] = (
        video_df["video_c1_consensus"].astype(str)
        + video_df["video_c2_consensus"].astype(str)
        + video_df["video_c3_consensus"].astype(str)
    )

    rng = np.random.default_rng(args.seed)

    split_map: dict[str, str] = {}
    for _, group in video_df.groupby("label_combo"):
        group_map = assign_group_split(
            video_ids=group["video_id"].tolist(),
            rng=rng,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        split_map.update(group_map)

    df["internal_split"] = df["video_id"].map(split_map)

    if df["internal_split"].isna().any():
        raise ValueError("Some rows did not receive an internal split.")

    summary = summarise(df)
    summary["seed"] = args.seed
    summary["train_ratio"] = args.train_ratio
    summary["val_ratio"] = args.val_ratio
    summary["test_ratio"] = args.test_ratio
    summary["stratification"] = "video-level label combination: video_c1/video_c2/video_c3"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Saved split manifest: {output_csv}")
    print(f"Saved summary:        {summary_json}")
    print()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
