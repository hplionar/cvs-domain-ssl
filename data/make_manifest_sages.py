from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


FRAME_RATER_COLUMNS = [
    "c1_rater1", "c1_rater2", "c1_rater3",
    "c2_rater1", "c2_rater2", "c2_rater3",
    "c3_rater1", "c3_rater2", "c3_rater3",
]

VIDEO_RATER_COLUMNS = FRAME_RATER_COLUMNS.copy()

CONFIDENCE_COLUMNS = [
    "confidence_rater1",
    "confidence_rater2",
    "confidence_rater3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a labelled-frame manifest for the SAGES CVS Challenge dataset."
    )

    parser.add_argument(
        "--sages-root",
        type=str,
        required=True,
        help="Path to SAGES_CVS_Challenge_2024 root.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="metadata/sages_frames.csv",
        help="Output manifest CSV path.",
    )

    parser.add_argument(
        "--summary-json",
        type=str,
        default="metadata/sages_manifest_summary.json",
        help="Output summary JSON path.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train"],
        help="Official SAGES split to process. Only train has labels.",
    )

    return parser.parse_args()


def majority_vote(row: pd.Series, columns: list[str]) -> int:
    return int(row[columns].astype(int).sum() >= 2)


def vote_count(row: pd.Series, columns: list[str]) -> int:
    return int(row[columns].astype(int).sum())


def validate_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_frame_map(frame_dir: Path) -> pd.DataFrame:
    frame_map_path = frame_dir / "frame_map.csv"

    if not frame_map_path.exists():
        raise FileNotFoundError(f"Missing frame_map.csv: {frame_map_path}")

    frame_map = pd.read_csv(frame_map_path)

    required = [
        "extracted_frame_id",
        "file_name",
        "source_frame_id",
        "timestamp_sec",
    ]
    validate_columns(frame_map, required, frame_map_path)

    frame_map["source_frame_id"] = frame_map["source_frame_id"].astype(int)
    frame_map["extracted_frame_id"] = frame_map["extracted_frame_id"].astype(int)

    return frame_map


def read_frame_labels(label_dir: Path) -> pd.DataFrame:
    frame_csv = label_dir / "frame.csv"

    if not frame_csv.exists():
        raise FileNotFoundError(f"Missing frame.csv: {frame_csv}")

    frame_labels = pd.read_csv(frame_csv)

    required = ["frame_id", *FRAME_RATER_COLUMNS]
    validate_columns(frame_labels, required, frame_csv)

    frame_labels["frame_id"] = frame_labels["frame_id"].astype(int)

    for col in FRAME_RATER_COLUMNS:
        frame_labels[col] = frame_labels[col].astype(int)

    return frame_labels


def read_video_labels(label_dir: Path) -> dict[str, Any]:
    video_csv = label_dir / "video.csv"

    if not video_csv.exists():
        raise FileNotFoundError(f"Missing video.csv: {video_csv}")

    video_df = pd.read_csv(video_csv)

    required = [*VIDEO_RATER_COLUMNS, *CONFIDENCE_COLUMNS]
    validate_columns(video_df, required, video_csv)

    if len(video_df) != 1:
        raise ValueError(f"Expected one row in {video_csv}, found {len(video_df)}")

    row = video_df.iloc[0]

    result: dict[str, Any] = {}

    for col in VIDEO_RATER_COLUMNS:
        result[f"video_{col}"] = int(row[col])

    for col in CONFIDENCE_COLUMNS:
        result[col] = float(row[col])

    result["video_c1_votes"] = int(row[["c1_rater1", "c1_rater2", "c1_rater3"]].astype(int).sum())
    result["video_c2_votes"] = int(row[["c2_rater1", "c2_rater2", "c2_rater3"]].astype(int).sum())
    result["video_c3_votes"] = int(row[["c3_rater1", "c3_rater2", "c3_rater3"]].astype(int).sum())

    result["video_c1_consensus"] = int(result["video_c1_votes"] >= 2)
    result["video_c2_consensus"] = int(result["video_c2_votes"] >= 2)
    result["video_c3_consensus"] = int(result["video_c3_votes"] >= 2)

    result["video_cvs_consensus"] = int(
        result["video_c1_consensus"]
        and result["video_c2_consensus"]
        and result["video_c3_consensus"]
    )

    result["confidence_mean"] = float(row[CONFIDENCE_COLUMNS].astype(float).mean())

    return result


def build_manifest(sages_root: Path, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_root = sages_root / split / "labels"
    video_root = sages_root / split / "videos"
    frame_root = sages_root / "derived" / split / "frames"

    if not label_root.exists():
        raise FileNotFoundError(f"Label root not found: {label_root}")

    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    if not frame_root.exists():
        raise FileNotFoundError(f"Derived frame root not found: {frame_root}")

    label_dirs = sorted([p for p in label_root.iterdir() if p.is_dir()])
    video_paths = sorted(video_root.glob("*.mp4"))

    rows: list[dict[str, Any]] = []

    missing_videos: list[str] = []
    missing_frame_dirs: list[str] = []
    missing_frame_images: list[str] = []
    frame_label_mismatches: list[str] = []

    for label_dir in label_dirs:
        video_id = label_dir.name
        video_path = video_root / f"{video_id}.mp4"
        video_frame_dir = frame_root / video_id

        if not video_path.exists():
            missing_videos.append(video_id)
            continue

        if not video_frame_dir.exists():
            missing_frame_dirs.append(video_id)
            continue

        frame_labels = read_frame_labels(label_dir)
        video_labels = read_video_labels(label_dir)
        frame_map = read_frame_map(video_frame_dir)

        merged = frame_labels.merge(
            frame_map,
            left_on="frame_id",
            right_on="source_frame_id",
            how="left",
            validate="one_to_one",
        )

        if merged["file_name"].isna().any():
            frame_label_mismatches.append(video_id)
            continue

        for _, row in merged.iterrows():
            frame_file = video_frame_dir / str(row["file_name"])

            if not frame_file.exists():
                missing_frame_images.append(str(frame_file))
                continue

            c1_cols = ["c1_rater1", "c1_rater2", "c1_rater3"]
            c2_cols = ["c2_rater1", "c2_rater2", "c2_rater3"]
            c3_cols = ["c3_rater1", "c3_rater2", "c3_rater3"]

            c1_votes = vote_count(row, c1_cols)
            c2_votes = vote_count(row, c2_cols)
            c3_votes = vote_count(row, c3_cols)

            c1_consensus = int(c1_votes >= 2)
            c2_consensus = int(c2_votes >= 2)
            c3_consensus = int(c3_votes >= 2)

            frame_id = int(row["frame_id"])
            extracted_frame_id = int(row["extracted_frame_id"])

            output_row: dict[str, Any] = {
                "sample_id": f"{video_id}_{frame_id}",
                "dataset": "sages",
                "split": split,
                "official_split": split,
                "video_id": video_id,
                "frame_id": frame_id,
                "source_frame_id": int(row["source_frame_id"]),
                "extracted_frame_id": extracted_frame_id,
                "sequence_index": extracted_frame_id,
                "timestamp_sec": float(row["timestamp_sec"]),
                "source_fps": 30.0,
                "label_interval_sec": 5.0,
                "video_relative_path": str(video_path.relative_to(sages_root)),
                "relative_path": str(frame_file.relative_to(sages_root)),
                "label_source": "manual_frame_raters",
                "c1_rater1": int(row["c1_rater1"]),
                "c1_rater2": int(row["c1_rater2"]),
                "c1_rater3": int(row["c1_rater3"]),
                "c2_rater1": int(row["c2_rater1"]),
                "c2_rater2": int(row["c2_rater2"]),
                "c2_rater3": int(row["c2_rater3"]),
                "c3_rater1": int(row["c3_rater1"]),
                "c3_rater2": int(row["c3_rater2"]),
                "c3_rater3": int(row["c3_rater3"]),
                "c1_votes": c1_votes,
                "c2_votes": c2_votes,
                "c3_votes": c3_votes,
                "c1_consensus": c1_consensus,
                "c2_consensus": c2_consensus,
                "c3_consensus": c3_consensus,
                "cvs_consensus": int(c1_consensus and c2_consensus and c3_consensus),
            }

            output_row.update(video_labels)
            rows.append(output_row)

    manifest = pd.DataFrame(rows)

    expected_rows = len(label_dirs) * 18

    summary: dict[str, Any] = {
        "dataset": "sages",
        "split": split,
        "label_dirs": len(label_dirs),
        "videos": len(video_paths),
        "expected_labelled_frames": expected_rows,
        "manifest_rows": int(len(manifest)),
        "expected_frames_per_video": 18,
        "source_fps": 30.0,
        "label_interval_sec": 5.0,
        "clip_duration_sec": 90.0,
        "missing_videos": len(missing_videos),
        "missing_frame_dirs": len(missing_frame_dirs),
        "missing_frame_images": len(missing_frame_images),
        "frame_label_mismatches": len(frame_label_mismatches),
    }

    if not manifest.empty:
        summary.update(
            {
                "unique_videos_in_manifest": int(manifest["video_id"].nunique()),
                "c1_positive_frames": int(manifest["c1_consensus"].sum()),
                "c2_positive_frames": int(manifest["c2_consensus"].sum()),
                "c3_positive_frames": int(manifest["c3_consensus"].sum()),
                "complete_cvs_positive_frames": int(manifest["cvs_consensus"].sum()),
                "video_c1_positive": int(
                    manifest.drop_duplicates("video_id")["video_c1_consensus"].sum()
                ),
                "video_c2_positive": int(
                    manifest.drop_duplicates("video_id")["video_c2_consensus"].sum()
                ),
                "video_c3_positive": int(
                    manifest.drop_duplicates("video_id")["video_c3_consensus"].sum()
                ),
                "video_complete_cvs_positive": int(
                    manifest.drop_duplicates("video_id")["video_cvs_consensus"].sum()
                ),
            }
        )

    if missing_videos:
        summary["missing_video_examples"] = missing_videos[:10]

    if missing_frame_dirs:
        summary["missing_frame_dir_examples"] = missing_frame_dirs[:10]

    if missing_frame_images:
        summary["missing_frame_image_examples"] = missing_frame_images[:10]

    if frame_label_mismatches:
        summary["frame_label_mismatch_examples"] = frame_label_mismatches[:10]

    return manifest, summary


def main() -> None:
    args = parse_args()

    sages_root = Path(args.sages_root)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    manifest, summary = build_manifest(
        sages_root=sages_root,
        split=args.split,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(output_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Saved manifest: {output_csv}")
    print(f"Saved summary:  {summary_json}")
    print()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
