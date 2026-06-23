"""Build and validate the Endoscapes frame manifest."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXPECTED_FRAMES = 58_813
EXPECTED_VIDEOS = 201
SPLITS = ("train", "val", "test")
IMAGE_PATTERN = re.compile(r"^(?P<video_id>\d+)_(?P<frame_num>\d+)\.jpg$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validated Endoscapes frame manifest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/paths.local.yaml"),
        help="Path to the machine-specific YAML configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional manifest CSV path.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional validation summary JSON path.",
    )
    return parser.parse_args()


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    required = {"endoscapes_root", "metadata_dir"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    return config


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    path = dataset_root / "all_metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Endoscapes metadata not found: {path}")

    metadata = pd.read_csv(path)
    required = {
        "vid",
        "frame",
        "avg_cvs",
        "C1",
        "C2",
        "C3",
        "is_ds_keyframe",
        "cvs_annotator_1",
        "cvs_annotator_2",
        "cvs_annotator_3",
        "mask_path",
        "label_path",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    duplicate_count = int(metadata.duplicated(["vid", "frame"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Metadata contains {duplicate_count} duplicate (vid, frame) rows."
        )

    return metadata


def index_images(dataset_root: Path) -> pd.DataFrame:
    records: list[dict] = []

    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

        for image_path in split_dir.glob("*.jpg"):
            match = IMAGE_PATTERN.fullmatch(image_path.name)
            if match is None:
                raise ValueError(f"Unexpected Endoscapes filename: {image_path.name}")

            records.append(
                {
                    "vid": int(match.group("video_id")),
                    "frame": int(match.group("frame_num")),
                    "split": split,
                    "relative_path": image_path.relative_to(dataset_root).as_posix(),
                }
            )

    images = pd.DataFrame.from_records(records)
    if images.empty:
        raise ValueError("No JPG images were found in train, val, or test.")

    duplicate_count = int(images.duplicated(["vid", "frame"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Image index contains {duplicate_count} duplicate identifiers."
        )

    videos_per_split = images.groupby("vid")["split"].nunique()
    leakage_count = int((videos_per_split > 1).sum())
    if leakage_count:
        leaked = videos_per_split[videos_per_split > 1].index.tolist()
        raise ValueError(f"Videos appear in multiple splits: {leaked}")

    return images


def parse_annotation_vector(value: object, column: str, sample_id: str) -> list[int]:
    try:
        vector = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"Could not parse {column} for sample {sample_id}: {value!r}"
        ) from error

    if (
        not isinstance(vector, list)
        or len(vector) != 3
        or any(item not in (0, 1) for item in vector)
    ):
        raise ValueError(
            f"Invalid {column} vector for sample {sample_id}: {vector!r}"
        )
    return vector


def add_manual_targets(manifest: pd.DataFrame) -> pd.DataFrame:
    annotator_columns = [
        "cvs_annotator_1",
        "cvs_annotator_2",
        "cvs_annotator_3",
    ]
    criteria = ["c1", "c2", "c3"]
    manual_mask = manifest["is_cvs_annotated"]
    manual = manifest.loc[manual_mask]

    annotations = np.array(
        [
            [
                parse_annotation_vector(row[column], column, row["sample_id"])
                for column in annotator_columns
            ]
            for _, row in manual.iterrows()
        ],
        dtype=np.int8,
    )

    stored_values = manual[
        ["c1_dataset_value", "c2_dataset_value", "c3_dataset_value"]
    ].to_numpy()
    computed_values = annotations.mean(axis=1)
    disagreement_count = int(
        np.any(~np.isclose(stored_values, computed_values), axis=1).sum()
    )
    if disagreement_count:
        raise ValueError(
            f"{disagreement_count} keyframes disagree with their annotator averages."
        )

    votes = annotations.sum(axis=1)
    consensus = (votes >= 2).astype(np.int8)

    for criterion_index, criterion in enumerate(criteria):
        vote_column = f"{criterion}_manual_votes"
        target_column = f"{criterion}_consensus"
        manifest[vote_column] = pd.Series(pd.NA, index=manifest.index, dtype="Int8")
        manifest[target_column] = pd.Series(pd.NA, index=manifest.index, dtype="Int8")
        manifest.loc[manual_mask, vote_column] = votes[:, criterion_index]
        manifest.loc[manual_mask, target_column] = consensus[:, criterion_index]

    manifest["cvs_consensus"] = pd.Series(
        pd.NA, index=manifest.index, dtype="Int8"
    )
    manifest.loc[manual_mask, "cvs_consensus"] = consensus.all(axis=1).astype(
        np.int8
    )
    return manifest


def build_manifest(metadata: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    joined = metadata.merge(
        images,
        on=["vid", "frame"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    correspondence = joined["_merge"].value_counts().to_dict()
    if correspondence.get("left_only", 0) or correspondence.get("right_only", 0):
        raise ValueError(f"Metadata-image mismatch: {correspondence}")

    manifest = joined.drop(columns="_merge").rename(
        columns={
            "vid": "video_id",
            "frame": "frame_num",
            "C1": "c1_dataset_value",
            "C2": "c2_dataset_value",
            "C3": "c3_dataset_value",
            "is_ds_keyframe": "is_cvs_annotated",
        }
    )

    manifest["sample_id"] = (
        manifest["video_id"].astype(str)
        + "_"
        + manifest["frame_num"].astype(str)
    )
    manifest["label_source"] = np.where(
        manifest["is_cvs_annotated"], "manual", "forward_fill"
    )

    manifest = manifest.sort_values(
        ["video_id", "frame_num"], kind="stable"
    ).reset_index(drop=True)
    manifest["sequence_index"] = (
        manifest.groupby("video_id", sort=False).cumcount().astype("int32")
    )

    manifest = add_manual_targets(manifest)

    ordered_columns = [
        "sample_id",
        "video_id",
        "frame_num",
        "sequence_index",
        "split",
        "relative_path",
        "is_cvs_annotated",
        "label_source",
        "c1_dataset_value",
        "c2_dataset_value",
        "c3_dataset_value",
        "c1_manual_votes",
        "c2_manual_votes",
        "c3_manual_votes",
        "c1_consensus",
        "c2_consensus",
        "c3_consensus",
        "cvs_consensus",
        "avg_cvs",
        "cvs_annotator_1",
        "cvs_annotator_2",
        "cvs_annotator_3",
        "mask_path",
        "label_path",
    ]
    return manifest[ordered_columns]


def validate_manifest(manifest: pd.DataFrame) -> None:
    if len(manifest) != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES} frames, found {len(manifest)}."
        )

    video_count = int(manifest["video_id"].nunique())
    if video_count != EXPECTED_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_VIDEOS} videos, found {video_count}."
        )

    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values.")

    annotated = manifest["is_cvs_annotated"]
    target_columns = [
        "c1_manual_votes",
        "c2_manual_votes",
        "c3_manual_votes",
        "c1_consensus",
        "c2_consensus",
        "c3_consensus",
        "cvs_consensus",
    ]
    if manifest.loc[annotated, target_columns].isna().any().any():
        raise ValueError("At least one annotated frame is missing a manual target.")
    if manifest.loc[~annotated, target_columns].notna().any().any():
        raise ValueError("At least one non-keyframe has a manual target.")


def make_summary(manifest: pd.DataFrame) -> dict:
    annotated = manifest[manifest["is_cvs_annotated"]]

    return {
        "total_frames": int(len(manifest)),
        "total_videos": int(manifest["video_id"].nunique()),
        "annotated_keyframes": int(len(annotated)),
        "non_keyframes": int((~manifest["is_cvs_annotated"]).sum()),
        "frames_per_split": {
            key: int(value)
            for key, value in manifest["split"].value_counts().sort_index().items()
        },
        "videos_per_split": {
            key: int(value)
            for key, value in (
                manifest[["video_id", "split"]]
                .drop_duplicates()["split"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "annotated_keyframes_per_split": {
            key: int(value)
            for key, value in annotated["split"].value_counts().sort_index().items()
        },
        "majority_positive_keyframes": {
            criterion: int(annotated[f"{criterion}_consensus"].sum())
            for criterion in ("c1", "c2", "c3")
        },
        "complete_cvs_positive_keyframes": int(annotated["cvs_consensus"].sum()),
        "duplicate_samples": int(manifest["sample_id"].duplicated().sum()),
        "missing_images": 0,
        "extra_images": 0,
        "videos_in_multiple_splits": 0,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_repo_path(repo_root, args.config)
    config = load_config(config_path)

    dataset_root = resolve_repo_path(repo_root, config["endoscapes_root"])
    metadata_dir = resolve_repo_path(repo_root, config["metadata_dir"])
    output_path = (
        resolve_repo_path(repo_root, args.output)
        if args.output
        else metadata_dir / "endoscapes_frames.csv"
    )
    summary_path = (
        resolve_repo_path(repo_root, args.summary)
        if args.summary
        else metadata_dir / "endoscapes_manifest_summary.json"
    )

    metadata = load_metadata(dataset_root)
    images = index_images(dataset_root)
    manifest = build_manifest(metadata, images)
    validate_manifest(manifest)
    summary = make_summary(manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest written: {output_path}")
    print(f"Summary written:  {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
