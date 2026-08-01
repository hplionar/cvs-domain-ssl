#!/usr/bin/env python3
"""Verify the SAGES download and reconcile it against the existing manifest.

Checks, in order of how badly a wrong answer would hurt:

1. Video duration against the labelled timestamp span. The manifest records 18
   timepoints at 5-second spacing, which reads as a 90-second clip; the
   downloaded file sizes (up to 276 MB) do not fit that. If the videos are
   substantially longer than the labelled span, there is far more unlabelled
   footage available for SSL than assumed, and the corpus design changes.
2. Official splits against the internal split. The repository holds 1000 videos
   across train and test; the manifest covers 700 in a 560/70/70 internal split.
   Any disagreement must be resolved before pretraining, not after.
3. Label file integrity and column names, including the per-rater confidence
   fields, which the manifest does not carry.

Usage:
    python scripts/verify_sages_download.py --root ~/datasets/sages \
        --manifest metadata/sages_frames_internal_split.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def probe(path: Path) -> dict[str, float | int | str] | None:
    """Read container metadata with ffprobe, falling back to PyAV."""
    if shutil.which("ffprobe"):
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
                 "-show_entries", "format=duration,size",
                 "-of", "json", str(path)],
                text=True, stderr=subprocess.DEVNULL,
            )
            data = json.loads(out)
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})
            num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
            fps = float(num) / float(den) if float(den or 0) else 0.0
            return {
                "duration_s": float(fmt.get("duration", 0.0)),
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": round(fps, 2),
                "frames": int(stream.get("nb_frames", 0) or 0),
            }
        except Exception:
            pass

    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return {
                "duration_s": float(container.duration or 0) / 1_000_000,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "fps": round(float(stream.average_rate or 0), 2),
                "frames": stream.frames,
            }
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="~/datasets/sages")
    parser.add_argument("--manifest", default="metadata/sages_frames_internal_split.csv")
    args = parser.parse_args()

    root = Path(args.root).expanduser()

    # -- inventory --------------------------------------------------------

    print("=" * 68)
    print("INVENTORY")
    print("=" * 68)

    for split in ("train", "test"):
        videos = sorted((root / split / "videos").glob("*.mp4"))
        labels = sorted((root / split / "labels").glob("*/"))
        total_gib = sum(v.stat().st_size for v in videos) / 1024**3
        print(f"{split:6s}  videos on disk: {len(videos):4d}   "
              f"label dirs: {len(labels):4d}   {total_gib:6.2f} GiB")

    # -- video geometry versus labelled span ------------------------------

    print()
    print("=" * 68)
    print("DURATION vs LABELLED SPAN")
    print("=" * 68)

    videos = sorted((root / "train" / "videos").glob("*.mp4"))
    if not videos:
        print("No videos downloaded yet; skipping.")
    else:
        rows = []
        for path in videos:
            uid = path.stem
            info = probe(path)
            if info is None:
                print(f"  could not probe {uid}; install ffmpeg or `pip install av`")
                break

            frame_csv = root / "train" / "labels" / uid / "frame.csv"
            span = n_labels = None
            if frame_csv.is_file():
                frames = pd.read_csv(frame_csv)
                n_labels = len(frames)
                for col in ("timestamp_sec", "timestamp", "time_sec", "sec"):
                    if col in frames.columns:
                        span = float(frames[col].max() - frames[col].min())
                        break

            rows.append({
                "video": uid[:8],
                "MB": round(path.stat().st_size / 1024**2, 1),
                "duration_s": round(info["duration_s"], 1),
                "res": f"{info['width']}x{info['height']}",
                "fps": info["fps"],
                "labels": n_labels,
                "label_span_s": span,
                "labelled_%": (
                    round(100 * span / info["duration_s"], 1)
                    if span and info["duration_s"] else None
                ),
            })

        if rows:
            table = pd.DataFrame(rows)
            print(table.to_string(index=False))
            print()
            print(f"median duration: {table.duration_s.median():.1f} s "
                  f"({table.duration_s.median()/60:.1f} min)")
            covered = table["labelled_%"].dropna()
            if len(covered):
                print(f"labelled span covers {covered.median():.1f}% of the video "
                      f"(median)")
                if covered.median() < 50:
                    print("\n  NOTE: most footage is unlabelled. The SSL corpus is "
                          "considerably larger than the labelled clip count "
                          "suggests, and the assumption that SAGES covers only the "
                          "CVS window does not hold.")

    # -- official splits versus internal split ---------------------------

    print()
    print("=" * 68)
    print("SPLITS")
    print("=" * 68)

    splits_path = root / "splits" / "subchallenge_c_splits.json"
    if splits_path.is_file():
        official = json.loads(splits_path.read_text())
        print(f"keys: {list(official)[:10]}")
        for key, value in official.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} entries, e.g. {value[:2]}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys -> {list(value)[:5]}")
    else:
        print(f"not found at {splits_path}")

    manifest_path = Path(args.manifest)
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        print()
        print(f"manifest: {len(manifest)} rows, "
              f"{manifest.video_id.nunique()} videos")
        print(manifest.groupby("internal_split").video_id.nunique().to_dict())

        on_disk = {p.stem for p in (root / "train" / "videos").glob("*.mp4")}
        in_manifest = set(manifest.video_id.astype(str))
        overlap = on_disk & in_manifest
        print(f"downloaded videos also in manifest: {len(overlap)}/{len(on_disk)}")
        if on_disk and not overlap:
            print("  WARNING: no downloaded video id appears in the manifest. "
                  "The manifest may use a different identifier scheme.")
            print(f"  disk:     {sorted(on_disk)[:2]}")
            print(f"  manifest: {sorted(in_manifest)[:2]}")
    else:
        print(f"\nmanifest not found at {manifest_path}")

    # -- label columns ----------------------------------------------------

    print()
    print("=" * 68)
    print("LABEL COLUMNS")
    print("=" * 68)

    label_dirs = sorted((root / "train" / "labels").glob("*/"))
    if label_dirs:
        sample = label_dirs[0]
        for name in ("frame.csv", "video.csv"):
            path = sample / name
            if path.is_file():
                frame = pd.read_csv(path)
                print(f"\n{name}  ({len(frame)} rows)")
                print(f"  columns: {list(frame.columns)}")
                print(frame.head(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())