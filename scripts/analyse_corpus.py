#!/usr/bin/env python3
"""Characterise a video corpus for self-supervised pretraining.

Produces the measurements needed to justify corpus design decisions rather than
assert them, and the numbers for the corpus section of the methods chapter.

Four analyses:

**Temporal decorrelation.** Frozen-encoder features are extracted at a high
frame rate and their cosine similarity computed against increasing temporal lag.
The lag at which similarity falls to the between-video baseline is the point
beyond which two frames carry effectively independent information. This sets the
SSL clip sampling stride: sampling more densely than the decorrelation time adds
redundancy rather than data.

**Effective sample size.** Frame counts overstate corpus size when consecutive
frames are near-identical. Dividing video duration by the decorrelation time
gives a defensible estimate of independent samples, which is the figure that
should appear in the write-up rather than the raw frame count.

**Inter-rater agreement.** Fleiss' kappa per CVS criterion from the three
per-rater columns. A criterion on which trained surgeons disagree substantially
is one where a model's errors are correspondingly less interpretable — relevant
to framing automated assessment as a precondition problem rather than as
automation of a settled standard.

**Geometry and prevalence.** Resolution, duration and frame-rate distributions
across the corpus, plus label prevalence. Resolution heterogeneity is a proxy
for camera and institutional diversity.

Usage:
    python scripts/analyse_corpus.py --root ~/datasets/sages --split train \\
        --encoder dinov3_b --max-videos 20 --out outputs/corpus_analysis
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CRITERIA = ("c1", "c2", "c3")
RATERS = (1, 2, 3)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def probe_video(path: Path) -> dict[str, Any] | None:
    """Container metadata via ffprobe, falling back to PyAV."""
    if shutil.which("ffprobe"):
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,avg_frame_rate",
                 "-show_entries", "format=duration",
                 "-of", "json", str(path)],
                text=True, stderr=subprocess.DEVNULL,
            )
            data = json.loads(out)
            stream = data.get("streams", [{}])[0]
            num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
            fps = float(num) / float(den) if float(den or 0) else 0.0
            return {
                "duration_s": float(data.get("format", {}).get("duration", 0.0)),
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "fps": round(fps, 2),
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
            }
    except Exception:
        return None


def analyse_geometry(video_paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in video_paths:
        info = probe_video(path)
        if info is None:
            continue
        info["video"] = path.stem
        info["size_mb"] = round(path.stat().st_size / 1024**2, 1)
        info["resolution"] = f"{info['width']}x{info['height']}"
        rows.append(info)

    if not rows:
        return {"error": "no videos could be probed"}

    frame = pd.DataFrame(rows)
    resolutions = Counter(frame.resolution)

    return {
        "num_videos": len(frame),
        "duration_s": {
            "mean": round(float(frame.duration_s.mean()), 2),
            "std": round(float(frame.duration_s.std(ddof=1)), 3) if len(frame) > 1 else 0.0,
            "min": round(float(frame.duration_s.min()), 2),
            "max": round(float(frame.duration_s.max()), 2),
        },
        "fps": sorted(frame.fps.unique().tolist()),
        "resolutions": dict(resolutions.most_common()),
        "resolution_entropy": round(_entropy(list(resolutions.values())), 3),
        "size_mb": {
            "median": round(float(frame.size_mb.median()), 1),
            "min": float(frame.size_mb.min()),
            "max": float(frame.size_mb.max()),
        },
        "total_frames": int((frame.duration_s * frame.fps).sum()),
    }


def _entropy(counts: list[int]) -> float:
    """Shannon entropy in bits over a categorical distribution.

    Reported for resolution because a corpus drawn from many camera systems is
    more diverse in appearance than one from a single system, and a single
    number is easier to compare across corpora than a full distribution.
    """
    total = sum(counts)
    probs = np.array([c / total for c in counts if c])
    return float(-(probs * np.log2(probs)).sum())


# --------------------------------------------------------------------------
# inter-rater agreement
# --------------------------------------------------------------------------


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for ``[N, R]`` binary ratings of N items by R raters.

    Measures agreement above what would occur by chance given the observed
    marginal prevalence, which matters here: on a criterion achieved in 12% of
    frames, three raters agreeing on "not achieved" is mostly a consequence of
    rarity rather than of shared judgement.
    """
    n_items, n_raters = ratings.shape
    if n_items == 0 or n_raters < 2:
        return float("nan")

    positives = ratings.sum(axis=1)
    counts = np.stack([n_raters - positives, positives], axis=1)

    # Per-item agreement
    p_i = (counts * (counts - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()

    # Chance agreement from marginal category proportions
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = (p_j**2).sum()

    if np.isclose(p_e, 1.0):
        return float("nan")  # degenerate: one category only
    return float((p_bar - p_e) / (1 - p_e))


def interpret_kappa(value: float) -> str:
    """Landis and Koch (1977) benchmarks. Conventional, not authoritative."""
    if np.isnan(value):
        return "undefined"
    for threshold, label in [
        (0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
        (0.60, "moderate"), (0.80, "substantial"),
    ]:
        if value <= threshold or (threshold == 0.80 and value <= 1.0):
            if value <= threshold:
                return label
    return "almost perfect"


def analyse_agreement(label_dirs: list[Path]) -> dict[str, Any]:
    per_criterion: dict[str, list[np.ndarray]] = {c: [] for c in CRITERIA}
    confidences: list[list[float]] = []

    for directory in label_dirs:
        frame_csv = directory / "frame.csv"
        if frame_csv.is_file():
            frame = pd.read_csv(frame_csv)
            for criterion in CRITERIA:
                cols = [f"{criterion}_rater{r}" for r in RATERS]
                if all(c in frame.columns for c in cols):
                    per_criterion[criterion].append(frame[cols].to_numpy())

        video_csv = directory / "video.csv"
        if video_csv.is_file():
            video = pd.read_csv(video_csv)
            cols = [f"confidence_rater{r}" for r in RATERS]
            if all(c in video.columns for c in cols):
                confidences.append(video[cols].to_numpy().ravel().tolist())

    result: dict[str, Any] = {"criteria": {}}
    for criterion, blocks in per_criterion.items():
        if not blocks:
            continue
        ratings = np.concatenate(blocks, axis=0)
        kappa = fleiss_kappa(ratings)
        majority = (ratings.sum(axis=1) >= 2).astype(int)
        unanimous = ((ratings.sum(axis=1) == 0) | (ratings.sum(axis=1) == 3))
        result["criteria"][criterion] = {
            "n_frames": int(ratings.shape[0]),
            "fleiss_kappa": round(kappa, 4),
            "interpretation": interpret_kappa(kappa),
            "prevalence_majority": round(float(majority.mean()), 4),
            "prevalence_per_rater": [
                round(float(ratings[:, i].mean()), 4) for i in range(ratings.shape[1])
            ],
            "unanimous_fraction": round(float(unanimous.mean()), 4),
            "split_decision_fraction": round(float(1 - unanimous.mean()), 4),
        }

    if confidences:
        flat = np.array(confidences)
        result["rater_confidence"] = {
            "n_videos": int(flat.shape[0]),
            "mean": round(float(flat.mean()), 4),
            "std": round(float(flat.std(ddof=1)), 4),
            "distribution": {
                str(v): int(c) for v, c in
                zip(*np.unique(flat.round(2), return_counts=True))
            },
        }

    return result


# --------------------------------------------------------------------------
# temporal decorrelation
# --------------------------------------------------------------------------


def analyse_decorrelation(
    video_paths: list[Path],
    *,
    encoder_name: str,
    sample_fps: float,
    max_lag_s: float,
    device_name: str,
    random_init: bool,
) -> dict[str, Any]:
    """Feature similarity against temporal lag.

    Frames are sampled at ``sample_fps`` and encoded once. Similarity is then
    computed for every lag up to ``max_lag_s``, averaged within videos, and
    compared against a between-video baseline computed from frames drawn from
    different procedures. The lag at which within-video similarity reaches that
    baseline is the decorrelation time.
    """
    import torch

    from data.transforms import build_transform_from_spec
    from models.encoders import build_encoder
    import models.encoders.dinov3_encoder  # noqa: F401
    import models.encoders.mae_encoder  # noqa: F401
    import models.encoders.videomae_encoder  # noqa: F401
    import models.encoders.vjepa2_encoder  # noqa: F401

    try:
        import av
    except ImportError:
        return {"error": "PyAV required for frame extraction; pip install av"}

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    kwargs = {"random_init": True} if random_init else {}
    encoder = build_encoder(encoder_name, **kwargs).to(device).eval()
    transform = build_transform_from_spec(encoder.preprocess_spec, train=False)

    if encoder.modality != "image":
        return {"error": f"{encoder_name} is a video encoder; use an image encoder here"}

    per_video: list[np.ndarray] = []
    video_means: list[np.ndarray] = []

    for path in video_paths:
        frames = _sample_frames(path, sample_fps)
        if len(frames) < 4:
            continue
        batch = torch.stack([transform(f) for f in frames]).to(device)

        with torch.inference_mode():
            features = []
            for start in range(0, len(batch), 32):
                out = encoder(batch[start : start + 32])
                features.append(out.tokens.float().mean(dim=1).cpu())
            feats = torch.cat(features)

        feats = torch.nn.functional.normalize(feats, dim=1).numpy()
        per_video.append(feats)
        video_means.append(feats.mean(axis=0))

    if len(per_video) < 2:
        return {"error": "need at least two videos"}

    max_lag = int(round(max_lag_s * sample_fps))
    lags = np.arange(1, max_lag + 1)
    similarity = np.full(len(lags), np.nan)

    for index, lag in enumerate(lags):
        values = [
            float((feats[:-lag] * feats[lag:]).sum(axis=1).mean())
            for feats in per_video
            if len(feats) > lag
        ]
        if values:
            similarity[index] = float(np.mean(values))

    # Between-video baseline: what similarity looks like for unrelated frames
    stacked = np.stack(video_means)
    between = stacked @ stacked.T
    baseline = float(between[~np.eye(len(stacked), dtype=bool)].mean())

    # Decorrelation time: first lag reaching the between-video baseline
    decorrelation_lag = None
    for index, value in enumerate(similarity):
        if not np.isnan(value) and value <= baseline:
            decorrelation_lag = float(lags[index] / sample_fps)
            break

    # Half-decay is reported as a robust alternative that always exists
    if not np.isnan(similarity[0]):
        target = baseline + (similarity[0] - baseline) / 2
        half_decay = next(
            (float(lags[i] / sample_fps) for i, v in enumerate(similarity)
             if not np.isnan(v) and v <= target),
            None,
        )
    else:
        half_decay = None

    return {
        "encoder": encoder.checkpoint_id,
        "random_init": random_init,
        "sample_fps": sample_fps,
        "num_videos": len(per_video),
        "frames_per_video": int(np.mean([len(f) for f in per_video])),
        "similarity_by_lag_s": {
            round(float(lag / sample_fps), 2): (None if np.isnan(v) else round(float(v), 4))
            for lag, v in zip(lags, similarity)
        },
        "between_video_baseline": round(baseline, 4),
        "decorrelation_time_s": decorrelation_lag,
        "half_decay_time_s": half_decay,
    }


def _sample_frames(path: Path, sample_fps: float) -> list:
    """Decode frames at approximately ``sample_fps``, returning PIL images."""
    import av

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        native = float(stream.average_rate or 30.0)
        step = max(1, int(round(native / sample_fps)))
        for index, frame in enumerate(container.decode(stream)):
            if index % step == 0:
                frames.append(frame.to_image())
    return frames


def effective_sample_size(
    geometry: dict[str, Any], decorrelation: dict[str, Any], num_videos_total: int
) -> dict[str, Any]:
    """Independent samples implied by the decorrelation time.

    The frame count is what the corpus contains; this is what it is worth. The
    difference between them is the point that needs stating explicitly in the
    write-up, since a frame count invites comparison against corpora whose
    frames are far less redundant.
    """
    tau = decorrelation.get("decorrelation_time_s") or decorrelation.get("half_decay_time_s")
    duration = geometry.get("duration_s", {}).get("mean")

    if not tau or not duration:
        return {"error": "decorrelation time or duration unavailable"}

    per_video = duration / tau
    fps = geometry.get("fps", [30.0])
    frames_per_video = duration * (fps[0] if fps else 30.0)

    return {
        "decorrelation_time_s": round(tau, 2),
        "basis": "decorrelation_time" if decorrelation.get("decorrelation_time_s") else "half_decay",
        "effective_samples_per_video": round(per_video, 1),
        "frames_per_video": round(frames_per_video),
        "redundancy_factor": round(frames_per_video / per_video, 1),
        "corpus_frames": int(frames_per_video * num_videos_total),
        "corpus_effective_samples": int(per_video * num_videos_total),
        "num_videos": num_videos_total,
    }


# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted((root / args.split / "videos").glob("*.mp4"))
    labels = sorted(p for p in (root / args.split / "labels").glob("*") if p.is_dir())
    sample = videos[: args.max_videos]

    report: dict[str, Any] = {
        "root": str(root),
        "split": args.split,
        "videos_available": len(videos),
        "videos_sampled": len(sample),
        "label_dirs": len(labels),
    }

    print("=" * 70)
    print(f"CORPUS ANALYSIS — {root.name}/{args.split}")
    print("=" * 70)
    print(f"videos on disk: {len(videos)}   label dirs: {len(labels)}   "
          f"sampled for geometry: {len(sample)}")

    # -- geometry ---------------------------------------------------------

    if sample:
        geometry = analyse_geometry(sample)
        report["geometry"] = geometry
        print("\n" + "-" * 70)
        print("GEOMETRY")
        print("-" * 70)
        print(f"  duration      {geometry['duration_s']['mean']} s "
              f"(sd {geometry['duration_s']['std']})")
        print(f"  frame rate    {geometry['fps']}")
        print(f"  file size     median {geometry['size_mb']['median']} MB, "
              f"range {geometry['size_mb']['min']}–{geometry['size_mb']['max']}")
        print(f"  resolutions   {geometry['resolutions']}")
        print(f"  entropy       {geometry['resolution_entropy']} bits "
              f"({len(geometry['resolutions'])} distinct)")
    else:
        geometry = {}
        print("\nNo videos on disk; skipping geometry and decorrelation.")

    # -- agreement --------------------------------------------------------

    if labels:
        agreement = analyse_agreement(labels)
        report["agreement"] = agreement
        print("\n" + "-" * 70)
        print("INTER-RATER AGREEMENT")
        print("-" * 70)
        print(f"  {'':4s}  {'kappa':>7s}  {'':14s}  {'prev':>6s}  {'split':>6s}")
        for criterion, stats in agreement.get("criteria", {}).items():
            print(f"  {criterion:4s}  {stats['fleiss_kappa']:7.4f}  "
                  f"{stats['interpretation']:14s}  "
                  f"{stats['prevalence_majority']:6.3f}  "
                  f"{stats['split_decision_fraction']:6.3f}")
        if "rater_confidence" in agreement:
            conf = agreement["rater_confidence"]
            print(f"\n  rater confidence  mean {conf['mean']:.3f} "
                  f"(sd {conf['std']:.3f}) over {conf['n_videos']} videos")
        print("\n  'split' is the fraction of frames where the three raters did "
              "not agree unanimously.")

    # -- decorrelation ----------------------------------------------------

    if sample and not args.skip_decorrelation:
        print("\n" + "-" * 70)
        print("TEMPORAL DECORRELATION")
        print("-" * 70)
        print(f"  encoder {args.encoder}, sampling at {args.sample_fps} fps "
              f"(this decodes video and may take a few minutes)")

        decorrelation = analyse_decorrelation(
            sample[: args.decorrelation_videos],
            encoder_name=args.encoder,
            sample_fps=args.sample_fps,
            max_lag_s=args.max_lag_s,
            device_name=args.device,
            random_init=args.random_init,
        )
        report["decorrelation"] = decorrelation

        if "error" in decorrelation:
            print(f"  {decorrelation['error']}")
        else:
            print(f"  between-video baseline   {decorrelation['between_video_baseline']}")
            print(f"  decorrelation time       {decorrelation['decorrelation_time_s']} s")
            print(f"  half-decay time          {decorrelation['half_decay_time_s']} s")
            print("\n  similarity by lag:")
            for lag, value in list(decorrelation["similarity_by_lag_s"].items())[:12]:
                if value is not None:
                    bar = "#" * int(max(0, value) * 40)
                    print(f"    {lag:5.2f}s  {value:6.4f}  {bar}")

            effective = effective_sample_size(geometry, decorrelation, len(videos))
            report["effective_sample_size"] = effective
            if "error" not in effective:
                print("\n" + "-" * 70)
                print("EFFECTIVE SAMPLE SIZE")
                print("-" * 70)
                print(f"  decorrelation time        {effective['decorrelation_time_s']} s "
                      f"(basis: {effective['basis']})")
                print(f"  frames per video          {effective['frames_per_video']}")
                print(f"  effective samples/video   {effective['effective_samples_per_video']}")
                print(f"  redundancy factor         {effective['redundancy_factor']}x")
                print(f"\n  corpus ({effective['num_videos']} videos):")
                print(f"    raw frames              {effective['corpus_frames']:,}")
                print(f"    effective samples       {effective['corpus_effective_samples']:,}")
                print("\n  Report the effective figure, not the frame count. A frame "
                      "count invites\n  comparison against corpora whose frames are far "
                      "less redundant.")

    path = out_dir / f"corpus_analysis_{args.split}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWritten to {path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="~/datasets/sages")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--out", default="outputs/corpus_analysis")

    p.add_argument("--max-videos", type=int, default=50,
                   help="videos sampled for geometry")
    p.add_argument("--decorrelation-videos", type=int, default=10,
                   help="videos used for decorrelation; each is fully decoded")

    p.add_argument("--encoder", default="dinov3_b",
                   help="image encoder for feature extraction")
    p.add_argument("--random-init", action="store_true",
                   help="build the encoder from config instead of downloading weights; "
                        "geometry is representative but features are not")
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--max-lag-s", type=float, default=20.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-decorrelation", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())