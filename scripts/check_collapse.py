#!/usr/bin/env python3
"""Detect representation collapse after latent-prediction pretraining.

A falling JEPA loss does not establish that anything was learned. The loss is
``L1(predictor(context), target_encoder(x))``, and it can be reduced either by
predicting well or by the target encoder producing outputs with less structure
to predict. The degenerate limit is every patch mapping to the same vector, at
which point the loss approaches zero and the representation is worthless.

Reconstruction objectives cannot fail this way, because the target is the input
pixels. That asymmetry is why this check exists for the V-JEPA arm and not for
the VideoMAE one, and why comparing the two on training loss alone would be
meaningless.

Three measures, computed on patch tokens from real clips:

**Per-dimension variance.** Averaged across the feature axis. Collapse drives it
toward zero.

**Effective rank.** The participation ratio of the covariance eigenvalues,
``(sum e)^2 / sum(e^2)``, giving the number of dimensions the representation
actually spans. A 1024-dimensional encoder using 3 dimensions has collapsed
whatever its loss says.

**Mean pairwise cosine similarity between patches.** Near 1 means all patches
carry the same representation, which is collapse in its most direct form.

Usage:
    python scripts/check_collapse.py \\
        --checkpoint outputs/ssl/vjepa2_sages/latest.pt \\
        --video-dir ~/datasets/sages/train/videos --num-clips 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _participation_ratio(matrix: torch.Tensor) -> float:
    """``(sum e)^2 / sum(e^2)`` over eigenvalues: the number of directions
    carrying appreciable weight, rather than the number that are merely
    non-zero."""
    eigenvalues = torch.linalg.eigvalsh(matrix).clamp(min=0)
    total = eigenvalues.sum()
    return (total**2 / (eigenvalues**2).sum()).item() if total > 0 else 0.0


def feature_statistics(tokens: torch.Tensor) -> dict[str, float]:
    """Statistics on ``[N, D]`` patch tokens pooled across clips."""
    tokens = tokens.float()

    variance = tokens.var(dim=0).mean().item()

    # Both ranks are reported because they answer different questions and
    # disagree substantially in practice.
    #
    # Uncentred detects *collapse*: mapping every patch to one constant vector
    # is rank one in this sense. Centring would subtract exactly that constant
    # and leave isotropic noise reading as full rank, so a covariance-based
    # measure cannot see the failure this script exists to catch.
    #
    # Centred describes *how much the representation varies*, which is the
    # quantity meant by "the encoder uses N of its D dimensions". A large shared
    # offset across patches dominates the uncentred spectrum and drags that
    # figure down even where variation is high-dimensional: VideoMAE ViT-B on
    # surgical video measures 2.6 uncentred against 31.9 centred. Reading the
    # uncentred number as capacity usage understates it by an order of
    # magnitude.
    effective_rank = _participation_ratio((tokens.T @ tokens) / max(len(tokens), 1))

    centred = tokens - tokens.mean(dim=0, keepdim=True)
    covariance = (centred.T @ centred) / max(len(centred) - 1, 1)
    centred_rank = _participation_ratio(covariance)

    normalised = torch.nn.functional.normalize(tokens, dim=1)
    sample = normalised[torch.randperm(len(normalised))[:512]]
    similarity = sample @ sample.T
    off_diagonal = similarity[~torch.eye(len(sample), dtype=torch.bool)]

    return {
        "mean_variance": variance,
        "effective_rank": effective_rank,
        "centred_rank": centred_rank,
        "centred_rank_fraction": centred_rank / tokens.shape[1],
        "dimensions": tokens.shape[1],
        "rank_fraction": effective_rank / tokens.shape[1],
        "mean_pairwise_cosine": off_diagonal.mean().item(),
        "feature_norm": tokens.norm(dim=1).mean().item(),
    }


def verdict(before: dict[str, float], after: dict[str, float]) -> tuple[str, list[str]]:
    """Judge collapse from the change in structure, not from absolute values."""
    problems = []

    variance_ratio = after["mean_variance"] / max(before["mean_variance"], 1e-12)
    rank_ratio = after["effective_rank"] / max(before["effective_rank"], 1e-12)

    if variance_ratio < 0.1:
        problems.append(
            f"feature variance fell to {variance_ratio:.1%} of the original"
        )
    if rank_ratio < 0.5:
        problems.append(
            f"effective rank fell to {rank_ratio:.1%} of the original "
            f"({before['effective_rank']:.1f} -> {after['effective_rank']:.1f})"
        )
    if after["mean_pairwise_cosine"] > 0.95:
        problems.append(
            f"patches are near-identical (mean cosine "
            f"{after['mean_pairwise_cosine']:.3f})"
        )
    centred_ratio = after["centred_rank"] / max(before["centred_rank"], 1e-12)
    if centred_ratio < 0.5:
        problems.append(
            f"centred rank fell to {centred_ratio:.1%} of the original "
            f"({before['centred_rank']:.1f} -> {after['centred_rank']:.1f}); "
            f"the representation lost variation, not just scale"
        )
    if after["rank_fraction"] < 0.01:
        problems.append(
            f"representation spans {after['effective_rank']:.1f} of "
            f"{after['dimensions']} dimensions"
        )

    return ("COLLAPSED" if problems else "healthy"), problems


@torch.no_grad()
def encode(model, clips: torch.Tensor, device: torch.device) -> torch.Tensor:
    outputs = []
    for index in range(len(clips)):
        hidden = model(pixel_values_videos=clips[index : index + 1].to(device))
        hidden = hidden.last_hidden_state if hasattr(hidden, "last_hidden_state") else hidden
        outputs.append(hidden.squeeze(0).cpu())
    return torch.cat(outputs, dim=0)


def main() -> int:
    args = parse_args()

    import copy

    from transformers import VJEPA2Model

    from data.ssl_clip_dataset import ClipTransform, SSLClipDataset

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    checkpoint_name = config.get("model", {}).get("checkpoint", args.base)

    reference = VJEPA2Model.from_pretrained(checkpoint_name).encoder.eval().to(device)

    trained = copy.deepcopy(reference)
    state = payload.get("target_encoder")
    if state is None:
        raise SystemExit(
            f"{args.checkpoint} contains no target_encoder state. Collapse is a "
            f"property of the target encoder, so it cannot be assessed without it."
        )
    trained.load_state_dict(state)
    trained.eval().to(device)

    model_config = reference.config if hasattr(reference, "config") else None
    data = config.get("data", {})
    dataset = SSLClipDataset(
        args.video_dir or data["video_dir"],
        num_frames=data.get("num_frames", 64),
        stride=data.get("stride", 4),
        clips_per_video=1,
        transform=ClipTransform(image_size=data.get("image_size", 256), train=False),
        decode_size=data.get("decode_size", 288),
        limit_videos=args.num_clips,
    )
    clips = torch.stack([dataset[i]["pixel_values"] for i in range(min(args.num_clips, len(dataset)))])

    print(f"checkpoint  {args.checkpoint}")
    print(f"base        {checkpoint_name}")
    print(f"step        {payload.get('step')}")
    print(f"clips       {len(clips)}")
    print()

    before = feature_statistics(encode(reference, clips, device))
    after = feature_statistics(encode(trained, clips, device))

    print(f"  {'measure':<26} {'before':>12} {'after':>12} {'ratio':>8}")
    for key in ("mean_variance", "effective_rank", "centred_rank",
                "mean_pairwise_cosine", "feature_norm"):
        ratio = after[key] / before[key] if before[key] else float("nan")
        print(f"  {key:<26} {before[key]:12.4f} {after[key]:12.4f} {ratio:8.3f}")
    print(f"  {'uncentred / dims':<26} "
          f"{before['effective_rank']:6.1f}/{before['dimensions']:<5d} "
          f"{after['effective_rank']:6.1f}/{after['dimensions']:<5d}")
    print(f"  {'centred / dims':<26} "
          f"{before['centred_rank']:6.1f}/{before['dimensions']:<5d} "
          f"{after['centred_rank']:6.1f}/{after['dimensions']:<5d}")
    print("\n  uncentred rank detects collapse to a constant; centred rank "
          "describes how much\n  the representation varies. Quote the centred "
          "figure for capacity usage.")

    status, problems = verdict(before, after)
    print(f"\n  verdict: {status}")
    for problem in problems:
        print(f"    - {problem}")
    if status == "healthy":
        print("    representation retains its variance and dimensionality")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"before": before, "after": after, "status": status, "problems": problems},
            indent=2,
        ))
        print(f"\n  written to {args.json}")

    return 0 if status == "healthy" else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video-dir", default=None)
    p.add_argument("--base", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--num-clips", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())