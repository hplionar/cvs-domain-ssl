#!/usr/bin/env python3
"""Extract frozen encoder features into a memory-mapped fp16 cache.

A frozen encoder is a deterministic function, so re-running it every epoch to
train a head of a few thousand parameters is pure waste. More importantly,
adaptation gain is a *difference* of two measurements: any stochastic variation
between the baseline run and the adapted run contaminates it. Extracting once
makes the probe protocol provably identical across arms.

Output layout, one directory per (encoder, dataset, split):

    <out>/
      tokens.npy      memmap  [N, num_tokens, D]  fp16
      prefix.npy      memmap  [N, num_prefix, D]  fp16   (omitted if none)
      targets.npy             [N, 3]              float32
      index.csv               sample_id, video_id, row
      manifest.json           provenance + integrity record

`.npy` memmaps rather than WebDataset shards: probe training needs *random*
access under shuffling, which sequential shard streaming serves poorly. The SSL
pretraining corpus is a different problem with a different answer.

Usage:
    # exercise the GPU path with synthetic data, no dataset required
    python scripts/extract_features.py --encoder mae_b --smoke --out /tmp/smoke

    # real extraction
    python scripts/extract_features.py \
        --encoder mae_b --dataset endoscapes --split train \
        --dataset-root /path/to/endoscapes \
        --manifest-path metadata/endoscapes_frames.csv \
        --out /path/to/cache/mae_b/endoscapes/train
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.transforms import build_transform_from_spec
from models.encoders import build_encoder
from models.encoders.base_encoder import BaseEncoder

# Registration side effects: importing populates the registry.
import models.encoders.dinov2_encoder  # noqa: F401
import models.encoders.dinov3_encoder  # noqa: F401
import models.encoders.mae_encoder  # noqa: F401
import models.encoders.videomae_encoder  # noqa: F401
import models.encoders.vjepa2_encoder  # noqa: F401


# Datasets differ in which key carries the input tensor and the identifier.
DATASET_KEYS = {
    "endoscapes": {"input": "image", "id": "sample_id"},
    "sages": {"input": "image", "id": "sample_id"},
    "sages_clip": {"input": "frames", "id": "target_sample_id"},
}


# --------------------------------------------------------------------------
# synthetic dataset for smoke testing
# --------------------------------------------------------------------------


class SyntheticDataset(Dataset):
    """Random tensors matching an encoder's declared input shape.

    Lets the GPU path, memory footprint, throughput and writer be exercised on a
    machine that does not yet hold the real data.
    """

    def __init__(self, encoder: BaseEncoder, length: int = 64) -> None:
        spec = encoder.preprocess_spec
        self.length = length
        self.shape = (
            (3, spec.image_size, spec.image_size)
            if encoder.modality == "image"
            else (spec.num_frames, 3, spec.image_size, spec.image_size)
        )
        self.generator = torch.Generator().manual_seed(0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        g = torch.Generator().manual_seed(index)
        return {
            "image": torch.randn(*self.shape, generator=g),
            "sample_id": f"synthetic_{index:06d}",
            "video_id": f"vid_{index // 8:03d}",
            "target": torch.zeros(3, dtype=torch.float32),
        }


# --------------------------------------------------------------------------
# dataset construction
# --------------------------------------------------------------------------


def build_dataset(args: argparse.Namespace, encoder: BaseEncoder) -> tuple[Dataset, dict[str, str]]:
    if args.smoke:
        return SyntheticDataset(encoder, length=args.smoke_size), DATASET_KEYS["endoscapes"]

    # Checked before any filesystem access so an incompatible pairing reports
    # the real problem rather than a missing-file error.
    if encoder.modality == "video" and args.dataset != "sages_clip":
        raise ValueError(
            f"Encoder {args.encoder!r} is a video model but --dataset "
            f"{args.dataset!r} yields single frames. Use --dataset sages_clip."
        )
    if encoder.modality == "image" and args.dataset == "sages_clip":
        raise ValueError(
            f"Encoder {args.encoder!r} is an image model but --dataset "
            f"sages_clip yields clips."
        )

    transform = build_transform_from_spec(encoder.preprocess_spec, train=False)

    if args.dataset == "endoscapes":
        from data.datasets import EndoscapesDataset

        dataset = EndoscapesDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split=args.split,
            mode="supervised",
            transform=transform,
        )
    elif args.dataset == "sages":
        from data.sages_datasets import SAGESFrameDataset

        dataset = SAGESFrameDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split=args.split,
            mode="supervised",
            transform=transform,
        )
    elif args.dataset == "sages_clip":
        from data.sages_sequence_datasets import SAGESClipDataset

        spec = encoder.preprocess_spec
        dataset = SAGESClipDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split=args.split,
            mode="supervised",
            clip_length=spec.num_frames,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return dataset, DATASET_KEYS[args.dataset]


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def build_manifest(args: argparse.Namespace, encoder: BaseEncoder, count: int) -> dict[str, Any]:
    """Record everything that determines the cache contents.

    Two caches whose manifests differ were not produced under an identical
    protocol, and adaptation gains computed across them are not comparable.
    ``train_probe_cached.py`` checks this rather than assuming it.
    """
    return {
        "encoder": encoder.describe(),
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": count,
        "transform": {
            "type": "deterministic_eval",
            **asdict(encoder.preprocess_spec),
        },
        "extraction": {
            "batch_size": args.batch_size,
            "amp_dtype": "float16" if args.amp else "float32",
            "cache_dtype": "float16",
            "reduction": args.reduction,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python": platform.python_version(),
            "git_commit": git_commit(),
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------
# reduction
# --------------------------------------------------------------------------


def apply_reduction(tokens: torch.Tensor, encoder: BaseEncoder, mode: str) -> torch.Tensor:
    """Optionally shrink the cache at the cost of what the head can learn.

    none     full grid; required for spatial attentive pooling
    spatial  mean over H,W keeping the temporal axis; temporal MIL only
    full     mean over all tokens; linear probe only, irreversible
    """
    if mode == "none":
        return tokens

    layout = encoder.token_layout
    if mode == "full":
        return tokens.mean(dim=1, keepdim=True)

    if mode == "spatial":
        if not layout.is_spatiotemporal:
            return tokens.mean(dim=1, keepdim=True)
        t, h, w = layout.grid
        b, n, d = tokens.shape
        return tokens.reshape(b, t, h * w, d).mean(dim=2)

    raise ValueError(f"Unknown reduction: {mode}")


def reduced_token_count(encoder: BaseEncoder, mode: str) -> int:
    layout = encoder.token_layout
    if mode == "none":
        return layout.num_tokens
    if mode == "full":
        return 1
    if mode == "spatial":
        return layout.grid[0] if layout.is_spatiotemporal else 1
    raise ValueError(f"Unknown reduction: {mode}")


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def iter_batches(loader: DataLoader, keys: dict[str, str]) -> Iterator[tuple[torch.Tensor, list[str], list[str], torch.Tensor | None]]:
    for batch in loader:
        inputs = batch.get(keys["input"], batch.get("image"))
        ids = batch.get(keys["id"], batch.get("sample_id"))
        video_ids = batch.get("video_id", ["" for _ in ids])
        if torch.is_tensor(video_ids):
            video_ids = [str(v.item()) for v in video_ids]
        yield inputs, [str(i) for i in ids], [str(v) for v in video_ids], batch.get("target")


def extract(args: argparse.Namespace) -> int:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("WARNING: CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)

    encoder_kwargs: dict[str, Any] = {}
    if args.model_name:
        encoder_kwargs["model_name"] = args.model_name
    if args.checkpoint:
        if args.random_init:
            raise SystemExit("--checkpoint and --random-init are mutually exclusive.")
        encoder_kwargs["adapted_checkpoint"] = args.checkpoint
    if args.random_init:
        encoder_kwargs["random_init"] = True
        print("WARNING: --random-init uses untrained weights with real architecture "
              "dimensions. Throughput and memory are representative; features are not.",
              file=sys.stderr)
    try:
        encoder = build_encoder(args.encoder, **encoder_kwargs)
    except TypeError as exc:
        if args.checkpoint and "adapted_checkpoint" in str(exc):
            raise SystemExit(
                f"Encoder {args.encoder!r} does not support --checkpoint. Only "
                f"encoders with a continued-pretraining path accept it."
            ) from exc
        raise
    encoder = encoder.to(device)
    if not encoder.is_frozen:
        encoder.freeze()

    dataset, keys = build_dataset(args, encoder)
    n = len(dataset)

    layout = encoder.token_layout
    n_tokens = reduced_token_count(encoder, args.reduction)
    dim = layout.dim

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gib = n * n_tokens * dim * 2 / 1024**3
    print(f"encoder      {args.encoder}  ({encoder.checkpoint_id})")
    print(f"grid         {layout.grid}  ->  {n_tokens} tokens x {dim} dim  [{args.reduction}]")
    print(f"samples      {n}")
    print(f"cache size   {gib:.2f} GiB")
    print(f"device       {device}  amp={'fp16' if args.amp else 'off'}")
    print(f"output       {out_dir}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    tokens_path = out_dir / "tokens.npy"
    tokens_mm = np.lib.format.open_memmap(
        tokens_path, mode="w+", dtype=np.float16, shape=(n, n_tokens, dim)
    )
    prefix_mm = None
    if layout.num_prefix_tokens and args.reduction == "none":
        prefix_mm = np.lib.format.open_memmap(
            out_dir / "prefix.npy",
            mode="w+",
            dtype=np.float16,
            shape=(n, layout.num_prefix_tokens, dim),
        )

    targets = np.zeros((n, 3), dtype=np.float32)
    rows: list[tuple[str, str, int]] = []

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=False,
    )

    autocast = torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cursor = 0
    warmup_done_at = None
    started = time.time()
    with torch.inference_mode():
        for inputs, ids, video_ids, target in iter_batches(loader, keys):
            inputs = inputs.to(device, non_blocking=True)
            with autocast:
                out = encoder(inputs)

            reduced = apply_reduction(out.tokens.float(), encoder, args.reduction)
            size = reduced.shape[0]

            tokens_mm[cursor : cursor + size] = reduced.half().cpu().numpy()
            if prefix_mm is not None and out.prefix is not None:
                prefix_mm[cursor : cursor + size] = out.prefix.float().half().cpu().numpy()
            if target is not None:
                targets[cursor : cursor + size] = target.numpy()

            for offset, (sample_id, video_id) in enumerate(zip(ids, video_ids)):
                rows.append((sample_id, video_id, cursor + offset))

            cursor += size
            if warmup_done_at is None:
                # Exclude CUDA context creation and kernel autotuning from the
                # steady-state rate, which is the figure that transfers to
                # other hardware.
                warmup_done_at = (time.time(), cursor)
            if args.log_every and (cursor // args.batch_size) % args.log_every == 0:
                rate = cursor / max(time.time() - started, 1e-6)
                remaining = (n - cursor) / max(rate, 1e-6)
                print(f"  {cursor:>7}/{n}  {rate:7.1f} samples/s  eta {remaining/60:5.1f} min")

    if cursor != n:
        raise RuntimeError(
            f"Wrote {cursor} rows but the dataset declares {n}. The cache is "
            f"incomplete and must not be used."
        )

    tokens_mm.flush()
    if prefix_mm is not None:
        prefix_mm.flush()
    np.save(out_dir / "targets.npy", targets)

    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "video_id", "row"])
        writer.writerows(rows)

    manifest = build_manifest(args, encoder, cursor)
    manifest["shapes"] = {
        "tokens": [n, n_tokens, dim],
        "prefix": None if prefix_mm is None else [n, layout.num_prefix_tokens, dim],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    elapsed = time.time() - started
    print(f"\nWrote {cursor} samples in {elapsed/60:.2f} min ({cursor/elapsed:.1f}/s overall)")

    if warmup_done_at is not None and cursor > warmup_done_at[1]:
        warm_elapsed = time.time() - warmup_done_at[0]
        warm_samples = cursor - warmup_done_at[1]
        print(f"Steady state: {warm_samples/max(warm_elapsed, 1e-6):.1f} samples/s "
              f"(first batch excluded)")

    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"Peak VRAM:    {peak_alloc:.2f} GiB allocated, "
              f"{peak_reserved:.2f} GiB reserved, of {total:.1f} GiB")
        if peak_reserved > total:
            print("              WARNING: reserved exceeds device memory. Under "
                  "WSL2 the driver spills to host memory over PCIe instead of "
                  "raising OOM, which degrades throughput several-fold. Reduce "
                  "--batch-size; on native Linux this would have failed outright.")
        print("              A single run cannot separate fixed overhead from "
              "per-sample cost. Sweep --batch-size and fit "
              "peak = fixed + marginal * batch_size.")

    print(f"Manifest: {out_dir / 'manifest.json'}")
    return 0


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--encoder", required=True, help="registry name, e.g. mae_b, videomae_b, vjepa2_l")
    p.add_argument("--model-name", default=None, help="explicit checkpoint id, overriding the variant default")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to an encoder produced by continued pretraining, e.g. "
        "outputs/ssl/videomae_b_sages/encoder_final.pt. This is the adapted arm; "
        "without it the encoder is the published baseline checkpoint",
    )
    p.add_argument("--dataset", default="endoscapes", choices=sorted(DATASET_KEYS))
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--out", required=True)

    p.add_argument(
        "--reduction",
        default="none",
        choices=["none", "spatial", "full"],
        help="none keeps the full grid and is required for spatial attentive pooling",
    )

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True, help="fp16 autocast (V100-safe; bf16 is not)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-every", type=int, default=10, help="log every N batches; 0 disables")

    p.add_argument("--smoke", action="store_true", help="synthetic inputs; no dataset needed")
    p.add_argument(
        "--random-init",
        action="store_true",
        help="build the encoder from config with real architecture dimensions "
        "instead of downloading weights; for throughput and memory testing",
    )
    p.add_argument("--smoke-size", type=int, default=64)
    p.add_argument("--dry-run", action="store_true", help="report shapes and cache size, write nothing")

    args = p.parse_args()
    if not args.smoke and not args.dry_run:
        missing = [f for f in ("dataset_root", "manifest_path") if getattr(args, f) is None]
        if missing:
            p.error(f"--{' and --'.join(m.replace('_', '-') for m in missing)} required without --smoke")
    return args


if __name__ == "__main__":
    raise SystemExit(extract(parse_args()))