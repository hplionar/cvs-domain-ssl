#!/usr/bin/env python3
"""Continued VideoMAE pretraining on surgical video.

Resumes the masked-reconstruction objective from a Kinetics-pretrained
checkpoint using unlabelled surgical clips. This is arm E2.1 of the objective
comparison.

Resumability
------------
Jobs are bounded by the cluster walltime, so a run spans several jobs and must
resume exactly. The checkpoint therefore carries more than model weights:

    model, optimizer, GradScaler, LR scheduler, global step, epoch, RNG states

Omitting the ``GradScaler`` state is the subtle one — the loss scale would
re-warm from its initial value on every resume, producing a loss spike at each
job boundary that is indistinguishable from genuine training instability, and
costing several hundred wasted steps each time.

Precision
---------
fp16 autocast with ``GradScaler`` throughout. Not bf16: the target hardware is
Volta (sm_70), which supports only fp16 and fp32, and code developed on a newer
local GPU will run bf16 successfully and then fail on the cluster.

Usage
-----
    python train/pretrain_videomae.py --config configs/ssl/videomae_b_sages.yaml
    python train/pretrain_videomae.py --config ... --max-steps 100 --timing-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.ssl_clip_dataset import ClipTransform, SSLClipDataset


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


class TubeMaskGenerator:
    """VideoMAE tube masking: one spatial mask shared by all temporal positions.

    Not independent random masking per frame. With a high mask ratio and
    temporally redundant video, an unmasked patch in an adjacent frame makes
    reconstruction near-trivial, and the objective degenerates into copying.
    Masking the same spatial location across the whole tube removes that
    shortcut, which is the design VideoMAE reports as necessary at 90% masking.
    """

    def __init__(self, temporal: int, spatial: int, mask_ratio: float = 0.9) -> None:
        self.temporal = temporal
        self.spatial = spatial
        self.num_masked = int(round(mask_ratio * spatial))
        if not 0 < self.num_masked < spatial:
            raise ValueError(
                f"mask_ratio {mask_ratio} masks {self.num_masked} of {spatial} "
                f"spatial positions; must leave at least one visible."
            )

    def __call__(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.rand(self.spatial, generator=generator)
        keep = noise.argsort()[self.num_masked :]
        spatial_mask = torch.ones(self.spatial, dtype=torch.bool)
        spatial_mask[keep] = False
        return spatial_mask.repeat(self.temporal)

    def batch(self, size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.stack([self(generator) for _ in range(size)])


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def cosine_with_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    history: list[dict[str, float]] = field(default_factory=list)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    state: TrainState,
    config: dict[str, Any],
) -> None:
    """Write a checkpoint that restores training exactly.

    Written to a temporary path and renamed, so a job killed mid-write leaves
    the previous checkpoint intact rather than a truncated file that fails to
    load on resume.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "history": state.history,
        "config": config,
        "bias_repair": getattr(save_checkpoint, "_bias_repair", None),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> TrainState:
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])

    rng = payload.get("rng", {})
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])

    return TrainState(
        step=payload["step"],
        epoch=payload["epoch"],
        best_loss=payload.get("best_loss", float("inf")),
        history=payload.get("history", []),
    )


# --------------------------------------------------------------------------
# signal handling
# --------------------------------------------------------------------------


class RequeueHandler:
    """Sets a flag on SIGUSR1 or SIGTERM so the loop exits after a clean save.

    Slurm can be asked to signal a job before its walltime expires. Saving at
    that point costs one checkpoint write; being killed instead costs whatever
    progress was made since the last periodic save.
    """

    def __init__(self) -> None:
        self.should_stop = False
        self.reason = ""
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass  # not available on this platform or thread

    def _handle(self, signum: int, frame) -> None:  # noqa: ANN001
        self.should_stop = True
        self.reason = signal.Signals(signum).name
        print(f"\n[signal] {self.reason} received; will checkpoint and exit.", flush=True)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load the checkpoint and repair its attention biases.

    The repair is not optional here. VideoMAE stores attention bias in the BEiT
    layout (``q_bias``/``v_bias``, key bias fixed at zero) while transformers
    5.x expects ``query/key/value.bias`` and ships no conversion, so
    ``from_pretrained`` silently replaces trained biases with newly initialised
    ones.

    Applying it in the wrapper but not here would be worse than skipping it
    entirely: the E1 baseline would use the published weights while the E2
    adapted run started from different ones, and adaptation gain — a difference
    between the two — would absorb that discrepancy.
    """
    from transformers import VideoMAEForPreTraining

    from models.encoders.videomae_encoder import repair_qkv_bias

    checkpoint = config["model"]["checkpoint"]
    model = VideoMAEForPreTraining.from_pretrained(checkpoint)

    repair = repair_qkv_bias(model, checkpoint)
    if repair.get("status") == "failed":
        raise RuntimeError(
            f"Attention bias repair failed for {checkpoint}: {repair}. Training "
            f"from an encoder that does not match the published checkpoint would "
            f"invalidate the adaptation-gain comparison."
        )

    if config["model"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model.to(device), repair


def build_dataset(config: dict[str, Any]) -> SSLClipDataset:
    data = config["data"]
    excluded = data.get("exclude_video_ids", [])

    # augment=False gives a deterministic centre crop, which the overfitting
    # check needs: with random crops the model never sees the same input twice
    # and cannot memorise, so a flat loss would be uninformative.
    transform = ClipTransform(
        image_size=data.get("image_size", 224),
        scale=tuple(data.get("crop_scale", (0.5, 1.0))),
        train=data.get("augment", True),
    )
    return SSLClipDataset(
        data["video_dir"],
        num_frames=data.get("num_frames", 16),
        stride=data.get("stride", 4),
        clips_per_video=data.get("clips_per_video", 40),
        transform=transform,
        exclude_video_ids=excluded,
        seed=config.get("seed", 0),
        decode_size=data.get("decode_size", 256),
        reader_cache_size=data.get("reader_cache_size", 4),
        limit_videos=data.get("limit_videos"),
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    for key, value in (args.override or {}):
        node = config
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)

    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 0)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device; running on CPU.", file=sys.stderr)

    # -- data -------------------------------------------------------------

    dataset = build_dataset(config)
    train = config["train"]
    loader = DataLoader(
        dataset,
        batch_size=train["batch_size"],
        shuffle=True,
        num_workers=train.get("num_workers", 8),
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=train.get("num_workers", 8) > 0,
        prefetch_factor=train.get("prefetch_factor", 4) if train.get("num_workers", 8) > 0 else None,
    )

    # -- model ------------------------------------------------------------

    model, bias_repair = build_model(config, device)
    model_config = model.config
    temporal = model_config.num_frames // model_config.tubelet_size
    spatial = (model_config.image_size // model_config.patch_size) ** 2
    masker = TubeMaskGenerator(temporal, spatial, mask_ratio=train.get("mask_ratio", 0.9))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train["lr"],
        weight_decay=train.get("weight_decay", 0.05),
        betas=tuple(train.get("betas", (0.9, 0.95))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    steps_per_epoch = len(loader)
    total_steps = train.get("max_steps") or steps_per_epoch * train["epochs"]
    warmup_steps = train.get("warmup_steps", int(0.05 * total_steps))

    # -- resume -----------------------------------------------------------

    latest = output_dir / "latest.pt"
    if latest.is_file() and not args.no_resume:
        state = load_checkpoint(latest, model=model, optimizer=optimizer, scaler=scaler, device=device)
        print(f"Resumed from {latest} at step {state.step}, epoch {state.epoch}")
    else:
        state = TrainState()

    # -- report -----------------------------------------------------------

    info = dataset.describe()
    print("=" * 68)
    print(f"checkpoint    {config['model']['checkpoint']}")
    print(f"videos        {info['num_videos']}  (excluded {len(info['excluded_video_ids'])})")
    print(f"clips         {info['num_clips']}  "
          f"({info['clips_per_video']}/video, {info['num_frames']}f @ stride {info['stride']}, "
          f"span {info['clip_span_frames']} frames)")
    if info.get("limit_videos"):
        print(f"              LIMITED to {info['limit_videos']} videos "
              f"(overfitting check; not a real run)")
    if not config["data"].get("augment", True):
        print("              augmentation OFF (deterministic centre crop)")
    print(f"decode        {info['decode_size']}px  ->  "
          f"{train['batch_size'] * info['num_frames'] * 3 * info['decode_size']**2 / 1024**2:.0f} MiB/batch uint8")
    if info["videos_with_overlapping_clips"]:
        print(f"              WARNING: {info['videos_with_overlapping_clips']} videos have "
              f"overlapping clips; quota exceeds non-overlapping capacity")
    print(f"batch         {train['batch_size']}  ->  {steps_per_epoch} steps/epoch")
    print(f"schedule      {total_steps} total steps, {warmup_steps} warmup, "
          f"mask ratio {train.get('mask_ratio', 0.9)}")
    print(f"bias repair   {bias_repair.get('status')}"
          + (f" ({bias_repair.get('num_repaired')} tensors)"
             if bias_repair.get("num_repaired") else ""))
    print(f"device        {device}  amp=fp16")
    print(f"output        {output_dir}")
    print(f"start         step {state.step}")
    print("=" * 68, flush=True)

    if args.dry_run:
        print("--dry-run: nothing trained.")
        return 0

    # -- loop -------------------------------------------------------------

    save_checkpoint._bias_repair = bias_repair  # recorded in every checkpoint

    requeue = RequeueHandler()
    mask_generator = torch.Generator().manual_seed(seed + state.step)
    save_every = train.get("save_every_steps", 500)
    log_every = train.get("log_every_steps", 20)

    model.train()
    started = time.time()
    window_loss, window_count = 0.0, 0

    while state.step < total_steps and not requeue.should_stop:
        state.epoch += 1
        for batch in loader:
            if state.step >= total_steps or requeue.should_stop:
                break

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            mask = masker.batch(pixel_values.shape[0], mask_generator).to(device)

            scale = cosine_with_warmup(state.step, warmup_steps, total_steps,
                                       train.get("min_lr_ratio", 0.0))
            for group in optimizer.param_groups:
                group["lr"] = train["lr"] * scale

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(pixel_values=pixel_values, bool_masked_pos=mask).loss

            if not torch.isfinite(loss):
                print(f"[step {state.step}] non-finite loss; skipping batch", flush=True)
                optimizer.zero_grad(set_to_none=True)
                state.step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if train.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            state.step += 1
            window_loss += loss.item()
            window_count += 1

            if state.step % log_every == 0:
                mean_loss = window_loss / max(window_count, 1)
                elapsed = time.time() - started
                rate = state.step / max(elapsed, 1e-6)
                remaining = (total_steps - state.step) / max(rate, 1e-6)
                row = {
                    "step": state.step,
                    "epoch": state.epoch,
                    "loss": round(mean_loss, 5),
                    "lr": round(train["lr"] * scale, 8),
                    "scale": scaler.get_scale(),
                }
                state.history.append(row)
                print(f"step {state.step:>7}/{total_steps}  loss {mean_loss:.4f}  "
                      f"lr {train['lr']*scale:.2e}  scale {scaler.get_scale():.0f}  "
                      f"{rate:.2f} it/s  eta {remaining/3600:.1f}h", flush=True)
                window_loss, window_count = 0.0, 0

            if state.step % save_every == 0:
                save_checkpoint(latest, model=model, optimizer=optimizer,
                                scaler=scaler, state=state, config=config)

    # -- finish -----------------------------------------------------------

    save_checkpoint(latest, model=model, optimizer=optimizer, scaler=scaler,
                    state=state, config=config)
    with open(output_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(state.history, fh, indent=2)

    complete = state.step >= total_steps
    if complete:
        encoder_path = output_dir / "encoder_final.pt"
        torch.save(
            {"model": model.videomae.state_dict(), "config": config, "step": state.step},
            encoder_path,
        )
        print(f"\nTraining complete at step {state.step}. Encoder: {encoder_path}")
    else:
        print(f"\nStopped at step {state.step}/{total_steps} "
              f"({requeue.reason or 'loop exit'}). Resume with the same command.")

    elapsed = time.time() - started
    print(f"Elapsed {elapsed/3600:.2f}h  |  {state.step} steps  |  "
          f"{state.step/max(elapsed,1e-6):.2f} it/s")
    if device.type == "cuda":
        print(f"Peak VRAM {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GiB")

    # Exit 0 when finished, 99 when more work remains, so a chained job can
    # distinguish completion from a walltime stop without parsing logs.
    return 0 if complete else 99


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="report configuration, train nothing")
    p.add_argument("--override", nargs=2, action="append", metavar=("KEY", "VALUE"),
                   help="override a config value, e.g. --override train.batch_size 4")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())