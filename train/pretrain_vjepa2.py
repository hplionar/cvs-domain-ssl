#!/usr/bin/env python3
"""Continued V-JEPA 2 pretraining on surgical video — experiment E2.2.

The counterpart to `pretrain_videomae.py`. Where VideoMAE reconstructs masked
pixels, this predicts the *representations* of masked regions, which is the
distinction the whole comparison rests on.

Why this file exists at all
---------------------------
`transformers` ships `VJEPA2Model` but no `VJEPA2ForPreTraining`. The forward
path is there — the predictor correctly consumes only context tokens, and its
source comments that "in VJEPA training a separate encoder is used for target" —
but `VJEPA2Model.forward` computes

    target_hidden_state = apply_masks(sequence_output, target_mask)

from the *same* encoder, in the same pass, with gradients flowing. Trained that
way the objective collapses: the model minimises the loss by driving all
representations to a constant. The stop-gradient EMA target is not an
optimisation detail in JEPA, it is what makes the objective well-posed.

This module supplies the missing pieces: an EMA target encoder, multi-block 3D
masking, and the latent prediction loss.

Asymmetry with the VideoMAE arm
-------------------------------
VideoMAE discards its decoder and reinitialises it by design, so continued
pretraining starts from a state the checkpoint anticipates. Here the *predictor*
comes from the checkpoint but the EMA target encoder has no saved counterpart
and must be initialised from the context encoder. The teacher therefore begins
identical to the student rather than carrying its own pretraining history. This
is inherent to resuming JEPA from a released checkpoint and is recorded in the
run manifest; see docs/experimental_design.md §5.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.ssl_clip_dataset import ClipTransform, SSLClipDataset
from train.pretrain_videomae import (
    RequeueHandler,
    TrainState,
    cosine_with_warmup,
    set_seed,
)


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


class MultiBlockMaskGenerator:
    """V-JEPA multi-block 3D masking.

    Samples rectangular blocks spanning the full temporal extent of the token
    grid and unions them into a target region; the complement is the context.

    Blocks rather than independent random tokens: with independent sampling the
    context almost always contains a spatial neighbour of every target token, so
    prediction reduces to local interpolation. Contiguous blocks force the model
    to infer a region from its surroundings, which is what makes the target a
    *representation* problem rather than a smoothing one.

    Blocks span all temporal positions for the same reason VideoMAE uses tube
    masking: on redundant video, an unmasked frame at the same spatial location
    makes the target visible.
    """

    def __init__(
        self,
        grid: tuple[int, int, int],
        *,
        num_blocks: int = 8,
        scale: tuple[float, float] = (0.15, 0.2),
        aspect: tuple[float, float] = (0.75, 1.5),
        min_context: int = 16,
        max_attempts: int = 20,
    ) -> None:
        self.temporal, self.height, self.width = grid
        self.num_blocks = num_blocks
        self.scale = scale
        self.aspect = aspect
        self.min_context = min_context
        self.max_attempts = max_attempts
        self.num_tokens = self.temporal * self.height * self.width

    def _sample_block(self, rng: random.Random) -> tuple[int, int, int, int]:
        area = self.height * self.width
        target_area = rng.uniform(*self.scale) * area
        ratio = rng.uniform(*self.aspect)

        block_h = max(1, min(self.height, int(round(math.sqrt(target_area * ratio)))))
        block_w = max(1, min(self.width, int(round(math.sqrt(target_area / ratio)))))
        top = rng.randint(0, self.height - block_h)
        left = rng.randint(0, self.width - block_w)
        return top, left, block_h, block_w

    def __call__(self, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(context_indices, target_indices)`` as 1-D long tensors."""
        for _ in range(self.max_attempts):
            spatial = torch.zeros(self.height, self.width, dtype=torch.bool)
            for _ in range(self.num_blocks):
                top, left, block_h, block_w = self._sample_block(rng)
                spatial[top : top + block_h, left : left + block_w] = True

            # Blocks span the temporal axis, as tube masking does.
            mask = spatial.flatten().repeat(self.temporal)
            context = (~mask).nonzero(as_tuple=True)[0]
            target = mask.nonzero(as_tuple=True)[0]

            if context.numel() >= self.min_context and target.numel() > 0:
                return context, target

        # Sampling can fail to leave enough context when blocks overlap little;
        # falling back to a fixed split is preferable to raising mid-epoch.
        split = self.num_tokens // 2
        order = torch.randperm(self.num_tokens, generator=torch.Generator().manual_seed(0))
        return order[:split].sort().values, order[split:].sort().values

    def batch(self, size: int, rng: random.Random) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Masks for a batch.

        Every sample in the batch shares one mask. V-JEPA's implementation does
        the same: the predictor sorts and gathers by position, and per-sample
        masks of differing lengths cannot be batched into a single tensor.
        """
        context, target = self(rng)
        return (
            [context.unsqueeze(0).expand(size, -1)],
            [target.unsqueeze(0).expand(size, -1)],
        )


# --------------------------------------------------------------------------
# EMA target encoder
# --------------------------------------------------------------------------


class TargetEncoder(nn.Module):
    """Exponential moving average of the context encoder, updated without gradient.

    The stop-gradient is what prevents collapse. If gradients reached the target,
    the loss could be minimised by making every representation identical, which
    is exactly what `VJEPA2Model.forward` permits by deriving the target from the
    same forward pass.
    """

    def __init__(self, encoder: nn.Module, momentum: float = 0.998) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
        self.momentum = momentum

    @torch.no_grad()
    def update(self, source: nn.Module, momentum: float | None = None) -> None:
        m = self.momentum if momentum is None else momentum
        for target_param, source_param in zip(self.encoder.parameters(), source.parameters()):
            target_param.mul_(m).add_(source_param.detach(), alpha=1.0 - m)
        # Buffers (normalisation statistics, position tables) are copied rather
        # than averaged; averaging integer or non-float buffers is undefined.
        for target_buf, source_buf in zip(self.encoder.buffers(), source.buffers()):
            target_buf.copy_(source_buf)

    @torch.no_grad()
    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values_videos=pixel_values_videos).last_hidden_state

    def train(self, mode: bool = True) -> "TargetEncoder":
        # Always eval: the target must be a deterministic function of its input.
        return super().train(False)


def momentum_schedule(step: int, total: int, start: float = 0.998, end: float = 1.0) -> float:
    """Momentum rising toward 1 over training.

    A faster-moving teacher early on lets the target track the student while
    both are changing quickly; a nearly frozen teacher later stabilises the
    objective. This is the schedule used by V-JEPA and BYOL.
    """
    return end - (end - start) * (math.cos(math.pi * min(step / max(total, 1), 1.0)) + 1) / 2


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------


def jepa_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 between predicted and target representations.

    L1 rather than L2, following V-JEPA: squared error is dominated by the few
    dimensions with the largest activations, and the objective is agreement
    across the representation rather than on its loudest components.
    """
    return F.smooth_l1_loss(prediction, target.detach())


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    target: TargetEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    state: TrainState,
    config: dict[str, Any],
) -> None:
    """Write a resumable checkpoint.

    The EMA target encoder is saved alongside everything else. Omitting it would
    reset the teacher to the student on every resume, discarding the averaging
    history and producing a loss discontinuity at each job boundary — the same
    class of error as dropping the GradScaler state, and harder to notice.
    """
    payload = {
        "model": model.state_dict(),
        "target_encoder": target.encoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "history": state.history,
        "config": config,
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
    target: TargetEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> TrainState:
    payload = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(payload["model"])
    if "target_encoder" not in payload:
        raise KeyError(
            "Checkpoint has no target_encoder. Resuming would reinitialise the "
            "teacher from the student and discard the EMA history."
        )
    target.encoder.load_state_dict(payload["target_encoder"])
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
# training
# --------------------------------------------------------------------------


def build_model(config: dict[str, Any], device: torch.device) -> tuple[nn.Module, TargetEncoder]:
    from transformers import VJEPA2Model

    model = VJEPA2Model.from_pretrained(config["model"]["checkpoint"])
    if config["model"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    model = model.to(device)

    target = TargetEncoder(
        model.encoder, momentum=config["train"].get("ema_momentum", 0.998)
    ).to(device)
    return model, target


def build_dataset(config: dict[str, Any]) -> SSLClipDataset:
    data = config["data"]
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
        exclude_video_ids=data.get("exclude_video_ids", []),
        seed=config.get("seed", 0),
        decode_size=data.get("decode_size", 256),
        reader_cache_size=data.get("reader_cache_size", 4),
        limit_videos=data.get("limit_videos"),
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    for key, value in (args.override or []):
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

    model, target = build_model(config, device)
    cfg = model.config

    image_size = getattr(cfg, "crop_size", None) or cfg.image_size
    num_frames = getattr(cfg, "frames_per_clip", None) or cfg.num_frames
    grid = (
        num_frames // cfg.tubelet_size,
        image_size // cfg.patch_size,
        image_size // cfg.patch_size,
    )
    masker = MultiBlockMaskGenerator(
        grid,
        num_blocks=train.get("num_blocks", 8),
        scale=tuple(train.get("block_scale", (0.15, 0.2))),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train["lr"],
        weight_decay=train.get("weight_decay", 0.04),
        betas=tuple(train.get("betas", (0.9, 0.95))),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    steps_per_epoch = len(loader)
    total_steps = train.get("max_steps") or steps_per_epoch * train["epochs"]
    warmup_steps = train.get("warmup_steps", int(0.05 * total_steps))

    latest = output_dir / "latest.pt"
    if latest.is_file() and not args.no_resume:
        state = load_checkpoint(latest, model=model, target=target,
                                optimizer=optimizer, scaler=scaler, device=device)
        print(f"Resumed from {latest} at step {state.step}, epoch {state.epoch}")
    else:
        state = TrainState()

    info = dataset.describe()
    print("=" * 68)
    print(f"checkpoint    {config['model']['checkpoint']}")
    print(f"videos        {info['num_videos']}  clips {info['num_clips']}")
    if info.get("limit_videos"):
        print(f"              LIMITED to {info['limit_videos']} videos (not a real run)")
    print(f"token grid    {grid}  ->  {grid[0]*grid[1]*grid[2]} tokens")
    sample_ctx, sample_tgt = masker(random.Random(0))
    print(f"masking       {train.get('num_blocks', 8)} blocks  ->  "
          f"{sample_ctx.numel()} context / {sample_tgt.numel()} target "
          f"({100*sample_tgt.numel()/(grid[0]*grid[1]*grid[2]):.0f}% masked)")
    print(f"batch         {train['batch_size']}  ->  {steps_per_epoch} steps/epoch")
    print(f"schedule      {total_steps} steps, {warmup_steps} warmup")
    print(f"ema           {train.get('ema_momentum', 0.998)} -> 1.0 (cosine)")
    print(f"device        {device}  amp=fp16")
    print(f"output        {output_dir}")
    print(f"start         step {state.step}")
    print("=" * 68, flush=True)

    if args.dry_run:
        print("--dry-run: nothing trained.")
        return 0

    requeue = RequeueHandler()
    mask_rng = random.Random(seed + state.step)
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

            clips = batch["pixel_values"].to(device, non_blocking=True)
            context_mask, target_mask = masker.batch(clips.shape[0], mask_rng)
            context_mask = [m.to(device) for m in context_mask]
            target_mask = [m.to(device) for m in target_mask]

            scale = cosine_with_warmup(state.step, warmup_steps, total_steps,
                                       train.get("min_lr_ratio", 0.0))
            for group in optimizer.param_groups:
                group["lr"] = train["lr"] * scale

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                # Target: EMA encoder over the full clip, no gradient.
                with torch.no_grad():
                    full = target(clips)
                    targets = torch.gather(
                        full, 1,
                        target_mask[0].unsqueeze(-1).expand(-1, -1, full.shape[-1]),
                    )
                    # Normalising the target is what V-JEPA relies on to keep
                    # the objective from being satisfied by scale alone.
                    targets = F.layer_norm(targets, (targets.shape[-1],))

                out = model(pixel_values_videos=clips,
                            context_mask=context_mask, target_mask=target_mask)
                prediction = out.predictor_output.last_hidden_state
                loss = jepa_loss(prediction, targets)

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

            momentum = momentum_schedule(
                state.step, total_steps, start=train.get("ema_momentum", 0.998)
            )
            target.update(model.encoder, momentum=momentum)

            state.step += 1
            window_loss += loss.item()
            window_count += 1

            if state.step % log_every == 0:
                mean_loss = window_loss / max(window_count, 1)
                elapsed = time.time() - started
                rate = state.step / max(elapsed, 1e-6)
                state.history.append({
                    "step": state.step, "epoch": state.epoch,
                    "loss": round(mean_loss, 5),
                    "lr": round(train["lr"] * scale, 8),
                    "ema": round(momentum, 6),
                    "scale": scaler.get_scale(),
                })
                print(f"step {state.step:>7}/{total_steps}  loss {mean_loss:.4f}  "
                      f"lr {train['lr']*scale:.2e}  ema {momentum:.5f}  "
                      f"{rate:.2f} it/s  eta {(total_steps-state.step)/max(rate,1e-6)/3600:.1f}h",
                      flush=True)
                window_loss, window_count = 0.0, 0

            if state.step % save_every == 0:
                save_checkpoint(latest, model=model, target=target, optimizer=optimizer,
                                scaler=scaler, state=state, config=config)

    save_checkpoint(latest, model=model, target=target, optimizer=optimizer,
                    scaler=scaler, state=state, config=config)
    with open(output_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(state.history, fh, indent=2)

    complete = state.step >= total_steps
    if complete:
        encoder_path = output_dir / "encoder_final.pt"
        torch.save({"model": model.encoder.state_dict(), "config": config,
                    "step": state.step}, encoder_path)
        print(f"\nTraining complete at step {state.step}. Encoder: {encoder_path}")
    else:
        print(f"\nStopped at step {state.step}/{total_steps} "
              f"({requeue.reason or 'loop exit'}). Resubmit to resume.")

    elapsed = time.time() - started
    print(f"Elapsed {elapsed/3600:.2f}h  |  {state.step} steps  |  "
          f"{state.step/max(elapsed,1e-6):.2f} it/s")
    if device.type == "cuda":
        print(f"Peak VRAM {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GiB")

    return 0 if complete else 99


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--override", nargs=2, action="append", metavar=("KEY", "VALUE"))
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())