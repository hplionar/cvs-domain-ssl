"""ViT-MAE continued pretraining on surgical frames — experiment E2.3.

The image-modality counterpart to ``pretrain_videomae.py``. Same checkpoint
contract, same resume semantics, same signal handling; the differences are that
the corpus is frames rather than clips and that masking happens inside the
model.

Masking
-------
``ViTMAEForPreTraining`` performs its own random masking in ``forward`` at
``config.mask_ratio``, so there is no mask generator here. This is the opposite
of the situation in ``models/encoders/mae_encoder.py``, which has to *disable*
that masking for feature extraction: correct behaviour during pretraining,
catastrophic during extraction.

The published ratio is 0.75, retained unmodified. VideoMAE uses 0.90 because
adjacent video frames are highly redundant and a lower ratio makes the task
trivial; single frames have no such redundancy.

Augmentation
------------
Uses ``build_transform_from_spec(spec, train=True)`` rather than
``build_ssl_train_transform``. The latter applies ColorJitter with hue 0.05 at
p=0.8 plus RandomGrayscale, and ``data/transforms.py`` records that hue
perturbation degrades the tissue-colour cue CVS criterion C2 depends on.

That is not a hypothetical concern here. C2 is the highest-scoring criterion in
all eight SAGES arms measured so far, so it carries a substantial share of the
signal, and augmenting the colour cue away during pretraining risks damaging
exactly what works. Set ``data.strong_augment: true`` to use the stronger recipe
if that is worth testing as its own arm.

Learning rate
-------------
MAE's linear scaling rule is ``blr * batch_size / 256`` with a base of 1.5e-4.
At batch 64 that is 3.75e-5. The config should carry the scaled value, not the
base: the same mistake cost a full run on VideoMAE, where 1.5e-4 was used flat
at batch 4 and the loss rose with the learning rate then returned to its
starting value as it decayed. Net progress over 700 steps was nil.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.ssl_frame_dataset import SSLFrameDataset
from data.transforms import build_ssl_train_transform, build_transform_from_spec


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------


def cosine_with_warmup(step: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base rate."""
    if step < warmup:
        return step / max(warmup, 1)
    if total <= warmup:
        return min_ratio
    progress = (step - warmup) / (total - warmup)
    cosine = 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


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
        torch.set_rng_state(
            rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"]
        )
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

    Slurm sends SIGUSR1 before the walltime with ``--signal=B:USR1@600``.
    Exiting at a step boundary with a written checkpoint is the difference
    between resuming and repeating the work.
    """

    def __init__(self) -> None:
        self.should_stop = False
        signal.signal(signal.SIGUSR1, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:  # noqa: ANN001, ARG002
        print(f"\n[signal] {signal.Signals(signum).name} received; "
              f"will checkpoint and exit.", flush=True)
        self.should_stop = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    """Load the checkpoint for continued pretraining.

    No bias repair here, unlike VideoMAE: ViT-MAE stores attention bias under
    the names transformers expects, so ``from_pretrained`` loads it correctly.
    The mismatch that silently zeroed 24 encoder biases in the VideoMAE arm has
    no counterpart in this architecture.
    """
    from transformers import ViTMAEForPreTraining

    checkpoint = config["model"]["checkpoint"]
    model = ViTMAEForPreTraining.from_pretrained(checkpoint)

    ratio = float(config["model"].get("mask_ratio", 0.75))
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            f"mask_ratio must lie in (0, 1), got {ratio}. A ratio of 0 leaves "
            f"nothing to reconstruct and the objective becomes degenerate."
        )
    model.config.mask_ratio = ratio

    if config["model"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model.to(device)


def build_dataset(config: dict[str, Any], model) -> SSLFrameDataset:
    data = config["data"]

    image_size = int(data.get("image_size", model.config.image_size))

    if data.get("strong_augment", False):
        # The stronger recipe, including colour jitter and grayscale. Available
        # as its own arm; not the default, for the reason in the module
        # docstring.
        transform = build_ssl_train_transform(
            image_size=image_size,
            scale=tuple(data.get("crop_scale", (0.2, 1.0))),
        )
    elif data.get("augment", True):
        from models.encoders.base_encoder import PreprocessSpec

        spec = PreprocessSpec(
            image_size=image_size,
            mean=tuple(data.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(data.get("std", (0.229, 0.224, 0.225))),
            interpolation=data.get("interpolation", "bicubic"),
        )
        transform = build_transform_from_spec(spec, train=True)
    else:
        # Deterministic centre crop. The overfitting check needs this: with
        # random crops the model never sees the same input twice and cannot
        # memorise, so a flat loss would be uninformative.
        from models.encoders.base_encoder import PreprocessSpec

        spec = PreprocessSpec(
            image_size=image_size,
            mean=tuple(data.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(data.get("std", (0.229, 0.224, 0.225))),
            interpolation=data.get("interpolation", "bicubic"),
        )
        transform = build_transform_from_spec(spec, train=False)

    return SSLFrameDataset(
        data["frames_dir"],
        frames_per_video=data.get("frames_per_video"),
        transform=transform,
        exclude_video_ids=data.get("exclude_video_ids", []),
        limit_videos=data.get("limit_videos"),
    )


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    for key, value in args.override:
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = yaml.safe_load(value)
        except yaml.YAMLError:
            node[parts[-1]] = value

    seed = int(config.get("seed", 0))
    set_seed(seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device; this will be extremely slow.", flush=True)

    model = build_model(config, device)
    dataset = build_dataset(config, model)

    train_cfg = config["train"]
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 12))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=train_cfg.get("prefetch_factor", 4) if num_workers > 0 else None,
    )

    steps_per_epoch = len(loader)
    epochs = int(train_cfg.get("epochs", 20))
    total_steps = int(train_cfg.get("max_steps", steps_per_epoch * epochs))
    warmup = int(train_cfg.get("warmup_steps", max(1, total_steps // 20)))
    base_lr = float(train_cfg["lr"])
    min_ratio = float(train_cfg.get("min_lr_ratio", 0.0))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    state = TrainState()
    latest = output_dir / "latest.pt"
    if latest.is_file() and not args.no_resume:
        state = load_checkpoint(
            latest, model=model, optimizer=optimizer, scaler=scaler, device=device
        )
        print(f"Resumed from {latest} at step {state.step}.", flush=True)

    corpus = dataset.describe()
    print("=" * 68)
    print(f"checkpoint    {config['model']['checkpoint']}")
    print(f"corpus        {corpus['num_frames']} frames over "
          f"{corpus['num_videos']} videos")
    print(f"              {config['data']['frames_dir']}")
    print(f"augment       {'strong' if config['data'].get('strong_augment') else ('standard' if config['data'].get('augment', True) else 'none (deterministic)')}")
    print(f"mask ratio    {model.config.mask_ratio}")
    print(f"batch         {batch_size}  ->  {steps_per_epoch} steps/epoch")
    print(f"schedule      {total_steps} total steps, {warmup} warmup")
    print(f"lr            {base_lr:.2e}")
    print(f"device        {device}  amp={'fp16' if device.type == 'cuda' else 'off'}")
    print(f"output        {output_dir}")
    print(f"start         step {state.step}")
    print("=" * 68, flush=True)

    if args.dry_run:
        print("--dry-run: nothing trained.")
        return 0

    requeue = RequeueHandler()
    log_every = int(train_cfg.get("log_every_steps", 20))
    save_every = int(train_cfg.get("save_every_steps", 200))

    started = time.time()
    model.train()
    complete = state.step >= total_steps

    while state.step < total_steps and not requeue.should_stop:
        state.epoch += 1
        running, seen = 0.0, 0

        for pixel_values in loader:
            if state.step >= total_steps or requeue.should_stop:
                break

            pixel_values = pixel_values.to(device, non_blocking=True)

            lr = base_lr * cosine_with_warmup(state.step, warmup, total_steps, min_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(pixel_values=pixel_values).loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {state.step}. Training cannot "
                    f"continue; the checkpoint at {latest} is the last valid state."
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if train_cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_cfg["grad_clip"])
                )
            scaler.step(optimizer)
            scaler.update()

            state.step += 1
            running += loss.item()
            seen += 1

            if state.step % log_every == 0:
                mean_loss = running / max(seen, 1)
                elapsed = time.time() - started
                rate = state.step / max(elapsed, 1e-6)
                eta = (total_steps - state.step) / max(rate, 1e-6)
                print(
                    f"step {state.step:7d}/{total_steps}  loss {mean_loss:.4f}  "
                    f"lr {lr:.2e}  scale {scaler.get_scale():.0f}  "
                    f"{rate:.2f} it/s  eta {eta/3600:.1f}h",
                    flush=True,
                )
                state.history.append(
                    {"step": state.step, "epoch": state.epoch,
                     "loss": round(mean_loss, 5), "lr": lr,
                     "scale": scaler.get_scale()}
                )
                running, seen = 0.0, 0

            if state.step % save_every == 0:
                save_checkpoint(latest, model=model, optimizer=optimizer,
                                scaler=scaler, state=state, config=config)

        if state.step >= total_steps:
            complete = True

    save_checkpoint(latest, model=model, optimizer=optimizer, scaler=scaler,
                    state=state, config=config)
    (output_dir / "history.json").write_text(json.dumps(state.history, indent=2))

    if complete:
        # The encoder alone, in the form extract_features.py expects. The
        # decoder is a pretraining artefact and is not part of the
        # representation being evaluated.
        torch.save(
            {"model": model.vit.state_dict(),
             "step": state.step,
             "config": config},
            output_dir / "encoder_final.pt",
        )
        print(f"\nTraining complete at step {state.step}. "
              f"Encoder: {output_dir / 'encoder_final.pt'}", flush=True)
    else:
        print(f"\nStopped at step {state.step}/{total_steps}. "
              f"Resume with the same command.", flush=True)

    elapsed = time.time() - started
    print(f"Elapsed {elapsed/3600:.2f}h  |  {state.step} steps", flush=True)
    if device.type == "cuda":
        print(f"Peak VRAM {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GiB",
              flush=True)

    return 0 if complete else 99


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs=2, action="append", default=[],
                   metavar=("KEY", "VALUE"),
                   help="dotted key and value, e.g. --override train.lr 3.75e-5")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore latest.pt and start from the base checkpoint")
    p.add_argument("--dry-run", action="store_true",
                   help="report the resolved configuration and exit")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
