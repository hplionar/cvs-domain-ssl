"""DINO continued pretraining on surgical frames — experiment E2.4.

The third image arm, alongside ViT-MAE (masked reconstruction) and I-JEPA
(latent prediction). Self-distillation is the objective SMIL and SurgeNetXL both
use, and the one behind DINOv2, which tops the frozen ranking here at 0.4593 on
SAGES.

Collapse is the failure this file is built around
-------------------------------------------------
DINO admits a degenerate solution: if the teacher maps every input to the same
output, the student matches it trivially and the loss falls to a low, stable
value. Centring and sharpening exist to prevent that, and they pull in opposite
directions — centring spreads the teacher output, sharpening concentrates it.
Stability depends on the balance holding.

The balance depends on batch size, because the centre is an EMA of the batch
mean over a 65,536-dimensional output. At batch 1024 that estimate is stable; at
batch 8 it is not. The memory sweep in `scripts/dino_memory_sweep.py` measured
batch 64 fitting in 23.06 GiB of 31.7 with the full 2+8 recipe, which is why
published defaults are used here rather than mitigations.

**Collapse does not announce itself in the loss.** The V-JEPA arm in this
project reduced its loss by 76% while downstream mAP fell 0.127, and issue #272
on facebookresearch/dinov2 reports DINOv2 reaching 50% on a binary medical task
after 300 epochs — chance — with no answer from the maintainers. Twelve hours
can produce a degenerate encoder with a clean-looking log.

So `feature_statistics` from `scripts/check_collapse.py` runs on the teacher
output every `collapse_every_steps` and is logged beside the loss. The
uncentred effective rank is the figure that matters: mapping every input to one
vector is rank one in that sense, whereas centring would subtract exactly that
constant and leave isotropic noise reading as full rank.

Augmentation
------------
`MultiCropTransform` omits colour jitter and grayscale by default, because hue
perturbation degrades the tissue-colour cue CVS criterion C2 depends on, and C2
is the strongest criterion in all eight SAGES arms measured so far. Set
`data.colour_jitter: true` to restore the reference recipe as its own arm.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import signal
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

from data.multicrop import MultiCropTransform, multi_crop_collate
from data.ssl_frame_dataset import SSLFrameDataset


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


class DINOHead(nn.Module):
    """Projection head: MLP, l2-normalised bottleneck, weight-normed output.

    The output dimension is large — 65,536 by default — and that is deliberate.
    DINO's loss is a cross-entropy over these dimensions treated as a soft
    codebook, and a wide codebook gives the centring statistic room to spread
    mass rather than concentrating it. It also means the final layer alone holds
    roughly 17M parameters against ViT-B's 86M.

    The weight norm is initialised with unit magnitude and left ungrouped: the
    reference implementation freezes the magnitude for the first epoch, which is
    omitted here because it interacts poorly with resuming mid-schedule.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        *,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOModel(nn.Module):
    """Backbone plus projection head, pooled to a single vector per view."""

    def __init__(self, backbone: nn.Module, head: DINOHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        """Run a list of views, concatenating outputs in the given order.

        Views of differing resolution cannot be batched into one tensor, so they
        are grouped by size. Order is preserved so the loss can identify which
        rows correspond to which view index.
        """
        sizes = [v.shape[-1] for v in views]
        order = sorted(range(len(views)), key=lambda i: sizes[i])

        outputs: list[torch.Tensor | None] = [None] * len(views)
        i = 0
        while i < len(order):
            size = sizes[order[i]]
            group = [order[i]]
            while i + 1 < len(order) and sizes[order[i + 1]] == size:
                i += 1
                group.append(order[i])
            i += 1

            batch = torch.cat([views[j] for j in group], dim=0)
            hidden = self.backbone(pixel_values=batch).last_hidden_state
            # CLS where the architecture has one, else mean-pooled patches.
            pooled = hidden[:, 0] if hidden.shape[1] > 1 else hidden.mean(dim=1)
            projected = self.head(pooled)

            per_view = projected.shape[0] // len(group)
            for k, j in enumerate(group):
                outputs[j] = projected[k * per_view : (k + 1) * per_view]

        return torch.cat(outputs, dim=0)  # type: ignore[arg-type]


class DINOLoss(nn.Module):
    """Cross-entropy against a centred, sharpened teacher.

    The centre is an EMA of the teacher's batch mean, subtracted before the
    softmax. Without it one output dimension can dominate and the teacher
    collapses to a constant; sharpening — a temperature below the student's —
    pushes the opposite way, toward one-hot outputs. Neither alone is stable.

    Matched pairs are excluded: a student view is never asked to predict the
    teacher's output for the *same* view, since that is trivially satisfiable
    and carries no local-to-global signal.
    """

    def __init__(
        self,
        out_dim: int,
        *,
        num_global: int = 2,
        student_temp: float = 0.1,
        centre_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_global = num_global
        self.student_temp = student_temp
        self.centre_momentum = centre_momentum
        self.register_buffer("centre", torch.zeros(1, out_dim))

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        *,
        num_views: int,
        teacher_temp: float,
    ) -> torch.Tensor:
        student = (student_out / self.student_temp).chunk(num_views)
        teacher = F.softmax(
            (teacher_out - self.centre) / teacher_temp, dim=-1
        ).detach().chunk(self.num_global)

        total, terms = 0.0, 0
        for t_index, t_out in enumerate(teacher):
            for s_index, s_out in enumerate(student):
                if s_index == t_index:
                    continue  # matched pair carries no signal
                total = total + torch.sum(-t_out * F.log_softmax(s_out, dim=-1), dim=-1).mean()
                terms += 1

        self._update_centre(teacher_out)
        return total / max(terms, 1)

    @torch.no_grad()
    def _update_centre(self, teacher_out: torch.Tensor) -> None:
        batch_centre = teacher_out.mean(dim=0, keepdim=True)
        self.centre = (
            self.centre * self.centre_momentum
            + batch_centre * (1 - self.centre_momentum)
        )


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------


def cosine_schedule(step: int, total: int, start: float, end: float,
                    warmup: int = 0, warmup_start: float = 0.0) -> float:
    """Linear warmup then cosine interpolation from ``start`` to ``end``."""
    if warmup and step < warmup:
        return warmup_start + (start - warmup_start) * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * progress))


def linear_warmup_value(step: int, warmup: int, start: float, end: float) -> float:
    """Teacher temperature: warms from ``start`` to ``end`` then holds.

    A sharp teacher early in training is what prevents the uniform-output mode
    of collapse; the reference implementation warms it over the first 30 epochs.
    """
    if step >= warmup:
        return end
    return start + (end - start) * step / max(warmup, 1)


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    history: list[dict[str, float]] = field(default_factory=list)


def save_checkpoint(path: Path, *, student, teacher, criterion, optimizer,
                    scaler, state: TrainState, config: dict[str, Any]) -> None:
    """Write a checkpoint that restores training exactly.

    The teacher is saved separately and explicitly: it is the encoder that gets
    evaluated, and collapse is a property of the teacher rather than the
    student. Losing it would make the run unassessable.

    Written to a temporary path and renamed, so a job killed mid-write leaves
    the previous checkpoint intact.
    """
    payload = {
        "model": student.state_dict(),
        "teacher": teacher.state_dict(),
        "centre": criterion.centre,
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


def load_checkpoint(path: Path, *, student, teacher, criterion, optimizer,
                    scaler, device) -> TrainState:
    payload = torch.load(path, map_location=device, weights_only=False)
    student.load_state_dict(payload["model"])
    teacher.load_state_dict(payload["teacher"])
    if "centre" in payload:
        criterion.centre = payload["centre"].to(device)
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
        step=payload["step"], epoch=payload["epoch"],
        best_loss=payload.get("best_loss", float("inf")),
        history=payload.get("history", []),
    )


class RequeueHandler:
    """Sets a flag on SIGUSR1 or SIGTERM so the loop exits after a clean save."""

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


def build_models(config: dict[str, Any], device: torch.device):
    """Student and teacher, initialised identically.

    The teacher is a deep copy rather than a second `from_pretrained`: they must
    start from the same weights, and two loads of the same checkpoint are equal
    only if nothing in the loading path is stochastic.
    """
    from transformers import AutoModel

    checkpoint = config["model"]["checkpoint"]
    backbone = AutoModel.from_pretrained(checkpoint)
    dim = backbone.config.hidden_size

    head_cfg = config["model"].get("head", {})
    out_dim = int(head_cfg.get("out_dim", 65536))

    student = DINOModel(
        backbone,
        DINOHead(
            dim, out_dim,
            hidden_dim=int(head_cfg.get("hidden_dim", 2048)),
            bottleneck_dim=int(head_cfg.get("bottleneck_dim", 256)),
            num_layers=int(head_cfg.get("num_layers", 3)),
        ),
    )
    teacher = copy.deepcopy(student)

    # The teacher is updated by EMA, never by gradient. Leaving it trainable
    # would double optimiser state for no purpose and silently allow the loss
    # to update it through any path that forgot to detach.
    for param in teacher.parameters():
        param.requires_grad = False

    if config["model"].get("gradient_checkpointing", False):
        student.backbone.gradient_checkpointing_enable()

    return student.to(device), teacher.to(device), dim, out_dim


def build_dataset(config: dict[str, Any]) -> tuple[SSLFrameDataset, MultiCropTransform]:
    data = config["data"]
    transform = MultiCropTransform(
        global_size=int(data.get("global_size", 224)),
        local_size=int(data.get("local_size", 96)),
        num_local=int(data.get("num_local", 8)),
        global_scale=tuple(data.get("global_scale", (0.4, 1.0))),
        local_scale=tuple(data.get("local_scale", (0.05, 0.4))),
        colour_jitter=bool(data.get("colour_jitter", False)),
    )
    dataset = SSLFrameDataset(
        data["frames_dir"],
        frames_per_video=data.get("frames_per_video"),
        transform=transform,
        exclude_video_ids=data.get("exclude_video_ids", []),
        limit_videos=data.get("limit_videos"),
    )
    return dataset, transform


@torch.no_grad()
def teacher_statistics(teacher: DINOModel, views: list[torch.Tensor]) -> dict[str, float]:
    """Collapse diagnostics on the teacher's pooled backbone features.

    Computed on the backbone output rather than the projection head: the head is
    discarded after pretraining, so the representation that matters downstream
    is what the backbone produces.
    """
    from scripts.check_collapse import feature_statistics

    hidden = teacher.backbone(pixel_values=views[0]).last_hidden_state
    pooled = hidden[:, 0] if hidden.shape[1] > 1 else hidden.mean(dim=1)
    stats = feature_statistics(pooled.float().cpu())
    return {
        "collapse_rank": stats["effective_rank"],
        "collapse_rank_fraction": stats["rank_fraction"],
        "collapse_variance": stats["mean_variance"],
        "collapse_cosine": stats["mean_pairwise_cosine"],
    }


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

    set_seed(int(config.get("seed", 0)))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device; this will be extremely slow.", flush=True)

    student, teacher, dim, out_dim = build_models(config, device)
    dataset, transform = build_dataset(config)

    train_cfg = config["train"]
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 12))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
        collate_fn=multi_crop_collate,
    )

    steps_per_epoch = len(loader)
    epochs = int(train_cfg.get("epochs", 20))
    total_steps = int(train_cfg.get("max_steps", steps_per_epoch * epochs))
    warmup = int(train_cfg.get("warmup_steps", max(1, total_steps // 20)))
    base_lr = float(train_cfg["lr"])

    criterion = DINOLoss(
        out_dim,
        num_global=2,
        student_temp=float(train_cfg.get("student_temp", 0.1)),
        centre_momentum=float(train_cfg.get("centre_momentum", 0.9)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=base_lr,
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(train_cfg.get("weight_decay", 0.04)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    state = TrainState()
    latest = output_dir / "latest.pt"
    if latest.is_file() and not args.no_resume:
        state = load_checkpoint(latest, student=student, teacher=teacher,
                                criterion=criterion, optimizer=optimizer,
                                scaler=scaler, device=device)
        print(f"Resumed from {latest} at step {state.step}.", flush=True)

    corpus = dataset.describe()
    teacher_temp_warmup = int(train_cfg.get("teacher_temp_warmup_steps", total_steps // 10))

    print("=" * 70)
    print(f"checkpoint    {config['model']['checkpoint']}  (dim {dim})")
    print(f"head          {out_dim} output dims")
    print(f"corpus        {corpus['num_frames']} frames over {corpus['num_videos']} videos")
    print(f"augment       {transform}")
    print(f"batch         {batch_size} x {transform.num_views} views  ->  {steps_per_epoch} steps/epoch")
    print(f"schedule      {total_steps} total steps, {warmup} warmup")
    print(f"lr            {base_lr:.2e}  ->  {float(train_cfg.get('min_lr', 1e-6)):.2e}")
    print(f"teacher temp  {train_cfg.get('teacher_temp_start', 0.04)} -> "
          f"{train_cfg.get('teacher_temp_end', 0.07)} over {teacher_temp_warmup} steps")
    print(f"ema momentum  {train_cfg.get('ema_start', 0.996)} -> {train_cfg.get('ema_end', 1.0)}")
    print(f"centre mom.   {train_cfg.get('centre_momentum', 0.9)}")
    print(f"device        {device}  amp={'fp16' if device.type == 'cuda' else 'off'}")
    print(f"output        {output_dir}")
    print(f"start         step {state.step}")
    print("=" * 70, flush=True)

    if args.dry_run:
        print("--dry-run: nothing trained.")
        return 0

    requeue = RequeueHandler()
    log_every = int(train_cfg.get("log_every_steps", 20))
    save_every = int(train_cfg.get("save_every_steps", 200))
    collapse_every = int(train_cfg.get("collapse_every_steps", 500))
    grad_clip = train_cfg.get("grad_clip", 3.0)

    started = time.time()
    student.train()
    teacher.train()
    complete = state.step >= total_steps

    while state.step < total_steps and not requeue.should_stop:
        state.epoch += 1
        running, seen = 0.0, 0

        for views in loader:
            if state.step >= total_steps or requeue.should_stop:
                break

            views = [v.to(device, non_blocking=True) for v in views]

            lr = cosine_schedule(
                state.step, total_steps, base_lr,
                float(train_cfg.get("min_lr", 1e-6)), warmup=warmup, warmup_start=0.0,
            )
            wd = cosine_schedule(
                state.step, total_steps,
                float(train_cfg.get("weight_decay", 0.04)),
                float(train_cfg.get("weight_decay_end", 0.4)),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = wd

            teacher_temp = linear_warmup_value(
                state.step, teacher_temp_warmup,
                float(train_cfg.get("teacher_temp_start", 0.04)),
                float(train_cfg.get("teacher_temp_end", 0.07)),
            )

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                with torch.no_grad():
                    teacher_out = teacher(views[:2])   # global views only
                student_out = student(views)           # all views
                loss = criterion(student_out, teacher_out,
                                 num_views=len(views), teacher_temp=teacher_temp)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {state.step}. The checkpoint at "
                    f"{latest} is the last valid state."
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()

            # EMA update. Momentum rises to 1.0, so the teacher slows and
            # eventually freezes; that is what stabilises the target late in
            # training, at the cost of the target no longer tracking the student.
            momentum = cosine_schedule(
                state.step, total_steps,
                float(train_cfg.get("ema_start", 0.996)),
                float(train_cfg.get("ema_end", 1.0)),
            )
            with torch.no_grad():
                for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                    t_param.mul_(momentum).add_(s_param.detach(), alpha=1 - momentum)

            state.step += 1
            running += loss.item()
            seen += 1

            if state.step % log_every == 0:
                mean_loss = running / max(seen, 1)
                elapsed = time.time() - started
                rate = state.step / max(elapsed, 1e-6)
                row = {
                    "step": state.step, "epoch": state.epoch,
                    "loss": round(mean_loss, 5), "lr": lr, "wd": round(wd, 5),
                    "teacher_temp": round(teacher_temp, 5),
                    "ema": round(momentum, 6),
                }
                print(
                    f"step {state.step:7d}/{total_steps}  loss {mean_loss:.4f}  "
                    f"lr {lr:.2e}  t_temp {teacher_temp:.4f}  ema {momentum:.5f}  "
                    f"{rate:.2f} it/s  eta {(total_steps-state.step)/max(rate,1e-6)/3600:.1f}h",
                    flush=True,
                )
                state.history.append(row)
                running, seen = 0.0, 0

            if collapse_every and state.step % collapse_every == 0:
                stats = teacher_statistics(teacher, views)
                print(
                    f"  [collapse] rank {stats['collapse_rank']:.1f} "
                    f"({100*stats['collapse_rank_fraction']:.1f}% of {dim})  "
                    f"var {stats['collapse_variance']:.4f}  "
                    f"cos {stats['collapse_cosine']:.4f}",
                    flush=True,
                )
                if state.history:
                    state.history[-1].update(stats)
                if stats["collapse_rank"] < 2.0:
                    raise RuntimeError(
                        f"Teacher effective rank {stats['collapse_rank']:.2f} at "
                        f"step {state.step}: the representation has collapsed to "
                        f"approximately one direction. Continuing would waste the "
                        f"remaining walltime producing a degenerate encoder."
                    )

            if state.step % save_every == 0:
                save_checkpoint(latest, student=student, teacher=teacher,
                                criterion=criterion, optimizer=optimizer,
                                scaler=scaler, state=state, config=config)

        if state.step >= total_steps:
            complete = True

    save_checkpoint(latest, student=student, teacher=teacher, criterion=criterion,
                    optimizer=optimizer, scaler=scaler, state=state, config=config)
    (output_dir / "history.json").write_text(json.dumps(state.history, indent=2))

    if complete:
        # The teacher backbone is the encoder that gets evaluated: it is the EMA
        # of the student and is what DINO's own linear-probe protocol uses. The
        # projection head is a pretraining artefact and is not saved.
        torch.save(
            {"model": teacher.backbone.state_dict(),
             "step": state.step, "config": config},
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
                   metavar=("KEY", "VALUE"))
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
