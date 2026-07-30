#!/usr/bin/env python3
"""Train a CVS probe on cached encoder features.

Reads the memory-mapped fp16 caches written by ``scripts/extract_features.py``
and trains either a mean-pooling or an attentive (MIL) head. Because the encoder
is frozen and its outputs are precomputed, a full grid search with seed
repetition costs seconds rather than GPU-hours.

Three protocol rules are enforced here rather than left to discipline:

1. **The same grid is searched for every arm.** Feature norms differ between
   objective families, so a single fixed learning rate silently favours whichever
   arm its scale happens to suit — and that bias lands directly in adaptation
   gain. Equal search effort per arm is the defensible protocol.

2. **Every configuration is run with multiple seeds.** Adaptation gain is a
   difference of two noisy measurements. If seed variance is ±0.015 and the
   measured gain is 0.008, nothing has been demonstrated.

3. **Model selection uses validation mAP, never validation loss.** Under
   ``pos_weight`` the loss changes scale entirely and is not comparable across
   configurations.

Raw validation logits are saved at the selected epoch so that decision
thresholds can be tuned afterwards without retraining (audit finding F3).

Usage:
    python train/train_probe_cached.py \
        --train-features cache/mae_b/endoscapes/train \
        --val-features   cache/mae_b/endoscapes/val \
        --head mean --seeds 3 \
        --output-dir outputs/probe/mae_b_endoscapes
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eval.metrics import compute_multilabel_metrics_from_logits
from models.heads.attentive_head import build_head


# Default search grid. Applied identically to every arm.
DEFAULT_GRID = {
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4, 1e-2],
    "dropout": [0.0, 0.1],
}

# Fields of a cache manifest that must agree for two caches to be comparable.
PROTOCOL_FIELDS = ("transform", "extraction")


# --------------------------------------------------------------------------
# cache loading
# --------------------------------------------------------------------------


class CachedFeatures(Dataset):
    """Token grids plus targets from an extraction directory.

    Caches are fp16 on disk. Small caches are read into RAM; large ones stay
    memory-mapped, since a full SAGES video cache is tens of gigabytes.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        in_memory: bool | None = None,
        memory_budget_gib: float = 4.0,
        video_ids: Iterable[str] | None = None,
    ) -> None:
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No manifest.json in {self.directory}. Run extract_features.py first."
            )
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())

        tokens_path = self.directory / "tokens.npy"
        size_gib = tokens_path.stat().st_size / 1024**3
        if in_memory is None:
            in_memory = size_gib <= memory_budget_gib
        self.in_memory = in_memory
        self.size_gib = size_gib

        self.tokens = np.load(tokens_path, mmap_mode=None if in_memory else "r")
        self.targets = np.load(self.directory / "targets.npy")

        with open(self.directory / "index.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.sample_ids = [r["sample_id"] for r in rows]
        self.video_ids = [r["video_id"] for r in rows]

        self.rows = np.arange(len(self.sample_ids))
        if video_ids is not None:
            keep = set(video_ids)
            self.rows = np.array(
                [i for i, v in enumerate(self.video_ids) if v in keep], dtype=np.int64
            )
            if self.rows.size == 0:
                raise ValueError("Video-level subset selected zero samples.")

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.rows[index])
        tokens = torch.from_numpy(np.asarray(self.tokens[row], dtype=np.float32))
        return tokens, torch.from_numpy(self.targets[row])

    @property
    def feature_dim(self) -> int:
        return int(self.tokens.shape[2])

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[1])

    def unique_video_ids(self) -> list[str]:
        return sorted({self.video_ids[i] for i in self.rows})

    def pooled(self) -> torch.Tensor:
        """Mean over tokens, computed once.

        For the mean-pooling head, pooling is a fixed linear reduction that does
        not depend on any parameter, so doing it per epoch is wasted work.
        Precomputing is mathematically identical and turns a 1568-token cache
        into a single vector per sample.
        """
        chunks = []
        for start in range(0, len(self), 4096):
            rows = self.rows[start : start + 4096]
            block = np.asarray(self.tokens[rows], dtype=np.float32)
            chunks.append(torch.from_numpy(block).mean(dim=1))
        return torch.cat(chunks, dim=0)

    def all_targets(self) -> torch.Tensor:
        return torch.from_numpy(self.targets[self.rows])


class PooledFeatures(Dataset):
    """Precomputed pooled vectors, restored to [1, D] so heads see a grid."""

    def __init__(self, pooled: torch.Tensor, targets: torch.Tensor) -> None:
        self.pooled = pooled
        self.targets = targets

    def __len__(self) -> int:
        return self.pooled.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pooled[index].unsqueeze(0), self.targets[index]


# --------------------------------------------------------------------------
# protocol verification
# --------------------------------------------------------------------------


def verify_protocol(a: dict[str, Any], b: dict[str, Any], *, label_a: str, label_b: str) -> None:
    """Refuse to proceed if two caches were not produced identically.

    Adaptation gain is only meaningful when the two measurements differ solely
    in the encoder weights. Comparing caches built with different preprocessing
    or different reductions produces a number that looks like a result and is
    not one.
    """
    problems = []
    for field_name in PROTOCOL_FIELDS:
        if a.get(field_name) != b.get(field_name):
            problems.append(
                f"  {field_name}:\n    {label_a}: {a.get(field_name)}\n"
                f"    {label_b}: {b.get(field_name)}"
            )
    layout_a = a.get("encoder", {}).get("token_layout")
    layout_b = b.get("encoder", {}).get("token_layout")
    if layout_a != layout_b:
        problems.append(f"  token_layout:\n    {label_a}: {layout_a}\n    {label_b}: {layout_b}")

    if problems:
        raise ValueError(
            "Caches were not produced under an identical protocol, so results "
            "from them are not comparable:\n" + "\n".join(problems)
        )


def verify_same_encoder(train: dict[str, Any], val: dict[str, Any]) -> None:
    a = train.get("encoder", {}).get("checkpoint_id")
    b = val.get("encoder", {}).get("checkpoint_id")
    if a != b:
        raise ValueError(
            f"Train cache used encoder {a!r} but validation cache used {b!r}."
        )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


@dataclass
class RunResult:
    config: dict[str, float]
    seed: int
    best_epoch: int
    best_map: float
    best_metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    return negatives / positives.clamp(min=1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for tokens, target in loader:
        out = model(tokens.to(device, non_blocking=True))
        logits.append(out.logits.float().cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def train_one_run(
    config: dict[str, float],
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    pos_weight: torch.Tensor | None,
) -> RunResult:
    set_seed(seed)

    head = build_head(
        args.head,
        feature_dim,
        dropout=float(config["dropout"]),
        hidden_dim=args.attn_hidden,
        num_branches=args.attn_branches,
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else pos_weight.to(device)
    )

    best = RunResult(config=config, seed=seed, best_epoch=0, best_map=-1.0, best_metrics={})
    best_logits: np.ndarray | None = None
    best_targets: np.ndarray | None = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss, total_n = 0.0, 0
        for tokens, target in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = criterion(head(tokens).logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * tokens.shape[0]
            total_n += tokens.shape[0]
        scheduler.step()

        logits, targets = evaluate(head, val_loader, device)
        metrics = compute_multilabel_metrics_from_logits(targets, logits)
        row = {"epoch": epoch, "train_loss": total_loss / max(total_n, 1), **metrics}
        best.history.append(row)

        # Selection on mAP, never on loss: pos_weight changes the loss scale.
        current = metrics["mAP"]
        if not np.isnan(current) and current > best.best_map:
            best.best_map = float(current)
            best.best_epoch = epoch
            best.best_metrics = {k: float(v) for k, v in metrics.items()}
            best_logits, best_targets = logits, targets
            patience_left = args.patience
        else:
            patience_left -= 1
            if args.patience > 0 and patience_left <= 0:
                break

    best.best_logits = best_logits  # type: ignore[attr-defined]
    best.best_targets = best_targets  # type: ignore[attr-defined]
    return best


def build_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    grid = {
        "lr": args.lr or DEFAULT_GRID["lr"],
        "weight_decay": args.weight_decay or DEFAULT_GRID["weight_decay"],
        "dropout": args.dropout or DEFAULT_GRID["dropout"],
    }
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_cache = CachedFeatures(args.train_features, in_memory=args.in_memory)
    val_cache = CachedFeatures(args.val_features, in_memory=args.in_memory)

    verify_same_encoder(train_cache.manifest, val_cache.manifest)
    verify_protocol(
        train_cache.manifest, val_cache.manifest, label_a="train", label_b="val"
    )
    if args.compare_with:
        other = json.loads((Path(args.compare_with) / "manifest.json").read_text())
        verify_protocol(train_cache.manifest, other, label_a="this arm", label_b="other arm")

    encoder_id = train_cache.manifest["encoder"]["checkpoint_id"]
    feature_dim = train_cache.feature_dim

    # Pooling is parameter-free, so for the mean head it is computed once
    # instead of once per epoch. Mathematically identical, far cheaper.
    precompute = args.head in {"mean", "meanpool", "linear"}
    if precompute:
        train_ds: Dataset = PooledFeatures(train_cache.pooled(), train_cache.all_targets())
        val_ds: Dataset = PooledFeatures(val_cache.pooled(), val_cache.all_targets())
    else:
        train_ds, val_ds = train_cache, val_cache

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos_weight = compute_pos_weight(train_cache.all_targets()) if args.pos_weight else None

    grid = build_grid(args)
    seeds = list(range(args.seed_base, args.seed_base + args.seeds))

    print(f"encoder     {encoder_id}")
    print(f"head        {args.head}"
          + (f" (branches={args.attn_branches})" if not precompute else " [pooling precomputed]"))
    print(f"train/val   {len(train_ds)} / {len(val_ds)} samples"
          f"  | {train_cache.num_tokens} tokens x {feature_dim} dim")
    print(f"grid        {len(grid)} configs x {len(seeds)} seeds = {len(grid)*len(seeds)} runs")
    print(f"pos_weight  {'on' if pos_weight is not None else 'off'}")
    print()

    started = time.time()
    results: list[RunResult] = []
    for config in grid:
        for seed in seeds:
            results.append(
                train_one_run(config, seed, train_loader, val_loader, feature_dim, args, device, pos_weight)
            )

    # Select on the mean across seeds, not on the single best run: choosing the
    # luckiest seed would inflate the reported result and its variance estimate.
    aggregated: list[dict[str, Any]] = []
    for config in grid:
        matching = [r for r in results if r.config == config]
        maps = [r.best_map for r in matching]
        aggregated.append(
            {
                "config": config,
                "mean_map": float(np.mean(maps)),
                "std_map": float(np.std(maps, ddof=1)) if len(maps) > 1 else 0.0,
                "seeds": {r.seed: r.best_map for r in matching},
                "best_epochs": [r.best_epoch for r in matching],
            }
        )
    aggregated.sort(key=lambda entry: entry["mean_map"], reverse=True)
    winner = aggregated[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_runs = [r for r in results if r.config == winner["config"]]
    for run in best_runs:
        with open(output_dir / f"history_seed{run.seed}.json", "w", encoding="utf-8") as fh:
            json.dump(run.history, fh, indent=2)
        logits = getattr(run, "best_logits", None)
        if logits is not None:
            np.savez(
                output_dir / f"val_logits_seed{run.seed}.npz",
                logits=logits,
                targets=getattr(run, "best_targets"),
            )

    summary = {
        "encoder": train_cache.manifest["encoder"],
        "protocol": {f: train_cache.manifest.get(f) for f in PROTOCOL_FIELDS},
        "head": {
            "kind": args.head,
            "branches": args.attn_branches,
            "hidden_dim": args.attn_hidden,
            "pooling_precomputed": precompute,
        },
        "search": {
            "grid": {k: (getattr(args, k) or DEFAULT_GRID[k]) for k in DEFAULT_GRID},
            "seeds": seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "pos_weight": bool(args.pos_weight),
        },
        "selected": winner,
        "all_configs": aggregated,
        "elapsed_sec": round(time.time() - started, 1),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"best config  {winner['config']}")
    print(f"val mAP      {winner['mean_map']:.4f} +/- {winner['std_map']:.4f}"
          f"  (n={len(seeds)} seeds)")
    print(f"best epochs  {winner['best_epochs']}")
    print(f"elapsed      {summary['elapsed_sec']:.1f}s")
    print(f"written to   {output_dir}")

    if winner["std_map"] > 0 and winner["mean_map"] > 0:
        print(f"\nSeed spread is +/-{winner['std_map']:.4f} mAP. An adaptation gain "
              f"smaller than roughly {2*winner['std_map']:.4f} is not distinguishable "
              f"from noise at this seed count.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--train-features", required=True)
    p.add_argument("--val-features", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--compare-with",
        default=None,
        help="another arm's cache directory; fails if the protocols differ",
    )

    p.add_argument("--head", default="mean", choices=["mean", "meanpool", "linear", "attentive", "attn", "mil", "abmil"])
    p.add_argument("--attn-hidden", type=int, default=128)
    p.add_argument("--attn-branches", type=int, default=1, choices=[1, 3])

    p.add_argument("--lr", type=float, nargs="*", default=None, help="override the default grid")
    p.add_argument("--weight-decay", type=float, nargs="*", default=None)
    p.add_argument("--dropout", type=float, nargs="*", default=None)

    p.add_argument("--seeds", type=int, default=3, help="three is the protocol minimum")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="0 disables early stopping")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pos-weight", action="store_true")
    p.add_argument("--in-memory", action="store_true", default=None)
    p.add_argument("--device", default="cuda")

    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())