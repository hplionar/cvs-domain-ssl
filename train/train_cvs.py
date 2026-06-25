from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets import EndoscapesDataset
from data.transforms import build_transform
from eval.metrics import compute_multilabel_metrics_from_logits
from models.cvs_model import CVSModel
from models.encoders.dinov2_encoder import DINOv2Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CVS classifier.")

    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--encoder", type=str, default="dinov2")
    parser.add_argument("--variant", type=str, default="base")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> CVSModel:
    if args.encoder != "dinov2":
        raise ValueError(f"Currently only dinov2 is supported, got {args.encoder}")

    encoder = DINOv2Encoder(variant=args.variant, pretrained=True)

    model = CVSModel(
        encoder=encoder,
        feature_dim=encoder.feature_dim,
        num_labels=3,
        dropout=args.dropout,
        freeze_encoder=True,
    )

    return model


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = EndoscapesDataset(
        manifest_path=args.manifest_path,
        dataset_root=args.dataset_root,
        split="train",
        mode="supervised",
        transform=build_transform(mode="supervised", split="train"),
    )

    val_dataset = EndoscapesDataset(
        manifest_path=args.manifest_path,
        dataset_root=args.dataset_root,
        split="val",
        mode="supervised",
        transform=build_transform(mode="supervised", split="val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(train_dataset)
    print(val_dataset)

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    limit_batches: int | None = None,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    limit_batches: int | None = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_logits = []

    for batch_idx, batch in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(targets.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits_np = np.concatenate(all_logits, axis=0)

    metrics = compute_multilabel_metrics_from_logits(y_true, logits_np)
    val_loss = total_loss / max(total_samples, 1)

    return val_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_loaders(args)

    model = build_model(args).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_map = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            limit_batches=args.limit_train_batches,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            limit_batches=args.limit_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"mAP={val_metrics['mAP']:.4f} | "
            f"mean_bacc={val_metrics['mean_bacc']:.4f}"
        )

        if val_metrics["mAP"] > best_map:
            best_map = val_metrics["mAP"]
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
            )

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best validation mAP: {best_map:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()