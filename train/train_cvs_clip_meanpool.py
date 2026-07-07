from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.sequence_datasets import EndoscapesClipDataset
from data.sages_sequence_datasets import SAGESClipDataset
from data.transforms import build_transform
from eval.metrics import compute_multilabel_metrics_from_logits
from models.encoders.dinov2_encoder import DINOv2Encoder
from models.heads.linear_head import LinearCVSHead


class ClipMeanPoolCVSModel(nn.Module):
    """Encode each frame independently, mean-pool over time, then classify CVS."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_labels: int = 3,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = LinearCVSHead(
            input_dim=feature_dim,
            num_labels=num_labels,
            dropout=dropout,
            use_layernorm=True,
        )
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, C, H, W]
        if frames.ndim != 5:
            raise ValueError(f"Expected frames with shape [B,T,C,H,W], got {frames.shape}")

        batch_size, clip_length, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch_size * clip_length, channels, height, width)

        if self.freeze_encoder:
            with torch.no_grad():
                flat_features = self.encoder(flat_frames)
        else:
            flat_features = self.encoder(flat_frames)

        features = flat_features.reshape(batch_size, clip_length, -1)
        pooled = features.mean(dim=1)

        logits = self.head(pooled)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CVS classifier on sparse clips using frame-feature mean pooling.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="sages",
        choices=["endoscapes", "sages"],
    )
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--encoder", type=str, default="dinov2")
    parser.add_argument("--variant", type=str, default="base")

    parser.add_argument("--clip-length", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--target-position", type=str, default="end", choices=["end", "center"])

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)

    parser.add_argument(
        "--use-pos-weight",
        action="store_true",
        help="Use positive-class weights computed from the training split.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> ClipMeanPoolCVSModel:
    if args.encoder != "dinov2":
        raise ValueError(f"Currently only dinov2 is supported, got {args.encoder}")

    encoder = DINOv2Encoder(variant=args.variant, pretrained=True)

    model = ClipMeanPoolCVSModel(
        encoder=encoder,
        feature_dim=encoder.feature_dim,
        num_labels=3,
        dropout=args.dropout,
        freeze_encoder=True,
    )

    return model


def build_datasets(args: argparse.Namespace):
    train_transform = build_transform(mode="supervised", split="train")
    val_transform = build_transform(mode="supervised", split="val")

    if args.dataset == "sages":
        train_dataset = SAGESClipDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split="train",
            mode="supervised",
            clip_length=args.clip_length,
            stride=args.stride,
            target_position=args.target_position,
            transform=train_transform,
        )

        val_dataset = SAGESClipDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split="val",
            mode="supervised",
            clip_length=args.clip_length,
            stride=args.stride,
            target_position=args.target_position,
            transform=val_transform,
        )

    elif args.dataset == "endoscapes":
        train_dataset = EndoscapesClipDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split="train",
            mode="supervised",
            clip_length=args.clip_length,
            stride=args.stride,
            target_position=args.target_position,
            transform=train_transform,
        )

        val_dataset = EndoscapesClipDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split="val",
            mode="supervised",
            clip_length=args.clip_length,
            stride=args.stride,
            target_position=args.target_position,
            transform=val_transform,
        )

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_dataset, val_dataset


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(args)

    pin_memory = args.device == "cuda" and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(train_dataset)
    print(val_dataset)

    return train_loader, val_loader


def compute_pos_weight_from_manifest(
    manifest_path: str,
    dataset: str,
    split: str = "train",
) -> torch.Tensor:
    df = pd.read_csv(manifest_path)
    target_cols = ["c1_consensus", "c2_consensus", "c3_consensus"]

    if dataset == "sages":
        df = df[df["internal_split"] == split].copy()

    elif dataset == "endoscapes":
        df = df[
            (df["split"] == split)
            & (df["is_cvs_annotated"] == True)
        ].copy()

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if df.empty:
        raise ValueError(f"No rows found for dataset={dataset}, split={split}")

    positives = df[target_cols].sum().to_numpy(dtype=np.float32)
    total = len(df)
    negatives = total - positives

    pos_weight = negatives / np.maximum(positives, 1.0)

    print("Training positive counts:", positives)
    print("Training negative counts:", negatives)
    print("Using pos_weight:", pos_weight)

    return torch.tensor(pos_weight, dtype=torch.float32)


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

        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(frames)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = frames.size(0)
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

        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(frames)
        loss = criterion(logits, targets)

        batch_size = frames.size(0)
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


def save_config(output_dir: Path, args: argparse.Namespace) -> None:
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir, args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Clip length: {args.clip_length}")
    print(f"Stride: {args.stride}")
    print(f"Target position: {args.target_position}")

    train_loader, val_loader = build_loaders(args)

    model = build_model(args).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    if args.use_pos_weight:
        pos_weight = compute_pos_weight_from_manifest(
            manifest_path=args.manifest_path,
            dataset=args.dataset,
            split="train",
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_map = -1.0
    history: list[dict[str, Any]] = []

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

        current_map = val_metrics["mAP"]
        if not np.isnan(current_map) and current_map > best_map:
            best_map = current_map
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
