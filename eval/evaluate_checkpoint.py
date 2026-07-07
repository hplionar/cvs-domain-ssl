from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.datasets import EndoscapesDataset
from data.sages_datasets import SAGESFrameDataset
from data.transforms import build_transform
from eval.metrics import compute_multilabel_metrics_from_logits
from models.cvs_model import CVSModel
from models.encoders.dinov2_encoder import DINOv2Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved CVS checkpoint.")

    parser.add_argument("--dataset", choices=["endoscapes", "sages"], required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--manifest-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--encoder", type=str, default="dinov2")
    parser.add_argument("--variant", type=str, default="base")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-batches", type=int, default=None)

    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    if args.dataset == "endoscapes":
        return EndoscapesDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split=args.split,
            mode="supervised",
            transform=build_transform(mode="supervised", split=args.split),
        )

    if args.dataset == "sages":
        return SAGESFrameDataset(
            manifest_path=args.manifest_path,
            dataset_root=args.dataset_root,
            split=args.split,
            mode="supervised",
            transform=build_transform(mode="supervised", split=args.split),
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    limit_batches: int | None = None,
) -> tuple[float, dict[str, float], int]:
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
    loss = total_loss / max(total_samples, 1)

    return loss, metrics, total_samples


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")

    dataset = build_dataset(args)
    print(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    criterion = nn.BCEWithLogitsLoss()

    loss, metrics, samples = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        limit_batches=args.limit_batches,
    )

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "split": args.split,
        "samples": samples,
        "checkpoint_path": str(checkpoint_path),
        "loss": loss,
        **metrics,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved evaluation result to: {output_json}")


if __name__ == "__main__":
    main()
