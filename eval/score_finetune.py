#!/usr/bin/env python3
"""Score a saved fine-tuning checkpoint on the held-out splits.

`train/finetune_cvs.py` reports the best validation mAP and saves the
corresponding weights, and nothing scores them afterwards. Every fine-tuning
figure in this project is therefore a selection-split figure: 0.6246 for DINOv3,
0.5999 for DINOv2, 0.6003 for full fine-tuning. None has been measured on data
withheld from the choice of configuration or epoch.

That matters more here than it usually would. The regression of held-out on
selection-split mAP across fourteen frozen arms has slope 0.691, and the two
depth arms dropped 0.049 and 0.087 when scored held out. A validation figure of
0.62 is consistent with a held-out figure near 0.55, and the difference is the
difference between a headline claim and an overstatement.

The checkpoint stores the arguments it was trained with, so the encoder, the
number of unfrozen blocks and the dropout are read from it rather than supplied
again: passing them by hand is how a scored checkpoint comes to differ from the
one that was trained.

Usage:
    python eval/score_finetune.py \\
        --checkpoint ../outputs/finetune/dinov3_b_lr3e-4/best.pt \\
        --dataset-root ../datasets/SAGES_CVS_Challenge_2024 \\
        --manifest-path metadata/sages_frames_internal_split.csv \\
        --official-manifest metadata/sages_frames_official_test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.transforms import build_transform_from_spec
from eval.metrics import compute_multilabel_metrics_from_logits
from models.encoders import build_encoder
from train.finetune_cvs import (
    FineTuneModel, transformer_blocks, unfreeze_last_blocks,
)


def collate(batch):
    """Keep only the image and the target.

    Defined here rather than imported. SAGESFrameDataset returns fourteen keys,
    most of them strings used for provenance, and the default collate would
    batch those into lists and carry them through every worker. The identical
    function exists in the trainer, but importing it couples this script to a
    file that has been overwritten several times; duplicating six lines is
    cheaper than the coupling.
    """
    return (torch.stack([b["image"] for b in batch]),
            torch.stack([torch.as_tensor(b["target"]) for b in batch]))

import models.encoders.dinov2_encoder  # noqa: F401
import models.encoders.dinov3_encoder  # noqa: F401
import models.encoders.mae_encoder  # noqa: F401
import models.encoders.vit_sup_encoder  # noqa: F401


def build_loader(manifest: str, split: str, root: str, spec, batch: int,
                 workers: int, dataset_name: str = "sages") -> DataLoader:
    """A loader for whichever dataset the checkpoint was trained on.

    The name comes from the checkpoint rather than from a flag, so a checkpoint
    cannot be scored against another dataset's frames -- which would produce a
    number rather than an error.
    """
    common = dict(manifest_path=manifest, dataset_root=root, split=split,
                  transform=build_transform_from_spec(spec))
    if dataset_name == "endoscapes":
        from data.datasets import EndoscapesDataset

        dataset = EndoscapesDataset(mode="supervised", **common)
    else:
        from data.sages_datasets import SAGESFrameDataset

        dataset = SAGESFrameDataset(**common)
    return DataLoader(dataset, batch_size=batch, shuffle=False,
                      num_workers=workers, pin_memory=True, collate_fn=collate)


@torch.no_grad()
def score(model, loader, device, amp: bool) -> dict[str, float]:
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        targets.append(y.numpy())
    logits = np.concatenate(logits)
    targets = np.concatenate(targets)
    metrics = compute_multilabel_metrics_from_logits(targets, logits)
    return {k: float(v) for k, v in metrics.items()}, logits, targets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="a best.pt from finetune_cvs")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--manifest-path", required=True,
                   help="the manifest the checkpoint was trained against")
    p.add_argument("--official-manifest", default=None,
                   help="optional: also score the official test split")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--save-logits", action="store_true",
                   help="write per-split logits beside the checkpoint, so the "
                        "stratified analyses can read them without rescoring")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    path = Path(args.checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trained = payload["args"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"

    print(f"checkpoint     {path}")
    print(f"encoder        {trained['encoder']}"
          + ("  [from scratch]" if trained.get("from_scratch") else ""))
    print(f"trained with   backbone lr {trained['backbone_lr']:.0e}, "
          f"{trained['unfreeze_blocks']} blocks, dropout {trained['dropout']}")
    print(f"selected at    epoch {payload['epoch']}, val mAP {payload['val_map']:.4f}\n")

    kwargs = {"freeze": False}
    if trained.get("checkpoint"):
        kwargs["adapted_checkpoint"] = trained["checkpoint"]
    encoder = build_encoder(trained["encoder"], **kwargs)

    # The head's shape does not depend on how much of the encoder was unfrozen,
    # but the reconstruction is kept faithful anyway: a state dict that loads
    # with unexpected or missing keys would score a different model from the one
    # that was trained, and load_state_dict is set to catch that.
    n_blocks = -1 if trained.get("from_scratch") else trained["unfreeze_blocks"]
    unfreeze_last_blocks(encoder, n_blocks,
                         bool(trained.get("unfreeze_embeddings")
                              or trained.get("from_scratch")))
    model = FineTuneModel(encoder, encoder.token_layout.dim,
                          float(trained["dropout"])).to(device)
    model.load_state_dict(payload["model"], strict=True)

    results: dict[str, Any] = {
        "checkpoint": str(path), "encoder": trained["encoder"],
        "backbone_lr": trained["backbone_lr"],
        "unfreeze_blocks": trained["unfreeze_blocks"],
        "from_scratch": bool(trained.get("from_scratch")),
        "val_map_selected": payload["val_map"], "epoch": payload["epoch"],
        "splits": {},
    }

    dataset_name = trained.get("dataset", "sages")
    # Endoscapes has one held-out split and it is the manifest's own, so the
    # "internal" label would be misleading there.
    test_name = "test" if dataset_name == "endoscapes" else "internal_test"
    splits = [("val", args.manifest_path, "val"),
              (test_name, args.manifest_path, "test")]
    if args.official_manifest:
        splits.append(("official_test", args.official_manifest, "test"))

    print(f"{'split':<16}{'mAP':>9}{'BAcc':>9}{'AUC':>9}{'n':>8}")
    for name, manifest, split in splits:
        loader = build_loader(manifest, split, args.dataset_root,
                              encoder.preprocess_spec, args.batch_size,
                              args.num_workers, dataset_name)
        metrics, logits, targets = score(model, loader, device, amp)
        results["splits"][name] = metrics
        print(f"{name:<16}{metrics['mAP']:>9.4f}"
              f"{metrics.get('mean_bacc', float('nan')):>9.4f}"
              f"{metrics.get('mean_auc', float('nan')):>9.4f}{len(targets):>8}")
        if args.save_logits:
            np.savez(path.parent / f"logits_{name}.npz",
                     logits=logits, targets=targets)

    # The quantity the project's own shrinkage measurement predicts, reported so
    # that a figure quoted from validation can be checked against it.
    val = results["splits"]["val"]["mAP"]
    for name in ("internal_test", "official_test", "test"):
        if name in results["splits"]:
            change = results["splits"][name]["mAP"] - val
            print(f"\n  {name} minus validation: {change:+.4f}")
    print("\n  Frozen-arm shrinkage across fourteen encoders has slope 0.691,")
    print("  and the two depth arms fell 0.049 and 0.087 when scored held out.")
    print("  A validation figure should not be quoted where a held-out one exists.")

    out = path.parent / "held_out_metrics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
