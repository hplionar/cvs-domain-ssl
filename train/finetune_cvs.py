#!/usr/bin/env python3
"""Fine-tune a frozen-probe encoder, sweeping the backbone learning rate.

Every result in this project concerns frozen features. Whether the ranking those
results establish survives fine-tuning is untested, and is the main threat to
their external validity: if an encoder that leads frozen trails once the backbone
is updated, the comparison applies to frozen features only and must say so.

Testing that with a single recipe per arm is unsound. A null would be
indistinguishable from a badly chosen learning rate, and the encoder is
pretrained while the head is not, so a rate suitable for one destroys the other.
This script therefore varies **only** the backbone learning rate and holds
everything else fixed, so that the result is a sensitivity curve rather than a
point that might be an artefact of one setting.

    frozen    the control. Should reproduce the existing frozen-probe figure;
              if it does not, something differs beyond the backbone update and
              the comparison is invalid.
    1e-6      conservative
    1e-5      typical for ViT fine-tuning at this scale
    1e-4      aggressive; expected to degrade

**Report the curve, not the best point.** Selecting the best of four rates on
validation is a selection, and Section 2.6 of the notes measures what selection
costs on this benchmark. State the rate chosen, state that it was chosen on
validation, and report the whole sweep.

**Only the last blocks are updated.** Fine-tuning all twelve blocks of a ViT-B on
10,080 frames overfits and is not what published surgical work does; SwinCVS
updates a backbone under a comparable budget. The count is exposed rather than
fixed so that it can be reported.

Usage:
    python train/finetune_cvs.py \\
        --encoder dinov3_b \\
        --dataset-root ../datasets/SAGES_CVS_Challenge_2024 \\
        --manifest-path metadata/sages_frames_internal_split.csv \\
        --backbone-lr 1e-5 --unfreeze-blocks 4 \\
        --output-dir ../outputs/finetune/dinov3_b_lr1e-5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.transforms import build_transform_from_spec
from eval.metrics import compute_multilabel_metrics_from_logits
from models.encoders import build_encoder
from models.heads.linear_head import LinearCVSHead

# Registration side effects, as in scripts/extract_features.py.
import models.encoders.dinov2_encoder  # noqa: F401
import models.encoders.dinov3_encoder  # noqa: F401
import models.encoders.ijepa_encoder  # noqa: F401
import models.encoders.mae_encoder  # noqa: F401
import models.encoders.vit_sup_encoder  # noqa: F401


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transformer_blocks(encoder: nn.Module) -> list[nn.Module]:
    """The encoder's sequence of transformer blocks.

    Located by structure rather than by name, since the HuggingFace classes
    differ between families: Dinov2Model and ViTMAEModel nest the block list at
    ``.encoder.layer``, DINOv3ViTModel at ``.layer``. Raising rather than
    guessing, because silently unfreezing nothing would produce a run that looks
    like fine-tuning and is not.
    """
    inner = getattr(encoder, "model", encoder)
    # Verified against the checkpoints in use: DINOv3ViTModel nests its stack
    # at .model.layer, ViTMAEModel exposes .layers, and the Dinov2 and ViT
    # families use .encoder.layer. Ordered most specific first so that a nested
    # match is not shadowed by a shallower one.
    for path in (("model", "layer"), ("layers",), ("encoder", "layer"),
                 ("layer",), ("encoder", "layers"), ("blocks",),
                 ("encoder", "blocks")):
        node = inner
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None and isinstance(node, (nn.ModuleList, nn.Sequential)):
            return list(node)
    raise RuntimeError(
        f"Could not locate the transformer blocks of {type(inner).__name__}. "
        f"Partial unfreezing cannot proceed; add the attribute path to "
        f"transformer_blocks()."
    )


def unfreeze_last_blocks(encoder: nn.Module, n: int,
                         embeddings: bool = False) -> tuple[int, int]:
    """Unfreeze the final n blocks and the final normalisation layer.

    The final norm is included because it sits after the last block and its
    statistics would otherwise be fixed to the pretraining distribution while the
    blocks beneath it move.
    """
    for p in encoder.parameters():
        p.requires_grad_(False)

    blocks = transformer_blocks(encoder)
    if n < 0:
        n = len(blocks)
    if n == 0 and not embeddings:
        return 0, sum(p.numel() for p in encoder.parameters())
    if n > len(blocks):
        raise SystemExit(f"--unfreeze-blocks {n} exceeds the {len(blocks)} blocks "
                         f"this encoder has.")
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad_(True)

    inner = getattr(encoder, "model", encoder)
    for name in ("layernorm", "norm", "final_layernorm"):
        mod = getattr(inner, name, None)
        if isinstance(mod, nn.Module):
            for p in mod.parameters():
                p.requires_grad_(True)
            break

    if embeddings:
        # A randomly initialised patch projection left frozen would mean the
        # model never trains from scratch: the transformer would sit on a fixed
        # random projection of pixels.
        emb = getattr(inner, "embeddings", None)
        if emb is None:
            raise RuntimeError(
                f"No embeddings module on {type(inner).__name__}, so the patch "
                f"projection cannot be unfrozen."
            )
        for p in emb.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    return trainable, frozen


class FineTuneModel(nn.Module):
    """Encoder plus mean-pooled linear head, matching the frozen probe.

    Pooling is the mean over patch tokens and the head is LayerNorm plus Linear,
    exactly as in the frozen protocol, so that a difference between this and the
    frozen result is attributable to updating the backbone and not to a change
    of head.
    """

    def __init__(self, encoder: nn.Module, dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = LinearCVSHead(dim, dropout=dropout)

    def forward(self, x):
        out = self.encoder(x)
        return self.head(out.tokens.mean(dim=1))


def build_loaders(args, spec):
    from data.sages_datasets import SAGESFrameDataset

    transform = build_transform_from_spec(spec)
    common = dict(manifest_path=args.manifest_path, dataset_root=args.dataset_root,
                  transform=transform)
    train = SAGESFrameDataset(split="train", **common)
    val = SAGESFrameDataset(split="val", **common)
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers, pin_memory=True, drop_last=True),
        DataLoader(val, batch_size=args.batch_size * 2, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True),
    )


def unpack(batch, device):
    """SAGESFrameDataset yields a dict or a tuple depending on its mode."""
    if isinstance(batch, dict):
        x = batch.get("image", batch.get("pixel_values"))
        y = batch.get("labels", batch.get("target"))
    else:
        x, y = batch[0], batch[-1]
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True).float()


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    logits, targets = [], []
    for batch in loader:
        x, y = unpack(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        logits.append(out.float().cpu().numpy())
        targets.append(y.cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True, help="registry name, e.g. dinov3_b")
    p.add_argument("--from-scratch", action="store_true",
                   help="discard the pretrained weights and reinitialise the "
                        "encoder, keeping its architecture. The control that "
                        "says what pretraining is worth: Kornblith et al. find "
                        "fine-tuning beats random initialisation by 0.6 points "
                        "on Stanford Cars and 0.2 on FGVC Aircraft, both tasks "
                        "whose concepts ImageNet does not contain, while buying "
                        "a seventeenfold convergence speedup. Whether the same "
                        "holds here is untested.")
    p.add_argument("--checkpoint", default=None,
                   help="adapted encoder weights, if fine-tuning an adapted arm")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="0 freezes the encoder, which is the control")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-blocks", type=int, default=4,
                   help="trailing transformer blocks to update. -1 unfreezes "
                        "every block, which is what --from-scratch forces and "
                        "what a pretrained control must match to be comparable.")
    p.add_argument("--unfreeze-embeddings", action="store_true",
                   help="also update the patch-embedding projection and prefix "
                        "tokens. Implied by --from-scratch: a frozen random "
                        "embedding would leave the model training on a fixed "
                        "random projection of pixels.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"

    kwargs = {"freeze": False}
    if args.checkpoint:
        kwargs["adapted_checkpoint"] = args.checkpoint
    encoder = build_encoder(args.encoder, **kwargs)

    if args.from_scratch:
        if args.checkpoint:
            raise SystemExit("--from-scratch and --checkpoint are exclusive: "
                             "one discards the weights the other loads.")
        # Reinitialise in place rather than constructing from config, so that
        # the architecture, preprocessing and token layout are identical to the
        # pretrained arm by construction and the only difference is the weights.
        # HuggingFace's _init_weights implements each model's published
        # initialisation, so this is the initialisation the checkpoint itself
        # started from.
        inner = getattr(encoder, "model", encoder)
        inner.apply(inner._init_weights)
        if hasattr(inner, "post_init"):
            inner.post_init()
        print("encoder reinitialised: pretrained weights discarded")

    frozen_control = args.backbone_lr <= 0
    n_blocks = 0 if frozen_control else args.unfreeze_blocks
    unfreeze_emb = args.unfreeze_embeddings
    if args.from_scratch:
        # Every parameter is random, so freezing any of it would measure a fixed
        # random projection. Reported rather than applied silently, because a
        # pretrained control must be given the same settings to be comparable.
        n_blocks, unfreeze_emb = -1, True
        print("from scratch: every block and the patch embedding unfrozen")
    trainable, frozen = unfreeze_last_blocks(encoder, n_blocks, unfreeze_emb)

    model = FineTuneModel(encoder, encoder.token_layout.dim, args.dropout).to(device)
    train_loader, val_loader = build_loaders(args, encoder.preprocess_spec)

    # Two parameter groups. A single rate across a pretrained encoder and a fresh
    # head either leaves the head undertrained or destroys the encoder.
    groups = [{"params": [p for p in model.head.parameters()], "lr": args.head_lr}]
    backbone = [p for p in model.encoder.parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.BCEWithLogitsLoss()

    print(f"encoder        {encoder.checkpoint_id}"
          + ("  [weights discarded]" if args.from_scratch else ""))
    print(f"blocks         {n_blocks} of {len(transformer_blocks(encoder))} unfrozen"
          + ("  (frozen control)" if frozen_control else ""))
    print(f"parameters     {trainable:,} trainable, {frozen:,} frozen, "
          f"{sum(p.numel() for p in model.head.parameters()):,} in the head")
    print(f"learning rate  head {args.head_lr:.0e}, backbone "
          f"{args.backbone_lr:.0e}" + ("  (no update)" if frozen_control else ""))
    print(f"schedule       {args.epochs} epochs, batch {args.batch_size}, "
          f"amp={'fp16' if amp else 'off'}\n")

    best, best_epoch, history = -1.0, 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x, y = unpack(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * x.shape[0]
            n += x.shape[0]
        sched.step()

        logits, targets = evaluate(model, val_loader, device, amp)
        metrics = compute_multilabel_metrics_from_logits(targets, logits)
        row = {"epoch": epoch, "train_loss": total / max(n, 1),
               **{k: float(v) for k, v in metrics.items()}}
        history.append(row)
        marker = ""
        if not np.isnan(metrics["mAP"]) and metrics["mAP"] > best:
            best, best_epoch = float(metrics["mAP"]), epoch
            marker = "  *"
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_map": best, "args": vars(args)},
                       out_dir / "best.pt")
        print(f"epoch {epoch:>3}/{args.epochs}  loss {row['train_loss']:.4f}  "
              f"val mAP {metrics['mAP']:.4f}{marker}", flush=True)

    elapsed = time.time() - started
    print(f"\nbest val mAP {best:.4f} at epoch {best_epoch}")
    print(f"elapsed {elapsed/3600:.2f}h")
    if device.type == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GiB")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump({"encoder": encoder.checkpoint_id,
                   "from_scratch": bool(args.from_scratch),
                   "args": vars(args),
                   "blocks_unfrozen": n_blocks, "trainable_params": trainable,
                   "best_val_map": best, "best_epoch": best_epoch,
                   "elapsed_hours": elapsed / 3600, "history": history}, fh, indent=2)
    print(f"written to {out_dir/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
