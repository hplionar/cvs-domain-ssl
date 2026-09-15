#!/usr/bin/env python3
"""Layer-wise learning-rate decay and warmup for the fine-tuning recipe.

Two symptoms in the current results point at the same omission. Unfreezing all
twelve blocks scores 0.5190 on the official test split against 0.5650 for four
blocks, and five of six Endoscapes runs peak at epoch 1 or 2 of fifteen.
Unfreezing more of an encoder should not be worse than unfreezing less, and a
model that peaks in its second epoch has not been given a schedule it can use.

Both follow from every backbone parameter sharing one learning rate. Early
blocks of a pretrained ViT hold general features -- edges, texture, layout --
and late blocks hold the task-specific ones. A rate suited to adapting block 12
destroys block 3, so unfreezing the whole stack trades a good initialisation for
noise, and the run peaks early because the useful adaptation happens before the
damage accumulates. Restricting to four blocks was a crude way of avoiding this;
decay is the direct one.

**Layer-wise decay.** Each depth gets its own rate:

    lr(layer) = backbone_lr * gamma ** (n_layers - depth)

With gamma = 0.75 and twelve blocks, block 12 trains at the full rate, block 9
at 0.42 of it, block 1 at 0.03, and the patch embedding lower still. Standard
in BEiT, MAE and DeiT-III fine-tuning, and the reason those recipes unfreeze
everything without the failure above.

**Warmup.** One epoch of linear warmup before the cosine decay, so the first
optimiser steps -- taken with a randomly initialised head producing large,
uninformative gradients -- do not move the backbone far.

Both default to off (gamma = 1.0, zero warmup epochs), so every existing
checkpoint remains reproducible and the comparison between recipes is a
controlled one rather than a replacement.

Drop-path is not included: it is a model-construction argument in
transformers, so it cannot be switched on after the encoder is built, and
adding it would mean changing how every arm is loaded.

Run from the repository root:
    python layerwise_decay_patch.py
"""

from __future__ import annotations

import pathlib


OPTIMIZER_OLD = '''    groups = [{"params": [p for p in model.head.parameters()], "lr": args.head_lr}]
    backbone = [p for p in model.encoder.parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)'''

OPTIMIZER_NEW = '''    groups = [{"params": [p for p in model.head.parameters()], "lr": args.head_lr,
               "name": "head"}]
    groups += backbone_groups(model.encoder, encoder, args.backbone_lr,
                              args.layer_decay)
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    # Warmup then cosine. The first steps are taken with a randomly initialised
    # head, whose gradients say nothing about the encoder, so the backbone
    # should not move far while they dominate.
    def lr_scale(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        progress = ((epoch - args.warmup_epochs)
                    / max(args.epochs - args.warmup_epochs, 1))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)'''

HELPER = '''

def backbone_groups(encoder_module, encoder, base_lr: float,
                    decay: float) -> list[dict]:
    """One parameter group per depth, with the rate decayed toward the input.

    A pretrained ViT holds general features in its early blocks and
    task-specific ones in its late blocks. One rate across the stack means a
    rate high enough to adapt the last block also overwrites the first, which is
    why unfreezing twelve blocks scored below unfreezing four. Decaying by depth
    lets the whole encoder be unfrozen without discarding what the early layers
    already encode.

    Parameters outside the transformer blocks are placed by where they sit:
    embeddings receive the most decayed rate, the terminal norm the least, since
    it is read directly by the head.

    With ``decay = 1.0`` every group receives ``base_lr``, which reproduces the
    single-group behaviour exactly.
    """
    blocks = transformer_blocks(encoder)
    n = len(blocks)

    # Map each trainable parameter to a depth: 0 for the embeddings, 1..n for
    # the blocks, n + 1 for anything after them.
    depth_of: dict[int, int] = {}
    for i, block in enumerate(blocks, start=1):
        for p in block.parameters():
            depth_of[id(p)] = i
    for name, p in encoder_module.named_parameters():
        if id(p) in depth_of:
            continue
        depth_of[id(p)] = 0 if "embed" in name else n + 1

    by_depth: dict[int, list] = {}
    for p in encoder_module.parameters():
        if p.requires_grad:
            by_depth.setdefault(depth_of.get(id(p), n + 1), []).append(p)

    groups = []
    for depth in sorted(by_depth):
        # The top of the stack keeps base_lr; each step toward the input scales
        # it by ``decay``. Anything past the last block is treated as the top.
        steps = max(n - depth, 0)
        groups.append({
            "params": by_depth[depth],
            "lr": base_lr * (decay ** steps),
            "name": f"depth_{depth}",
        })
    return groups

'''


def main() -> int:
    p = pathlib.Path("train/finetune_cvs.py")
    if not p.is_file():
        raise SystemExit("train/finetune_cvs.py not found; run from the repo root.")
    s = p.read_text()
    applied = 0

    def sub(old: str, new: str, marker: str, label: str) -> None:
        nonlocal s, applied
        if marker in s:
            print(f"  skip     {label}")
            return
        if old not in s:
            print(f"  NO MATCH {label}")
            return
        s = s.replace(old, new, 1)
        applied += 1
        print(f"  applied  {label}")

    sub(OPTIMIZER_OLD, OPTIMIZER_NEW, "backbone_groups(model.encoder",
        "1 optimizer groups and schedule")

    # The helper goes immediately before the function that uses it.
    sub("\ndef run(", HELPER + "\ndef run(", "def backbone_groups(",
        "2 helper function")

    sub('''    p.add_argument("--weight-decay", type=float, default=0.05)''',
        '''    p.add_argument("--layer-decay", type=float, default=1.0,
                   help="per-depth multiplier on the backbone rate, applied "
                        "toward the input: 0.75 is the usual value for ViT "
                        "fine-tuning. The default of 1.0 gives every depth the "
                        "same rate, reproducing the runs already scored.")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="linear warmup before the cosine decay. The default of "
                        "0 reproduces the existing schedule.")
    p.add_argument("--weight-decay", type=float, default=0.05)''',
        "--layer-decay", "3 flags")

    sub("import time", "import math\nimport time", "import math",
        "4 math import")

    sub('''    print(f"learning rate  head {args.head_lr:.0e}, backbone "
          f"{args.backbone_lr:.0e}"''',
        '''    if args.layer_decay < 1.0:
        rates = [g["lr"] for g in groups if g.get("name", "").startswith("depth")]
        print(f"layer decay    {args.layer_decay}, backbone rates "
              f"{min(rates):.1e} to {max(rates):.1e} across {len(rates)} depths")
    print(f"learning rate  head {args.head_lr:.0e}, backbone "
          f"{args.backbone_lr:.0e}"''',
        "layer decay    {args.layer_decay}", "5 report the rates")

    p.write_text(s)
    print(f"\n{applied}/5 applied")
    print()
    print("Check before submitting:")
    print("  grep -c 'layer_decay\\|backbone_groups\\|warmup_epochs' train/finetune_cvs.py")
    print("  python3 -c \"import ast;ast.parse(open('train/finetune_cvs.py').read())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
