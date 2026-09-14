#!/usr/bin/env python3
"""Add `freeze_last_blocks` to the DINO trainer.

The depth probe shows continued self-distillation improves DINOv3 at blocks 3,
6 and 9 by 0.023 to 0.041 and by exactly nothing at block 12, so the final
blocks discard the gain. DINO's loss acts on the representation the last block
produces, which is a candidate explanation: the objective reshapes the top of
the stack hardest, and that is where the damage appears.

This makes the hypothesis testable. Freezing the final k blocks of the student
means the loss cannot update them, and the question is whether the improvement
then survives into the output.

Three consequences, all of which belong in the write-up:

1. The teacher is an exponential moving average of the student, so freezing the
   student's last k blocks leaves the teacher's last k at their pretrained
   values -- an EMA of a constant is that constant. The loss is therefore
   computed against a target whose final blocks are the original representation.
   That is arguably the point rather than a side effect: the gradient reshapes
   the middle of the stack while the objective stays anchored to what already
   worked. It is also a confound if the result is positive, and must be stated.

2. The projection head is never frozen. It is initialised fresh and carries no
   pretrained representation, so freezing it would leave the loss with nothing
   to optimise through.

3. Fewer trainable parameters at a fixed learning rate is not a matched
   comparison. The fine-tuning sweep showed the optimum rate moves with the
   number of updated blocks. Either sweep the rate or state that it was held
   fixed and that the comparison is therefore conservative.

Run from the repository root:
    python dino_freeze_patch.py
"""

from __future__ import annotations

import pathlib
import sys


def patch(path: str, old: str, new: str, *, description: str) -> None:
    p = pathlib.Path(path)
    text = p.read_text()
    if new.strip().splitlines()[0] in text:
        print(f"  SKIP  {description} (already applied)")
        return
    if old not in text:
        raise SystemExit(
            f"FAILED: anchor for '{description}' not found in {path}.\n"
            f"Sought:\n{old[:300]}"
        )
    p.write_text(text.replace(old, new, 1))
    print(f"  OK    {description}")


def main() -> int:
    target = "train/pretrain_dino.py"
    if not pathlib.Path(target).is_file():
        raise SystemExit(f"{target} not found; run from the repository root.")

    patch(
        target,
        '''    if config["model"].get("gradient_checkpointing", False):
        student.backbone.gradient_checkpointing_enable()

    return student.to(device), teacher.to(device), dim, out_dim''',
        '''    freeze_last = int(config["model"].get("freeze_last_blocks", 0))
    if freeze_last:
        blocks = _transformer_blocks(student.backbone)
        if freeze_last > len(blocks):
            raise SystemExit(
                f"freeze_last_blocks={freeze_last} exceeds the {len(blocks)} "
                f"blocks this backbone has."
            )
        n_frozen = 0
        for block in blocks[-freeze_last:]:
            for param in block.parameters():
                param.requires_grad = False
                n_frozen += param.numel()
        # The terminal norm sits after the last block and is frozen with it:
        # leaving it trainable would let the loss rescale the output of blocks
        # it cannot otherwise reach, which is the effect being tested.
        inner = getattr(student.backbone, "model", student.backbone)
        for name in ("layernorm", "norm", "final_layernorm"):
            mod = getattr(inner, name, None)
            if isinstance(mod, torch.nn.Module):
                for param in mod.parameters():
                    param.requires_grad = False
                    n_frozen += param.numel()
                break
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"froze the last {freeze_last} of {len(blocks)} blocks and the "
              f"terminal norm: {n_frozen:,} parameters held fixed, "
              f"{trainable:,} trainable")

    if config["model"].get("gradient_checkpointing", False):
        student.backbone.gradient_checkpointing_enable()

    return student.to(device), teacher.to(device), dim, out_dim''',
        description="freeze the student's trailing blocks",
    )

    patch(
        target,
        '''    optimizer = torch.optim.AdamW(
        student.parameters(), lr=base_lr,''',
        '''    # Frozen parameters are excluded rather than passed with requires_grad
        # False: AdamW would otherwise allocate moment buffers for them, and
        # weight decay is applied to parameters in the group regardless of
        # whether they receive gradients.
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=base_lr,''',
        description="exclude frozen parameters from the optimiser",
    )

    # The block-locating helper, appended near the top-level helpers.
    p = pathlib.Path(target)
    text = p.read_text()
    helper = '''

def _transformer_blocks(backbone) -> list:
    """The backbone's sequence of transformer blocks.

    Located by structure rather than by name: DINOv3ViTModel nests its stack at
    .model.layer, ViTMAEModel exposes .layers, and the Dinov2 and ViT families
    use .encoder.layer. Raising rather than returning an empty list, because
    freezing nothing while reporting success would produce a run that looks like
    the experiment and is not.
    """
    inner = getattr(backbone, "model", backbone)
    for path in (("model", "layer"), ("layers",), ("encoder", "layer"),
                 ("layer",), ("encoder", "layers"), ("blocks",)):
        node = inner
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None and isinstance(
            node, (torch.nn.ModuleList, torch.nn.Sequential)
        ):
            return list(node)
    raise RuntimeError(
        f"Could not locate the transformer blocks of {type(inner).__name__}; "
        f"add its attribute path to _transformer_blocks()."
    )
'''
    if "_transformer_blocks" in text and "def _transformer_blocks" not in text:
        raise SystemExit("_transformer_blocks is referenced but not defined.")
    if "def _transformer_blocks" not in text:
        anchor = "\ndef save_checkpoint("
        if anchor not in text:
            raise SystemExit("Could not find an insertion point for the helper.")
        text = text.replace(anchor, helper + anchor, 1)
        p.write_text(text)
        print("  OK    added _transformer_blocks helper")
    else:
        print("  SKIP  _transformer_blocks helper (already present)")

    print("\nPatched. Verify with:")
    print("  python -c \"import ast;ast.parse(open('train/pretrain_dino.py').read());print('parses')\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
