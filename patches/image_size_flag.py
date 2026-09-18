"""Input resolution as a controlled factor.

DINOv3Encoder gains an optional image_size that overrides the checkpoint
config's 224 when computing the grid, preprocess spec and token layout (the
HF DINOv3 forward is resolution-flexible via rotary embeddings). finetune_cvs
exposes --image-size and passes it through; score_finetune reads it back from
the run's saved args so a checkpoint is always scored at the size it trained at.
"""
from pathlib import Path
def rep(path, old, new):
    p = Path(path); src = p.read_text()
    n = src.count(old); assert n == 1, f"{path}: expected 1 match, found {n}:\n{old[:80]}"
    p.write_text(src.replace(old, new)); print("patched", path)

rep("models/encoders/dinov3_encoder.py",
'''        adapted_checkpoint: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)
''',
'''        adapted_checkpoint: str | Path | None = None,
        freeze: bool = True,
        image_size: int | None = None,
    ) -> None:
        super().__init__(freeze=freeze)
        self._image_size_override = int(image_size) if image_size else None
''')

rep("models/encoders/dinov3_encoder.py",
'''        image_size = int(cfg_get(cfg, "image_size"))
        patch_size = int(cfg_get(cfg, "patch_size"))
''',
'''        image_size = self._image_size_override or int(cfg_get(cfg, "image_size"))
        patch_size = int(cfg_get(cfg, "patch_size"))
''')

rep("train/finetune_cvs.py",
'''    p.add_argument("--seed", type=int, default=0)
''',
'''    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=None,
                   help="input resolution in pixels; default is the encoder "
                        "checkpoint's own (224). Saved with the run so scoring "
                        "uses the same size.")
''')

rep("train/finetune_cvs.py",
'''    if args.checkpoint:
        kwargs["adapted_checkpoint"] = args.checkpoint
    encoder = build_encoder(args.encoder, **kwargs)
''',
'''    if args.checkpoint:
        kwargs["adapted_checkpoint"] = args.checkpoint
    if args.image_size:
        kwargs["image_size"] = args.image_size
    encoder = build_encoder(args.encoder, **kwargs)
''')

rep("eval/score_finetune.py",
'''        kwargs["adapted_checkpoint"] = trained["checkpoint"]
    encoder = build_encoder(trained["encoder"], **kwargs)
''',
'''        kwargs["adapted_checkpoint"] = trained["checkpoint"]
    if trained.get("image_size"):
        kwargs["image_size"] = trained["image_size"]
    encoder = build_encoder(trained["encoder"], **kwargs)
''')
