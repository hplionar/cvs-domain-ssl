#!/usr/bin/env python3
"""Add ``--layer-depths`` to scripts/extract_features.py.

Extraction currently caches the final layer only. This patch lets it cache a
selected set of intermediate layers alongside it, written to ``layers.npy`` with
shape ``[N, L, num_tokens, D]``, so that a head can learn which depths matter --
potentially differently for each CVS criterion.

Depths are relative, in (0, 1]. VideoMAE ViT-B has 12 blocks and V-JEPA 2 ViT-L
has 24, so ``0.25`` means block 3 in one and block 6 in the other. Absolute
indices would make the two arms incomparable.

``tokens.npy`` is still written and is still the final layer, so every existing
cache consumer keeps working and a depth cache is a strict superset of an
ordinary one.

Disk cost is linear in the number of depths: a SAGES train split for VideoMAE
ViT-B is 22.6 GiB at one layer, 90 GiB at four.

Run from the repository root:

    python apply_layer_depths_patch.py
    PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("scripts/extract_features.py")


def patch(old: str, new: str, *, description: str, marker: str) -> None:
    text = PATH.read_text()
    if marker in text:
        print(f"  SKIP  {description}")
        return
    if old not in text:
        raise SystemExit(
            f"FAILED: anchor for '{description}' not found.\nSought:\n{old[:300]}"
        )
    PATH.write_text(text.replace(old, new, 1))
    print(f"  OK    {description}")


# --------------------------------------------------------------------------
# 1. CLI flag
# --------------------------------------------------------------------------

patch(
    old='''    p.add_argument("--dry-run", action="store_true", help="report shapes and cache size, write nothing")''',
    new='''    p.add_argument(
        "--layer-depths",
        type=float,
        nargs="+",
        default=None,
        metavar="D",
        help=(
            "Relative depths in (0, 1] at which to additionally cache "
            "intermediate layers, e.g. --layer-depths 0.25 0.5 0.75 1.0. "
            "Relative rather than absolute so that a depth means the same "
            "fraction of the network in a 12-block and a 24-block encoder. "
            "Written to layers.npy as [N, L, tokens, D]; tokens.npy still holds "
            "the final layer. Disk cost is linear in the number of depths."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="report shapes and cache size, write nothing")''',
    description="--layer-depths flag",
    marker='"--layer-depths"',
)


# --------------------------------------------------------------------------
# 2. Validation and size reporting
# --------------------------------------------------------------------------

patch(
    old='''    gib = n * n_tokens * dim * 2 / 1024**3''',
    new='''    depths = None if args.layer_depths is None else tuple(args.layer_depths)
    if depths is not None:
        if args.reduction != "none":
            raise SystemExit(
                f"--layer-depths requires --reduction none, got "
                f"{args.reduction!r}. Reducing every layer would discard the "
                f"spatial structure the depth comparison is meant to examine."
            )
        if any(not 0.0 < d <= 1.0 for d in depths):
            raise SystemExit(
                f"Layer depths must lie in (0, 1], got {depths}. Depth 0 is the "
                f"embedding output, which precedes every transformer block."
            )

    gib = n * n_tokens * dim * 2 / 1024**3
    if depths is not None:
        gib += n * len(depths) * n_tokens * dim * 2 / 1024**3''',
    description="depth validation and size estimate",
    marker="--layer-depths requires --reduction none",
)

patch(
    old='''    print(f"cache size   {gib:.2f} GiB")''',
    new='''    if depths is not None:
        print(f"layers       {depths}  ->  extra [{n}, {len(depths)}, {n_tokens}, {dim}]")
    print(f"cache size   {gib:.2f} GiB")''',
    description="report requested depths",
    marker='print(f"layers       {depths}',
)


# --------------------------------------------------------------------------
# 3. Allocate the memmap
# --------------------------------------------------------------------------

patch(
    old='''    targets = np.zeros((n, 3), dtype=np.float32)''',
    new='''    layers_mm = None
    if depths is not None:
        layers_mm = np.lib.format.open_memmap(
            out_dir / "layers.npy",
            mode="w+",
            dtype=np.float16,
            shape=(n, len(depths), n_tokens, dim),
        )

    targets = np.zeros((n, 3), dtype=np.float32)''',
    description="allocate layers.npy",
    marker='out_dir / "layers.npy"',
)


# --------------------------------------------------------------------------
# 4. Request and write the layers
# --------------------------------------------------------------------------

patch(
    old='''            with autocast:
                out = encoder(inputs)''',
    new='''            with autocast:
                out = encoder(inputs, layer_depths=depths)''',
    description="request depths in the forward pass",
    marker="encoder(inputs, layer_depths=depths)",
)

patch(
    old='''            if prefix_mm is not None and out.prefix is not None:
                prefix_mm[cursor : cursor + size] = out.prefix.float().half().cpu().numpy()''',
    new='''            if prefix_mm is not None and out.prefix is not None:
                prefix_mm[cursor : cursor + size] = out.prefix.float().half().cpu().numpy()
            if layers_mm is not None:
                if out.hidden_states is None:
                    raise RuntimeError(
                        f"{type(encoder).__name__} returned no hidden states "
                        f"despite layer_depths={depths}. Writing the cache "
                        f"anyway would leave layers.npy silently zero-filled."
                    )
                layers_mm[cursor : cursor + size] = (
                    out.hidden_states.float().half().cpu().numpy()
                )''',
    description="write layers.npy",
    marker="returned no hidden states",
)

patch(
    old='''    tokens_mm.flush()
    if prefix_mm is not None:
        prefix_mm.flush()''',
    new='''    tokens_mm.flush()
    if prefix_mm is not None:
        prefix_mm.flush()
    if layers_mm is not None:
        layers_mm.flush()''',
    description="flush layers.npy",
    marker="layers_mm.flush()",
)


# --------------------------------------------------------------------------
# 5. Record it in the manifest
# --------------------------------------------------------------------------

patch(
    old='''        "prefix": None if prefix_mm is None else [n, layout.num_prefix_tokens, dim],
    }''',
    new='''        "prefix": None if prefix_mm is None else [n, layout.num_prefix_tokens, dim],
        "layers": None if layers_mm is None else [n, len(depths), n_tokens, dim],
    }
    # Recorded so that verify_protocol can refuse to compare a depth cache
    # against a final-layer-only one, and so the resolved block indices are
    # recoverable from the cache alone.
    manifest["layer_depths"] = None if depths is None else list(depths)''',
    description="record depths in the manifest",
    marker='manifest["layer_depths"]',
)


print("\nDone. Now run:")
print("  python -c \"import ast; ast.parse(open('scripts/extract_features.py').read())\"")
print("  PYTHONPATH=. python -m pytest tests/ -q")
