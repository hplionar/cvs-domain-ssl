#!/usr/bin/env python3
"""Exercise the three-arm head comparison on synthetic features.

NOT AN EXPERIMENT. This is a wiring check: it proves the cached-feature probe
can load prefix tokens, build each of the three heads, train them under one
grid and one seed set, and produce a comparison that the verifier accepts. The
mAP values it prints describe fabricated data and carry no information about
CVS, about Endoscapes, or about SMIL.

The synthetic labels are constructed so that each criterion needs *both* a
sparse patch cue and the global token, which is the regime in which a fusion
design should help. That makes it a test of whether the implementation is
capable of expressing the intended advantage, not evidence that the advantage
exists on real features.

Usage:
    python scripts/_smoke_head_comparison.py /tmp/smoke
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

N_TOKENS = 32
DIM = 32
RNG_SEED = 0


def make_cache(path: Path, n: int, rng: np.random.Generator) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    tokens = rng.standard_normal((n, N_TOKENS, DIM)).astype(np.float32) * 0.3
    prefix = rng.standard_normal((n, 1, DIM)).astype(np.float32) * 0.3

    # Each criterion is the AND of a sparse patch cue and a global cue, so
    # neither branch alone is sufficient. Which patch carries the cue varies per
    # sample, which is what makes uniform averaging a poor aggregator: the cue
    # is diluted by 1/N at a different index every time.
    targets = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        for c in range(3):
            patch_cue = rng.random() < 0.5
            global_cue = rng.random() < 0.5
            slot = int(rng.integers(0, N_TOKENS))
            tokens[i, slot, c] += 3.0 if patch_cue else -3.0
            prefix[i, 0, 8 + c] += 3.0 if global_cue else -3.0
            targets[i, c] = float(patch_cue and global_cue)

    np.save(path / "tokens.npy", tokens.astype(np.float16))
    np.save(path / "prefix.npy", prefix.astype(np.float16))
    np.save(path / "targets.npy", targets)

    with open(path / "index.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "video_id", "row"])
        for i in range(n):
            writer.writerow([f"s{i:05d}", f"v{i // 16:03d}", i])

    manifest = {
        "encoder": {
            "checkpoint_id": "synthetic",
            "token_layout": {"grid": [8, 4], "dim": DIM, "num_prefix_tokens": 1},
        },
        "transform": {"type": "deterministic_eval", "image_size": 224},
        "extraction": {"cache_dtype": "float16", "reduction": "none"},
        "shapes": {"tokens": [n, N_TOKENS, DIM], "prefix": [n, 1, DIM]},
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoke_heads")
    rng = np.random.default_rng(RNG_SEED)

    train = make_cache(root / "cache/train", 600, rng)
    val = make_cache(root / "cache/val", 300, rng)

    for head in ("mean", "attentive", "fusion"):
        extra = ["--global-source", "cls"] if head == "fusion" else []
        print(f"\n{'=' * 62}\narm: {head}\n{'=' * 62}", flush=True)
        subprocess.run(
            [
                sys.executable, "-m", "train.train_probe_cached",
                "--train-features", str(train),
                "--val-features", str(val),
                "--head", head, *extra,
                "--seeds", "3",
                "--epochs", "30",
                "--patience", "10",
                "--lr", "1e-3", "3e-3",
                "--weight-decay", "0.0",
                "--dropout", "0.0",
                "--attn-hidden", "128", "512",
                "--device", "cpu",
                "--output-dir", str(root / "out" / head),
            ],
            check=True,
        )

    subprocess.run(
        [
            sys.executable, "scripts/compare_heads.py",
            "--run", f"mean={root / 'out/mean'}",
            "--run", f"attentive={root / 'out/attentive'}",
            "--run", f"fusion={root / 'out/fusion'}",
            "--reference", "attentive",
            "--output-md", str(root / "head_comparison.md"),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
