#!/usr/bin/env python3
"""Compare encoder weights before and after continued pretraining.

The first question about any SSL run is whether it changed the encoder at all,
and a training loss that decreases does not answer it: a masked-reconstruction
objective can be driven down by the decoder alone while the encoder barely
moves. That failure produces a plausible loss curve, a valid checkpoint, and an
adaptation gain of zero, discovered only after the probe has been run.

Reports, per parameter and aggregated per block:

    relative change = ||w_after - w_before||_2 / ||w_before||_2

Two patterns are worth acting on. **No movement anywhere in the encoder** means
gradients are not reaching it — a frozen module, a misconfigured optimiser, or
a checkpoint saved from the wrong object. **Movement concentrated entirely in
the last blocks** is normal and expected; movement concentrated in the first
blocks is not, and usually indicates too high a learning rate.

Usage:
    python scripts/compare_weights.py \\
        --before MCG-NJU/videomae-base \\
        --after outputs/ssl/videomae_b_sages/latest.pt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ARCHITECTURES = {
    "videomae": {
        "reference_class": "VideoMAEForPreTraining",
        "encoder_attr": "videomae",
        "state_prefix": "videomae.",
        "repair_bias": True,
    },
    "vjepa2": {
        "reference_class": "VJEPA2Model",
        "encoder_attr": "encoder",
        "state_prefix": "encoder.",
        "repair_bias": False,
    },
}


def detect_architecture(checkpoint_name: str, payload: dict[str, Any] | None = None) -> str:
    """Identify the architecture from the checkpoint identifier or payload.

    Comparing a VideoMAE reference against a V-JEPA checkpoint produces a
    well-formed but meaningless report: every key is missing on both sides and
    the tool reports zero shared parameters rather than failing. Detecting the
    architecture makes that impossible.
    """
    haystack = (checkpoint_name or "").lower()
    if payload:
        haystack += " " + str(payload.get("config", {}).get("model", {}).get("checkpoint", "")).lower()

    for key in ("vjepa2", "vjepa"):
        if key in haystack:
            return "vjepa2"
    if "videomae" in haystack:
        return "videomae"

    raise ValueError(
        f"Cannot identify the architecture from {checkpoint_name!r}. Supported: "
        f"{sorted(ARCHITECTURES)}. Pass --architecture explicitly."
    )


def load_reference(name: str, architecture: str) -> dict[str, torch.Tensor]:
    """Encoder weights from the original checkpoint.

    For VideoMAE the attention biases are repaired first. Without that, the
    comparison is against weights the training run never started from, and the
    36 biases appear as zero-norm parameters whose movement is discarded.
    V-JEPA has no equivalent problem: its checkpoint stores query, key and value
    biases under the names transformers expects.
    """
    spec = ARCHITECTURES[architecture]

    if architecture == "videomae":
        from transformers import VideoMAEForPreTraining

        from models.encoders.videomae_encoder import repair_qkv_bias

        model = VideoMAEForPreTraining.from_pretrained(name)
        repair_qkv_bias(model, name)
        encoder = model.videomae
    else:
        from transformers import VJEPA2Model

        encoder = VJEPA2Model.from_pretrained(name).encoder

    return {k: v.detach().cpu().float() for k, v in encoder.state_dict().items()}


def load_trained(path: Path, architecture: str | None = None) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    """Encoder weights from a training checkpoint or an exported encoder.

    Handles both ``latest.pt`` (full training state, encoder keys prefixed with
    ``videomae.``) and ``encoder_final.pt`` (encoder only).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = {k: payload.get(k) for k in ("step", "epoch") if k in payload}

    base = payload.get("config", {}).get("model", {}).get("checkpoint", "")
    architecture = architecture or detect_architecture(str(path), payload)
    prefix = ARCHITECTURES[architecture]["state_prefix"]

    state = payload.get("model", payload)
    encoder = {
        k[len(prefix):]: v.detach().cpu().float()
        for k, v in state.items()
        if k.startswith(prefix)
    }
    if not encoder:
        encoder = {k: v.detach().cpu().float() for k, v in state.items()}

    meta["base_checkpoint"] = base
    return encoder, meta, architecture


def block_of(name: str) -> str:
    """Group a parameter name into an architectural block for aggregation."""
    # VideoMAE nests blocks under encoder.layer.N; V-JEPA names them layer.N.
    match = re.search(r"(?:encoder\.)?layer\.(\d+)\.", name)
    if match:
        return f"layer_{int(match.group(1)):02d}"
    if "embeddings" in name:
        return "embeddings"
    if name.startswith("layernorm") or name.endswith("layernorm.weight"):
        return "final_norm"
    return "other"


def compare(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    zero_threshold: float = 1e-8,
) -> dict[str, Any]:
    shared = sorted(set(before) & set(after))
    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))

    rows: list[dict[str, Any]] = []
    for name in shared:
        w0, w1 = before[name], after[name]
        if w0.shape != w1.shape:
            rows.append({"name": name, "error": f"shape {tuple(w0.shape)} vs {tuple(w1.shape)}"})
            continue

        norm0 = w0.norm().item()
        delta = (w1 - w0).norm().item()
        rows.append({
            "name": name,
            "block": block_of(name),
            "numel": w0.numel(),
            "norm_before": norm0,
            "norm_after": w1.norm().item(),
            "abs_change": delta,
            # Relative change is undefined for an all-zero reference, which is
            # exactly the case for VideoMAE's reinitialised attention biases.
            "rel_change": delta / norm0 if norm0 > zero_threshold else float("nan"),
            "was_zero": norm0 <= zero_threshold,
        })

    blocks: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if "error" not in row and not row["was_zero"]:
            blocks[row["block"]].append(row["rel_change"])

    block_summary = {
        name: {
            "mean_rel_change": sum(values) / len(values),
            "max_rel_change": max(values),
            "num_params": len(values),
        }
        for name, values in sorted(blocks.items())
    }

    moved = [r for r in rows if "error" not in r and not r["was_zero"] and r["abs_change"] > zero_threshold]
    unchanged = [r for r in rows if "error" not in r and not r["was_zero"] and r["abs_change"] <= zero_threshold]

    return {
        "num_shared": len(shared),
        "only_in_before": only_before,
        "only_in_after": only_after,
        "num_moved": len(moved),
        "num_unchanged": len(unchanged),
        "unchanged_names": [r["name"] for r in unchanged][:20],
        "blocks": block_summary,
        "parameters": rows,
    }


def render(result: dict[str, Any], top: int = 10) -> None:
    print("=" * 68)
    print("ENCODER WEIGHT COMPARISON")
    print("=" * 68)
    print(f"shared parameters   {result['num_shared']}")
    print(f"moved               {result['num_moved']}")
    print(f"unchanged           {result['num_unchanged']}")

    if result["only_in_before"]:
        print(f"missing after       {len(result['only_in_before'])}: "
              f"{result['only_in_before'][:3]}")
    if result["only_in_after"]:
        print(f"new after           {len(result['only_in_after'])}: "
              f"{result['only_in_after'][:3]}")

    if result["num_moved"] == 0:
        print("\n  NO PARAMETER CHANGED. The encoder did not train. Check that it "
              "is not frozen,\n  that its parameters are in the optimiser, and "
              "that the checkpoint was saved\n  from the trained object.")
        return

    print("\nper block, mean relative change:")
    print(f"  {'block':14s} {'mean':>10s} {'max':>10s} {'params':>8s}")
    for name, stats in result["blocks"].items():
        bar = "#" * min(40, int(stats["mean_rel_change"] * 2000))
        print(f"  {name:14s} {stats['mean_rel_change']:10.5f} "
              f"{stats['max_rel_change']:10.5f} {stats['num_params']:8d}  {bar}")

    ranked = sorted(
        (r for r in result["parameters"] if "error" not in r and not r["was_zero"]),
        key=lambda r: r["rel_change"], reverse=True,
    )
    print(f"\nlargest {top} changes:")
    for row in ranked[:top]:
        print(f"  {row['rel_change']:8.5f}  {row['name']}")

    zeros = [r for r in result["parameters"] if r.get("was_zero")]
    if zeros:
        print(f"\n{len(zeros)} parameters had zero norm before training, so relative "
              f"change is undefined.\nFor VideoMAE these are the attention biases, "
              f"which load as zeros because the\ncheckpoint stores them under names "
              f"transformers does not expect.")

    if result["num_unchanged"]:
        print(f"\n{result['num_unchanged']} parameters did not move at all:")
        for name in result["unchanged_names"]:
            print(f"  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", default=None,
                        help="original checkpoint identifier; read from the "
                             "trained checkpoint's config when omitted")
    parser.add_argument("--architecture", default=None, choices=sorted(ARCHITECTURES),
                        help="override architecture detection")
    parser.add_argument("--after", required=True, help="trained checkpoint path")
    parser.add_argument("--json", default=None, help="write full comparison here")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    peek = torch.load(args.after, map_location="cpu", weights_only=False)
    recorded = peek.get("config", {}).get("model", {}).get("checkpoint")
    checkpoint_name = args.before or recorded
    if not checkpoint_name:
        raise SystemExit(
            "No base checkpoint recorded in the trained file; pass --before."
        )
    architecture = args.architecture or detect_architecture(checkpoint_name, peek)
    del peek

    before = load_reference(checkpoint_name, architecture)
    after, meta, architecture = load_trained(Path(args.after), architecture)

    print(f"architecture  {architecture}")
    print(f"reference     {checkpoint_name}")
    if meta:
        print(f"trained       step {meta.get('step')}, epoch {meta.get('epoch')}")
    print()

    result = compare(before, after)
    render(result, top=args.top)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in result.items() if k != "parameters"} |
                      {"parameters": result["parameters"]}, fh, indent=2)
        print(f"\nWritten to {args.json}")

    return 0 if result["num_moved"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())