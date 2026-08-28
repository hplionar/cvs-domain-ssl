#!/usr/bin/env python3
"""Print the continued-pretraining throughput table from the job logs.

The figures in the progress report's compute section come from here rather than
from notes, so the table can be regenerated and checked rather than trusted.

Reads the final summary lines each trainer prints — `Elapsed ... | N steps`,
`Peak VRAM ... GiB`, and the last `it/s` from the step log — and the batch size
from the run's config, which is stored inside the checkpoint.

Runs that never completed are reported as such rather than being given
extrapolated figures. A 60-step timing run tells you the rate but not the
elapsed time, and presenting the latter as measured invites exactly the
confusion this script exists to prevent.

Usage:
    python scripts/throughput_table.py
    python scripts/throughput_table.py --markdown
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LOGS = Path("/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs")
SSL = Path("/group/pmc085/hlionar/outputs/cvs-domain-ssl/ssl")

# Run directory -> (display name, input type, log filename stem)
RUNS = [
    ("videomae_b_sages", "VideoMAE ViT-B", "clips", "vmae_sages"),
    ("vjepa2_l_sages", "V-JEPA 2 ViT-L", "clips", "vjepa_sages"),
    ("dinov2_b_sages_sparse", "DINOv2 ViT-B", "frames", "dino_sages"),
    ("mae_b_sages_sparse", "ViT-MAE ViT-B", "frames", "mae_sages"),
]


def scan_logs(stem: str) -> dict:
    """Collect the last reported rate, peak VRAM and elapsed across a run's logs.

    A run may span several array elements. The elements that matter are the ones
    that did work: an element which found the schedule already complete exits in
    seconds and reports a meaningless rate, so those are skipped.
    """
    record: dict = {"rate": None, "vram": None, "elapsed": None, "steps": None}
    total_elapsed = 0.0

    for path in sorted(LOGS.glob(f"{stem}_*.out")):
        text = path.read_text(errors="replace")

        elapsed = re.findall(r"Elapsed\s+([\d.]+)h\s+\|\s+(\d+) steps", text)
        if elapsed:
            hours, steps = float(elapsed[-1][0]), int(elapsed[-1][1])
            # Elements that did no work report ~0h at an absurd rate.
            if hours > 0.01:
                total_elapsed += hours
                record["steps"] = steps

        vram = re.findall(r"Peak VRAM\s+([\d.]+) GiB", text)
        if vram:
            peak = float(vram[-1])
            if record["vram"] is None or peak > record["vram"]:
                record["vram"] = peak

        rates = re.findall(r"([\d.]+) it/s", text)
        if rates:
            # The last line of a substantive element, not of a no-op one.
            if elapsed and float(elapsed[-1][0]) > 0.01:
                record["rate"] = float(rates[-1])
            elif record["rate"] is None:
                record["rate"] = float(rates[-1])

    record["elapsed"] = total_elapsed if total_elapsed > 0 else None
    return record


def batch_from_checkpoint(run: str) -> int | None:
    """Batch size, read from the config embedded in the checkpoint."""
    for name in ("encoder_final.pt", "latest.pt"):
        path = SSL / run / name
        if not path.is_file():
            continue
        try:
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
            return int(payload.get("config", {}).get("train", {}).get("batch_size"))
        except Exception:  # noqa: BLE001 - a missing key is not worth failing on
            return None
    return None


def main() -> int:
    args = parse_args()
    rows = []

    for run, label, kind, stem in RUNS:
        if not (SSL / run).is_dir():
            rows.append((label, kind, None, None, None, None, "not run"))
            continue
        found = scan_logs(stem)
        batch = batch_from_checkpoint(run) if not args.no_torch else None
        complete = (SSL / run / "encoder_final.pt").is_file()
        rows.append((
            label, kind, batch, found["rate"], found["vram"], found["elapsed"],
            "complete" if complete else "incomplete",
        ))

    def cell(value, fmt: str) -> str:
        return "—" if value is None else format(value, fmt)

    if args.markdown:
        print("| Arm | Input | Batch | it/s | Peak VRAM | Elapsed |")
        print("|---|---|---:|---:|---:|---:|")
        for label, kind, batch, rate, vram, elapsed, status in rows:
            if status != "complete":
                print(f"| {label} | {kind} | — | — | — | — |")
                continue
            print(f"| {label} | {kind} | {cell(batch, 'd')} | {cell(rate, '.2f')} | "
                  f"{cell(vram, '.2f')} GiB | {cell(elapsed, '.2f')} h |")
    else:
        header = f"{'arm':18s} {'input':7s} {'batch':>6s} {'it/s':>7s} {'GiB':>7s} {'hours':>7s}  status"
        print(header)
        print("-" * len(header))
        for label, kind, batch, rate, vram, elapsed, status in rows:
            print(f"{label:18s} {kind:7s} {cell(batch,'d'):>6s} {cell(rate,'.2f'):>7s} "
                  f"{cell(vram,'.2f'):>7s} {cell(elapsed,'.2f'):>7s}  {status}")

        print()
        print("samples/s = it/s x batch:")
        for label, kind, batch, rate, vram, elapsed, status in rows:
            if status == "complete" and batch and rate:
                unit = "clips/s" if kind == "clips" else "frames/s"
                print(f"  {label:18s} {rate*batch:6.1f} {unit}")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--markdown", action="store_true",
                   help="emit a markdown table for pasting into the report")
    p.add_argument("--no-torch", action="store_true",
                   help="skip reading batch size from checkpoints; much faster")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
