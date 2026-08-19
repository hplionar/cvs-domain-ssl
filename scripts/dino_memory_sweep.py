#!/usr/bin/env python3
"""Measure the batch ceiling for DINO self-distillation on a single GPU.

WHY THIS EXISTS

DINO is the hardest of the three image objectives to run here, and the reason is
memory rather than compute. Each sample passes through **two** networks -- a
student and an EMA teacher -- and the multi-crop recipe sends 2 global views at
224 px plus 8 local views at 96 px, so a nominal batch of 32 is really 320
forward passes of which 64 also carry a backward.

The published recipe uses batch 1024 across many GPUs. That matters beyond
throughput: DINO's centring subtracts an EMA of the batch mean from the teacher
output, and at small batch that estimate is noisy, which is the mechanism behind
collapse to a constant. So the batch ceiling determines how aggressive the
mitigations have to be:

    comfortable (>= 64)   published defaults are probably fine
    tight (16-32)         raise centre momentum toward 0.996, warm up the
                          teacher temperature from 0.04, monitor collapse
    severe (<= 8)         all of the above plus gradient accumulation, and
                          reduce local crops from 8 to 4

Guessing this wrong is expensive in a specific way: collapse does not announce
itself. The loss falls smoothly while the representation degenerates, exactly as
it did in the V-JEPA arm, where a 76% loss reduction accompanied a 0.127 drop in
downstream mAP. A twelve-hour run can be entirely wasted without a single
warning in the log.

WHAT IT MEASURES

For each batch size: peak VRAM, forward time, backward time, and samples per
second, with the first two iterations discarded as warmup. It stops at the first
out-of-memory error rather than crashing, and reports what succeeded.

The student is DINOv2 ViT-B, the best encoder in the frozen ranking at 0.4593 on
SAGES and therefore the one worth adapting first.

USAGE

    python scripts/dino_memory_sweep.py
    python scripts/dino_memory_sweep.py --batch-sizes 4 8 16 --local-crops 4
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


class DINOHead(nn.Module):
    """The projection head DINO attaches to the backbone.

    Three layers to 2048, then a bottleneck to 256, then a weight-normalised
    linear layer to the output dimension. Included because it is where a
    surprising share of the memory goes: 65,536 output dimensions against a
    768-dimensional backbone means the final layer alone carries roughly 17M
    parameters, more than a fifth of ViT-B itself.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 65536,
                 hidden_dim: int = 2048, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


def build_pair(device: torch.device, checkpoint: str):
    """Student and teacher, each a backbone plus a projection head."""
    from transformers import AutoModel

    student_backbone = AutoModel.from_pretrained(checkpoint)
    teacher_backbone = AutoModel.from_pretrained(checkpoint)

    dim = student_backbone.config.hidden_size
    student = nn.ModuleDict({
        "backbone": student_backbone,
        "head": DINOHead(in_dim=dim),
    }).to(device)
    teacher = nn.ModuleDict({
        "backbone": teacher_backbone,
        "head": DINOHead(in_dim=dim),
    }).to(device)

    # The teacher is updated by EMA, never by gradient. Leaving it trainable
    # would roughly double optimiser state for no purpose.
    for param in teacher.parameters():
        param.requires_grad = False

    return student, teacher, dim


def forward_views(module: nn.ModuleDict, views: list[torch.Tensor]) -> torch.Tensor:
    """Run a list of views through backbone and head, concatenating outputs.

    Views of differing resolution cannot be batched into one tensor, so they are
    grouped by size. This mirrors the reference implementation and matters for
    the measurement: batching all 224 px views in one call and all 96 px views
    in another is substantially cheaper than ten separate calls.
    """
    by_size: dict[int, list[torch.Tensor]] = {}
    for view in views:
        by_size.setdefault(view.shape[-1], []).append(view)

    outputs = []
    for _, group in sorted(by_size.items()):
        batch = torch.cat(group, dim=0)
        hidden = module["backbone"](pixel_values=batch).last_hidden_state
        # CLS token where present, else mean-pooled patches. DINOv2 has a CLS;
        # this keeps the sweep usable if the backbone is swapped for one without.
        pooled = hidden[:, 0] if hidden.shape[1] > 1 else hidden.mean(dim=1)
        outputs.append(module["head"](pooled))
    return torch.cat(outputs, dim=0)


def measure(batch_size: int, *, device: torch.device, checkpoint: str,
            global_crops: int, local_crops: int, global_size: int,
            local_size: int, iterations: int, amp: bool) -> dict:
    """One batch size, or an out-of-memory report."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    student, teacher, dim = build_pair(device, checkpoint)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    globals_ = [torch.randn(batch_size, 3, global_size, global_size, device=device)
                for _ in range(global_crops)]
    locals_ = [torch.randn(batch_size, 3, local_size, local_size, device=device)
               for _ in range(local_crops)]

    forward_times, backward_times = [], []

    try:
        for i in range(iterations):
            torch.cuda.synchronize()
            t0 = time.time()

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                # The teacher sees only the global views; the student sees all
                # of them. That asymmetry is the local-to-global correspondence
                # DINO trains on.
                with torch.no_grad():
                    teacher_out = forward_views(teacher, globals_)
                student_out = forward_views(student, globals_ + locals_)

                # Stand-in for the cross-entropy against a centred, sharpened
                # teacher. The exact loss does not change the memory profile,
                # which is what this measures.
                # The teacher sees only the global views, the student all of
                # them, so the two differ in row count. Real DINO pairs each
                # teacher view against every student view except its own match;
                # for a memory measurement only the shapes matter, so the
                # teacher output is repeated to line up.
                t = teacher_out.softmax(dim=-1).detach()
                reps = student_out.shape[0] // t.shape[0]
                t = t.repeat(reps, 1)[: student_out.shape[0]]
                loss = -(t * student_out.log_softmax(dim=-1)).sum(dim=-1).mean()

            torch.cuda.synchronize()
            t1 = time.time()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            t2 = time.time()

            if i >= 2:  # discard warmup
                forward_times.append(t1 - t0)
                backward_times.append(t2 - t1)

        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        fwd = sum(forward_times) / max(len(forward_times), 1)
        bwd = sum(backward_times) / max(len(backward_times), 1)
        result = {
            "batch_size": batch_size,
            "status": "ok",
            "peak_gib": round(peak, 2),
            "forward_s": round(fwd, 4),
            "backward_s": round(bwd, 4),
            "step_s": round(fwd + bwd, 4),
            "samples_per_s": round(batch_size / max(fwd + bwd, 1e-9), 1),
        }

    except torch.cuda.OutOfMemoryError:
        result = {"batch_size": batch_size, "status": "oom"}

    del student, teacher, optimizer, scaler, globals_, locals_
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. This measurement is meaningless on CPU.")

    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    views = args.global_crops + args.local_crops
    print("=" * 72)
    print(f"device        {torch.cuda.get_device_name(device)}  {total:.1f} GiB")
    print(f"checkpoint    {args.checkpoint}")
    print(f"multi-crop    {args.global_crops} x {args.global_size} px + "
          f"{args.local_crops} x {args.local_size} px = {views} views/sample")
    print(f"amp           {'fp16' if args.amp else 'off'}")
    print("=" * 72)
    print(f"{'batch':>6}  {'peak GiB':>9}  {'fwd s':>7}  {'bwd s':>7}  "
          f"{'step s':>7}  {'samp/s':>7}")
    print("-" * 72)

    results = []
    for batch_size in args.batch_sizes:
        record = measure(
            batch_size, device=device, checkpoint=args.checkpoint,
            global_crops=args.global_crops, local_crops=args.local_crops,
            global_size=args.global_size, local_size=args.local_size,
            iterations=args.iterations, amp=args.amp,
        )
        results.append(record)

        if record["status"] == "oom":
            print(f"{batch_size:>6}  {'OOM':>9}")
            print("-" * 72)
            print("Stopping: larger batches will not fit either.")
            break

        print(f"{record['batch_size']:>6}  {record['peak_gib']:>9.2f}  "
              f"{record['forward_s']:>7.3f}  {record['backward_s']:>7.3f}  "
              f"{record['step_s']:>7.3f}  {record['samples_per_s']:>7.1f}")

    ok = [r for r in results if r["status"] == "ok"]
    print()
    if not ok:
        print("Nothing fits, even at the smallest batch requested. Reduce "
              "--local-crops, or the backbone is too large for this device.")
        return 1

    best = max(ok, key=lambda r: r["batch_size"])
    largest = best["batch_size"]

    print(f"Largest batch that fits: {largest}  "
          f"({best['peak_gib']:.2f} GiB of {total:.1f})")
    print()

    if largest >= 64:
        print("Comfortable. The published DINO defaults are probably usable;")
        print("centre momentum 0.9 and the standard temperature schedule.")
    elif largest >= 16:
        print("Tight. The centre is an EMA of the batch mean, and at this size")
        print("that estimate is noisy. Raise centre momentum toward 0.996 so it")
        print("averages over many more batches, warm the teacher temperature")
        print("from 0.04 to 0.07 over the first epochs, and call")
        print("scripts/check_collapse.py every few hundred steps -- collapse")
        print("shows in the representation long before it shows in the loss.")
    else:
        print("Severe. Add gradient accumulation over 8-16 micro-batches, and")
        print("consider reducing --local-crops from 8 to 4, which roughly")
        print("halves the compute and lets the batch double. Note that")
        print("accumulation fixes the gradient estimate but NOT the centring")
        print("statistic, which is still computed per micro-batch -- so the")
        print("centre momentum change is required regardless.")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "device": torch.cuda.get_device_name(device),
            "total_gib": round(total, 1),
            "checkpoint": args.checkpoint,
            "global_crops": args.global_crops,
            "local_crops": args.local_crops,
            "global_size": args.global_size,
            "local_size": args.local_size,
            "amp": args.amp,
            "results": results,
        }, indent=2))
        print(f"\nWritten to {path}")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="facebook/dinov2-base")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--global-crops", type=int, default=2)
    p.add_argument("--local-crops", type=int, default=8)
    p.add_argument("--global-size", type=int, default=224)
    p.add_argument("--local-size", type=int, default=96)
    p.add_argument("--iterations", type=int, default=5,
                   help="per batch size; the first two are discarded as warmup")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--out", default=None, help="write results as JSON")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
