#!/usr/bin/env python3
"""Compare probe heads trained on one cache, and refuse to compare unlike runs.

Answers a single question: on identical frozen features, does one aggregation
design beat another by more than seed noise?

The comparison is only meaningful if the arms differ in the head and in nothing
else, so this script verifies that before reporting anything. Two runs that used
different encoders, different caches, different seeds or different search grids
produce a difference that looks like a result and is not one.

Three things are reported, and they answer slightly different questions.

**Selected-config mAP.** Each arm's headline figure: the grid winner, chosen on
the mean across seeds, reported as mean +/- sd. This is what a paper would
print.

**Equal-grid mAP.** The same statistic recomputed over the intersection of the
arms' grids. It exists because the headline figure is a maximum over a grid, and
the maximum of several noisy estimates is biased upward by an amount that grows
with the number of configurations searched. The attention-based heads search
``hidden_dim`` and the mean head does not, so their grids are twice the size and
their headline numbers carry twice the selection bias. Restricting to the shared
subgrid removes that asymmetry. Fusion versus attentive is unaffected either
way, since those two search the same grid.

**Paired per-seed differences.** Both arms are trained from the same seeds on
the same data, so the runs pair. The paired sd is usually far smaller than the
unpaired sd because it cancels the component of seed variance that both arms
share, which makes a small real difference detectable at a seed count where the
unpaired comparison cannot resolve it.

Usage:
    python scripts/compare_heads.py \
        --run mean=outputs/probe/heads/mean \
        --run attentive=outputs/probe/heads/attentive \
        --run fusion=outputs/probe/heads/fusion \
        --reference attentive \
        --output-md metadata/experiments/head_comparison.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Search-grid axes that must be identical across arms. hidden_dim is excluded:
# it is only defined for heads with learned attention, and requiring it would
# make the mean head incomparable with anything by construction.
SHARED_GRID_KEYS = ("lr", "weight_decay", "dropout")

#: Value of hidden_dim used for the equal-grid comparison. Arms that do not
#: search hidden_dim match every value.
EQUAL_GRID_HIDDEN_DIM = 128


class ComparabilityError(ValueError):
    """Raised when the arms did not differ solely in the head."""


def load_run(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "results.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No results.json in {directory}. Run train/train_probe_cached.py first."
        )
    return json.loads(path.read_text())


def verify_comparable(runs: dict[str, dict[str, Any]]) -> None:
    """Fail loudly if the arms differ in anything except the head."""
    if len(runs) < 2:
        raise ComparabilityError("At least two arms are needed for a comparison.")

    names = list(runs)
    reference_name = names[0]
    reference = runs[reference_name]
    problems: list[str] = []

    def compare(label: str, extract) -> None:
        expected = extract(reference)
        for name in names[1:]:
            actual = extract(runs[name])
            if actual != expected:
                problems.append(
                    f"  {label}:\n    {reference_name}: {expected}\n    {name}: {actual}"
                )

    compare("encoder", lambda r: r.get("encoder"))
    compare("protocol", lambda r: r.get("protocol"))
    compare("seeds", lambda r: r.get("search", {}).get("seeds"))
    compare("epochs", lambda r: r.get("search", {}).get("epochs"))
    compare("patience", lambda r: r.get("search", {}).get("patience"))
    compare("pos_weight", lambda r: r.get("search", {}).get("pos_weight"))
    for key in SHARED_GRID_KEYS:
        compare(f"grid.{key}", lambda r, k=key: r.get("search", {}).get("grid", {}).get(k))

    kinds = [runs[name].get("head", {}).get("kind") for name in names]
    if len(set(kinds)) != len(kinds):
        problems.append(
            f"  head.kind: arms must differ in the head, got {kinds}. Two arms "
            f"with the same head are a seed-noise measurement, not a comparison."
        )

    if problems:
        raise ComparabilityError(
            "Arms did not differ solely in the head, so the difference between "
            "them is not attributable to the head:\n" + "\n".join(problems)
        )


def _seed_maps(entry: dict[str, Any]) -> dict[int, float]:
    """Per-seed mAP from an all_configs entry, with JSON's string keys undone."""
    return {int(seed): float(value) for seed, value in entry.get("seeds", {}).items()}


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def select(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Winner by mean across seeds, matching train_probe_cached.aggregate."""
    if not entries:
        raise ComparabilityError("No configurations to select from.")
    return max(entries, key=lambda e: e["mean_map"])


def equal_grid_entries(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Configurations restricted to the subgrid every arm shares.

    An arm that does not search ``hidden_dim`` has no such key in its configs,
    so all of its configurations qualify and its equal-grid figure equals its
    headline figure. That is correct: it never paid the extra selection bias.
    """
    entries = run.get("all_configs", [])
    filtered = [
        e for e in entries
        if e["config"].get("hidden_dim", EQUAL_GRID_HIDDEN_DIM) == EQUAL_GRID_HIDDEN_DIM
    ]
    return filtered or entries


def paired_delta(arm: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any] | None:
    """Per-seed difference between two arms at their own selected configs.

    Pairing is by seed, which is the whole point: both arms saw the same
    initialisation stream and the same shuffling, so the shared component of
    seed variance cancels.
    """
    a = _seed_maps(select(arm.get("all_configs", [])))
    b = _seed_maps(select(reference.get("all_configs", [])))
    shared = sorted(set(a) & set(b))
    if not shared:
        return None

    deltas = [a[s] - b[s] for s in shared]
    mean, sd = _mean_sd(deltas)
    return {
        "seeds": shared,
        "deltas": deltas,
        "mean_delta": mean,
        "sd_delta": sd,
        "wins": sum(d > 0 for d in deltas),
        "n": len(deltas),
    }


def summarise(runs: dict[str, dict[str, Any]], reference: str | None) -> dict[str, Any]:
    verify_comparable(runs)

    arms = []
    for name, run in runs.items():
        headline = select(run.get("all_configs", []))
        restricted = select(equal_grid_entries(run))
        seed_values = list(_seed_maps(headline).values())
        mean, sd = _mean_sd(seed_values)
        arms.append({
            "arm": name,
            "head": run.get("head", {}),
            "num_configs": len(run.get("all_configs", [])),
            "selected_config": headline["config"],
            "mean_map": mean,
            "sd_map": sd,
            "per_seed": _seed_maps(headline),
            "equal_grid_config": restricted["config"],
            "equal_grid_mean_map": restricted["mean_map"],
            "equal_grid_sd_map": restricted["std_map"],
            "equal_grid_num_configs": len(equal_grid_entries(run)),
        })

    summary: dict[str, Any] = {
        "encoder": next(iter(runs.values())).get("encoder"),
        "protocol": next(iter(runs.values())).get("protocol"),
        "seeds": next(iter(runs.values())).get("search", {}).get("seeds"),
        "arms": sorted(arms, key=lambda a: a["mean_map"], reverse=True),
    }

    if reference is not None:
        if reference not in runs:
            raise ComparabilityError(
                f"Reference arm {reference!r} is not among {sorted(runs)}."
            )
        summary["reference"] = reference
        summary["paired"] = {
            name: paired_delta(runs[name], runs[reference])
            for name in runs
            if name != reference
        }

    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    seeds = summary.get("seeds") or []
    lines = [
        "# Head comparison on identical cached features",
        "",
        f"Encoder: `{summary.get('encoder', {}).get('checkpoint_id', 'unknown')}`  ",
        f"Seeds: {seeds} (n={len(seeds)})",
        "",
        "Selection is on the mean across seeds, never on the best single run.",
        "",
        "| Arm | Head | Selected config | Val mAP (mean +/- sd) | Grid |",
        "|---|---|---|---:|---:|",
    ]
    for arm in summary["arms"]:
        cfg = arm["selected_config"]
        cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
        lines.append(
            f"| {arm['arm']} | {arm['head'].get('kind', '?')} | {cfg_str} | "
            f"{arm['mean_map']:.4f} +/- {arm['sd_map']:.4f} | {arm['num_configs']} |"
        )

    lines += [
        "",
        f"## Equal-grid comparison (hidden_dim = {EQUAL_GRID_HIDDEN_DIM})",
        "",
        "The table above reports each arm's grid winner. A maximum over a grid "
        "is biased upward, and the bias grows with grid size, so arms that "
        "search `hidden_dim` are flattered relative to arms that do not. "
        "Restricting every arm to the shared subgrid removes that asymmetry.",
        "",
        "| Arm | Selected config | Val mAP (mean +/- sd) | Grid |",
        "|---|---|---:|---:|",
    ]
    for arm in sorted(summary["arms"], key=lambda a: a["equal_grid_mean_map"], reverse=True):
        cfg = arm["equal_grid_config"]
        cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
        lines.append(
            f"| {arm['arm']} | {cfg_str} | {arm['equal_grid_mean_map']:.4f} +/- "
            f"{arm['equal_grid_sd_map']:.4f} | {arm['equal_grid_num_configs']} |"
        )

    if summary.get("paired"):
        reference = summary["reference"]
        lines += [
            "",
            f"## Paired per-seed difference against `{reference}`",
            "",
            "Both arms train from the same seeds on the same features, so the "
            "runs pair and the shared component of seed variance cancels.",
            "",
            "| Arm | Mean delta mAP | sd of delta | Noise floor (2 sd) | Resolved? | "
            "Seeds won | Per-seed deltas |",
            "|---|---:|---:|---:|:---:|---:|---|",
        ]
        for name, paired in summary["paired"].items():
            if paired is None:
                lines.append(f"| {name} | - | - | - | - | - | no shared seeds |")
                continue
            per_seed = ", ".join(f"{d:+.4f}" for d in paired["deltas"])
            floor = 2 * paired["sd_delta"]
            resolved = "yes" if abs(paired["mean_delta"]) > floor else "no"
            lines.append(
                f"| {name} | {paired['mean_delta']:+.4f} | {paired['sd_delta']:.4f} | "
                f"{floor:.4f} | {resolved} | {paired['wins']}/{paired['n']} | {per_seed} |"
            )

        n = max((p["n"] for p in summary["paired"].values() if p), default=0)
        lines += [
            "",
            f"The noise floor is that arm's own paired sd, not a pooled one: two "
            f"arms can differ in how stable their gap is. `Resolved? = no` means "
            f"the measured difference is inside the floor and the honest "
            f"statement is that nothing was demonstrated, whatever the sign. "
            f"With n={n} seeds this supports a direction, not a p-value; if a "
            f"difference lands near the boundary the seed count is the thing to "
            f"increase, not the interpretation.",
        ]

    lines += [
        "",
        "## Scope",
        "",
        "This measures head design on frozen features from one encoder. It is "
        "not a reproduction of SMIL (Wang et al., 2026): SMIL's backbone is DINO "
        "self-distillation initialised from Endo-FM, no checkpoint used here, "
        "and their published figure is not comparable with these numbers. The "
        "only transferable claim is the relative ordering of the three heads on "
        "identical features.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="an arm's output directory, named; repeat once per arm",
    )
    p.add_argument(
        "--reference",
        default=None,
        help="arm to compute paired per-seed differences against",
    )
    p.add_argument("--output-md", default=None)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    runs: dict[str, dict[str, Any]] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects NAME=DIR, got {spec!r}")
        name, directory = spec.split("=", 1)
        runs[name] = load_run(directory)

    summary = summarise(runs, args.reference)
    markdown = render_markdown(summary)
    print(markdown)

    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(markdown, encoding="utf-8")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
