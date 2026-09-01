#!/usr/bin/env python3
"""Patch bootstrap_strata.py: per-criterion strata and a power-matched control.

Two additions.

**Per-criterion reporting.** The stratified result currently averages C1, C2 and
C3 before reporting, which could hide the fact that the collapse is driven by
one criterion rather than by all three. The criteria differ substantially: C2 is
the most frequently achieved on both benchmarks and the most reliably annotated
(Fleiss' kappa 0.540 against 0.474 and 0.475), and their prevalences differ
sharply between strata -- C1 is 5.1% positive among unanimous frames and 31.2%
among contested ones. Reporting the delta per criterion says whether the effect
is general.

**A power-matched control.** The honest alternative explanation for finding no
distinguishable pairs on contested frames is that the contested stratum is
simply smaller: roughly 250 frames against 1,010. If the sixteen significant
unanimous comparisons survive when the unanimous stratum is randomly subsampled
to the contested stratum's size, sample size is not the explanation. If they
collapse, it is at least partly the explanation, and the finding must be
reported accordingly.

The subsample is drawn inside each bootstrap replicate, after the videos have
been resampled, so the video-level clustering is preserved. Prevalence is not
matched -- the contested stratum is far more positive -- so this controls for
size alone, which is the specific objection being tested.

Usage:
    python scripts/patch_bootstrap.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("eval/bootstrap_strata.py")

EDITS = [
    (
        "arguments",
        """    p.add_argument("--output-dir", required=True)""",
        """    p.add_argument("--output-dir", required=True)
    p.add_argument("--match-power", action="store_true",
                   help="additionally report the unanimous stratum subsampled to the "
                        "size of the contested stratum, so that the two are compared "
                        "at equal frame counts")
    p.add_argument("--per-criterion", action="store_true",
                   help="report the delta separately for C1, C2 and C3 rather than "
                        "only their mean")""",
    ),
    (
        "subsampling helper",
        """def stratum_auc(y: np.ndarray, s: np.ndarray, mask: np.ndarray, rows: np.ndarray) -> float:
    sel = rows[mask[rows]]
    return safe_auc(y[sel], s[sel]) if sel.size else float("nan")""",
        """def stratum_auc(y: np.ndarray, s: np.ndarray, mask: np.ndarray, rows: np.ndarray,
                cap: int | None = None, rng: np.random.Generator | None = None) -> float:
    \"\"\"AUC over the rows of one stratum, optionally capped in size.

    ``cap`` subsamples the selected rows without replacement, which is used only
    to compare two strata at equal frame counts. The subsample is drawn after
    the videos have been resampled, so the clustering is preserved.
    \"\"\"
    sel = rows[mask[rows]]
    if cap is not None and sel.size > cap:
        sel = (rng or np.random.default_rng(0)).choice(sel, size=cap, replace=False)
    return safe_auc(y[sel], s[sel]) if sel.size else float("nan")""",
    ),
]

APPEND_BEFORE = """    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)"""

APPEND_BLOCK = '''    # --- per criterion -----------------------------------------------------
    if args.per_criterion:
        print(f"\\nDelta by criterion")
        print(f"{'arm':<20}{'crit':<5}{'unan':>9}{'cont':>9}{'delta':>9}{'95% CI':>22}")
        results["per_criterion"] = {}
        for name, arm in arms.items():
            probs = arm["probs"]
            results["per_criterion"][name] = {}
            for j, c in enumerate(CRITERIA):
                def delta_c(rows: np.ndarray, probs=probs, j=j, c=c) -> float:
                    u = stratum_auc(labels[c], probs[:, j], unanimous[c], rows)
                    k = stratum_auc(labels[c], probs[:, j], ~unanimous[c], rows)
                    return u - k if np.isfinite(u) and np.isfinite(k) else float("nan")

                pu = stratum_auc(labels[c], probs[:, j], unanimous[c], all_rows)
                pk = stratum_auc(labels[c], probs[:, j], ~unanimous[c], all_rows)
                ci = boot.interval(delta_c)
                results["per_criterion"][name][c] = {
                    "unanimous": pu, "contested": pk, "delta": pu - pk, **ci}
                print(f"{name:<20}{c:<5}{pu:>9.4f}{pk:>9.4f}{pu - pk:>9.4f}"
                      f"   [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")

    # --- power-matched control ---------------------------------------------
    if args.match_power and len(arms) >= 2:
        # The contested stratum is roughly a quarter the size of the unanimous
        # one, so the absence of distinguishable pairs there could reflect
        # sample size rather than an absence of difference. Subsampling the
        # unanimous stratum to the contested size, inside each replicate,
        # separates the two explanations.
        caps = {c: int((~unanimous[c]).sum()) for c in CRITERIA}
        print(f"\\nPower-matched control: unanimous subsampled to the contested size")
        print(f"  frames per criterion  " + "  ".join(f"{c} {caps[c]}" for c in CRITERIA))

        rng = np.random.default_rng(args.seed + 1)

        def spread_capped(rows: np.ndarray) -> float:
            means = []
            for arm in arms.values():
                vals = [stratum_auc(labels[c], arm["probs"][:, j], unanimous[c], rows,
                                    cap=caps[c], rng=rng)
                        for j, c in enumerate(CRITERIA)]
                if all(np.isfinite(v) for v in vals):
                    means.append(float(np.mean(vals)))
            return float(max(means) - min(means)) if len(means) == len(arms) else float("nan")

        point = spread_capped(all_rows)
        ci = boot.interval(spread_capped)
        results["spread"]["unanimous_power_matched"] = {"point": point, **ci}
        print(f"  {'spread':<12}{point:>8.4f}   [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
              f"   (full unanimous {results['spread']['unanimous']['point']:.4f})")

        significant = 0
        for a, b in combinations(arms, 2):
            pa, pb = arms[a]["probs"], arms[b]["probs"]

            def diff_capped(rows: np.ndarray, pa=pa, pb=pb) -> float:
                vals = []
                for j, c in enumerate(CRITERIA):
                    x = stratum_auc(labels[c], pa[:, j], unanimous[c], rows,
                                    cap=caps[c], rng=rng)
                    y = stratum_auc(labels[c], pb[:, j], unanimous[c], rows,
                                    cap=caps[c], rng=rng)
                    if np.isfinite(x) and np.isfinite(y):
                        vals.append(x - y)
                return float(np.mean(vals)) if vals else float("nan")

            ci = boot.interval(diff_capped)
            if np.isfinite(ci["ci_low"]) and ci["ci_low"] * ci["ci_high"] > 0:
                significant += 1
            results["pairwise"].setdefault(f"{a}-{b}", {})["unanimous_power_matched"] = {
                "diff": diff_capped(all_rows), **ci}

        total = len(list(combinations(arms, 2)))
        full_u = sum(1 for p in results["pairwise"].values()
                     if p.get("unanimous") and
                     p["unanimous"]["ci_low"] * p["unanimous"]["ci_high"] > 0)
        full_c = sum(1 for p in results["pairwise"].values()
                     if p.get("contested") and
                     p["contested"]["ci_low"] * p["contested"]["ci_high"] > 0)
        print(f"\\n  pairwise intervals excluding zero, out of {total}:")
        print(f"    unanimous, full            {full_u}")
        print(f"    unanimous, size-matched    {significant}")
        print(f"    contested                  {full_c}")
        print("  If the size-matched count stays well above the contested count, the")
        print("  absence of distinguishable pairs on contested frames is not explained")
        print("  by the size of that stratum.")

'''


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default=str(TARGET))
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"not found: {path}")
        return 1
    source = path.read_text(encoding="utf-8")

    applied, already, missing = [], [], []
    for name, old, new in EDITS:
        if new in source:
            already.append(name)
        elif old in source:
            source = source.replace(old, new, 1)
            applied.append(name)
        else:
            missing.append(name)

    if "power-matched control" in source:
        already.append("reporting blocks")
    elif APPEND_BEFORE in source:
        source = source.replace(APPEND_BEFORE, APPEND_BLOCK + APPEND_BEFORE, 1)
        applied.append("reporting blocks")
    else:
        missing.append("reporting blocks")

    for n in already:
        print(f"  already   {n}")
    for n in applied:
        print(f"  applied   {n}")
    for n in missing:
        print(f"  NO MATCH  {n}")

    if missing:
        print("\nAnchors did not match. Nothing written.")
        return 1
    if not applied:
        print("\nNothing to do.")
        return 0
    if args.check:
        print("\n--check: nothing written.")
        return 0

    path.write_text(source, encoding="utf-8")
    import ast
    try:
        ast.parse(source)
    except SyntaxError as exc:
        print(f"SYNTAX ERROR: {exc}\nRevert with: git checkout {path}")
        return 1
    print(f"\nwritten to {path}, parses cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
