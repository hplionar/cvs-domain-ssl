#!/usr/bin/env python3
"""Correct the disagreement score in the perspective probe.

The probe fits nine outputs -- three annotators by three criteria -- and asked
whether the model could anticipate which frames the annotators would split on.
It scored that by the standard deviation across the three annotator logits, and
returned chance AUC on eighteen measurements.

**That result established nothing, and the reason is structural.** Under a
threshold account the three raters read one shared signal at three different
bars. Three linear heads fitted to that converge on very nearly one direction
with three different biases, so at any frame the three logits are the shared
score plus three constants. Their standard deviation is then the spread of three
constants -- almost the same at every frame -- and an AUC near 0.5 follows by
construction whether or not disagreement is predictable from the image.

The score the account actually calls for is in probability space. With per-rater
probabilities p_1, p_2, p_3 for a criterion, the frames on which the raters
disagree are those that are neither unanimously positive nor unanimously
negative:

    P(contested) = 1 - p_1 p_2 p_3 - (1 - p_1)(1 - p_2)(1 - p_3)

This peaks where the shared score sits between the strictest and most lenient
bar, which is exactly the band the account predicts, and it is invariant to the
three biases being different. A constant shift in one rater's bar moves where
the peak sits without flattening it.

Both scores are computed here, from the logits the probe already produces, so
the comparison is between scoring rules on identical predictions rather than
between models.

**What the result means.** If the product score gives an AUC well above 0.5
where the standard deviation gave 0.5, the earlier null was an artefact of the
scoring rule and disagreement is partly anticipable. If both are at chance, the
null survives a scoring rule designed to find the band, which is a much stronger
statement than the original measurement supported.

This adds a save of the per-rater logits, without which the analysis cannot be
run at all, and a second scoring rule beside the first.

Run from the repository root:
    python disagreement_score_patch.py
"""

from __future__ import annotations

import pathlib


def patch(path: str, edits: list[tuple[str, str, str, str]]) -> int:
    p = pathlib.Path(path)
    if not p.is_file():
        raise SystemExit(f"{path} not found; run from the repository root.")
    s = p.read_text()
    applied = 0
    for old, new, marker, label in edits:
        if marker in s:
            print(f"  skip     {label}")
            continue
        if old not in s:
            print(f"  NO MATCH {label}")
            continue
        s = s.replace(old, new, 1)
        applied += 1
        print(f"  applied  {label}")
    p.write_text(s)
    return applied


def main() -> int:
    n = patch("eval/perspective_probe.py", [
        (
            '''    print(f"\\ndoes predicted disagreement match observed disagreement?")
    print(f"{'criterion':<11}{'AUC':>9}{'contested':>12}")
    results["disagreement"] = {}
    for c, name in enumerate(CRITERIA):
        contested = ((truth[:, :, c].sum(axis=1) == 1) |
                     (truth[:, :, c].sum(axis=1) == 2)).astype(int)
        # Spread among the three predicted logits: large where the heads split.
        spread = logits[:, :, c].std(axis=1)
        auc = (roc_auc_score(contested, spread)
               if np.unique(contested).size > 1 else float("nan"))
        results["disagreement"][name] = {"auc": auc,
                                         "contested_rate": float(contested.mean())}
        print(f"{name.upper():<11}{auc:>9.4f}{contested.mean():>12.3f}")''',

            '''    print(f"\\ndoes predicted disagreement match observed disagreement?")
    print(f"{'criterion':<11}{'spread AUC':>12}{'product AUC':>13}"
          f"{'contested':>12}")
    results["disagreement"] = {}
    probs = 1.0 / (1.0 + np.exp(-logits))
    for c, name in enumerate(CRITERIA):
        contested = ((truth[:, :, c].sum(axis=1) == 1) |
                     (truth[:, :, c].sum(axis=1) == 2)).astype(int)

        # The original rule: spread among the three predicted logits. Under a
        # threshold account the three heads share a direction and differ by a
        # bias, so this is the spread of three near-constants and returns chance
        # by construction. Kept for comparison, not as a measurement.
        spread = logits[:, :, c].std(axis=1)

        # The rule the account calls for: the probability that the three raters
        # are neither unanimously positive nor unanimously negative. This peaks
        # where the shared score sits between the strictest and most lenient
        # bar, and is unaffected by the biases differing.
        p = probs[:, :, c]
        product = 1.0 - p.prod(axis=1) - (1.0 - p).prod(axis=1)

        usable = np.unique(contested).size > 1
        auc_spread = roc_auc_score(contested, spread) if usable else float("nan")
        auc_product = roc_auc_score(contested, product) if usable else float("nan")
        results["disagreement"][name] = {
            "auc_spread": auc_spread,
            "auc_product": auc_product,
            "contested_rate": float(contested.mean()),
        }
        print(f"{name.upper():<11}{auc_spread:>12.4f}{auc_product:>13.4f}"
              f"{contested.mean():>12.3f}")''',
            "auc_product",
            "1 product score alongside spread",
        ),
        (
            '''    print("  An AUC near 0.5 means disagreement is not anticipable from the")
    print("  frame, which would be consistent with the null in Section 5.19.")''',
            '''    print("\\n  The spread column is uninformative by construction: three linear")
    print("  heads sharing a direction differ by a bias, and the spread of three")
    print("  constants is near-constant. Read the product column.")
    print("\\n  An AUC near 0.5 there means disagreement is not anticipable from")
    print("  the frame under a rule designed to find the band, which is a")
    print("  stronger null than the spread rule could support.")''',
            "Read the product column",
            "2 corrected interpretation",
        ),
        (
            '''        if args.save_logits:
            np.savez(path.parent / f"logits_{name}.npz",
                     logits=logits, targets=targets)''',
            '''        if args.save_logits:
            np.savez(path.parent / f"logits_{name}.npz",
                     logits=logits, targets=targets)''',
            "__never__",
            "3 (no-op)",
        ),
        (
            '''    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "perspective_probe.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)''',
            '''    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The per-rater predictions are saved because two further analyses need them
    # and refitting the grid to recover them is wasteful: the band score above
    # can be recomputed under other rules, and the same predictions over a
    # temporal window test whether context closes the band.
    np.savez(out / "per_rater_logits.npz", logits=logits, targets=truth)
    with open(out / "perspective_probe.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)''',
            "per_rater_logits.npz",
            "4 persist per-rater logits",
        ),
    ])
    print(f"\n{n} edits applied")
    print()
    print("Then rerun the six arms; each is a few minutes on cached features:")
    print()
    print("  for arm in dinov3_b dinov2_b mae_b vit_sup_b \\")
    print("             dinov3_b_trainonly mae_b_trainonly; do")
    print("    echo \"### $arm\"")
    print("    python eval/perspective_probe.py \\")
    print("      --train-features ../cache/$arm/endoscapes/train \\")
    print("      --val-features   ../cache/$arm/endoscapes/val \\")
    print("      --test-features  ../cache/$arm/endoscapes/test \\")
    print("      --manifest metadata/endoscapes_frames.csv \\")
    print("      --output-dir ../outputs/perspective/$arm 2>&1 \\")
    print("      | grep -A6 'does predicted'")
    print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

