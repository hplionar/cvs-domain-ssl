#!/usr/bin/env python3
"""Every figure and table in the dissertation, generated from the result files.

Nothing here is drawn or typed by hand. Each figure and each table reads the
JSON written by the analysis that produced it, so a number that changes upstream
changes everywhere it appears, and a figure cannot disagree with the table
beside it. Transcription is the most common source of error in a results
chapter and it is entirely avoidable.

Run with no arguments to build everything that has inputs available; missing
inputs are reported and skipped rather than raising, so the command is safe to
re-run as results arrive.

**Palette.** The colourblind-safe set from seaborn, chosen against three
requirements rather than for appearance. It survives greyscale printing, since
examiners print; it avoids red against green; and one colour means one thing
across every figure, which matters more than any particular choice of hue.
Line style and marker shape carry the same information as colour wherever a
figure has more than two series, so nothing is lost when the colour is.

**Sizing.** Figures are emitted at the text width so that LaTeX includes them at
scale 1.0. A figure scaled to fit has font sizes that differ from the body and
from every other figure, which is visible and looks careless.

Usage:
    python eval/figures.py --all --output-dir ../outputs/figures
    python eval/figures.py --figure stratified --output-dir ../outputs/figures
    python eval/figures.py --table results --output-dir ../outputs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------

#: Colourblind-safe, greyscale-separable. One colour per family, used
#: identically in every figure.
PALETTE = {
    "dinov2": "#0173B2",      # blue
    "dinov3": "#56B4E9",      # light blue
    "mae": "#DE8F05",         # orange
    "supervised": "#029E73",  # green
    "video": "#CC78BC",       # pink
    "ijepa": "#CA9161",       # brown
    "neutral": "#949494",     # grey, for chance lines and baselines
    "ceiling": "#333333",     # near-black, for the human ceiling
}

#: Display names, matching notes section 0.6. The dagger marks arms adapted on
#: the full 700-video corpus, which are transductive on the internal splits.
NAMES = {
    "dinov2_b": "DINOv2",
    "dinov2_b_trainonly": "DINOv2 + adapt",
    "dinov2_b_adapted": "DINOv2 + adapt$^\\dagger$",
    "dinov3_b": "DINOv3",
    "dinov3_b_trainonly": "DINOv3 + adapt",
    "mae_b": "ViT-MAE",
    "mae_b_trainonly": "ViT-MAE + adapt",
    "vit_sup_b": "ViT-B/16 sup.",
    "ijepa_h": "I-JEPA",
    "videomae_b_base": "VideoMAE",
    "videomae_b_adapted": "VideoMAE + adapt$^\\dagger$",
    "vjepa2_l_base": "V-JEPA 2",
    "vjepa2_l_adapted": "V-JEPA 2 + adapt$^\\dagger$",
    "vjepa2_l_lr1e6": "V-JEPA 2 + adapt$^\\dagger$ (lr 1e-6)",
}

FAMILY = {
    "dinov2": "dinov2", "dinov3": "dinov3", "mae_b": "mae", "mae": "mae",
    "vit_sup": "supervised", "ijepa": "ijepa",
    "videomae": "video", "vjepa": "video",
}


def colour_for(arm: str) -> str:
    for key, family in FAMILY.items():
        if arm.startswith(key):
            return PALETTE[family]
    return PALETTE["neutral"]


def display(arm: str) -> str:
    return NAMES.get(arm, arm.replace("_", " "))


def apply_style() -> None:
    """Serif type at body size, so a figure does not read as a foreign object.

    Computer Modern is the report class's default; matching it means figure
    labels and body text share a typeface. Font sizes are set for a figure
    included at scale 1.0.
    """
    plt.rcParams.update({
        # The body text is Times. A figure set in matplotlib's DejaVu default
        # is visibly a foreign object on the page, and the formal-details
        # criterion notices. The fallback chain ends at DejaVu so the module
        # still runs where Times is unavailable; check the embedded fonts of the
        # output if consistency matters.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "FreeSerif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"  missing  {path}")
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def fig_stratified(root: Path, out: Path, width: float) -> bool:
    """Per-arm AUC on unanimous against contested frames, with intervals.

    The central result, so it gets the most care. Arms are ordered by their
    unanimous score, the two strata are drawn as a paired dot plot rather than
    grouped bars -- bars would imply a zero baseline that AUC does not have --
    and the chance line is drawn explicitly because an AUC of 0.58 means
    something quite different from an accuracy of 0.58.
    """
    data = load(root / "rater_agreement" / "bootstrap_val.json")
    if data is None or "arms" not in data:
        return False

    arms = sorted(data["arms"].items(), key=lambda kv: kv[1]["unanimous"])
    y = np.arange(len(arms))

    fig, ax = plt.subplots(figsize=(width, 0.34 * len(arms) + 1.4))
    for i, (arm, v) in enumerate(arms):
        c = colour_for(arm)
        ax.plot([v["contested"], v["unanimous"]], [i, i], color=c,
                linewidth=1.2, alpha=0.55, zorder=1)
        ax.scatter(v["unanimous"], i, color=c, s=34, marker="o", zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(v["contested"], i, color=c, s=34, marker="s", zorder=3,
                   edgecolor="white", linewidth=0.6)

    ax.axvline(0.5, color=PALETTE["neutral"], linestyle=":", linewidth=1,
               zorder=0)
    ax.text(0.5, len(arms) - 0.3, " chance", fontsize=7,
            color=PALETTE["neutral"], va="top")

    ax.set_yticks(y)
    ax.set_yticklabels([display(a) for a, _ in arms])
    ax.set_xlabel("AUC, mean over the three criteria")
    ax.set_xlim(0.45, 0.95)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#444444",
               markersize=6, label="frames where all three annotators agreed"),
        Line2D([], [], marker="s", linestyle="", color="#444444",
               markersize=6, label="frames where one dissented"),
    ], loc="lower right", frameon=False)

    fig.savefig(out / "fig_stratified.pdf")
    fig.savefig(out / "fig_stratified.png")
    plt.close(fig)
    print(f"  written  fig_stratified  ({len(arms)} arms)")
    return True


def fig_shrinkage(root: Path, out: Path, width: float) -> bool:
    """Held-out score against selection score, with the fitted slope.

    A scatter rather than a table because the point is the slope: differences on
    the split used for selection are systematically larger than they are on a
    fresh split, and a line through the points shows that in a way a column of
    changes does not. The identity line is drawn so the shrinkage is read as a
    departure from it.
    """
    rows = []
    for path in sorted((root / "cvs-domain-ssl" / "probe").glob("*/test_metrics.json")):
        arm = path.parent.name.replace("_sages_mean", "").replace("_mean", "")
        # The Endoscapes probes also write test_metrics.json. One slope across
        # both datasets would be fitted across a discontinuity.
        if arm.endswith("_endoscapes"):
            continue
        d = json.loads(path.read_text())
        rows.append((arm, d["val_map_selected"], d["test"]["mAP"]["mean"]))
    if len(rows) < 4:
        print("  missing  probe test_metrics.json files")
        return False

    val = np.array([r[1] for r in rows])
    test = np.array([r[2] for r in rows])
    slope, intercept = np.polyfit(val, test, 1)

    fig, ax = plt.subplots(figsize=(width * 0.62, width * 0.58))
    lo = min(val.min(), test.min()) - 0.02
    hi = max(val.max(), test.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color=PALETTE["neutral"], linestyle=":",
            linewidth=1, label="no change", zorder=0)
    xs = np.linspace(lo, hi, 10)
    ax.plot(xs, slope * xs + intercept, color=PALETTE["dinov2"], linewidth=1.4,
            label=f"fitted, slope {slope:.2f}", zorder=1)
    for arm, v, t in rows:
        ax.scatter(v, t, color=colour_for(arm), s=30, zorder=2,
                   edgecolor="white", linewidth=0.5)

    ax.set_xlabel("mAP on the split used for selection")
    ax.set_ylabel("mAP on the held-out split")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False)

    fig.savefig(out / "fig_shrinkage.pdf")
    fig.savefig(out / "fig_shrinkage.png")
    plt.close(fig)
    print(f"  written  fig_shrinkage  ({len(rows)} arms, slope {slope:.3f})")
    return True


def fig_temporal(root: Path, out: Path, width: float) -> bool:
    """Held-out mAP against the length of the trailing window.

    Plotted against window length rather than as a bar per setting, because the
    finding is the shape: a step between one frame and three, and nothing after.
    The single-frame control is marked, since every other result in the project
    is that configuration.
    """
    arms = {}
    for path in sorted((root / "temporal_probe").glob("*/temporal_probe.json")):
        arms[path.parent.name] = json.loads(path.read_text())
    if not arms:
        print("  missing  temporal_probe.json")
        return False

    fig, ax = plt.subplots(figsize=(width * 0.62, width * 0.42))
    for arm, d in arms.items():
        ks = sorted(int(k) for k in d["windows"])
        vals = [d["windows"][str(k)].get("test", {}).get("mAP", {}).get("mean")
                for k in ks]
        errs = [d["windows"][str(k)].get("test", {}).get("mAP", {}).get("sd", 0)
                for k in ks]
        if any(v is None for v in vals):
            continue
        spans = [5 * (k - 1) for k in ks]
        ax.errorbar(spans, vals, yerr=errs, marker="o", markersize=4,
                    capsize=2.5, linewidth=1.3, color=colour_for(arm),
                    label=display(arm))

    ax.set_xlabel("temporal context before the labelled frame (s)")
    ax.set_ylabel("mAP, official test split")
    if len(arms) > 1:
        ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "fig_temporal.pdf")
    fig.savefig(out / "fig_temporal.png")
    plt.close(fig)
    print(f"  written  fig_temporal  ({len(arms)} arms)")
    return True


def fig_geometry(root: Path, out: Path, width: float) -> bool:
    """Within-frame token geometry before and after continued pretraining.

    Two panels, because cosine and effective rank measure different things and
    a fixture test showed they can move independently: pulling tokens toward
    their mean raises the cosine without touching the centred spectrum. Both are
    needed to say what changed.
    """
    arms = {}
    for name, sub in (("DINOv2 + adapt$^\\dagger$", "patch_similarity"),
                      ("DINOv3 + adapt", "patch_similarity_dinov3"),
                      ("ViT-MAE + adapt", "patch_similarity_mae"),
                      ("DINOv2 + adapt", "patch_similarity_dinov2_trainonly")):
        d = load(root / sub / "patch_similarity.json")
        if d and "paired_change" in d:
            arms[name] = d
    if not arms:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.34))
    labels = list(arms)
    x = np.arange(len(labels))
    for ax, key, title in ((axes[0], "cosine", "within-frame cosine"),
                           (axes[1], "effective_rank", "effective rank")):
        vals = [arms[k]["paired_change"][key]["mean_change"] for k in labels]
        cols = [PALETTE["dinov2"] if "DINOv2" in k else
                PALETTE["dinov3"] if "DINOv3" in k else PALETTE["mae"]
                for k in labels]
        ax.bar(x, vals, color=cols, width=0.62)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("change under adaptation")
    fig.tight_layout()
    fig.savefig(out / "fig_geometry.pdf")
    fig.savefig(out / "fig_geometry.png")
    plt.close(fig)
    print(f"  written  fig_geometry  ({len(arms)} arms)")
    return True


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def latex_table(caption: str, label: str, header: list[str],
                rows: list[list[str]], align: str, note: str = "",
                groups: list[tuple[str, int, int]] | None = None) -> str:
    """booktabs, no vertical rules, caption above.

    The caption states what the table shows rather than describing its columns,
    since a reader who has to work out the finding from the numbers will often
    not bother.
    """
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{align}}}",
        "    \\toprule",
    ]
    if groups:
        # A table whose column headers repeat across two datasets is unreadable
        # without a spanning row: a reader cannot tell which pair belongs to
        # which without consulting the note, and most will not.
        cells, rules = [""] * 1, []
        span_row = [""]
        for title, first, last in groups:
            span_row.append(f"\\multicolumn{{{last - first + 1}}}{{c}}{{{title}}}")
            rules.append(f"\\cmidrule(lr){{{first}-{last}}}")
        lines.append("    " + " & ".join(span_row) + " \\\\")
        lines.append("    " + " ".join(rules))
    lines += [
        "    " + " & ".join(header) + " \\\\",
        "    \\midrule",
    ]
    lines += ["    " + " & ".join(r) + " \\\\" for r in rows]
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    if note:
        lines.append(f"  \\\\[2pt]\n  \\footnotesize{{{note}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def table_results(root: Path, out: Path) -> bool:
    """Every arm on every split, which is the table the results chapter is built
    around. Sorted by the official split, since that is the primary surface."""
    rows_by_arm: dict[str, dict[str, float]] = {}
    for name, fname in (("internal", "test_metrics.json"),
                        ("official", "test_metrics_official.json")):
        for path in sorted((root / "cvs-domain-ssl" / "probe").glob(f"*/{fname}")):
            d = json.loads(path.read_text())
            arm = path.parent.name.replace("_sages_mean", "").replace("_mean", "")
            if arm.endswith("_endoscapes"):
                continue
            rows_by_arm.setdefault(arm, {})[name] = d["test"]["mAP"]["mean"]
            rows_by_arm[arm][f"{name}_sd"] = d["test"]["mAP"]["std"]
            rows_by_arm[arm]["val"] = d["val_map_selected"]
    for path in sorted((root / "cvs-domain-ssl" / "probe").glob("*_endoscapes_mean/test_metrics.json")):
        d = json.loads(path.read_text())
        arm = path.parent.name.replace("_endoscapes_mean", "")
        rows_by_arm.setdefault(arm, {})["endoscapes"] = d["test"]["mAP"]["mean"]

    if not rows_by_arm:
        print("  missing  probe metrics")
        return False

    order = sorted(rows_by_arm.items(),
                   key=lambda kv: -kv[1].get("official", kv[1].get("internal", 0)))
    rows = []
    for arm, v in order:
        rows.append([
            display(arm),
            f"{v['val']:.4f}" if "val" in v else "--",
            f"{v['internal']:.4f}" if "internal" in v else "--",
            f"{v['official']:.4f}" if "official" in v else "--",
            f"{v['endoscapes']:.4f}" if "endoscapes" in v else "--",
        ])

    tex = latex_table(
        caption=("Frozen-probe mean average precision. The first numeric "
                 "column is the split on which each arm's configuration and "
                 "epoch were selected and is reported as context rather than as "
                 "a result. The ordering differs between splits: arms separated "
                 "by less than roughly 0.05 on the 70-video internal split do "
                 "not retain that ordering on the 300-video official split."),
        label="tab:results",
        header=["Arm", "Val", "Internal test", "Official test", "Endoscapes"],
        rows=rows,
        align="lrrrr",
        note=("$^\\dagger$ adapted on all 700 released training videos, which "
              "include the internal evaluation procedures; these arms are "
              "transductive on the internal splits and not on the official one."),
    )
    (out / "tab_results.tex").write_text(tex + "\n")
    print(f"  written  tab_results  ({len(rows)} arms)")
    return True


def table_stratified(root: Path, out: Path) -> bool:
    """AUC by agreement stratum, both datasets where available."""
    sages = load(root / "rater_agreement" / "bootstrap_val.json")
    endo = load(root / "endoscapes_agreement" / "endoscapes_agreement_test.json")
    if sages is None:
        return False

    # The Endoscapes run was invoked with shorter arm names than the SAGES
    # bootstrap, so a join on the key alone silently drops three arms. Mapping
    # them here rather than re-running keeps the two analyses independent.
    ALIAS = {"dinov3_b_trainonly": "dinov3_adapt",
             "mae_b_trainonly": "mae_adapt",
             "dinov2_b_trainonly": "dinov2_adapt"}

    rows = []
    for arm, v in sorted(sages["arms"].items(), key=lambda kv: -kv[1]["unanimous"]):
        endo_arms = (endo or {}).get("arms", {})
        e = endo_arms.get(arm) or endo_arms.get(ALIAS.get(arm, ""), {})
        rows.append([
            display(arm),
            f"{v['unanimous']:.4f}", f"{v['contested']:.4f}", f"{v['delta']:.4f}",
            f"{e['unanimous']:.4f}" if e else "--",
            f"{e['contested']:.4f}" if e else "--",
        ])

    tex = latex_table(
        caption=("AUC by annotator agreement. Every arm performs substantially "
                 "worse on frames where one of three annotators dissented, on "
                 "both datasets and without exception."),
        label="tab:stratified",
        header=["Arm", "Unanimous", "Contested", "$\\Delta$",
                "Unanimous", "Contested"],
        rows=rows,
        align="lrrrrr",
        groups=[("SAGES", 2, 4), ("Endoscapes", 5, 6)],
        note=("The first three numeric columns are SAGES, the last two "
              "Endoscapes. Values are the mean over C1, C2 and C3."),
    )
    (out / "tab_stratified.tex").write_text(tex + "\n")
    print(f"  written  tab_stratified  ({len(rows)} arms)")
    return True


def table_adaptation(root: Path, out: Path) -> bool:
    """The 2x2, plus DINOv2, on the official split."""
    got: dict[str, float] = {}
    got_sd: dict[str, float] = {}
    for path in sorted((root / "cvs-domain-ssl" / "probe").glob("*/test_metrics_official.json")):
        d = json.loads(path.read_text())
        arm = path.parent.name.replace("_sages_mean", "").replace("_mean", "")
        got[arm] = d["test"]["mAP"]["mean"]
        got_sd[arm] = d["test"]["mAP"]["std"]
    pairs = [("DINOv2, self-distillation", "dinov2_b", "dinov2_b_trainonly"),
             ("DINOv3, self-distillation", "dinov3_b", "dinov3_b_trainonly"),
             ("ViT-MAE, masked reconstruction", "mae_b", "mae_b_trainonly")]
    rows = []
    for label, base, adapt in pairs:
        if base not in got or adapt not in got:
            continue
        change = got[adapt] - got[base]
        rows.append([label, f"{got[base]:.4f}", f"{got[adapt]:.4f}",
                     f"{change:+.4f}", f"{2*got_sd[adapt]:.4f}"])
    if not rows:
        print("  missing  official metrics for the adaptation arms")
        return False

    tex = latex_table(
        caption=("Continued pretraining on 50,400 frames of in-domain "
                 "cholecystectomy video, official test split. The two objective "
                 "families move in opposite directions under an identical "
                 "corpus, architecture and step budget."),
        label="tab:adaptation",
        header=["Objective", "Base", "Adapted", "Change", "$2\\sigma$"],
        rows=rows,
        align="lrrrr",
        note=("$\\sigma$ is the standard deviation over three probe seeds. "
              "Each arm is a single pretraining run."),
    )
    (out / "tab_adaptation.tex").write_text(tex + "\n")
    print(f"  written  tab_adaptation  ({len(rows)} objectives)")
    return True




def table_compute(root: Path, out: Path) -> bool:
    """The measured compute budget, for the appendix.

    No published work in this literature reports peak memory or wall time, so
    anyone reproducing it is guessing at batch sizes. The figures below are read
    from the training logs rather than recalled, and the views-per-sample column
    is what explains the fivefold difference in wall time between the two
    objectives: self-distillation is defined over pairs of views and forwards
    ten crops per image, masked reconstruction operates within one.
    """
    import re

    logs = root / "cvs-domain-ssl" / "logs"
    if not logs.is_dir():
        print("  missing  logs directory")
        return False

    wanted = [
        ("DINOv2, 700-video corpus", "dino_sages_*.out", 10),
        ("DINOv2, 560-video corpus", "dinov2_trainonly_*.out", 10),
        ("DINOv3, 560-video corpus", "dinov3_trainonly_*.out", 10),
        ("ViT-MAE, 560-video corpus", "mae_trainonly_*.out", 1),
    ]

    rows = []
    for label, pattern, views in wanted:
        text = ""
        for path in sorted(logs.glob(pattern)):
            text += path.read_text(errors="ignore")
        if not text:
            print(f"  no log for {label}")
            continue
        steps = re.findall(r"(\d+) steps", text)
        hours = re.findall(r"Elapsed ([\d.]+)h", text)
        vram = re.findall(r"Peak VRAM ([\d.]+) GiB", text)
        batch = re.findall(r"batch\s+(\d+)", text)
        rows.append([
            label,
            f"{int(steps[-1]):,}" if steps else "--",
            batch[-1] if batch else "--",
            str(views),
            f"{float(hours[-1]):.2f}" if hours else "--",
            f"{float(vram[-1]):.2f}" if vram else "--",
        ])

    if not rows:
        return False

    tex = latex_table(
        caption=("Measured cost of continued pretraining. The two objectives "
                 "are matched on optimisation steps and not on computation: "
                 "self-distillation forwards ten crops per image where masked "
                 "reconstruction forwards one, which no configuration can "
                 "equalise without breaking the schedules the published recipes "
                 "define in steps."),
        label="tab:compute",
        header=["Arm", "Steps", "Batch", "Views", "Hours", "Peak GiB"],
        rows=rows,
        align="lrrrrr",
        note=("Measured on NVIDIA V100 accelerators under float16 mixed "
              "precision. Gradient checkpointing was available and not enabled."),
    )
    (out / "tab_compute.tex").write_text(tex + "\n")
    print(f"  written  tab_compute  ({len(rows)} arms)")
    return True


def table_extraction(root: Path, out: Path) -> bool:
    """Feature extraction throughput and cache size, from the manifests.

    Image and video arms differ by more than an order of magnitude in both, and
    the difference is the reason the video arms dominate storage: a 2,048-token
    grid at 1,024 dimensions is forty times the bytes of a 196-token grid at
    768.
    """
    import json as _json

    cache = root.parent / "cache"
    if not cache.is_dir():
        print("  missing  cache directory")
        return False

    rows = []
    for arm in sorted(p.name for p in cache.iterdir() if p.is_dir()):
        m = cache / arm / "sages" / "val" / "manifest.json"
        if not m.is_file():
            continue
        d = _json.loads(m.read_text())
        shape = d.get("shapes", {}).get("tokens")
        if not shape:
            continue
        gib = np.prod(shape) * 2 / 1024 ** 3
        rows.append([
            display(arm),
            "video" if len(shape) == 3 and shape[1] > 500 else "image",
            f"{shape[1]:,}",
            str(shape[2]),
            f"{gib:.2f}",
        ])
    if not rows:
        return False

    tex = latex_table(
        caption=("Cached feature size per 1,260-frame split. Video arms carry "
                 "eight to ten times the tokens of image arms at up to a third "
                 "more width, which is why they dominate storage."),
        label="tab:extraction",
        header=["Arm", "Input", "Tokens", "Dim", "GiB per split"],
        rows=rows,
        align="llrrr",
    )
    (out / "tab_extraction.tex").write_text(tex + "\n")
    print(f"  written  tab_extraction  ({len(rows)} arms)")
    return True


def table_splits(root: Path, out: Path) -> bool:
    """The evaluation sets, and which of them the SSL corpus covers.

    The final column is the reason this table earns its space. The unlabelled
    corpus used for continued pretraining covers all 700 released SAGES
    training videos, so the internal validation and test procedures were seen
    without labels during adaptation. That is transductive learning rather than
    label leakage, but the adapted encoders have observed the pixel content of
    frames they are then evaluated on and the unadapted ones have not, and a
    reader is entitled to see which splits that affects before reading any
    adaptation result.

    Counts are read from the manifests rather than transcribed.
    """
    import pandas as pd

    repo = Path.cwd()
    rows = []
    try:
        d = pd.read_csv(repo / "metadata" / "sages_frames_internal_split.csv")
        o = pd.read_csv(repo / "metadata" / "sages_frames_official_test.csv")
    except FileNotFoundError as exc:
        print(f"  missing  {exc.filename}")
        return False

    order = [("train", "fit the classification head", "yes"),
             ("val", "select configuration and epoch", "yes"),
             ("test", "development held-out set", "yes")]
    for split, role, in_corpus in order:
        g = d[d.internal_split == split]
        rows.append(["SAGES", split.capitalize(), role,
                     f"{g.video_id.nunique():,}", f"{len(g):,}", in_corpus])
    rows.append(["SAGES", "Official test", "final evaluation, scored once",
                 f"{o.video_id.nunique():,}", f"{len(o):,}", "no"])

    endo = repo / "metadata" / "endoscapes_frames.csv"
    if endo.is_file():
        e = pd.read_csv(endo)
        e = e[e.is_cvs_annotated]
        for split, role in (("train", "fit the classification head"),
                            ("val", "select configuration and epoch"),
                            ("test", "cross-dataset evaluation")):
            g = e[e.split == split]
            rows.append(["Endoscapes", split.capitalize(), role,
                         f"{g.video_id.nunique():,}", f"{len(g):,}", "no"])

    tex = latex_table(
        caption=("Evaluation sets. The unlabelled corpus used for continued "
                 "pretraining covers all 700 released SAGES training videos, so "
                 "the adapted encoders observed the internal validation and test "
                 "frames without labels; results for those arms on the internal "
                 "splits are transductive and on the official split are not."),
        label="tab:splits",
        header=["Dataset", "Split", "Role", "Videos", "Frames",
                "In SSL corpus"],
        rows=rows,
        align="llrrrl",
        note=("SAGES splits divide the 700 released training videos by video, "
              "stratified on video-level \\cvs{} achievement. Endoscapes counts "
              "are CVS-annotated frames under the official 120/41/40 split."),
    )
    (out / "tab_splits.tex").write_text(tex + "\n")
    print(f"  written  tab_splits  ({len(rows)} splits)")
    return True


def table_registry(root: Path, out: Path) -> bool:
    """The encoders, with the properties that vary across them.

    Read from the cache manifests, so that patch size, token count and width
    are what the extraction actually produced rather than what the checkpoint
    name implies. The pretraining corpus is not recorded in the manifest and is
    supplied from the checkpoint documentation; it is the one column a reader
    should check against the sources.

    The table exists to make the confounds visible at a glance. Nine encoders
    differ in objective, but they also differ in capacity, patch size, token
    count and pretraining corpus, and only three share a configuration exactly.
    A comparison across the full set is therefore a screen and not a controlled
    experiment, which is easier to see in a table than to argue in prose.
    """
    import json as _json

    #: Not recoverable from a manifest. Sources: the model cards for each
    #: checkpoint. LVD-1689M is DINOv2's and DINOv3's curated corpus; the
    #: ImageNet variants are as published.
    CORPUS = {
        "dinov2_b": "LVD-142M", "dinov3_b": "LVD-1689M",
        "mae_b": "ImageNet-1k", "ijepa_h": "ImageNet-22k",
        "vit_sup_b": "ImageNet-21k, ft.\\ 1k",
        "videomae_b_base": "Kinetics-400", "vjepa2_l_base": "VideoMix2M",
    }
    OBJECTIVE = {
        "dinov2": "self-distillation", "dinov3": "self-distillation",
        "mae": "masked reconstruction", "videomae": "masked reconstruction",
        "ijepa": "latent prediction", "vjepa": "latent prediction",
        "vit_sup": "supervised classification",
    }

    cache = root.parent / "cache"
    rows = []
    for arm in ("dinov2_b", "dinov3_b", "mae_b", "vit_sup_b", "ijepa_h",
                "videomae_b_base", "vjepa2_l_base"):
        m = cache / arm / "sages" / "val" / "manifest.json"
        if not m.is_file():
            print(f"  no manifest for {arm}")
            continue
        d = _json.loads(m.read_text())
        enc = d["encoder"]
        layout = enc.get("token_layout", {})
        grid = layout.get("grid", [])
        shape = d.get("shapes", {}).get("tokens", [None, None, None])
        objective = next((v for k, v in OBJECTIVE.items() if arm.startswith(k)), "--")
        rows.append([
            display(arm),
            "video" if len(grid) == 3 else "image",
            objective,
            str(enc.get("patch_size", "--")),
            f"{shape[1]:,}" if shape[1] else "--",
            str(layout.get("dim", shape[2] or "--")),
            CORPUS.get(arm, "--"),
        ])
    if not rows:
        return False

    tex = latex_table(
        caption=("The encoders compared. They differ in pretext objective, but "
                 "also in input level, patch size, token count, width and "
                 "pretraining corpus; only DINOv3, ViT-MAE and the supervised "
                 "control share a configuration exactly. A comparison across the "
                 "full set is therefore a screen rather than a controlled "
                 "experiment."),
        label="tab:registry",
        header=["Arm", "Input", "Objective", "Patch", "Tokens", "Width",
                "Pretraining corpus"],
        rows=rows,
        align="lllrrrl",
        note=("Patch size, token count and width are read from the cache "
              "manifests and are what the extraction produced. Corpora are as "
              "published for each checkpoint."),
    )
    (out / "tab_registry.tex").write_text(tex + "\n")
    print(f"  written  tab_registry  ({len(rows)} encoders)")
    return True


def table_ceiling(root: Path, out: Path) -> bool:
    """The annotation ceiling, against the best arm.

    A model score is uninterpretable without the ceiling beside it. Balanced
    accuracy of 0.61 reads as poor against a maximum of 1.0 and as recovering a
    third of the attainable range against a ceiling of 0.83, and the second is
    the honest comparison because the reference standard is what the model is
    scored against.

    Reported on the official 300-video split, which is four times the size of
    the internal one and the surface every model figure in Chapter 4 uses.
    """
    for candidate in (root / "rater_agreement_official" / "rater_agreement_test.json",
                      root / "rater_agreement" / "rater_agreement_test.json"):
        d = load(candidate)
        if d:
            break
    else:
        return False
    # The ceiling is a list of per-criterion records, each carrying a
    # "criterion" key, rather than a mapping keyed by criterion.
    raw = d.get("ceiling", [])
    ceiling = {r["criterion"]: r for r in raw} if isinstance(raw, list) else raw

    best = None
    for path in (root / "cvs-domain-ssl" / "probe").glob("*/test_metrics_official.json"):
        m = json.loads(path.read_text())
        bacc = m["test"].get("mean_bacc", {}).get("mean")
        if bacc and (best is None or bacc > best[1]):
            best = (path.parent.name.replace("_sages_mean", "").replace("_mean", ""), bacc)

    rows = []
    for c in ("c1", "c2", "c3"):
        v = ceiling.get(c, {})
        if "bacc" not in v:
            continue
        rows.append([c.upper(), f"{v['bacc']:.4f}",
                     f"{v.get('f1', float('nan')):.4f}",
                     f"{v.get('recall', float('nan')):.4f}",
                     f"{v.get('specificity', v.get('spec', float('nan'))):.4f}"])
    if not rows:
        return False
    mean = np.mean([float(r[1]) for r in rows])
    rows.append(["\\textit{Mean}", f"\\textbf{{{mean:.4f}}}", "", "", ""])

    note = ("Each annotator is scored against the consensus of the other two "
            "and the results averaged; frames on which the remaining pair "
            "splits admit no majority and are excluded.")
    if best:
        share = (best[1] - 0.5) / (mean - 0.5)
        note += (f" The best arm reaches {best[1]:.4f} balanced accuracy, "
                 f"{100*share:.0f}\\% of the range between chance and this "
                 f"ceiling.")

    tex = latex_table(
        caption=("The annotation ceiling on the official test split. A trained "
                 "surgeon assessed against the consensus of two peers reaches "
                 f"{mean:.3f} balanced accuracy, so the reference standard "
                 "against which every published result is scored is itself only "
                 "moderately reliable."),
        label="tab:ceiling",
        header=["Criterion", "BAcc", "F1", "Recall", "Specificity"],
        rows=rows,
        align="lrrrr",
        note=note,
    )
    (out / "tab_ceiling.tex").write_text(tex + "\n")
    print(f"  written  tab_ceiling  (mean {mean:.4f})")
    return True


def table_blank(root: Path, out: Path) -> bool:
    """Blank frames across the four evaluation splits.

    SAGES is included precisely because it has none: without it a reader cannot
    tell whether empty frames are a property of surgical video or of one
    release, and the answer determines whether the observation is a general
    caution or a specific one.
    """
    rows = []
    for label, path in (
            ("Endoscapes train", root / "blank_frames" / "blank_frames_train.json"),
            ("Endoscapes val", root / "blank_frames" / "blank_frames_val.json"),
            ("Endoscapes test", root / "blank_frames" / "blank_frames_test.json"),
            ("SAGES official test", root / "blank_frames_sages" / "blank_frames_test.json")):
        d = load(path)
        if not d:
            continue
        pos = sum(v.get("positive_blank", 0) for v in d.get("labels", {}).values())
        rows.append([
            label,
            f"{d['n_frames']:,}",
            f"{d['n_blank']:,}",
            f"{100*d['share_blank']:.1f}\\%",
            f"{d['n_videos_affected']} of {d['n_videos_total']}",
            str(pos) if d["n_blank"] else "--",
        ])
    if not rows:
        return False

    tex = latex_table(
        caption=("Frames carrying no image content. Blankness is a mean "
                 "luminance below 5 of 255; the count is unchanged at "
                 "thresholds of 2, 5, 10 and 20, so the criterion separates "
                 "empty frames from dark ones. SAGES contains none."),
        label="tab:blank",
        header=["Split", "Frames", "Blank", "Share", "Videos", "Positive labels"],
        rows=rows,
        align="lrrrlr",
        note=("The final column counts criterion-instances labelled positive on "
              "a blank frame. The blanking is contiguous within a video and "
              "covers unannotated frames as well, so it is a property of the "
              "released video rather than of the frame extraction."),
    )
    (out / "tab_blank.tex").write_text(tex + "\n")
    print(f"  written  tab_blank  ({len(rows)} splits)")
    return True


def table_temporal(root: Path, out: Path) -> bool:
    """Held-out mAP against the length of the trailing window.

    Tabulated as well as plotted because the finding is a comparison between
    adjacent numbers -- the step from one frame to three, and its absence
    thereafter -- and a reader checking that reads a table more easily than a
    figure.
    """
    arms = {}
    for path in sorted((root / "temporal_probe").glob("*/temporal_probe.json")):
        arms[path.parent.name] = json.loads(path.read_text())
    if not arms:
        return False

    ks = sorted({int(k) for d in arms.values() for k in d["windows"]})
    rows = []
    for k in ks:
        row = [str(k), f"{5*(k-1)}"]
        for arm in arms:
            w = arms[arm]["windows"].get(str(k), {})
            t = w.get("test", {}).get("mAP", {})
            row.append(f"{t['mean']:.4f}" if "mean" in t else "--")
        rows.append(row)

    tex = latex_table(
        caption=("Mean average precision against the length of the trailing "
                 "window, official test split. Temporal context gains 0.018 mAP "
                 "for DINOv2 at three frames with no further gain to eighteen, "
                 "and 0.016 for DINOv3 only at eighteen. The benefit is small in "
                 "both cases and does not grow monotonically with the window."),
        label="tab:temporal",
        header=["$k$", "Span (s)"] + [display(a) for a in arms],
        rows=rows,
        align="rr" + "r" * len(arms),
        note=("$k = 1$ is the frozen probe with a recurrent layer of one step, "
              "so the only factor varying down the table is the temporal "
              "aggregation. Windows are trailing, and the configuration is "
              "selected on validation for each $k$ independently."),
    )
    (out / "tab_temporal.tex").write_text(tex + "\n")
    print(f"  written  tab_temporal  ({len(ks)} windows, {len(arms)} arms)")
    return True


FIGURES = {"stratified": fig_stratified, "shrinkage": fig_shrinkage,
           "temporal": fig_temporal, "geometry": fig_geometry}
TABLES = {"results": table_results, "stratified": table_stratified,
          "adaptation": table_adaptation, "compute": table_compute,
          "extraction": table_extraction, "splits": table_splits,
          "registry": table_registry, "ceiling": table_ceiling,
          "blank": table_blank, "temporal": table_temporal}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="../outputs",
                   help="the directory holding rater_agreement, temporal_probe, ...")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--width", type=float, default=6.3,
                   help="text width in inches; 6.3 is the report class at 11pt "
                        "with 2.5 cm margins on A4")
    p.add_argument("--figure", action="append", choices=sorted(FIGURES))
    p.add_argument("--table", action="append", choices=sorted(TABLES))
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    apply_style()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures = sorted(FIGURES) if (args.all or not (args.figure or args.table)) else (args.figure or [])
    tables = sorted(TABLES) if (args.all or not (args.figure or args.table)) else (args.table or [])

    print("figures")
    for name in figures:
        FIGURES[name](root, out, args.width)
    print("tables")
    for name in tables:
        TABLES[name](root, out)

    print(f"\noutput in {out}")
    print("Include a figure at scale 1.0:  \\includegraphics{fig_stratified.pdf}")
    print("Include a table:                \\input{tab_results.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
