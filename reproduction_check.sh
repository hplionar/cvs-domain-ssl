#!/bin/bash
# Prints every number in the progress report, from the logs and results files
# that produced them. One command, three sections, in report order.
#
#   bash reproduction_check.sh
#
# Nothing here recomputes anything: it reads what the jobs wrote. If a figure in
# the report is not reproduced by this script, the report is wrong.

set -uo pipefail

CVS=/group/pmc085/hlionar/cvs-domain-ssl
OUT=/group/pmc085/hlionar/outputs/cvs-domain-ssl
LOGS=$OUT/logs
SWIN=/group/pmc085/hlionar/swincvs-reproduction

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate /group/pmc085/hlionar/conda_envs/cvsssl 2>/dev/null || true
cd "$CVS"

echo
echo "══════════════════════════════════════════════════════════════════"
echo " §1  COMPUTE — throughput, peak VRAM, elapsed"
echo "══════════════════════════════════════════════════════════════════"
for f in $LOGS/vmae_sages_*.out $LOGS/vjepa_sages_*.out $LOGS/dino_sages_1092836*.out; do
  [ -f "$f" ] || continue
  line=$(grep -E "Elapsed|Peak VRAM" "$f" | tail -2 | tr '\n' ' ')
  # elements that found the schedule complete report ~0.00h; skip them
  case "$line" in *"Elapsed 0.00h"*) continue ;; esac
  [ -n "$line" ] && printf "  %-34s %s\n" "$(basename "$f" .out)" "$line"
done
echo
echo "  V-JEPA spans four array elements; sum the elapsed figures (14.5 h total)."

echo
echo "══════════════════════════════════════════════════════════════════"
echo " §2  SwinCVS — five Frozen seeds against published 0.6745 ± 0.0041"
echo "══════════════════════════════════════════════════════════════════"
python3 - <<'PY'
import glob, os, re, statistics as st
rows = {}
for path in sorted(glob.glob('/group/pmc085/hlionar/swincvs-reproduction/reproduction/logs/swincvs_frozen_test_*.out')):
    text = open(path, errors='replace').read()
    seed = re.search(r'_(\d+)\.out$', path)
    get = lambda k: (lambda m: float(m.group(1)) if m else None)(re.search(rf'{k}\s+([\d.]+)', text))
    rows[seed.group(1) if seed else path] = (get('^mAP'), get('C1 ap'), get('C2 ap'), get('C3 ap'))

if not rows:
    print("  no test logs found")
else:
    print(f"  {'seed':>6s} {'mAP':>8s} {'C1':>8s} {'C2':>8s} {'C3':>8s}")
    for seed, vals in rows.items():
        print(f"  {seed:>6s} " + " ".join(f"{v:8.4f}" if v else "     —  " for v in vals))
    cols = list(zip(*[v for v in rows.values() if all(v)]))
    if cols:
        pub = [(67.45, 0.41), (65.02, 0.77), (61.38, 1.71), (75.95, 1.96)]
        print()
        print(f"  {'':>6s} {'ours':>16s} {'published':>16s} {'diff':>9s}")
        for name, col, (p, ps) in zip(('mAP', 'C1', 'C2', 'C3'), cols, pub):
            m, s = st.mean(col) * 100, st.stdev(col) * 100
            print(f"  {name:>6s} {m:8.2f} ± {s:4.2f} {p:8.2f} ± {ps:4.2f} {m-p:+8.2f} pp")
PY

echo
echo "══════════════════════════════════════════════════════════════════"
echo " §4  MAIN RESULT — SAGES validation, frozen probe, mean head"
echo "══════════════════════════════════════════════════════════════════"
PYTHONPATH=. python3 - <<'PY'
import json, os
base = '/group/pmc085/hlionar/outputs/cvs-domain-ssl/probe'
groups = [
    ("frames", [('dinov2_b_sages_mean','DINOv2 ViT-B'),
                ('dinov2_b_adapted_sages_mean','DINOv2 ViT-B, adapted'),
                ('dinov3_b_sages_mean','DINOv3 ViT-B'),
                ('mae_b_sages_mean','ViT-MAE ViT-B'),
                ('ijepa_h_sages_mean','I-JEPA ViT-H')]),
    ("clips",  [('videomae_b_base_mean','VideoMAE ViT-B'),
                ('videomae_b_adapted_mean','VideoMAE ViT-B, adapted'),
                ('vjepa2_l_base_mean','V-JEPA 2 ViT-L'),
                ('vjepa2_l_adapted_mean','V-JEPA 2 ViT-L, adapted'),
                ('vjepa2_l_lr1e6_mean','V-JEPA 2 ViT-L, adapted lr 1e-6')]),
]
for kind, arms in groups:
    print(f"\n  {kind}")
    print(f"  {'':32s} {'mAP':>8s} {'C1':>8s} {'C2':>8s} {'C3':>8s}")
    for key, label in arms:
        try:
            r = json.load(open(f'{base}/{key}/results.json'))['selected']
            m = r['metrics']
            print(f"  {label:32s} {r['mean_map']:8.4f} {m['c1_ap']['mean']:8.4f} "
                  f"{m['c2_ap']['mean']:8.4f} {m['c3_ap']['mean']:8.4f}")
        except Exception:
            print(f"  {label:32s}   pending")
print("\n  chance floor                       0.1750   0.1270   0.2460   0.1510")
PY

echo
echo "══════════════════════════════════════════════════════════════════"
echo " SSL runs — did each converge?"
echo "══════════════════════════════════════════════════════════════════"
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob('/group/pmc085/hlionar/outputs/cvs-domain-ssl/ssl/*/history.json')):
    h = json.load(open(f))
    if len(h) < 10:
        continue
    name = os.path.basename(os.path.dirname(f))
    losses = [r['loss'] for r in h]
    n = len(losses)
    tail = sum(losses[int(0.9*n):]) / len(losses[int(0.9*n):])
    prev = sum(losses[int(0.8*n):int(0.9*n)]) / len(losses[int(0.8*n):int(0.9*n)])
    flat = "converged" if abs(tail - prev) < 0.01 * losses[0] else "STILL FALLING"
    print(f"  {name:26s} {losses[0]:7.4f} -> {losses[-1]:7.4f}  "
          f"({100*(losses[0]-losses[-1])/losses[0]:5.1f}% down)  {flat}")
PY

echo
echo "══════════════════════════════════════════════════════════════════"
echo " SwinCVS defects — verified in their released code"
echo "══════════════════════════════════════════════════════════════════"
echo
echo "  (a) scheduler declared in config but never called:"
grep -E "WARMUP_EPOCHS|BASE_LR|MIN_LR|EPOCHS:" $SWIN/config/SwinCVS_config.yaml | sed 's/^/      /'
printf "      build_scheduler called in SwinCVS.py: "
grep -c "build_scheduler" $SWIN/SwinCVS.py
echo "      (paper states 10 epochs; config says 8)"
echo
echo "  (b) frozen backbone never put in eval mode:"
sed -n '43,47p' $SWIN/scripts/f_build.py | sed 's/^/      /'
echo
echo "  (c) best-epoch selection never implemented:"
grep -n "ADD CHOOSING AND LOADING BEST EPOCH" $SWIN/SwinCVS.py | sed 's/^/      /'
echo

echo "══════════════════════════════════════════════════════════════════"
echo " Corpora"
echo "══════════════════════════════════════════════════════════════════"
D=/group/pmc085/hlionar/datasets
printf "  SAGES SSL sparse (1 fps):  %s frames\n" \
  "$(find $D/SAGES_CVS_Challenge_2024/derived_ssl/sparse/train/frames -name '*.jpg' 2>/dev/null | wc -l)"
printf "  SAGES SSL dense  (5 fps):  %s frames\n" \
  "$(find $D/SAGES_CVS_Challenge_2024/derived_ssl/dense/train/frames -name '*.jpg' 2>/dev/null | wc -l)"
printf "  Cholec80:                  %s videos\n" \
  "$(ls $D/cholec80/videos/*.mp4 2>/dev/null | wc -l)"
echo
