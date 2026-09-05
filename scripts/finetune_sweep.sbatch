#!/bin/bash
#SBATCH --job-name=finetune_sweep
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Every result in this project concerns frozen features, so whether the ranking
# they establish survives fine-tuning is the main threat to their external
# validity. Testing that with one recipe per arm would leave a null
# indistinguishable from a badly chosen learning rate, so only the backbone rate
# varies and everything else is held fixed. A backbone rate of zero freezes the
# encoder and is the control: it should reproduce the frozen-probe figure, and
# if it does not, something differs beyond the backbone update.
#
# Three arms rather than one, because the question is whether the ranking
# survives and not whether fine-tuning helps. Each arm is minutes.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for ARM in dinov3_b mae_b dinov2_b; do
  for LR in 0 1e-6 1e-5 1e-4; do
    echo; echo "################################ $ARM  backbone lr $LR"
    python -u train/finetune_cvs.py \
      --encoder $ARM \
      --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
      --manifest-path metadata/sages_frames_internal_split.csv \
      --backbone-lr $LR --unfreeze-blocks 4 \
      --epochs 15 --batch-size 32 \
      --output-dir $ROOT/outputs/finetune/${ARM}_lr$LR \
      || echo "FAILED: $ARM lr $LR"
  done
done

echo; echo "################################ summary"
python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("/group/pmc085/hlionar/outputs/finetune/*/results.json")):
    d = json.load(open(f))
    name = os.path.basename(os.path.dirname(f))
    if name == "smoke":
        continue
    rows.append((name, d["args"]["encoder"], d["args"]["backbone_lr"],
                 d["best_val_map"], d["best_epoch"]))
print("%-24s %-10s %10s %10s %7s" % ("run", "encoder", "backbone lr", "val mAP", "epoch"))
for r in sorted(rows, key=lambda r: (r[1], r[2])):
    print("%-24s %-10s %10.0e %10.4f %7d" % r)
print()
print("frozen probe, validation: dinov2_b 0.4593 | dinov3_b 0.4584 | mae_b 0.3900")
print("frozen probe, official test: 0.4159 | 0.4167 | 0.3517")
print()
print("Read the lr = 0 rows first: they should reproduce the frozen figures.")
print("Then whether the ordering between arms is the same at the best rate as")
print("it is frozen. That is the question; the magnitudes are secondary.")
PY
