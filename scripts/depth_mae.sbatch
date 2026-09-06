#!/bin/bash
#SBATCH --job-name=depth_mae
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# DINOv3 gains from adaptation at blocks 3, 6 and 9 and exactly nothing at block
# 12, so the final blocks discard the improvement. The semantic-diffusion account
# predicts this is specific to self-distillation, and ViT-MAE is the test: its
# within-frame token geometry did not degrade under adaptation, so its depth
# curves should not separate the way DINOv3's do.
#
# A null here would tie the depth result to the geometry result. A matching
# effect would mean the discarding is general and the geometry account is
# incomplete.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for spec in "mae_b:-" "mae_b_trainonly:$ROOT/outputs/cvs-domain-ssl/ssl/mae_b_sages_trainonly/encoder_final.pt"; do
  arm=${spec%%:*}; ckpt=${spec#*:}
  extra=""; [ "$ckpt" != "-" ] && extra="--checkpoint $ckpt"
  for SPLIT in train val; do
    echo; echo "=== $arm / $SPLIT ==="
    python -u scripts/extract_features.py \
      --encoder mae_b $extra \
      --dataset sages --split "$SPLIT" \
      --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
      --manifest-path metadata/sages_frames_internal_split.csv \
      --out $ROOT/cache_depth/$arm/sages/$SPLIT \
      --reduction none --layer-depths 0.25 0.5 0.75 1.0 \
      --batch-size 32 --num-workers 12
  done
done

python -u eval/depth_probe.py \
  --arm mae_b --arm mae_b_trainonly \
  --cache-root $ROOT/cache_depth \
  --output-dir $ROOT/outputs/depth_probe_mae

echo
echo "DINOv3 for comparison, validation mAP by block:"
echo "  base    3: 0.2215  6: 0.3341  9: 0.4567  12: 0.4584"
echo "  adapted 3: 0.2445  6: 0.3753  9: 0.4922  12: 0.4582"
