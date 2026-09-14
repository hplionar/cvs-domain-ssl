#!/bin/bash
#SBATCH --job-name=depth_fine
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The DINO projection head reads the CLS token alone. The 196 patch tokens at
# the final block therefore receive no gradient through the loss; they move only
# because block 12's weights are pulled to serve CLS. That predicts the damage
# is confined to the last block or two rather than accumulating through the
# stack, and it is a different account from semantic diffusion, which requires a
# loss that reads the patches.
#
# Blocks 9, 10, 11 and 12 at four relative depths separate the two. A sharp drop
# between 11 and 12 with 11 near base is neglect. A graded decline from 9 is
# something upstream, where the tokens still feed forward into what the loss
# reads.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for spec in "dinov3_b:-" "dinov3_b_trainonly:$ROOT/outputs/cvs-domain-ssl/ssl/dinov3_b_sages_trainonly/encoder_final.pt"; do
  arm=${spec%%:*}; ckpt=${spec#*:}
  extra=""; [ "$ckpt" != "-" ] && extra="--checkpoint $ckpt"
  echo; echo "=== $arm ==="
  python -u scripts/extract_features.py \
    --encoder dinov3_b $extra \
    --dataset sages --split val \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache_depth_fine/$arm/sages/val \
    --reduction none --layer-depths 0.75 0.8333 0.9167 1.0 \
    --batch-size 32 --num-workers 12
done

echo
echo "coarse depths, validation mAP:"
echo "  base    3: 0.2215  6: 0.3341  9: 0.4567  12: 0.4584"
echo "  adapted 3: 0.2445  6: 0.3753  9: 0.4922  12: 0.4582"
