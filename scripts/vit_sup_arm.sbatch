#!/bin/bash
#SBATCH --job-name=vit_sup_arm
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The control the objective comparison lacks. Every other arm is
# self-supervised, so the ranking cannot say whether self-supervision
# contributes anything over a strong supervised backbone -- and SwinCVS, the
# strongest reproducible published result, is supervised ImageNet transfer.
#
# google/vit-base-patch16-224 matches dinov3_b and mae_b on architecture, patch
# size, token count, hidden dimension and input resolution. It normalises to
# 0.5/0.5 rather than the ImageNet statistics, which is recorded in the manifest
# and is a property of the checkpoint rather than a protocol deviation.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for SPLIT in train val test; do
  echo; echo "=== vit_sup_b / sages / ${SPLIT} ==="
  python -u scripts/extract_features.py \
    --encoder vit_sup_b \
    --dataset sages --split "$SPLIT" \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache/vit_sup_b/sages/$SPLIT \
    --reduction none \
    --batch-size 64 --num-workers 12
done

echo; echo "=== probe ==="
python -u train/train_probe_cached.py \
  --train-features $ROOT/cache/vit_sup_b/sages/train \
  --val-features   $ROOT/cache/vit_sup_b/sages/val \
  --head mean --seeds 3 --in-memory \
  --output-dir $ROOT/outputs/cvs-domain-ssl/probe/vit_sup_b_sages_mean

echo; echo "=== held-out score ==="
python -u eval/score_test_cache.py \
  --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/vit_sup_b_sages_mean \
  --train-features $ROOT/cache/vit_sup_b/sages/train \
  --val-features   $ROOT/cache/vit_sup_b/sages/val \
  --test-features  $ROOT/cache/vit_sup_b/sages/test

echo
echo "compare, held-out test mAP:  dinov2_b 0.4508 | dinov3_b 0.4115 | mae_b 0.3624"
