#!/bin/bash
#SBATCH --job-name=mae_endoscapes
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# A pipeline sanity check, not an experiment.
#
# THE QUESTION. Frozen VideoMAE on SAGES scores 0.30 mAP. Frozen DINOv2 on
# Endoscapes scores 0.5208 under the same probe. Either SAGES is a much harder
# measurement, or something in the video path is wrong.
#
# ViT-MAE ViT-B isolates one variable. It is masked reconstruction, like
# VideoMAE, but operates on single frames, so it uses the image path DINOv2
# used and needs no clip construction. Endoscapes ships at 1 fps and a
# 16-frame clip would span 16 seconds against the 2.1 seconds VideoMAE was
# adapted on, so avoiding clips avoids a confound as well as the code.
#
# READING THE RESULT
#
#   ~0.50   The image path is sound and masked reconstruction is not the
#           problem. The remaining suspects are the video clip construction and
#           SAGES itself.
#   ~0.30   The result travels with the objective rather than the dataset,
#           which would be a finding rather than a bug -- but would also warrant
#           checking the image path directly against DINOv2 on the same cache.
#   <0.20   Below the 0.175 chance floor for these prevalences. That would
#           indicate a genuine defect and everything downstream is suspect.
#
# Endoscapes val is 2,331 annotated frames and train 6,960, so extraction is
# minutes rather than hours: 197 tokens per frame against SAGES's 1,568.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

DATA=$ROOT/datasets/endoscapes
MANIFEST=metadata/endoscapes_frames.csv

echo "=============================================================="
echo "job      ${SLURM_JOB_ID}   node $(hostname)   started $(date -Is)"
echo "commit   $(git rev-parse --short HEAD)"
echo "=============================================================="

for SPLIT in val train; do
  echo
  echo "=== mae_b / endoscapes / ${SPLIT} ==="
  python -u scripts/extract_features.py \
    --encoder mae_b \
    --dataset endoscapes --split "$SPLIT" \
    --dataset-root "$DATA" --manifest-path "$MANIFEST" \
    --out "$ROOT/cache/mae_b/endoscapes/${SPLIT}" \
    --batch-size 64 --num-workers 12
done

echo
echo "=== probe: mae_b / endoscapes / mean head ==="
python -u train/train_probe_cached.py \
  --train-features $ROOT/cache/mae_b/endoscapes/train \
  --val-features   $ROOT/cache/mae_b/endoscapes/val \
  --head mean --seeds 3 --in-memory \
  --output-dir $ROOT/outputs/cvs-domain-ssl/probe/mae_b_endoscapes_mean

echo
echo "=============================================================="
echo "finished $(date -Is)"
echo
echo "Compare against, all validation mAP under the same probe:"
echo "  dinov2_b  endoscapes  0.5208"
echo "  dinov3_b  endoscapes  0.4797"
echo "  videomae  sages base  0.2545"
echo "  chance floor          ~0.175"
echo "=============================================================="
