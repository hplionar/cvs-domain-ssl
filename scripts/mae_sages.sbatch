#!/bin/bash
#SBATCH --job-name=mae_sages
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Fills the one missing cell: DINOv2 on SAGES frames.
#
# DINOv2 scores 0.5208 on Endoscapes and VideoMAE 0.2545 on SAGES, but those
# differ in both encoder and dataset. This run holds the encoder fixed and
# changes only the dataset, so the two effects separate.
#
# Frames, not clips: SAGESFrameDataset reads the 18 labelled frames per video
# that the clip dataset builds its windows around, so the label set is identical
# and only the input construction differs.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for SPLIT in val train; do
  echo; echo "=== mae_b / sages / ${SPLIT} ==="
  python -u scripts/extract_features.py \
    --encoder mae_b \
    --dataset sages --split "$SPLIT" \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache/mae_b/sages/$SPLIT \
    --batch-size 64 --num-workers 12
done

echo; echo "=== probe ==="
python -u train/train_probe_cached.py \
  --train-features $ROOT/cache/mae_b/sages/train \
  --val-features   $ROOT/cache/mae_b/sages/val \
  --head mean --seeds 3 --in-memory \
  --output-dir $ROOT/outputs/cvs-domain-ssl/probe/mae_b_sages_mean

echo
echo "compare:  dinov2 endoscapes 0.5208 | videomae sages 0.2545 | chance 0.175"
