#!/bin/bash
#SBATCH --job-name=eval_vjepa_lr1e6
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for SPLIT in val train; do
  python -u scripts/extract_features.py \
    --encoder vjepa2_l \
    --model-name facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --checkpoint $ROOT/outputs/cvs-domain-ssl/ssl/vjepa2_l_sages_lr1e6/encoder_final.pt \
    --dataset sages_clip --split $SPLIT \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache/vjepa2_l_lr1e6/sages/$SPLIT \
    --batch-size 16 --num-workers 12
done

python -u train/train_probe_cached.py \
  --train-features $ROOT/cache/vjepa2_l_lr1e6/sages/train \
  --val-features   $ROOT/cache/vjepa2_l_lr1e6/sages/val \
  --head mean --seeds 3 --in-memory \
  --output-dir $ROOT/outputs/cvs-domain-ssl/probe/vjepa2_l_lr1e6_mean

echo "compare: base 0.3356, adapted@4.7e-6 0.2085"
