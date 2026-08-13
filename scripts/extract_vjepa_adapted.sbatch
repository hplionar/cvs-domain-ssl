#!/bin/bash
#SBATCH --job-name=extract_vjepa
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
  python scripts/extract_features.py \
    --encoder vjepa2_l \
    --model-name facebook/vjepa2-vitl-fpc16-256-ssv2 \
    --checkpoint $ROOT/outputs/cvs-domain-ssl/ssl/vjepa2_l_sages/encoder_final.pt \
    --dataset sages_clip --split $SPLIT \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache/vjepa2_l_adapted/sages/$SPLIT \
    --batch-size 16 --num-workers 12
done
