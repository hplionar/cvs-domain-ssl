#!/bin/bash
#SBATCH --job-name=extract_vmae_tr
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

python scripts/extract_features.py \
  --encoder videomae_b \
  --checkpoint $ROOT/outputs/cvs-domain-ssl/ssl/videomae_b_sages/encoder_final.pt \
  --dataset sages_clip --split train \
  --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --out $ROOT/cache/videomae_b_adapted/sages/train \
  --batch-size 32 --num-workers 12
