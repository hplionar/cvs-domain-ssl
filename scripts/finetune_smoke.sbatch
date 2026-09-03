#!/bin/bash
#SBATCH --job-name=finetune_smoke
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD
python -u train/finetune_cvs.py \
  --encoder dinov3_b \
  --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --backbone-lr 1e-5 --unfreeze-blocks 4 --epochs 2 --batch-size 32 \
  --output-dir $ROOT/outputs/finetune/smoke
