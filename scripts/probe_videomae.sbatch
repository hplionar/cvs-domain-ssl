#!/bin/bash
#SBATCH --job-name=probe_videomae
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for ARM in adapted base; do
  echo "=== vjepa2_l $ARM ==="
  python train/train_probe_cached.py \
    --train-features $ROOT/cache/videomae_b_${ARM}/sages/train \
    --val-features   $ROOT/cache/videomae_b_${ARM}/sages/val \
    --head fusion --seeds 3 --in-memory \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/videomae_${ARM}_fusion
done
