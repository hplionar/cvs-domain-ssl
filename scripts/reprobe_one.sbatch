#!/bin/bash
#SBATCH --job-name=reprobe_one
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

python -u train/train_probe_cached.py \
  --train-features $ROOT/cache/dinov3_b/sages/train \
  --val-features   $ROOT/cache/dinov3_b/sages/val \
  --head mean --seeds 3 --in-memory \
  --output-dir $ROOT/outputs/cvs-domain-ssl/probe/dinov3_b_sages_mean

python -u eval/score_test_cache.py \
  --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/dinov3_b_sages_mean \
  --train-features $ROOT/cache/dinov3_b/sages/train \
  --val-features   $ROOT/cache/dinov3_b/sages/val \
  --test-features  $ROOT/cache/dinov3_b/sages/test
