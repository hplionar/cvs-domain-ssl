#!/bin/bash
#SBATCH --job-name=dinov2_trainonly_official
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The ninth arm on the official 300-video test split. Flags copied from the
# arm's own internal-split manifest so verify_protocol accepts the cache.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

python -u scripts/extract_features.py \
  --encoder dinov2_b \
  --checkpoint $ROOT/outputs/cvs-domain-ssl/ssl/dinov2_b_sages_trainonly/encoder_final.pt \
  --dataset sages --split test \
  --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_official_test.csv \
  --out $ROOT/cache/dinov2_b_trainonly/sages_official/test \
  --reduction none \
  --batch-size 64 --num-workers 12

python -u eval/score_test_cache.py \
  --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/dinov2_b_trainonly_mean \
  --train-features $ROOT/cache/dinov2_b_trainonly/sages/train \
  --val-features   $ROOT/cache/dinov2_b_trainonly/sages/val \
  --test-features  $ROOT/cache/dinov2_b_trainonly/sages_official/test \
  --output-name test_metrics_official.json
