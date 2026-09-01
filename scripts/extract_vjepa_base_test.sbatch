#!/bin/bash
#SBATCH --job-name=extract_vjepa_base_test
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Re-extraction. The first attempt omitted --model-name, so the registry default
# fpc64-256 was used in place of the fpc16-256-ssv2 checkpoint that produced the
# train and val caches. The substitution changed the token grid from [8,16,16]
# to [32,16,16], and verify_same_encoder refused the cache before it could be
# scored -- the number would otherwise have been for a different model.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

python -u scripts/extract_features.py \
  --encoder vjepa2_l \
  --model-name facebook/vjepa2-vitl-fpc16-256-ssv2 \
  --dataset sages_clip --split test \
  --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --out $ROOT/cache/vjepa2_l_base/sages/test \
  --reduction none \
  --batch-size 16 --num-workers 12

python -u eval/score_test_cache.py \
  --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/vjepa2_l_base_mean \
  --train-features $ROOT/cache/vjepa2_l_base/sages/train \
  --val-features   $ROOT/cache/vjepa2_l_base/sages/val \
  --test-features  $ROOT/cache/vjepa2_l_base/sages/test
