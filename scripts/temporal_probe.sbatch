#!/bin/bash
#SBATCH --job-name=temporal_probe
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Does temporal context help when the aggregator can use order? Trailing
# windows over frozen mean-pooled features, so k = 1 is the existing frozen
# probe and only the aggregation changes between windows.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for arm in dinov3_b dinov2_b; do
  echo; echo "################################ $arm"
  python -u eval/temporal_probe.py \
    --train-features $ROOT/cache/$arm/sages/train \
    --val-features   $ROOT/cache/$arm/sages/val \
    --test-features  $ROOT/cache/$arm/sages_official/test \
    --manifest       metadata/sages_frames_internal_split.csv \
    --test-manifest  metadata/sages_frames_official_test.csv \
    --windows 1 3 5 9 18 \
    --output-dir $ROOT/outputs/temporal_probe/$arm \
    || echo "FAILED: $arm"
done
