#!/bin/bash
#SBATCH --job-name=temporal_transformer
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The recurrent probe found a step at three frames and nothing beyond. Whether
# that is a property of the data or of the aggregator is untested: a recurrence
# must pass information through every intervening step, where attention can
# weight a frame five back as heavily as the one immediately preceding.
#
# Both heads reduce to the frozen probe at k = 1, so the comparison is between
# aggregators and not between protocols.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for arm in dinov2_b dinov3_b; do
  echo; echo "################################ $arm"
  python -u eval/temporal_probe.py \
    --train-features $ROOT/cache/$arm/sages/train \
    --val-features   $ROOT/cache/$arm/sages/val \
    --test-features  $ROOT/cache/$arm/sages_official/test \
    --manifest       metadata/sages_frames_internal_split.csv \
    --test-manifest  metadata/sages_frames_official_test.csv \
    --head transformer --windows 1 3 5 9 18 \
    --output-dir $ROOT/outputs/temporal_probe/$arm \
    || echo "FAILED: $arm"
done

echo
echo "recurrent head, official test mAP:"
echo "  dinov2_b  k=1 0.4096  k=3 0.4278  k=5 0.4265  k=9 0.4283  k=18 0.4250"
echo "  dinov3_b  k=1 0.4121  k=3 0.4190  k=5 0.4193  k=9 0.4182  k=18 0.4279"
