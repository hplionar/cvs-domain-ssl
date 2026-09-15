#!/bin/bash
#SBATCH --job-name=temporal_rescore
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The temporal probe wrote metrics and discarded its predictions. The
# across-encoder resolution on this split is 0.042 mAP and the temporal gain is
# 0.018, which reads as below resolution. But 0.042 is the median half-width
# between *different* encoders, and the temporal comparison is one encoder
# against itself on the same cached features: a paired bootstrap over videos
# cancels the shared variance, and only the predictions support it.

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
    --head lstm --windows 1 3 5 9 18 \
    --logits-prefix test_logits_official \
    --output-dir $ROOT/outputs/temporal_probe/$arm \
    || echo "FAILED: $arm"
done

echo
echo "previous run, recurrent head: dinov2_b k=1 0.4096, k=3 0.4278"
echo "across-encoder resolution at 300 videos: 0.042"
