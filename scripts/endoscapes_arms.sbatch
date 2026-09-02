#!/bin/bash
#SBATCH --job-name=endoscapes_arms
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The reference benchmark. Every published segmentation-free result is on the
# Endoscapes official 120/41/40 video split, giving 6,960 / 2,331 / 1,799
# CVS-annotated frames, so this is where the frozen-probe protocol can be placed
# against the literature.
#
# Batch size differs between arms because the existing train and val caches were
# built at different sizes, and verify_protocol compares the extraction block.
# Each arm is extracted at the size its own siblings used.
#
# The adapted arms were adapted on SAGES and are evaluated here on a different
# institution's data. That is domain transfer, not domain adaptation, and must
# be framed as such: the question is whether adaptation to one surgical corpus
# helps on another, which is a different question from whether it helps in
# domain.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD
ENDO=$ROOT/datasets/endoscapes

FAILED=()

# arm : encoder : batch : splits : checkpoint (or "-")
run () {
  local arm=$1 enc=$2 bs=$3 splits=$4 ckpt=$5
  for SPLIT in $splits; do
    echo; echo "=== $arm / endoscapes / $SPLIT ==="
    local extra=""
    [ "$ckpt" != "-" ] && extra="--checkpoint $ckpt"
    python -u scripts/extract_features.py \
      --encoder $enc $extra \
      --dataset endoscapes --split $SPLIT \
      --dataset-root $ENDO \
      --manifest-path metadata/endoscapes_frames.csv \
      --out $ROOT/cache/$arm/endoscapes/$SPLIT \
      --reduction none \
      --batch-size $bs --num-workers 12 \
      || { echo "EXTRACTION FAILED: $arm/$SPLIT"; FAILED+=("$arm $SPLIT"); return; }
  done

  python -u train/train_probe_cached.py \
    --train-features $ROOT/cache/$arm/endoscapes/train \
    --val-features   $ROOT/cache/$arm/endoscapes/val \
    --head mean --seeds 3 --in-memory \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/${arm}_endoscapes_mean \
    || { echo "PROBE FAILED: $arm"; FAILED+=("$arm probe"); return; }

  python -u eval/score_test_cache.py \
    --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/${arm}_endoscapes_mean \
    --train-features $ROOT/cache/$arm/endoscapes/train \
    --val-features   $ROOT/cache/$arm/endoscapes/val \
    --test-features  $ROOT/cache/$arm/endoscapes/test \
    --output-name test_metrics.json --logits-prefix test_logits \
    || { echo "SCORING FAILED: $arm"; FAILED+=("$arm scoring"); }
}

# Existing caches: only the test split is missing.
run dinov2_b            dinov2_b  32 "test"            -
run dinov3_b            dinov3_b  32 "test"            -
run mae_b               mae_b     64 "test"            -

# New arms: all three splits. Batch matched to the sibling of the same encoder.
run vit_sup_b           vit_sup_b 64 "train val test"  -
run dinov3_b_trainonly  dinov3_b  32 "train val test"  $ROOT/outputs/cvs-domain-ssl/ssl/dinov3_b_sages_trainonly/encoder_final.pt
run mae_b_trainonly     mae_b     64 "train val test"  $ROOT/outputs/cvs-domain-ssl/ssl/mae_b_sages_trainonly/encoder_final.pt

echo; echo "################ summary"
if [ ${#FAILED[@]} -eq 0 ]; then echo "all arms completed"
else printf 'failed: %s\n' "${FAILED[@]}"; fi
echo
echo "published, Endoscapes test mAP:"
echo "  SMIL 0.7066 | SwinCVS Frozen 0.6745 | SV2LSTG 0.6468 | SwinCVS E2E 0.6459"
echo "  ResNet-50 + surgical SSL 0.574 | ResNet-50 0.515"
echo "  own SwinCVS Frozen reproduction, 5 seeds: 0.6553 +/- 0.0058"
