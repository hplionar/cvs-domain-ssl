#!/bin/bash
#SBATCH --job-name=depth_probe
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Where in the encoder does CVS information live, and where does adaptation
# change the geometry?
#
# Every result in this project reads the final block. That is convention rather
# than measurement: for tasks defined over spatial arrangement, intermediate
# layers often carry more, and the semantic-diffusion account predicts exactly
# that here -- if global information diffuses into patch tokens with depth,
# earlier layers should retain more within-frame differentiation.
#
# Extracting the base and adapted arms at four depths answers two questions in
# one pass: whether the last layer is the right place to read, and where in the
# stack the adapted arm's geometry departs from the base.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for spec in "dinov3_b:-" "dinov3_b_trainonly:$ROOT/outputs/cvs-domain-ssl/ssl/dinov3_b_sages_trainonly/encoder_final.pt"; do
  arm=${spec%%:*}; ckpt=${spec#*:}
  extra=""; [ "$ckpt" != "-" ] && extra="--checkpoint $ckpt"
  for SPLIT in train val; do
    echo; echo "=== $arm / depths / $SPLIT ==="
    python -u scripts/extract_features.py \
      --encoder dinov3_b $extra \
      --dataset sages --split $SPLIT \
      --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
      --manifest-path metadata/sages_frames_internal_split.csv \
      --out $ROOT/cache_depth/$arm/sages/$SPLIT \
      --reduction none --layer-depths 0.25 0.5 0.75 1.0 \
      --batch-size 32 --num-workers 12 \
      || { echo "FAILED: $arm/$SPLIT"; break; }
  done
done
