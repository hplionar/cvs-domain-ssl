#!/bin/bash
#SBATCH --job-name=extract_depths
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Caches four layers per encoder at relative depths 0.25, 0.5, 0.75, 1.0, so a
# head can learn which depths carry CVS-relevant information -- potentially
# different depths for different criteria.
#
# WHY RELATIVE DEPTHS. VideoMAE ViT-B has 12 blocks and V-JEPA 2 ViT-L has 24,
# so 0.25 resolves to block 3 in one and block 6 in the other. Absolute indices
# would make "layer 6" mean a quarter of one network and half of the other.
#
# WHY THIS EXPERIMENT. Every SAGES Challenge submission finds C2 far easier than
# C1 and C3 -- Farm scores 85.30 against 56.65 and 65.32, and the ordering holds
# across all thirteen. C2 (hepatocystic triangle cleared) is a judgement about
# tissue absence and texture; C1 (counting two structures) and C3 (a spatial
# relation to the cystic plate) are configurational. If those are resolved at
# different levels of abstraction, a last-layer probe discards the evidence for
# at least one of them.
#
# The embedding output is deliberately unreachable: it precedes every
# transformer block and is close to a linear projection of pixels, so a
# "shallow layers do not help" conclusion should not be able to rest on it.
#
# DISK. tokens.npy is still written and still holds the final layer, so each of
# these is a strict superset of the existing cache rather than a replacement.
#
#   VideoMAE  1568 tokens x  768 dim   ~113 GiB train,  14 GiB val
#   V-JEPA    2048 tokens x 1024 dim   ~197 GiB train,  25 GiB val
#   Total across four arms and both splits: roughly 700 GiB.
#
# Free space was 1.2 PB at the time of writing; check before submitting.
#
# TIME. Extraction ran at 12-19 samples/s previously, and hidden states add
# little compute -- the layers are already produced by the forward pass and only
# the copy to host is extra. Budget is set from the slower V-JEPA rate with
# headroom.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

DATA=$ROOT/datasets/SAGES_CVS_Challenge_2024
MANIFEST=metadata/sages_frames_internal_split.csv
DEPTHS="0.25 0.5 0.75 1.0"

echo "=============================================================="
echo "job      ${SLURM_JOB_ID}   node $(hostname)   started $(date -Is)"
echo "commit   $(git rev-parse --short HEAD)"
echo "depths   ${DEPTHS}"
df -h /group/pmc085 | tail -1
echo "=============================================================="

extract () {
  local name=$1 encoder=$2 split=$3
  shift 3
  echo
  echo "=== ${name} / ${split} ==="
  # val first within each arm, so a misconfiguration surfaces in two minutes
  # rather than after an hour of writing the train split.
  python -u scripts/extract_features.py \
    --encoder "$encoder" \
    --dataset sages_clip --split "$split" \
    --dataset-root "$DATA" --manifest-path "$MANIFEST" \
    --out "$ROOT/cache_depth/${name}/sages/${split}" \
    --layer-depths $DEPTHS \
    --batch-size 16 --num-workers 12 \
    "$@"
}

VMAE_ADAPTED=$ROOT/outputs/cvs-domain-ssl/ssl/videomae_b_sages/encoder_final.pt
VJEPA_BASE=facebook/vjepa2-vitl-fpc16-256-ssv2
VJEPA_ADAPTED=$ROOT/outputs/cvs-domain-ssl/ssl/vjepa2_l_sages/encoder_final.pt

for SPLIT in val train; do
  extract videomae_b_base    videomae_b "$SPLIT"
  extract videomae_b_adapted videomae_b "$SPLIT" --checkpoint "$VMAE_ADAPTED"
  # extract vjepa2_l_base      vjepa2_l   "$SPLIT" --model-name "$VJEPA_BASE"
  # extract vjepa2_l_adapted   vjepa2_l   "$SPLIT" --model-name "$VJEPA_BASE" \
  #                                               --checkpoint "$VJEPA_ADAPTED"
done

echo
echo "=============================================================="
echo "finished $(date -Is)"
df -h /group/pmc085 | tail -1
echo "=============================================================="
