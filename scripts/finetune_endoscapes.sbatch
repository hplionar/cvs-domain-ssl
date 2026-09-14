#!/bin/bash
#SBATCH --job-name=finetune_endo
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Every fine-tuning result so far is SAGES. Endoscapes is where every published
# CVS method reports, so it is the only place this protocol can be compared
# against the literature rather than against a challenge leaderboard.
#
# Frozen here: DINOv2 0.4613, DINOv3 0.4125. Fine-tuning four blocks gained
# 0.148 held out on SAGES. A comparable gain would place this near SwinCVS
# Frozen, 0.6745 published and 0.6553 in this project's reproduction.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for ARM in dinov3_b dinov2_b; do
  for LR in 1e-5 1e-4 3e-4; do
    echo; echo "################################ $ARM  lr $LR"
    python -u train/finetune_cvs.py \
      --encoder $ARM --dataset endoscapes \
      --dataset-root $ROOT/datasets/endoscapes \
      --manifest-path metadata/endoscapes_frames.csv \
      --backbone-lr $LR --unfreeze-blocks 4 \
      --epochs 15 --batch-size 32 \
      --output-dir $ROOT/outputs/finetune_endo/${ARM}_lr$LR \
      || echo "FAILED: $ARM lr $LR"
  done
done

echo
echo "frozen Endoscapes test: dinov2_b 0.4613 | dinov3_b 0.4125"
echo "published: SwinCVS Frozen 0.6745 | own reproduction 0.6553 | SMIL 0.7066"
