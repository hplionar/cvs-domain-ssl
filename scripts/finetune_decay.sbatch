#!/bin/bash
#SBATCH --job-name=finetune_decay
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Two symptoms point at one omission. Unfreezing all twelve blocks scored 0.5190
# on the official test split against 0.5650 for four, and five of six Endoscapes
# runs peaked at epoch 1 or 2 of fifteen. Unfreezing more of an encoder should
# not be worse than unfreezing less, and a model peaking in its second epoch has
# not been given a schedule it can use. Both follow from every backbone
# parameter sharing one rate: a rate high enough to adapt block 12 overwrites
# block 3, so restricting to four blocks was a crude substitute for decaying the
# rate by depth.
#
# SELECTION RULE, fixed before the runs: the winner is chosen on validation mAP
# and that one configuration is scored on the official test split. Twelve
# candidates selected on a noisy estimate inflate the winner -- this project
# measured retention at about 0.69 -- so scoring all twelve and reporting the
# best would be the error the dissertation is about.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for ARM in dinov3_b dinov2_b; do
  for LR in 5e-5 1e-4 3e-4; do
    for GAMMA in 0.65 0.75; do
      TAG=${ARM}_lr${LR}_g${GAMMA}
      echo; echo "################################ $TAG"
      python -u train/finetune_cvs.py \
        --encoder $ARM \
        --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
        --manifest-path metadata/sages_frames_internal_split.csv \
        --backbone-lr $LR --unfreeze-blocks -1 \
        --layer-decay $GAMMA --warmup-epochs 1 \
        --epochs 10 --batch-size 32 \
        --output-dir $ROOT/outputs/finetune_decay/$TAG \
        || echo "FAILED: $TAG"
    done
  done
done

echo
echo "Comparators, held out on the official test split:"
echo "  4 blocks, no decay, lr 3e-4    val 0.6246   test 0.5650"
echo "  12 blocks, no decay, lr 1e-4   val 0.6003   test 0.5190"
echo "  frozen DINOv3                  val 0.4584   test 0.4167"
