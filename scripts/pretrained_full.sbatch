#!/bin/bash
#SBATCH --job-name=pretrained_full
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The control for the from-scratch arm. That arm trains all twelve blocks and
# the patch embedding because every parameter is random; the fine-tuning sweep
# trained four blocks and kept the embedding, which is right for a pretrained
# encoder but not comparable. Matching the settings makes the weights the only
# difference between the two.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for LR in 1e-4 3e-4 1e-3; do
  echo; echo "################################ pretrained, all blocks, lr $LR"
  python -u train/finetune_cvs.py \
    --encoder dinov3_b \
    --unfreeze-blocks -1 --unfreeze-embeddings \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --backbone-lr $LR --epochs 40 --batch-size 32 \
    --output-dir $ROOT/outputs/finetune/full_lr$LR \
    || echo "FAILED: lr $LR"
done

echo
echo "from scratch, same settings: 0.6387 at epoch 1, 0.6531 at epoch 2 (lr 1e-4)"
echo "The difference between these two is what pretraining is worth on this task."
