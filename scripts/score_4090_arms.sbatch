#!/bin/bash
#SBATCH --job-name=score_4090_arms
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Three checkpoints trained on the rented RTX 4090 (17 Sep 2026), each scored
# once on the official test split. Selection was made before this job:
#   finetune_decay winner on validation (0.6416): dinov3_b_lr3e-4_g0.75
#   hardware control: mae_b_sages_trainonly rerun on the 4090, then fine-tuned
#   Cholec80 corpus arm: mae_b adapted on Cholec80 video01-48, then fine-tuned
# Comparators, official test: mae_b_sages_trainonly_ft 0.546 (+0.037 over
# mae_b), dinov3_b four blocks 0.5650.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for d in $ROOT/outputs/finetune_decay/dinov3_b_lr3e-4_g0.75 \
         $ROOT/outputs/finetune/mae_b_sages_trainonly_4090_ft \
         $ROOT/outputs/finetune/mae_b_cholec80_ft; do
  echo; echo "################################ $(basename $d)"
  python -u eval/score_finetune.py \
    --checkpoint "$d/best.pt" \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --official-manifest metadata/sages_frames_official_test.csv \
    --save-logits || echo "FAILED: $(basename $d)"
done
