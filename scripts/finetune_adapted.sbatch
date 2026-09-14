#!/bin/bash
#SBATCH --job-name=finetune_adapted
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The dissertation is about domain-adaptive pretraining, every adaptation
# result is frozen-probe, and fine-tuning is worth three times what adaptation
# is. Whether adaptation survives when the encoder is allowed to move has never
# been measured, and it is the first thing an examiner asks.
#
# Each arm uses its unadapted parent's winning learning rate, so the only
# difference between an adapted arm and its parent is the initialisation.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD
SSL=$ROOT/outputs/cvs-domain-ssl/ssl

for spec in "dinov3_b:dinov3_b_sages_trainonly:3e-4" \
            "mae_b:mae_b_sages_trainonly:1e-4"; do
  enc=${spec%%:*}; rest=${spec#*:}; run=${rest%%:*}; lr=${rest#*:}
  echo; echo "################################ $run at $lr"
  python -u train/finetune_cvs.py \
    --encoder $enc --checkpoint $SSL/$run/encoder_final.pt \
    --dataset sages \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --backbone-lr $lr --unfreeze-blocks 4 \
    --epochs 15 --batch-size 32 \
    --output-dir $ROOT/outputs/finetune/${run}_ft \
    || echo "FAILED: $run"
done

echo
echo "unadapted parents, official test: dinov3_b 0.5650 | mae_b 0.5095"
echo "frozen adaptation effect: DINOv3 -0.015 | ViT-MAE +0.048"
