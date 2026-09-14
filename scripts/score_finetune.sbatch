#!/bin/bash
#SBATCH --job-name=score_finetune
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Every fine-tuning figure in this project is a validation figure: the trainer
# reports the best validation mAP and saves the weights, and nothing scores them
# afterwards. The regression of held-out on selection-split mAP across fourteen
# frozen arms has slope 0.691, so 0.6246 on validation is consistent with
# roughly 0.55 held out. No fine-tuning number is quotable until this has run.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for d in $ROOT/outputs/finetune/*/; do
  [ -f "$d/best.pt" ] || continue
  echo; echo "################################ $(basename $d)"
  python -u eval/score_finetune.py \
    --checkpoint "$d/best.pt" \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --official-manifest metadata/sages_frames_official_test.csv \
    --save-logits || echo "FAILED: $(basename $d)"
done

echo
echo "frozen probe, official test: dinov2_b 0.4159 | dinov3_b 0.4167 | mae_b 0.3517"
