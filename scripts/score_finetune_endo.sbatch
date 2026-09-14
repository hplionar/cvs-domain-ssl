#!/bin/bash
#SBATCH --job-name=score_finetune_endo
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The trainer reports validation and saves weights; nothing scores them. On
# SAGES that left every fine-tuning figure unquotable until this was run, and
# the shrinkage was 0.060. Endoscapes has 40 test videos against SAGES' 300, so
# the gap here may be larger and is worth knowing before any figure is used.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for d in $ROOT/outputs/finetune_endo/*/; do
  [ -f "$d/best.pt" ] || continue
  echo; echo "################################ $(basename $d)"
  python -u eval/score_finetune.py \
    --checkpoint "$d/best.pt" \
    --dataset-root $ROOT/datasets/endoscapes \
    --manifest-path metadata/endoscapes_frames.csv \
    --save-logits || echo "FAILED: $(basename $d)"
done

echo
echo "frozen Endoscapes test: dinov2_b 0.4613 | dinov3_b 0.4125"
