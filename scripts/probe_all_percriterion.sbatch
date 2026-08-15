#!/bin/bash
#SBATCH --job-name=probe_percrit
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Regenerates every mean-head probe so results.json carries the per-criterion
# breakdown added to aggregate(). The C1/C2/C3 split is the substantive result:
# all thirteen SAGES Challenge submissions find C2 far easier than C1 and C3,
# and the hypothesis under test predicts exactly that.
#
# Nothing about the training changes; only what is written out.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for ARM in videomae_b_adapted videomae_b_base vjepa2_l_adapted vjepa2_l_base vjepa2_l_lr1e6; do
  echo
  echo "=== ${ARM} ==="
  python -u train/train_probe_cached.py \
    --train-features $ROOT/cache/${ARM}/sages/train \
    --val-features   $ROOT/cache/${ARM}/sages/val \
    --head mean --seeds 3 --in-memory \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/${ARM}_mean
done
