#!/bin/bash
#SBATCH --job-name=mae_timing
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
cd /group/pmc085/hlionar/cvs-domain-ssl
export PYTHONPATH=$PWD

python -u train/pretrain_mae.py \
  --config configs/ssl/mae_b_sages.yaml \
  --override output_dir /tmp/mae_timing \
  --override train.max_steps 60 \
  --override train.warmup_steps 20 \
  --no-resume
