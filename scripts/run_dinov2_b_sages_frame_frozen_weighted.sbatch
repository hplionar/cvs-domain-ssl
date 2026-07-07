#!/bin/bash
#SBATCH --job-name=dinov2_b_sages_wbce
#SBATCH --account=pmc085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

cd /group/pmc085/hlionar/cvs-domain-ssl

mkdir -p /group/pmc085/hlionar/outputs/cvs-domain-ssl/logs

source ~/.bashrc
conda activate /group/pmc085/hlionar/conda_envs/vmamba

echo "Python: $(which python)"
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
PY

python -m train.train_cvs \
  --dataset sages \
  --dataset-root /group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --output-dir /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp004_dinov2_b_sages_frame_frozen_weighted \
  --encoder dinov2 \
  --variant base \
  --epochs 5 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --use-pos-weight

echo "Job finished at: $(date)"
