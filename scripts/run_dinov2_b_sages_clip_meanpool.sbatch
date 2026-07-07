#!/bin/bash
#SBATCH --job-name=dinov2_b_sages_clip
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

python -m train.train_cvs_clip_meanpool \
  --dataset sages \
  --dataset-root /group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --output-dir /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp005_dinov2_b_sages_clip_meanpool \
  --encoder dinov2 \
  --variant base \
  --clip-length 5 \
  --stride 1 \
  --target-position end \
  --epochs 5 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1

echo "Job finished at: $(date)"
