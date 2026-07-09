#!/bin/bash
#SBATCH --job-name=eval_endo_001_002
#SBATCH --account=pmc085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

cd /group/pmc085/hlionar/cvs-domain-ssl

mkdir -p /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval
mkdir -p /group/pmc085/hlionar/outputs/cvs-domain-ssl/logs

source ~/.bashrc
conda activate /group/pmc085/hlionar/conda_envs/vmamba

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
PY

echo "Evaluating Exp001: Endoscapes frame normal BCE"
python -m eval.evaluate_checkpoint \
  --dataset endoscapes \
  --dataset-root /group/pmc085/hlionar/datasets/endoscapes \
  --manifest-path metadata/endoscapes_frames.csv \
  --checkpoint-path /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp001_dinov2_b_frame_frozen/best.pt \
  --output-json /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/exp001_endoscapes_frame_test.json \
  --split test \
  --encoder dinov2 \
  --variant base \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda

echo "Evaluating Exp002: Endoscapes frame weighted BCE"
python -m eval.evaluate_checkpoint \
  --dataset endoscapes \
  --dataset-root /group/pmc085/hlionar/datasets/endoscapes \
  --manifest-path metadata/endoscapes_frames.csv \
  --checkpoint-path /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp002_dinov2_b_frame_frozen_weighted/best.pt \
  --output-json /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/exp002_endoscapes_frame_weighted_test.json \
  --split test \
  --encoder dinov2 \
  --variant base \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda

echo "All Endoscapes test evaluations completed at: $(date)"
ls -lh /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/exp00*_endoscapes*.json
