#!/bin/bash
#SBATCH --job-name=eval_sages_003_006
#SBATCH --account=pmc085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
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

echo "Evaluating Exp003: SAGES frame normal BCE"
python -m eval.evaluate_checkpoint \
  --dataset sages \
  --dataset-root /group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --checkpoint-path /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp003_dinov2_b_sages_frame_frozen/best.pt \
  --output-json /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/exp003_sages_frame_test.json \
  --split test \
  --encoder dinov2 \
  --variant base \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda

echo "Evaluating Exp004: SAGES frame weighted BCE"
python -m eval.evaluate_checkpoint \
  --dataset sages \
  --dataset-root /group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --checkpoint-path /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp004_dinov2_b_sages_frame_frozen_weighted/best.pt \
  --output-json /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/exp004_sages_frame_weighted_test.json \
  --split test \
  --encoder dinov2 \
  --variant base \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda

echo "Evaluating Exp005: SAGES clip mean-pool normal BCE"
python -m eval.evaluate_clip_checkpoint \
  --dataset sages \
  --dataset-root /group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --checkpoint-path /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp005_dinov2_b_sages_clip_meanpool/best.pt \
  --output-json /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/exp005_sages_clip_meanpool_test.json \
  --split test \
  --encoder dinov2 \
  --variant base \
  --clip-length 5 \
  --stride 1 \
  --target-position end \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda

echo "Evaluating Exp006: SAGES clip mean-pool weighted BCE"
python -m eval.evaluate_clip_checkpoint \
  --dataset sages \
  --dataset-root /group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_internal_split.csv \
  --checkpoint-path /group/pmc085/hlionar/outputs/cvs-domain-ssl/exp006_dinov2_b_sages_clip_meanpool_weighted/best.pt \
  --output-json /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/exp006_sages_clip_meanpool_weighted_test.json \
  --split test \
  --encoder dinov2 \
  --variant base \
  --clip-length 5 \
  --stride 1 \
  --target-position end \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda

echo "All test evaluations completed at: $(date)"

echo "Generated test evaluation files:"
ls -lh /group/pmc085/hlionar/outputs/cvs-domain-ssl/test_eval/*.json
