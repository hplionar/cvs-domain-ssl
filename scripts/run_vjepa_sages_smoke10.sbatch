#!/bin/bash
#SBATCH --job-name=vjepa_smoke10
#SBATCH --account=pmc085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

source ~/.bashrc
conda activate /group/pmc085/hlionar/conda_envs/vmamba

cd /group/pmc085/hlionar/external/jepa

mkdir -p /group/pmc085/hlionar/outputs/cvs-domain-ssl/logs
mkdir -p /group/pmc085/hlionar/outputs/cvs-domain-ssl/vjepa_smoke10

python - <<'PY'
import torch
import decord
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
print("Decord:", decord.__version__)
PY

export PYTHONPATH=/group/pmc085/hlionar/external/jepa:${PYTHONPATH:-}

python -m app.main \
  --fname configs/pretrain/sages/vitl16_sages_smoke10.yaml \
  --devices cuda:0

echo "Job finished at: $(date)"
ls -lh /group/pmc085/hlionar/outputs/cvs-domain-ssl/vjepa_smoke10
