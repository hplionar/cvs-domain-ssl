#!/bin/bash
#SBATCH --job-name=dino_sweep
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Finds the batch ceiling for DINO self-distillation before the trainer is
# written, so the design can account for it rather than be retrofitted.
#
# The measurement matters because DINO's centring subtracts an EMA of the batch
# mean from the teacher output, and at small batch that estimate is noisy --
# which is the mechanism behind collapse to a constant. The published recipe
# uses batch 1024 across many GPUs; a 32 GiB V100 will not come close, and the
# question is how far short.
#
# Collapse does not announce itself. In the V-JEPA arm the loss fell 76% while
# downstream mAP dropped 0.127, with nothing in the log to suggest a problem.
# Twelve hours can be spent producing a degenerate representation.
#
# Runs two configurations: the full 8 local crops, then 4. If the full recipe
# only fits at batch 8, halving the crops roughly doubles the affordable batch,
# and knowing that now is cheaper than discovering it after a failed run.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

OUT=$ROOT/outputs/cvs-domain-ssl/diagnostics
mkdir -p "$OUT"

echo "=============================================================="
echo "job      ${SLURM_JOB_ID}   node $(hostname)   started $(date -Is)"
echo "commit   $(git rev-parse --short HEAD)"
echo "=============================================================="

echo
echo "=== full recipe: 2 global + 8 local ==="
python -u scripts/dino_memory_sweep.py \
  --checkpoint facebook/dinov2-base \
  --batch-sizes 8 16 32 64 \
  --local-crops 8 \
  --out "$OUT/dino_sweep_8crops.json"

echo
echo "=== reduced: 2 global + 4 local ==="
python -u scripts/dino_memory_sweep.py \
  --checkpoint facebook/dinov2-base \
  --batch-sizes 8 16 32 64 128 \
  --local-crops 4 \
  --out "$OUT/dino_sweep_4crops.json"

echo
echo "=============================================================="
echo "finished $(date -Is)"
echo "=============================================================="
