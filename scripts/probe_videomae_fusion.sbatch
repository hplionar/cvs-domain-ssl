#!/bin/bash
#SBATCH --job-name=probe_vmae_fusion
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Resolves the anomaly recorded in docs/journal/weeks_3_4.md §4.2.
#
# An earlier fusion-head probe on these caches reported 0.6127 adapted against
# 0.5964 baseline. The mean head on the SAME caches gives 0.3031 and 0.2545 --
# roughly half. Those fusion runs were never written to disk, so neither figure
# can currently be reported.
#
# The grid is deliberately small. The fusion head cannot precompute pooling
# (its weights are learned), so every epoch pushes all 1568 tokens per sample
# through the attention module: measured at ~51 s per run-epoch, against the
# mean head's precomputed 72-run grid in ~10 minutes. The full 24-config grid
# would run to roughly 100 hours.
#
#   1 config x 3 seeds x 15 epochs, ~40 min per arm. hidden_dim pinned to 512
#   ~4 x 3 x 25 x 51 s ~= 4.3 h per arm, 8.5 h for both.
#
# The learning rates bracket the value the mean head selected on these same
# caches (3e-3), so the comparison is not disadvantaged by a grid centred
# somewhere else.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

echo "=============================================================="
echo "job      ${SLURM_JOB_ID}   node $(hostname)   started $(date -Is)"
echo "=============================================================="

for ARM in adapted base; do
  echo
  echo "=== videomae_b ${ARM} : fusion head ==="
  # -u so progress appears in the log rather than sitting in a buffer; two
  # earlier jobs ran five hours and produced a 25-byte log.
  # --in-memory because the default 4 GiB budget memory-maps a 22.6 GiB cache
  # and reads it sample-by-sample from GPFS with num_workers=0.
  python -u train/train_probe_cached.py \
    --train-features $ROOT/cache/videomae_b_${ARM}/sages/train \
    --val-features   $ROOT/cache/videomae_b_${ARM}/sages/val \
    --head fusion --seeds 3 --in-memory \
    --lr 3e-3 --attn-hidden 128 \
    --weight-decay 0.01 \
    --dropout 0.0 \
    --epochs 15 \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/videomae_b_${ARM}_fusion
done

echo
echo "=============================================================="
echo "finished $(date -Is)"
echo
echo "Compare against the mean-head result on the same caches:"
echo "  videomae_b_adapted_mean   0.3031 +/- 0.0112"
echo "  videomae_b_base_mean      0.2545 +/- 0.0010"
echo "=============================================================="
