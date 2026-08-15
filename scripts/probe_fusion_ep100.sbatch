#!/bin/bash
#SBATCH --job-name=probe_fusion_ep100
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=11:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Attempts to reproduce a fusion-head result that was printed to a terminal and
# never written to disk:
#
#   videomae_adapted_fusion   0.6127 +/- 0.0038   best epochs [12,  8, 15]
#   videomae_base_fusion      0.5964 +/- 0.0018   best epochs [21, 30, 18]
#   both selected lr 3e-3, weight_decay 0.0, dropout 0.1
#
# A rerun on the same caches with --epochs 15 gave 0.2972 and 0.2621, agreeing
# with the mean head to within a seed spread. One of the two is wrong.
#
# Four differences between the runs. This job tests the two most plausible:
#
#   epochs        the original used the default 100 and selected best epochs up
#                 to 30, so a cap of 15 truncates configurations that were still
#                 improving. This is the leading candidate.
#   weight_decay  the original selected 0.0; the rerun pinned 0.01.
#
# The other two differences -- --in-memory, and pinning attn hidden_dim to 512
# rather than sweeping [128, 512] -- are left as they are. Neither should move
# mAP by 0.3, and reproducing the full 24-config sweep would take ~100 hours at
# the fusion head's measured ~51 s per run-epoch.
#
# Cost: 2 configs (dropout 0.0/0.1) x 3 seeds x up to 100 epochs. Early stopping
# on val mAP means most runs will halt well before 100. Budgeted at 11 h.
#
# Writes to *_fusion_ep100 so the 15-epoch results are preserved for comparison.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

echo "=============================================================="
echo "job      ${SLURM_JOB_ID}   node $(hostname)   started $(date -Is)"
echo "testing  epochs 100 (was 15) and weight_decay 0.0 (was 0.01)"
echo "=============================================================="

for ARM in adapted base; do
  echo
  echo "=== videomae_b ${ARM} : fusion, 100 epochs, wd 0.0 ==="
  python -u train/train_probe_cached.py \
    --train-features $ROOT/cache/videomae_b_${ARM}/sages/train \
    --val-features   $ROOT/cache/videomae_b_${ARM}/sages/val \
    --head fusion --seeds 3 --in-memory \
    --lr 3e-3 --attn-hidden 512 \
    --weight-decay 0.0 \
    --dropout 0.0 0.1 \
    --epochs 100 \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/videomae_b_${ARM}_fusion_ep100
done

echo
echo "=============================================================="
echo "finished $(date -Is)"
echo
echo "Compare:"
echo "  target (lost run)          0.6127 / 0.5964"
echo "  fusion, 15 epochs, wd 0.01 0.2972 / 0.2621"
echo "  mean head                  0.3031 / 0.2545"
echo "=============================================================="
