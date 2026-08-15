#!/bin/bash
#SBATCH --job-name=fusion_h128
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The last untested difference between the run that reported 0.6127 / 0.5964 on
# these caches and the four reruns that all landed near 0.30 / 0.26.
#
# WHAT HAS ALREADY BEEN RULED OUT
#
#   Code changed          git diff fda40fc HEAD on attentive_head.py and
#                         train_probe_cached.py is empty; no commits to either
#                         file since 8 August.
#   Checkpoint changed    encoder_final.pt is dated 7 August and has not been
#                         rewritten. Both extractions read the same weights.
#   Bias-repair logic     `inherited_from_adapted_checkpoint` was introduced in
#                         5540cc8, the same commit that added --checkpoint at
#                         all. There was never a version that re-applied the
#                         repair during extraction.
#   Epoch budget          The 15-epoch reruns selected best epochs [3,2,3] and
#                         [1,2,1] -- nowhere near the cap. A 100-epoch rerun
#                         moved the result by 0.004.
#   Weight decay          0.0 versus 0.01 moved the result by 0.004.
#
# WHAT REMAINS
#
# The original used the default grid, which sweeps attn hidden_dim over
# [128, 512] and selects the winner. Every rerun pinned 512. If 128 is
# substantially better on these features, the original would have found it and
# the reruns systematically missed it.
#
# This job pins 128 and holds everything else at the values the reruns used, so
# hidden_dim is the only variable.
#
#   1 config x 3 seeds x 100 epochs, patience 20.
#   The original selected best epochs [12, 8, 15] and [21, 30, 18], so 100 with
#   early stopping should not truncate anything.
#
# READING THE RESULT
#
#   ~0.61  hidden_dim is the explanation. The reruns pinned the worse value and
#          the fusion head is genuinely much stronger than the mean head here.
#   ~0.30  every testable difference has now been tested. The 0.6127 figure
#          came from a run whose conditions cannot be reconstructed, and the
#          reproducible numbers stand.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

echo "=============================================================="
echo "job      ${SLURM_JOB_ID}   node $(hostname)   started $(date -Is)"
echo "commit   $(git rev-parse --short HEAD)"
echo "testing  attn hidden_dim 128 (reruns used 512)"
echo "=============================================================="

for ARM in adapted base; do
  echo
  echo "=== videomae_b ${ARM} : fusion, hidden_dim 128 ==="
  python -u train/train_probe_cached.py \
    --train-features $ROOT/cache/videomae_b_${ARM}/sages/train \
    --val-features   $ROOT/cache/videomae_b_${ARM}/sages/val \
    --head fusion --seeds 3 --in-memory \
    --lr 3e-3 --attn-hidden 128 \
    --weight-decay 0.0 \
    --dropout 0.1 \
    --epochs 100 \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/videomae_b_${ARM}_fusion_h128
done

echo
echo "=============================================================="
echo "finished $(date -Is)"
echo
echo "  target (unreproduced)      0.6127 / 0.5964"
echo "  fusion h512, 15 ep         0.2972 / 0.2621"
echo "  fusion h512, 100 ep wd0    0.3014 / 0.2624"
echo "  mean head                  0.3031 / 0.2545"
echo "=============================================================="
