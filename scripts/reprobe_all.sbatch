#!/bin/bash
#SBATCH --job-name=reprobe_all
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Re-run the probe for every arm with head-state saving enabled, then score the
# held-out internal test cache from the saved parameters. dinov3_b is omitted:
# it was done separately and reproduced its recorded configuration exactly.
#
# Arms are not allowed to abort one another. An arm that fails is reported and
# the loop continues, so one bad cache does not cost the other eight.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

FAILED=()

for arm in dinov2_b dinov2_b_adapted mae_b ijepa_h \
           videomae_b_base videomae_b_adapted \
           vjepa2_l_base vjepa2_l_adapted vjepa2_l_lr1e6; do
  case $arm in
    dinov2_b_adapted)   probe=dinov2_b_adapted_sages_mean ;;
    videomae_b_base)    probe=videomae_b_base_mean ;;
    videomae_b_adapted) probe=videomae_b_adapted_mean ;;
    vjepa2_l_base)      probe=vjepa2_l_base_mean ;;
    vjepa2_l_adapted)   probe=vjepa2_l_adapted_mean ;;
    vjepa2_l_lr1e6)     probe=vjepa2_l_lr1e6_mean ;;
    *)                  probe=${arm}_sages_mean ;;
  esac

  echo; echo "################################ $arm  ($(date +%H:%M:%S))"

  python -u train/train_probe_cached.py \
    --train-features $ROOT/cache/$arm/sages/train \
    --val-features   $ROOT/cache/$arm/sages/val \
    --head mean --seeds 3 --in-memory \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/$probe \
    || { echo "PROBE FAILED: $arm"; FAILED+=("$arm probe"); continue; }

  python -u eval/score_test_cache.py \
    --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/$probe \
    --train-features $ROOT/cache/$arm/sages/train \
    --val-features   $ROOT/cache/$arm/sages/val \
    --test-features  $ROOT/cache/$arm/sages/test \
    || { echo "SCORING FAILED: $arm"; FAILED+=("$arm scoring"); }
done

echo; echo "################################ summary"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "all arms completed"
else
  printf 'failed: %s\n' "${FAILED[@]}"
fi
