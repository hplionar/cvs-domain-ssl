#!/bin/bash
#SBATCH --job-name=extract_2x2
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The adaptation-by-objective cell. DINOv3 and ViT-MAE are both ViT-B/16 at
# 224 px with a 14x14 grid, adapted on the same 560-video corpus for the same
# 19,675 steps at the same batch size. Only the pretext objective differs.
#
# The corpus excludes the 140 validation and test procedures, so unlike every
# other adapted arm in the project these results are held out rather than
# transductive.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

FAILED=()

run_arm () {
  local arm=$1 encoder=$2 ckpt=$3
  echo; echo "################################ $arm  ($(date +%H:%M:%S))"
  for SPLIT in train val test; do
    python -u scripts/extract_features.py \
      --encoder $encoder \
      --checkpoint $ROOT/outputs/cvs-domain-ssl/ssl/$ckpt/encoder_final.pt \
      --dataset sages --split $SPLIT \
      --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
      --manifest-path metadata/sages_frames_internal_split.csv \
      --out $ROOT/cache/$arm/sages/$SPLIT \
      --reduction none \
      --batch-size 64 --num-workers 12 \
      || { echo "EXTRACTION FAILED: $arm/$SPLIT"; FAILED+=("$arm $SPLIT"); return; }
  done

  python -u train/train_probe_cached.py \
    --train-features $ROOT/cache/$arm/sages/train \
    --val-features   $ROOT/cache/$arm/sages/val \
    --head mean --seeds 3 --in-memory \
    --output-dir $ROOT/outputs/cvs-domain-ssl/probe/${arm}_mean \
    || { echo "PROBE FAILED: $arm"; FAILED+=("$arm probe"); return; }

  python -u eval/score_test_cache.py \
    --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/${arm}_mean \
    --train-features $ROOT/cache/$arm/sages/train \
    --val-features   $ROOT/cache/$arm/sages/val \
    --test-features  $ROOT/cache/$arm/sages/test \
    || { echo "SCORING FAILED: $arm"; FAILED+=("$arm scoring"); }
}

run_arm dinov3_b_trainonly dinov3_b dinov3_b_sages_trainonly
run_arm mae_b_trainonly    mae_b    mae_b_sages_trainonly

echo; echo "################################ summary"
if [ ${#FAILED[@]} -eq 0 ]; then echo "both arms completed"
else printf 'failed: %s\n' "${FAILED[@]}"; fi
