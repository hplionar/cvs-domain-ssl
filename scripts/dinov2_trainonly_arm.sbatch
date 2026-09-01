#!/bin/bash
#SBATCH --job-name=dinov2_trainonly_arm
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The arm that removes the corpus confound from the token-geometry finding.
#
# dinov2_b_adapted lost 36% of its within-frame effective rank in 99% of frames,
# while dinov3_b_trainonly and mae_b_trainonly did not. Four things separate
# them: corpus (700 videos against 560), patch size (14 against 16), register
# tokens (none against four), and base recipe. This arm holds the corpus fixed
# at 560 and leaves three.
#
# Both outcomes are informative. If the collapse recurs on the clean corpus it
# is architectural -- patch 14 or the absence of registers -- and the
# transductive corpus was not the cause. If it does not recur, the 700-video
# corpus was implicated, which is a different and smaller finding, since corpus
# overlap should not by itself change token geometry.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

ARM=dinov2_b_trainonly
CKPT=$ROOT/outputs/cvs-domain-ssl/ssl/dinov2_b_sages_trainonly/encoder_final.pt

if [ ! -f "$CKPT" ]; then
  echo "No checkpoint at $CKPT. The pretraining run has not completed."
  exit 1
fi

for SPLIT in train val test; do
  echo; echo "=== $ARM / sages / $SPLIT ==="
  python -u scripts/extract_features.py \
    --encoder dinov2_b \
    --checkpoint "$CKPT" \
    --dataset sages --split "$SPLIT" \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache/$ARM/sages/$SPLIT \
    --reduction none \
    --batch-size 64 --num-workers 12 \
    || { echo "EXTRACTION FAILED: $SPLIT"; exit 1; }
done

echo; echo "=== probe ==="
python -u train/train_probe_cached.py \
  --train-features $ROOT/cache/$ARM/sages/train \
  --val-features   $ROOT/cache/$ARM/sages/val \
  --head mean --seeds 3 --in-memory \
  --output-dir $ROOT/outputs/cvs-domain-ssl/probe/${ARM}_mean

echo; echo "=== held-out score ==="
python -u eval/score_test_cache.py \
  --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/${ARM}_mean \
  --train-features $ROOT/cache/$ARM/sages/train \
  --val-features   $ROOT/cache/$ARM/sages/val \
  --test-features  $ROOT/cache/$ARM/sages/test

echo; echo "=== token geometry against the unadapted base ==="
python -u eval/patch_similarity.py \
  --arm base=$ROOT/cache/dinov2_b/sages/val \
  --arm adapted=$ROOT/cache/$ARM/sages/val \
  --manifest metadata/sages_frames_internal_split.csv \
  --output-dir $ROOT/outputs/patch_similarity_dinov2_trainonly

echo; echo "=== spatial visualisation ==="
python -u eval/visualise_tokens.py \
  --base    $ROOT/cache/dinov2_b/sages/val \
  --adapted $ROOT/cache/$ARM/sages/val \
  --manifest metadata/sages_frames_internal_split.csv \
  --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
  --n-frames 6 --label dinov2_base dinov2_trainonly \
  --output-dir $ROOT/outputs/token_visualisation

echo
echo "compare, held-out test mAP:  dinov2_b 0.4508 | dinov2_b_adapted 0.4395"
echo "compare, within-frame cosine change:  700-video corpus +0.1357, rank -11.28"
echo "  A recurrence on the 560-video corpus implicates the architecture;"
echo "  its absence implicates the corpus."
