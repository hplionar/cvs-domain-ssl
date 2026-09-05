#!/bin/bash
#SBATCH --job-name=mae_dense_arm
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Does more data from the same videos help? The dense corpus is 252,000 frames
# from the 560 training videos against the sparse corpus's 50,400, at the same
# 19,675-step budget. Corpus size varies and video diversity does not, which is
# the prior question to whether adding institutions would help.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

ARM=mae_b_dense
CKPT=$ROOT/outputs/cvs-domain-ssl/ssl/mae_b_sages_dense/encoder_final.pt

for SPLIT in train val test; do
  python -u scripts/extract_features.py \
    --encoder mae_b --checkpoint $CKPT \
    --dataset sages --split $SPLIT \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache/$ARM/sages/$SPLIT \
    --reduction none --batch-size 64 --num-workers 12
done

python -u scripts/extract_features.py \
  --encoder mae_b --checkpoint $CKPT \
  --dataset sages --split test \
  --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_official_test.csv \
  --out $ROOT/cache/$ARM/sages_official/test \
  --reduction none --batch-size 64 --num-workers 12

python -u train/train_probe_cached.py \
  --train-features $ROOT/cache/$ARM/sages/train \
  --val-features   $ROOT/cache/$ARM/sages/val \
  --head mean --seeds 3 --in-memory \
  --output-dir $ROOT/outputs/cvs-domain-ssl/probe/${ARM}_mean

for spec in "sages/test:test_metrics.json:test_logits_internal" \
            "sages_official/test:test_metrics_official.json:test_logits_official"; do
  cache=${spec%%:*}; rest=${spec#*:}; name=${rest%%:*}; prefix=${rest#*:}
  python -u eval/score_test_cache.py \
    --probe-dir $ROOT/outputs/cvs-domain-ssl/probe/${ARM}_mean \
    --train-features $ROOT/cache/$ARM/sages/train \
    --val-features   $ROOT/cache/$ARM/sages/val \
    --test-features  $ROOT/cache/$ARM/$cache \
    --output-name $name --logits-prefix $prefix
done

echo
echo "compare, official test mAP:  mae_b 0.3517 | mae_b + adapt sparse 0.4000"
