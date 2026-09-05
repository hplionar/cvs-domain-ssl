#!/bin/bash
#SBATCH --job-name=depth9_test
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Block 9 outperformed block 12 by 0.034 mAP for the adapted arm, and the depth
# was chosen on validation. Section 2.6 measures what selection costs on this
# benchmark, so the claim needs a held-out figure before it can be reported.
#
# extract_features.py writes the selected depths to layers.npy and the final
# layer to tokens.npy regardless, and the probe reads tokens.npy. The
# post-processing step below substitutes the selected layer so that the ordinary
# probe and scoring pipeline read the intended block without modification.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

ARM=dinov3_b_trainonly_b9
CKPT=$ROOT/outputs/cvs-domain-ssl/ssl/dinov3_b_sages_trainonly/encoder_final.pt

for SPLIT in train val test; do
  python -u scripts/extract_features.py \
    --encoder dinov3_b --checkpoint "$CKPT" \
    --dataset sages --split "$SPLIT" \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --out $ROOT/cache/$ARM/sages/$SPLIT \
    --reduction none --layer-depths 0.75 \
    --batch-size 32 --num-workers 12
done

python -u scripts/extract_features.py \
  --encoder dinov3_b --checkpoint "$CKPT" \
  --dataset sages --split test \
  --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
  --manifest-path metadata/sages_frames_official_test.csv \
  --out $ROOT/cache/$ARM/sages_official/test \
  --reduction none --layer-depths 0.75 \
  --batch-size 32 --num-workers 12

python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path("/group/pmc085/hlionar/cache/dinov3_b_trainonly_b9")
for cache in sorted(p for p in root.rglob("*") if (p / "layers.npy").is_file()):
    manifest = json.loads((cache / "manifest.json").read_text())
    depths = manifest.get("layer_depths") or []
    if len(depths) != 1:
        raise SystemExit(f"{cache} holds {len(depths)} depths; expected one.")
    layers = np.load(cache / "layers.npy", mmap_mode="r")
    tokens = np.load(cache / "tokens.npy", mmap_mode="r")
    if layers.shape[0] != tokens.shape[0] or layers.shape[2:] != tokens.shape[1:]:
        raise SystemExit(f"{cache}: shapes disagree, {layers.shape} vs {tokens.shape}")
    np.save(cache / "tokens.npy", np.asarray(layers[:, 0]))
    manifest["tokens_source"] = f"block at relative depth {depths[0]}"
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  {cache.name}: tokens.npy now holds depth {depths[0]}")
PY

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
echo "block 12 for comparison, official test: dinov3_b 0.4167 | + adapt 0.4014"
echo "block 9 on validation was 0.4922 against block 12's 0.4582"
