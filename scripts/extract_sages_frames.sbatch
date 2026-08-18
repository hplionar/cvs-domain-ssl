#!/bin/bash
#SBATCH --job-name=extract_sages_frames
#SBATCH --account=pmc085
#SBATCH --partition=work
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Builds an SSL corpus of SAGES frames at two densities, so corpus density can
# be varied as a factor rather than fixed by whatever happened to be on disk.
#
# WHY THIS IS NEEDED. `derived/train/frames` currently holds 19 frames per
# procedure -- the 18 labelled ones plus a boundary frame. That is roughly
# 13,300 images for 700 procedures, and every one of them is also in the
# evaluation set. Continued pretraining on that corpus would be pretraining on
# exactly the frames the probe is scored on. No label leakage, since SSL uses no
# labels, but "domain adaptation" would mean something much narrower than
# intended.
#
# The video arms drew 28,000 clips from 1.9M source frames. These densities give
# the image arms a comparable footing:
#
#   sparse   1 fps    90 frames/procedure     63,000 total   ~6 GB
#   dense    5 fps   450 frames/procedure    315,000 total  ~30 GB
#
# A five-fold contrast, both decodable in hours, testing the same question as
# 1-vs-30 fps without 900 GB of JPEGs.
#
# NO GPU. This is video decoding, entirely CPU-bound, so it runs on the `work`
# partition (3-day walltime, 12 nodes) and leaves the two GPU nodes free for
# training. Check `sinfo -p work` before submitting.
#
# The extractor is the challenge's own tools/preprocess_videos.py rather than a
# fourth decoder path written here, so these frames are produced exactly as the
# labelled ones were. It requires --fps to be a multiple of 0.2.
#
# The labelled frames in derived/train/frames are NOT touched: output goes to
# derived_ssl/. Overwriting them would break every existing cache and manifest.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

DATA=/group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024
TOOL=$DATA/tools/preprocess_videos.py

echo "=============================================================="
echo "job      ${SLURM_JOB_ID}   node $(hostname)   started $(date -Is)"
echo "cpus     ${SLURM_CPUS_PER_TASK}"
df -h /group/pmc085 | tail -1
echo "=============================================================="

for SPEC in "1.0 sparse" "5.0 dense"; do
  set -- $SPEC
  FPS=$1
  NAME=$2
  OUT=$DATA/derived_ssl/${NAME}/train/frames

  echo
  echo "=== ${NAME}: ${FPS} fps -> ${OUT} ==="
  mkdir -p "$OUT"

  python -u "$TOOL" \
    --dataset-root "$DATA" \
    --split train \
    --frames-directory "$OUT" \
    --fps "$FPS" \
    --image-extension jpg \
    --jpeg-quality 95 \
    --summary-csv "$DATA/derived_ssl/${NAME}_extraction_summary.csv"

  echo "--- ${NAME} done at $(date -Is) ---"
  echo "videos:  $(ls "$OUT" | wc -l)"
  echo "frames:  $(find "$OUT" -name '*.jpg' | wc -l)"
  du -sh "$OUT"
done

echo
echo "=============================================================="
echo "finished $(date -Is)"
df -h /group/pmc085 | tail -1
echo
echo "Expected: sparse ~63,000 frames, dense ~315,000, over 700 procedures."
echo "A count far from those means videos were skipped -- check the summary CSV"
echo "before building a manifest on top of this."
echo "=============================================================="
