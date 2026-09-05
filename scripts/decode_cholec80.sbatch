#!/bin/bash
#SBATCH --job-name=decode_cholec80
#SBATCH --account=pmc085
#SBATCH --partition=work
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Cholec80 at 1 fps. The videos are full procedures of 30 to 60 minutes, so a
# whole-video decode gives roughly 2,000 frames each and 160,000 in total --
# three times the SAGES sparse corpus from a seventh of the videos, which is the
# wrong shape for a size-matched comparison. Frames are therefore subsampled to
# 630 per video after decoding, giving 50,400 to match the SAGES corpus exactly.
#
# No GPU: this is ffmpeg and file I/O.

set -euo pipefail
ROOT=/group/pmc085/hlionar
SRC=$ROOT/datasets/cholec80/videos
OUT=$ROOT/datasets/cholec80/derived_ssl/frames
mkdir -p "$OUT"

n=0
for video in "$SRC"/*.mp4; do
  name=$(basename "$video" .mp4)
  dir="$OUT/$name"
  if [ -f "$dir/.done" ]; then echo "skip $name"; continue; fi
  mkdir -p "$dir"
  ffmpeg -loglevel error -i "$video" -vf "fps=1,scale=-2:256" \
         -q:v 3 "$dir/frame_%05d.jpg" </dev/null
  touch "$dir/.done"
  n=$((n+1))
  echo "$name: $(ls "$dir" | wc -l) frames"
done
echo "decoded $n videos"
du -sh "$OUT"
