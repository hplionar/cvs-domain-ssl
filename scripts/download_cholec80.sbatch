#!/bin/bash
#SBATCH --job-name=dl_cholec80
#SBATCH --account=pmc085
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# 70 GB single archive. A previous attempt from a login shell died at 3% when
# the session ended; batching it removes that failure mode. -c resumes if this
# job is itself interrupted and requeued.

set -euo pipefail
DEST=/group/pmc085/hlionar/datasets/cholec80
cd "$DEST"

df -h /group/pmc085 | tail -1
wget -c --progress=dot:giga \
  https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip

ls -la cholec80.zip
unzip -o cholec80.zip -d "$DEST"
ls "$DEST/videos" | wc -l
df -h /group/pmc085 | tail -1
