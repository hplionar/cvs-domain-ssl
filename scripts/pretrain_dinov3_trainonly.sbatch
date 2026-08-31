#!/bin/bash
#SBATCH --job-name=dinov3_trainonly
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=1-2%1
#SBATCH --signal=B:USR1@600
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%A_%a.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%A_%a.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl

REPO=/group/pmc085/hlionar/cvs-domain-ssl
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

echo "job ${SLURM_JOB_ID} node $(hostname) started $(date -Is)"
echo "commit $(git rev-parse --short HEAD)"

trap 'echo "[sbatch] USR1 at $(date -Is)"; kill -USR1 "$PID" 2>/dev/null || true' USR1

python -u train/pretrain_dino.py --config configs/ssl/dinov3_b_sages_trainonly.yaml &
PID=$!
set +e
wait "$PID"
STATUS=$?
set -e

echo "trainer exited ${STATUS} at $(date -Is)"
case "$STATUS" in
  0)  echo "Complete." ; scancel "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true ;;
  99) echo "Walltime; next element resumes." ;;
  *)  echo "Failed; check the .err log." ; exit "$STATUS" ;;
esac
