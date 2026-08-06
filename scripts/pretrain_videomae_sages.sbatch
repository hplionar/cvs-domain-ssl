#!/bin/bash
#SBATCH --job-name=vmae_sages
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=1-2%1
#SBATCH --signal=B:USR1@600
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%A_%a.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%A_%a.err

# Continued VideoMAE pretraining on SAGES.
#
# Chained via a job array with %1, so array elements run strictly one at a
# time and each resumes the previous element's checkpoint. This needs no
# special permissions, unlike scontrol requeue, and works on any Slurm site.
# If an element finishes the schedule it exits 0 and cancels the remainder;
# if it stops at walltime it exits 99 and the next element continues.
#
# --signal=B:USR1@600 asks Slurm to signal 10 minutes before the walltime, so
# the trainer saves a checkpoint rather than being killed mid-epoch.

set -euo pipefail

REPO=/group/pmc085/hlionar/cvs-domain-ssl
ENV=/group/pmc085/hlionar/conda_envs/cvsssl
CONFIG=configs/ssl/videomae_b_sages.yaml

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "=============================================================="
echo "job          ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID:-n/a})"
echo "node         $(hostname)"
echo "started      $(date -Is)"
python - <<'PY'
import torch
print(f"torch        {torch.__version__}  cuda {torch.version.cuda}")
print(f"device       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"capability   sm_{cap[0]}{cap[1]}  bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
PY
echo "=============================================================="

# Forward SIGUSR1 to the trainer, which saves and exits cleanly.
trap 'echo "[sbatch] USR1 received"; kill -USR1 "$PID" 2>/dev/null || true' USR1

python train/pretrain_videomae.py --config "$CONFIG" &
PID=$!
wait "$PID" || STATUS=$?
STATUS=${STATUS:-0}

echo "trainer exited with status ${STATUS} at $(date -Is)"

if [ "$STATUS" -eq 0 ]; then
    echo "Schedule complete; cancelling remaining array elements."
    scancel "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true
elif [ "$STATUS" -eq 99 ]; then
    echo "Walltime reached with work remaining; next array element will resume."
else
    echo "Trainer failed (status ${STATUS}); cancelling remaining elements."
    scancel "${SLURM_ARRAY_JOB_ID}" 2>/dev/null || true
    exit "$STATUS"
fi