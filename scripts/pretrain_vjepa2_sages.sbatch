#!/bin/bash
#SBATCH --job-name=vjepa_sages
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

# Continued VideoMAE pretraining on SAGES — experiment E2.1.
#
# A single job rather than a chained array. The gpu partition allows 72 hours
# (verified: `sinfo -p gpu` reports 3-00:00:00), and the full 20-epoch schedule
# is an order of magnitude shorter than that, so the array chaining written
# against an assumed 12-hour ceiling is unnecessary complexity. Every resume
# boundary is a place where state can be dropped; having none is safer than
# handling them correctly.
#
# The checkpoint contract still matters, for node failure and preemption rather
# than for walltime. --signal=B:USR1@600 asks Slurm to signal ten minutes before
# the limit so the trainer saves rather than being killed mid-epoch; resubmitting
# the same command then resumes from latest.pt.

set -euo pipefail

REPO=/group/pmc085/hlionar/cvs-domain-ssl
ENV=/group/pmc085/hlionar/conda_envs/cvsssl
CONFIG=configs/ssl/vjepa2_l_sages.yaml

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# GPFS handles concurrent readers well, but eight dataloader workers each
# spawning decoder threads oversubscribes the allocated cores.
export OMP_NUM_THREADS=1

echo "=============================================================="
echo "job          ${SLURM_JOB_ID}"
echo "node         $(hostname)"
echo "started      $(date -Is)"
echo "config       ${CONFIG}"
echo "commit       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
python - <<'PY'
import torch
print(f"torch        {torch.__version__}  cuda {torch.version.cuda}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device       {torch.cuda.get_device_name(0)}  {total:.0f} GiB")
    print(f"capability   sm_{cap[0]}{cap[1]}  bf16={torch.cuda.is_bf16_supported()}")
    # Volta has no bf16; the trainer uses fp16 with GradScaler throughout.
    if cap[0] >= 8:
        print("             note: bf16 available here but NOT on V100; "
              "code must stay fp16 for portability")
else:
    raise SystemExit("No CUDA device visible to the job.")
PY
echo "=============================================================="

# Forward SIGUSR1 to the trainer so it checkpoints and exits cleanly.
trap 'echo "[sbatch] USR1 received at $(date -Is)"; kill -USR1 "$PID" 2>/dev/null || true' USR1

python train/pretrain_vjepa2.py --config "$CONFIG" &
PID=$!

set +e
wait "$PID"
STATUS=$?
set -e

echo "=============================================================="
echo "trainer exited ${STATUS} at $(date -Is)"

case "$STATUS" in
    0)
        echo "Schedule complete. Encoder written to the output directory."
        ;;
    99)
        echo "Stopped with work remaining. Resubmit this script to resume"
        echo "from latest.pt; no arguments change."
        ;;
    *)
        echo "Trainer failed. Check the .err log before resubmitting, since"
        echo "resuming will replay whatever caused the failure."
        exit "$STATUS"
        ;;
esac