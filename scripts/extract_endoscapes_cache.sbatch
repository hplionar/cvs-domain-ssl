#!/bin/bash
#SBATCH --job-name=extract_endoscapes
#SBATCH --account=pmc085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# Extract frozen encoder features for Endoscapes into the fp16 memmap cache that
# train/train_probe_cached.py reads.
#
# --reduction none is hard-coded, not a parameter. It is what makes
# extract_features.py write prefix.npy, and without prefix.npy the fusion head
# has no global branch. A "spatial" or "full" cache is unusable for this
# comparison and would have to be extracted again from scratch.
#
# Expected size. EndoscapesDataset in supervised mode keeps only
# is_cvs_annotated frames, so this is far smaller than the 58,813 released
# frames. dinov2_b at 224 px is patch 14 -> a 16x16 grid, 256 patch tokens plus
# one [CLS], 768 dim, fp16:
#
#   train   6,960 frames    2.55 GiB tokens + 10.2 MiB prefix
#   val     2,331 frames    0.85 GiB tokens +  3.4 MiB prefix
#   test    1,799 frames    0.66 GiB tokens +  2.6 MiB prefix
#
# dinov3_b is patch 16 -> 196 tokens, about 25% smaller.
#
# WEIGHTS. dinov2_* and mae_* are ungated and need no token. Only dinov3_* is
# gated, and it needs two separate things, both login-node work:
#
#   # 0. In a browser, signed in to the account whose token you will use, open
#   #    https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
#   #    and accept the terms. Authentication alone is not enough: without
#   #    access granted the download fails 403, not 401. Approval is a manual
#   #    review and is not immediate.
#
#   conda activate /group/pmc085/hlionar/conda_envs/cvsssl
#   export HF_HOME=/group/pmc085/hlionar/hf_cache      # not $HOME: quota
#   hf auth login                                      # NOT huggingface-cli
#   hf auth whoami                                     # must print your username
#
# `huggingface-cli login` is a no-op stub from huggingface_hub 1.x onward. It
# prints a deprecation hint, exits zero, and authenticates nothing, so the next
# call fails 401 as though no login had been attempted. Use `hf auth login`.
#
# Export HF_HOME *before* logging in. The token is written under $HF_HOME, so a
# login with the variable unset lands it in ~/.cache/huggingface and this job,
# which sets HF_HOME below, will not find it.
#
# PREFETCH. Kaya compute nodes may have no outbound network, so pull the weights
# on the login node first, whichever encoder you use:
#
#   export HF_HOME=/group/pmc085/hlionar/hf_cache
#   python -c "
#   from transformers import AutoModel
#   AutoModel.from_pretrained('facebook/dinov2-base'); print('cached ok')"
#
# The job sets HF_HUB_OFFLINE=1 so it reads that cache and fails loudly if the
# prefetch did not happen, rather than hanging on a dead socket for the walltime.
#
# Usage:
#   sbatch scripts/extract_endoscapes_cache.sbatch                  # dinov2_b
#   ENCODER=dinov3_b sbatch scripts/extract_endoscapes_cache.sbatch # once approved
#   ENCODER=mae_b sbatch scripts/extract_endoscapes_cache.sbatch
#   SPLITS="train val test" sbatch scripts/extract_endoscapes_cache.sbatch
#   DRY_RUN=1 bash scripts/extract_endoscapes_cache.sbatch   # shapes only, login node

set -euo pipefail

REPO="${REPO:-/group/pmc085/hlionar/cvs-domain-ssl}"
ENV="${ENV:-/group/pmc085/hlionar/conda_envs/cvsssl}"

ENCODER="${ENCODER:-dinov2_b}"
SPLITS="${SPLITS:-train val}"
DATASET_ROOT="${DATASET_ROOT:-/group/pmc085/hlionar/datasets/endoscapes}"
MANIFEST="${MANIFEST:-metadata/endoscapes_frames.csv}"
CACHE_ROOT="${CACHE_ROOT:-/group/pmc085/hlionar/cache}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DRY_RUN="${DRY_RUN:-0}"

export HF_HOME="${HF_HOME:-/group/pmc085/hlionar/hf_cache}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p /group/pmc085/hlionar/outputs/cvs-domain-ssl/logs

echo "=============================================================="
echo "job          ${SLURM_JOB_ID:-dry-run}"
echo "node         $(hostname)"
echo "started      $(date -Is)"
echo "encoder      ${ENCODER}"
echo "splits       ${SPLITS}"
echo "dataset      ${DATASET_ROOT}"
echo "HF_HOME      ${HF_HOME}"
echo "=============================================================="

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: no dataset at ${DATASET_ROOT}." >&2
  exit 1
fi

# Offline only inside the batch job. On a login node (DRY_RUN) leave the network
# available so a first-time download can still succeed.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export HF_HUB_OFFLINE=1
  echo "HF_HUB_OFFLINE=1 -- weights must already be in ${HF_HOME}"
fi

EXTRA=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA=(--dry-run)
fi

for SPLIT in ${SPLITS}; do
  OUT="${CACHE_ROOT}/${ENCODER}/endoscapes/${SPLIT}"
  echo
  echo "=============================================================="
  echo "split ${SPLIT}  ->  ${OUT}"
  echo "=============================================================="
  python scripts/extract_features.py \
    --encoder "${ENCODER}" \
    --dataset endoscapes \
    --split "${SPLIT}" \
    --dataset-root "${DATASET_ROOT}" \
    --manifest-path "${MANIFEST}" \
    --reduction none \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 8 \
    --device cuda \
    --out "${OUT}" \
    "${EXTRA[@]}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry run only, nothing written."
  exit 0
fi

echo
echo "Verifying prefix.npy landed in every split:"
for SPLIT in ${SPLITS}; do
  OUT="${CACHE_ROOT}/${ENCODER}/endoscapes/${SPLIT}"
  if [[ -f "${OUT}/prefix.npy" ]]; then
    echo "  ${SPLIT}: ok  ($(du -h "${OUT}/prefix.npy" | cut -f1))"
  else
    echo "  ${SPLIT}: MISSING prefix.npy -- the fusion arm cannot run" >&2
    exit 1
  fi
done

echo
echo "Next:  PREFLIGHT_ONLY=1 bash scripts/run_head_comparison.sbatch"
echo "finished     $(date -Is)"
