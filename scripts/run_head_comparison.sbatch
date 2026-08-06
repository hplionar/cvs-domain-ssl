#!/bin/bash
#SBATCH --job-name=head_comparison
#SBATCH --account=pmc085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# E5 extended: does aggregation matter, and does two-branch fusion beat plain
# attention MIL on identical frozen features?
#
# Three arms, one cache, one grid, three seeds, selection on the seed mean:
#
#   mean       MeanPoolHead        uniform average over patch tokens
#   attentive  AttentivePoolHead   gated attention MIL over patch tokens
#   fusion     FusionHead          [CLS] concatenated with the same MIL branch
#
# The fusion head follows the Methods section of SMIL (Wang et al., 2026,
# IJCARS). This is NOT a reproduction of SMIL. Their encoder is DINO
# self-distillation initialised from Endo-FM; nothing here uses that checkpoint,
# so their 70.66 mAP is not comparable with anything this job produces. The only
# question asked here is the relative ordering of the three heads.
#
# NO NEW DEPENDENCIES. Everything this job runs imports only torch, numpy,
# scikit-learn and the standard library, all already in the cvsssl environment.
# Nothing needs installing and requirements.txt is unchanged.
#
# ENCODER CHOICE. The fusion head's global branch needs a [CLS] token, so the
# encoder must emit prefix tokens and the cache must have been extracted with
# --reduction none. In this repository that means dinov3_s/dinov3_b or
# mae_b/mae_l; VideoMAE and V-JEPA 2 return prefix=None. There is no DINOv2
# entry in the BaseEncoder registry -- models/encoders/dinov2_encoder.py is a
# legacy wrapper returning a pooled [B, D] vector that cannot feed any head in
# models/heads/.
#
# dinov3_b is the default rather than mae_b because SMIL's h_ctx is a
# self-distilled [CLS] token. ViT-MAE is trained with no CLS objective, and the
# standard MAE linear-probe protocol pools patch tokens precisely because that
# token carries little on its own. Testing a fusion design on an encoder whose
# global token is near-empty answers a different question. Run mae_b afterwards
# to find out how much of the result depends on that.
#
# Usage:
#   sbatch scripts/run_head_comparison.sbatch
#   ENCODER=mae_b sbatch scripts/run_head_comparison.sbatch
#
#   # check the caches are usable without queueing a GPU job (login node, fast)
#   PREFLIGHT_ONLY=1 bash scripts/run_head_comparison.sbatch

set -euo pipefail

REPO="${REPO:-/group/pmc085/hlionar/cvs-domain-ssl}"
ENV="${ENV:-/group/pmc085/hlionar/conda_envs/cvsssl}"

ENCODER="${ENCODER:-dinov3_b}"
DATASET="${DATASET:-endoscapes}"
CACHE_ROOT="${CACHE_ROOT:-/group/pmc085/hlionar/cache}"
OUT_ROOT="${OUT_ROOT:-/group/pmc085/hlionar/outputs/cvs-domain-ssl/head_comparison}"
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-100}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

TRAIN_CACHE="${CACHE_ROOT}/${ENCODER}/${DATASET}/train"
VAL_CACHE="${CACHE_ROOT}/${ENCODER}/${DATASET}/val"
RUN_DIR="${OUT_ROOT}/${ENCODER}_${DATASET}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p /group/pmc085/hlionar/outputs/cvs-domain-ssl/logs "${RUN_DIR}"

echo "=============================================================="
echo "job          ${SLURM_JOB_ID:-preflight}"
echo "node         $(hostname)"
echo "started      $(date -Is)"
echo "env          ${ENV}"
echo "encoder      ${ENCODER}  dataset ${DATASET}"
echo "train cache  ${TRAIN_CACHE}"
echo "val cache    ${VAL_CACHE}"
echo "output       ${RUN_DIR}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Preflight. Failing here costs seconds; failing after three grid searches
# costs the job. Worse, a fusion arm that silently ran on mean-pooled patches
# instead of [CLS] would not fail at all -- it would answer a different
# question and look healthy doing it.
# ---------------------------------------------------------------------------
for SPLIT_CACHE in "${TRAIN_CACHE}" "${VAL_CACHE}"; do
  if [[ ! -f "${SPLIT_CACHE}/manifest.json" ]]; then
    echo "ERROR: no cache at ${SPLIT_CACHE}." >&2
    echo "Extract it first with scripts/extract_features.py --reduction none." >&2
    exit 1
  fi
  if [[ ! -f "${SPLIT_CACHE}/prefix.npy" ]]; then
    echo "ERROR: ${SPLIT_CACHE} has no prefix.npy, so the fusion head has no" >&2
    echo "global branch. Either the encoder emits no [CLS] token, or the cache" >&2
    echo "was extracted with --reduction spatial/full. Re-extract with" >&2
    echo "--reduction none, or run the fusion arm with --global-source" >&2
    echo "patch_mean and say so when reporting the result." >&2
    exit 1
  fi
done

python - "$TRAIN_CACHE" "$VAL_CACHE" <<'PY'
import json, sys
from pathlib import Path

for label, directory in zip(("train", "val"), sys.argv[1:3]):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    layout = manifest["encoder"]["token_layout"]
    shapes = manifest.get("shapes", {})
    print(f"{label:<6} {manifest['num_samples'] if 'num_samples' in manifest else '?':>6} samples"
          f"  grid {layout['grid']}  dim {layout['dim']}"
          f"  prefix {layout['num_prefix_tokens']}"
          f"  reduction {manifest['extraction']['reduction']}")
    if manifest["extraction"]["reduction"] != "none":
        raise SystemExit(
            f"ERROR: {label} cache used reduction "
            f"{manifest['extraction']['reduction']!r}. Spatial attentive pooling "
            f"and the fusion head both need the full grid. Re-extract with "
            f"--reduction none."
        )
    if layout["num_prefix_tokens"] == 0:
        raise SystemExit(
            f"ERROR: {label} cache declares no prefix tokens, so there is no "
            f"[CLS] for the fusion head's global branch."
        )
    if shapes.get("prefix") is None:
        print(f"WARNING: {label} manifest records no prefix shape; the cache "
              f"predates that field. prefix.npy exists, so this is cosmetic.")
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo
  echo "Preflight passed. Caches are usable. Submit with:"
  echo "  ENCODER=${ENCODER} sbatch scripts/run_head_comparison.sbatch"
  exit 0
fi

python - <<'PY'
import torch
print(f"torch        {torch.__version__}  cuda {torch.version.cuda}")
print(f"device       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PY

# ---------------------------------------------------------------------------
# Three arms. Identical cache, identical lr/weight_decay/dropout grid,
# identical seeds. The two attention arms additionally sweep hidden_dim over
# {128, 512} -- 128 is this project's default, 512 is what SMIL states -- so
# neither of them gets an attention-capacity advantage over the other.
# ---------------------------------------------------------------------------
COMMON_ARGS=(
  --train-features "${TRAIN_CACHE}"
  --val-features   "${VAL_CACHE}"
  --seeds "${SEEDS}"
  --epochs "${EPOCHS}"
  --device cuda
)

for HEAD in mean attentive fusion; do
  echo
  echo "=============================================================="
  echo "arm: ${HEAD}"
  echo "=============================================================="
  EXTRA=()
  if [[ "${HEAD}" == "fusion" ]]; then
    # Explicit, not 'auto': the preflight already proved prefix.npy exists, so
    # asking for cls here means a missing global branch aborts the run rather
    # than quietly substituting mean-pooled patches.
    EXTRA=(--global-source cls)
  fi
  python -m train.train_probe_cached \
    "${COMMON_ARGS[@]}" \
    --head "${HEAD}" \
    "${EXTRA[@]}" \
    --output-dir "${RUN_DIR}/${HEAD}"
done

# ---------------------------------------------------------------------------
# Comparison. Refuses to report if the arms differ in anything but the head.
# ---------------------------------------------------------------------------
echo
python scripts/compare_heads.py \
  --run "mean=${RUN_DIR}/mean" \
  --run "attentive=${RUN_DIR}/attentive" \
  --run "fusion=${RUN_DIR}/fusion" \
  --reference attentive \
  --output-md   "${RUN_DIR}/head_comparison.md" \
  --output-json "${RUN_DIR}/head_comparison.json"

echo
echo "Wrote ${RUN_DIR}/head_comparison.md"
echo "finished     $(date -Is)"
