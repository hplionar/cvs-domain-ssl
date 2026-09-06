#!/bin/bash
#SBATCH --job-name=from_scratch
#SBATCH --account=pmc085
#SBATCH --partition=ondemand-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.out
#SBATCH --error=/group/pmc085/hlionar/outputs/cvs-domain-ssl/logs/%x_%j.err

# The control that says what pretraining is worth. Kornblith et al. find
# fine-tuning beats random initialisation by 0.6 points on Stanford Cars and 0.2
# on FGVC Aircraft -- fine-grained tasks whose concepts ImageNet does not
# contain -- while buying a seventeenfold convergence speedup. CVS is further
# from ImageNet than either, so the same may hold, and no result in this project
# would detect it.
#
# The architecture, preprocessing and token layout are identical to the
# pretrained arm; only the weights differ. Every block is unfrozen, since
# freezing randomly initialised blocks would measure a random projection.

set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /group/pmc085/hlionar/conda_envs/cvsssl
ROOT=/group/pmc085/hlionar
cd $ROOT/cvs-domain-ssl
export PYTHONPATH=$PWD

for LR in 1e-4 3e-4 1e-3; do
  echo; echo "################################ from scratch, lr $LR"
  python -u train/finetune_cvs.py \
    --encoder dinov3_b --from-scratch \
    --dataset-root $ROOT/datasets/SAGES_CVS_Challenge_2024 \
    --manifest-path metadata/sages_frames_internal_split.csv \
    --backbone-lr $LR --epochs 40 --batch-size 32 \
    --output-dir $ROOT/outputs/finetune/scratch_lr$LR \
    || echo "FAILED: lr $LR"
done

echo
echo "pretrained DINOv3, validation mAP: frozen 0.4584 | fine-tuned 0.6246 at 3e-4"
echo "If from scratch approaches 0.62, pretraining buys convergence rather than"
echo "final accuracy, which is what Kornblith et al. find on fine-grained tasks."
