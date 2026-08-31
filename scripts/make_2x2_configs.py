#!/usr/bin/env python3
"""Generate the two train-only continued-pretraining configs for the 2x2.

The adaptation axis currently has no controlled test. Every adapted arm that
exists sits outside the architecture-matched cell: DINOv2 is patch 14, and the
other three are video encoders. So "does adaptation gain depend on the pretext
objective?" cannot be answered from what has been run -- DINOv2-adapted against
VideoMAE-adapted differs in objective, patch size, input level and architecture
at once.

These two configs complete the cell:

                      base            adapted
    DINOv3  ViT-B/16  have            this file
    ViT-MAE ViT-B/16  have            this file

Same architecture, patch size, token count, hidden dimension, input resolution,
corpus, and optimisation-step budget. Only the pretext objective differs,
crossed with adaptation.

Two departures from the existing configs, both deliberate:

**Corpus restricted to the 560 training videos.** The existing SSL corpus covers
all 700, so the internal validation and test videos were present, unlabelled,
during continued pretraining -- confirmed 28 August. That is not label leakage,
but it makes every current adaptation figure transductive. Excluding the 140
val and test videos makes these arms held-out on the internal splits without
waiting for the official test set. The exclusion list is written into the config
in full rather than referenced, so the config records exactly which procedures
were withheld.

**Epochs raised from 20 to 25.** 560 videos at 1 fps is 50,400 frames, 787
steps per epoch at batch 64. Twenty epochs would be 15,750 steps against the
DINOv2 arm's 19,680, and `warmup_steps` and `teacher_temp_warmup_steps` are
absolute counts, so a shorter schedule would also change their proportion of
training. Twenty-five epochs gives 19,675 -- matched to within 5 steps.

Everything else is copied from `dinov2_b_sages.yaml` and `mae_b_sages.yaml`
unchanged, including the decision to leave colour jitter off.

Usage:
    python scripts/make_2x2_configs.py \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --out-dir configs/ssl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FRAMES_DIR = ("/group/pmc085/hlionar/datasets/SAGES_CVS_Challenge_2024"
              "/derived_ssl/sparse/train/frames")
OUTPUT_ROOT = "/group/pmc085/hlionar/outputs/cvs-domain-ssl/ssl"

DINOV3 = """\
# DINOv3 ViT-B/16 continued pretraining, training videos only.
#
# Half of the adaptation-by-objective 2x2. The counterpart is
# mae_b_sages_trainonly.yaml: same architecture, patch size, token count,
# hidden dimension, resolution, corpus and step budget, differing only in the
# pretext objective.
#
# Why DINOv3 rather than DINOv2 for the controlled arm: DINOv3 is patch 16 and
# so is ViT-MAE, giving 14x14 = 196 tokens for both. DINOv2 is patch 14 and
# yields 256, which would confound objective with spatial granularity.
#
# CORPUS. The 560 internal training videos. The 140 validation and test videos
# are excluded by identifier below, so this encoder has never seen the frames it
# is evaluated on. The existing dinov2_b_sages_sparse arm was pretrained on all
# 700 and is therefore transductive; that arm is reported as exploratory.
#
# SCHEDULE. 25 epochs rather than 20: 50,400 frames at batch 64 is 787 steps per
# epoch, so 25 epochs is 19,675 steps, matching the 19,680 of the DINOv2 arm.
# warmup_steps and teacher_temp_warmup_steps are absolute, so holding the total
# fixed also holds their proportions fixed.
#
# Everything below other than the checkpoint, epochs, output_dir and exclusion
# list is copied unchanged from dinov2_b_sages.yaml.

seed: 0
output_dir: {output_root}/dinov3_b_sages_trainonly

model:
  checkpoint: facebook/dinov3-vitb16-pretrain-lvd1689m
  gradient_checkpointing: false
  head:
    out_dim: 65536
    hidden_dim: 2048
    bottleneck_dim: 256
    num_layers: 3

data:
  frames_dir: {frames_dir}
  global_size: 224
  local_size: 96
  num_local: 8
  global_scale: [0.4, 1.0]
  local_scale: [0.05, 0.4]
  colour_jitter: false
  frames_per_video: null
  exclude_video_ids:
{exclusions}

train:
  batch_size: 64
  epochs: 25
  lr: 1.25e-4
  min_lr: 1.0e-6
  weight_decay: 0.04
  weight_decay_end: 0.4
  betas: [0.9, 0.999]
  grad_clip: 3.0
  warmup_steps: 1000
  teacher_temp_start: 0.04
  teacher_temp_end: 0.07
  teacher_temp_warmup_steps: 6000
  student_temp: 0.1
  centre_momentum: 0.9
  ema_start: 0.996
  ema_end: 1.0
  num_workers: 12
  prefetch_factor: 2
  save_every_steps: 200
  log_every_steps: 20
  collapse_every_steps: 500
"""

MAE = """\
# ViT-MAE ViT-B/16 continued pretraining, training videos only.
#
# Half of the adaptation-by-objective 2x2. The counterpart is
# dinov3_b_sages_trainonly.yaml: same architecture, patch size, token count,
# hidden dimension, resolution, corpus and step budget, differing only in the
# pretext objective.
#
# CORPUS. The 560 internal training videos; the 140 validation and test videos
# are excluded by identifier below. See the counterpart config for why.
#
# SCHEDULE. 25 epochs rather than 20, giving 19,675 steps at 787 steps per
# epoch, matched to the DINOv3 arm and to the earlier DINOv2 arm.
#
# Everything below other than epochs, output_dir and the exclusion list is
# copied unchanged from mae_b_sages.yaml, including mask ratio 0.75 and the
# decision to leave the stronger augmentation recipe off.

seed: 0
output_dir: {output_root}/mae_b_sages_trainonly

model:
  checkpoint: facebook/vit-mae-base
  mask_ratio: 0.75
  gradient_checkpointing: false

data:
  frames_dir: {frames_dir}
  image_size: 224
  augment: true
  strong_augment: false
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  interpolation: bicubic
  frames_per_video: null
  exclude_video_ids:
{exclusions}

train:
  batch_size: 64
  epochs: 25
  lr: 3.75e-5
  min_lr_ratio: 0.01
  weight_decay: 0.05
  betas: [0.9, 0.95]
  grad_clip: 1.0
  warmup_steps: 1000
  num_workers: 12
  prefetch_factor: 4
  save_every_steps: 200
  log_every_steps: 20
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="metadata/sages_frames_internal_split.csv")
    p.add_argument("--out-dir", default="configs/ssl")
    p.add_argument("--frames-dir", default=FRAMES_DIR)
    p.add_argument("--output-root", default=OUTPUT_ROOT)
    args = p.parse_args()

    d = pd.read_csv(args.manifest)
    held_out = sorted(d[d.internal_split.isin(["val", "test"])].video_id.unique())
    kept = sorted(d[d.internal_split == "train"].video_id.unique())
    if set(held_out) & set(kept):
        raise SystemExit("Split overlap: a video appears in both train and val/test.")

    frames = Path(args.frames_dir)
    if frames.is_dir():
        present = {p.name for p in frames.iterdir() if p.is_dir()}
        missing = [v for v in held_out if v not in present]
        print(f"corpus directory      {len(present)} video directories")
        print(f"excluded              {len(held_out)}")
        print(f"remaining             {len(present - set(held_out))}")
        if missing:
            print(f"WARNING: {len(missing)} excluded ids are not in the corpus, "
                  f"e.g. {missing[:3]} -- the exclusion may be a no-op")
    else:
        print(f"corpus directory not visible from here: {frames}")

    exclusions = "\n".join(f"    - {v}" for v in held_out)
    fields = {"exclusions": exclusions, "frames_dir": args.frames_dir,
              "output_root": args.output_root}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, template in (("dinov3_b_sages_trainonly.yaml", DINOV3),
                           ("mae_b_sages_trainonly.yaml", MAE)):
        path = out_dir / name
        if path.exists():
            print(f"exists, not overwritten: {path}")
            continue
        path.write_text(template.format(**fields), encoding="utf-8")
        print(f"written {path}")

    n_frames = 560 * 90  # 1 fps over 90-second clips
    steps = (n_frames // 64) * 25
    print(f"\nexpected corpus  ~{n_frames:,} frames over {len(kept)} videos")
    print(f"expected steps   {n_frames // 64} per epoch x 25 = {steps:,}")
    print(f"                 (dinov2_b_sages_sparse ran 19,680)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
