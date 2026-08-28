# Weeks 3–4

| | |
|---|---|
| **Date** | 12 August 2026 |
| **Period covered** | 6 – 12 August 2026 |
| **Dissertation** | [Overleaf project](https://www.overleaf.com/project/69ae1aa3586d2832c13190e6) |
| **Code** | [github.com/hplionar/cvs-domain-ssl](https://github.com/hplionar/cvs-domain-ssl) |

---

## 1. Status

**GPU access resolved.** The `gpu` partition (17 nodes, V100 16 GB) remained
down throughout, but `ondemand-gpu` carries two V100 32 GB nodes that were idle
and available to `pmc085`. Weeks 1–2 reported the cluster as unavailable; that
was a partition-level error on my part, not an outage.

The consequence is a shorter walltime: `ondemand-gpu` allows 12 hours against
`gpu`'s 3 days. The chained job-array machinery written in weeks 1–2 is
therefore necessary rather than precautionary.

**Both self-supervised arms have now completed continued pretraining, been
extracted to frozen feature caches, and been probed.** RQ1 has a first answer on
SAGES.

---

## 2. The primary result

| Encoder | Objective | Baseline mAP | Adapted mAP | Adaptation gain |
|---|---|---:|---:|---:|
| VideoMAE ViT-B | masked reconstruction | 0.2545 ± 0.0010 | 0.3031 ± 0.0112 | **+0.0487** |
| V-JEPA 2 ViT-L | latent prediction | 0.3356 ± 0.0006 | 0.2085 ± 0.0004 | **−0.1271** |

Validation mAP on the SAGES internal split. Mean-pooling head, 24-configuration
grid × 3 seeds, selection on the seed mean, identical caches and protocol for
both arms.

**Masked reconstruction improved with domain adaptation. Latent prediction
degraded substantially** — a 38% relative drop, two orders of magnitude outside
its seed spread.

Two observations that matter as much as the headline:

**V-JEPA's unadapted baseline is the strongest of the four arms.** At 0.3356 it
exceeds even adapted VideoMAE. The starting representation was better and
continued pretraining destroyed it.

**The adapted VideoMAE features converge far faster.** Best epochs [5, 9, 5]
against the baseline's [26, 20, 11] — independent evidence that the adapted
representation suits the task better, separate from the mAP difference.

### 2.1 This contradicts the hypothesis, and matches independent work

The project predicted that objectives preserving global spatial structure would
adapt *better* to CVS, since the criteria are configurational. The result is the
reverse.

It is, however, consistent with **SurgRec** (Lu et al., 2026), which pretrained
both objectives on 214.5M frames and evaluated across 16 datasets:

| | Baseline | After surgical pretraining | Gain |
|---|---:|---:|---:|
| VideoMAE | 38.12 | 43.55 | **+5.43** |
| V-JEPA | 33.48 | 35.39 | **+1.91** |

Same ordering, from a different group, at 300× the corpus scale. Notably their
16-dataset benchmark contains **no CVS task**, despite Endoscapes-CVS appearing
in their pretraining corpus — so this is the first evidence for that ordering on
CVS specifically.

Reframing accordingly: prior work establishes that masked reconstruction adapts
better than latent prediction on standard surgical benchmarks. This project
predicted the ordering would reverse for CVS because the criteria are
configurational. **It did not.** That is a falsifiable prediction, tested, and
disconfirmed — which is a reportable outcome rather than a failure.

### 2.2 Why the result is not yet final

Three explanations remain, and the current design cannot separate them.

1. **Learning rate.** V-JEPA used 4.7e-6, VideoMAE 1.9e-5, both from the linear
   scaling rule at their respective batch sizes. Principled, but neither was
   verified to sit in a workable range.
2. **Checkpoint provenance.** The only V-JEPA 2 release at 16 frames is
   `vjepa2-vitl-fpc16-256-ssv2`, which carries supervised fine-tuning on
   Something-Something V2. Continued SSL may be disrupting a representation
   shaped by that supervision. VideoMAE's checkpoint is pure SSL.
3. **The objective genuinely transfers poorly to this domain**, which SurgRec
   independently supports.

A short V-JEPA run at a higher learning rate is the cheapest discriminating
test. If adaptation still degrades, (3) gains considerable weight.

---

## 3. Continued pretraining

| | VideoMAE ViT-B | V-JEPA 2 ViT-L |
|---|---|---|
| Checkpoint | `MCG-NJU/videomae-base` | `facebook/vjepa2-vitl-fpc16-256-ssv2` |
| Geometry | 16f @ stride 4, 224 px, 1568 tokens | 16f @ stride 4, 256 px, 2048 tokens |
| Parameters | 86M | 326M |
| Batch / lr | 32 / 1.9e-5 | 8 / 4.7e-6 |
| Steps | 17,500 (20 epochs) | 70,000 (20 epochs) |
| Wall clock | **5.6 h** | ~14.5 h across two jobs |
| Throughput | 27.8 clips/s | 2.5 clips/s |
| Loss | 0.601 → 0.569 | 0.263 → 0.062 |

### 3.1 Choosing the V-JEPA checkpoint

The `fpc64` releases require 64 frames at 256 px — 8,192 tokens against
VideoMAE's 1,568, roughly 90× the compute per sample and no comparability on
temporal extent. Measured on the local RTX 4060: 0.14 it/s at batch 1 against
VideoMAE's 3.25 it/s at batch 4.

`vjepa2-vitl-fpc16-256-ssv2` is the only released variant at 16 frames. It gives
2,048 tokens, close to VideoMAE's geometry and affordable. There is no ViT-B
release at all — the distilled base model discussed in weeks 1–2 does not exist
on HuggingFace.

**Limitation.** The `-ssv2` suffix denotes supervised fine-tuning on
Something-Something V2. Loading confirms the classification head is separable —
every unexpected key is `pooler.*` or `classifier.*`, and the encoder and
predictor load intact at 325.97M parameters — but the encoder weights were
updated during that fine-tuning. The two arms are therefore not matched on
supervision history.

### 3.2 Throughput tuning

VideoMAE was data-bound, not compute-bound:

| batch | workers | clips/s | peak VRAM |
|---:|---:|---:|---:|
| 8 | 4 | 3.5 | 3.49 GiB |
| 16 | 12 | 10.9 | 5.35 GiB |
| **32** | **24** | **18.6** | 9.03 GiB |
| 64 | 24 | 19.8 | 16.36 GiB |

The job script's original 8 CPUs left the GPU idle. Batch 64 gave no further
gain. Sustained throughput over the full run reached 27.8 clips/s as the reader
cache warmed.

For V-JEPA, batch 16 gave identical throughput to batch 8 at 7 GiB more memory —
compute-bound, the opposite regime. Disabling gradient checkpointing gave a 1.3×
speedup for 6 GiB, well within the 32 GiB available.

### 3.3 Verification that adaptation occurred

**VideoMAE** shows a clean depth-graded profile against the original checkpoint:

```
embeddings   0.19614     ← patch projection, largest
layer_00     0.13521
layer_05     0.10024
layer_11     0.07639     ← smallest
```

Monotonically decreasing with depth. That is what domain adaptation looks like
when low-level statistics differ: the patch projection relearns surgical
texture, while later blocks encoding more abstract structure move least.

**V-JEPA** required two measurements, which disagreed instructively. The
per-tensor mean reported 0.00012, flat across all blocks. A global norm ratio
over the same weights gives **0.026** for the encoder and **0.069** for the
predictor. The per-tensor mean buries concentrated change; the global figure is
the more faithful summary, and the tool's averaging is a weakness now recorded.

The predictor moving 2.6× more than the encoder is expected — it is the
task-specific head — but is also consistent with the predictor absorbing much of
the learning.

**Collapse check.** V-JEPA's representation geometry was essentially unchanged:
centred effective rank 45.72 → 45.72, variance ratio 0.950, mean pairwise cosine
0.257 → 0.263. No collapse. The loss fell because the predictor improved, not
because the target degenerated. This matters because a falling JEPA loss alone
cannot distinguish learning from collapse.

---

## 4. Aggregation

### 4.1 Head comparison on frozen DINOv2 features

Three heads, one cache, one grid, three seeds, Endoscapes:

| Head | val mAP | vs mean |
|---|---:|---:|
| mean pooling | 0.7028 ± 0.0025 | — |
| attention MIL | 0.7076 ± 0.0059 | +0.0048 |
| **fusion (global + MIL)** | **0.7126 ± 0.0016** | **+0.0098** |

The ordering matches SMIL's design prediction. The effect sits between two and
six standard deviations depending on which is used — suggestive, not conclusive,
on three seeds and one encoder.

### 4.2 An unresolved anomaly

An earlier fusion-head probe on the VideoMAE SAGES caches reported 0.6127
adapted against 0.5964 baseline. The mean head on **identical caches** gives
0.3031 and 0.2545 — roughly half the absolute mAP.

Those fusion runs were not written to disk and cannot be inspected. Either
aggregation matters enormously on this corpus, or the earlier run differed in
some unrecoverable way. **This must be resolved before either figure is
reported.**

Note also that the adaptation gain differs between the two heads: +0.0163 under
fusion, +0.0487 under mean pooling. If both are real, the choice of head
materially affects the measured gain, which is itself worth reporting.

### 4.3 Why the fusion head is expensive

The mean head **precomputes pooling once**, reducing `[10080, 1568, 768]` to
`[10080, 768]`. The 72-run grid then completes in about 10 minutes.

The fusion and attention heads **cannot precompute**, because their pooling is
learned. Every epoch pushes all 1,568 tokens per sample through the attention
module. Measured: 2 runs × 5 epochs took **507 seconds**, so the full 72-run
grid at up to 100 epochs would be on the order of 100 hours.

Two six-hour jobs timed out before this was understood. The practical
consequence is that fusion-head results need either a reduced grid, fewer
epochs, or a `--reduction spatial` cache giving 8 tokens instead of 1,568.

---

## 5. Feature caches

All extracted with `--reduction none`, so every head reads the same extraction:

| Cache | Samples | Size |
|---|---:|---:|
| `videomae_b_{adapted,base}/sages/train` | 10,080 | 22.61 GiB each |
| `videomae_b_{adapted,base}/sages/val` | 1,260 | 2.83 GiB each |
| `vjepa2_l_{adapted,base}/sages/train` | 10,080 | 39.38 GiB each |
| `vjepa2_l_{adapted,base}/sages/val` | 1,260 | 4.92 GiB each |

Extraction ran at 12–19 samples/s: roughly 9 minutes per VideoMAE split, 14 per
V-JEPA split. Each manifest records the encoder identifier, preprocessing, token
layout and environment, and `verify_protocol` refuses to probe two caches whose
manifests differ.

---

## 6. Defects found and fixed

| # | Defect | Consequence if unfixed |
|---|---|---|
| 1 | `extract_features.py` rejected `--checkpoint` for V-JEPA | The adapted V-JEPA arm had no route into evaluation; in a loop this failed silently and only baseline caches were written |
| 2 | `compare_weights.py` assumed VideoMAE throughout | Against V-JEPA it reported 588 parameters, 0 shared, every key missing — a meaningless comparison that did not error |
| 3 | V-JEPA trainer exits 10 on `USR1`, not 99 | The sbatch chain treated a clean checkpoint-and-exit as failure and cancelled remaining array elements |
| 4 | `CachedFeatures` defaults to a 4 GiB memory budget | A 39 GiB cache is memory-mapped and read sample-by-sample from GPFS with `num_workers=0`; `--in-memory` is required |
| 5 | Job scripts buffered stdout | Five hours of running produced a 25-byte log, with no way to distinguish progress from a hang. `python -u` now used throughout |
| 6 | Probe results not persisted before terminal closed | Two sets of results lost; all runs now go through `sbatch` |

Defects 1 and 2 are the same class as those found in review during weeks 1–2:
an architecture-specific code path silently doing the wrong thing for the other
arm. This is now the third instance, and it argues for testing every new tool
against both encoders rather than the one it was written for.

---

## 7. Next

1. **Resolve the fusion-head anomaly** (§4.2). Rerun fusion on the VideoMAE
   SAGES caches with a reduced grid and establish which absolute figure is
   correct. This blocks reporting either number.
2. **Discriminate the V-JEPA explanations** (§2.2). A short run at a higher
   learning rate separates the tuning hypothesis from the objective hypothesis.
3. **Move to the SAGES official test split.** All figures above are validation
   mAP with configuration selected on the same split. The 300-video official
   test set is labelled and public, gives four times the test data, and makes
   results comparable to the challenge paper.
4. **OOD generalisation to Endoscapes**, once the SAGES result is settled.
   Encoders adapted on SAGES, probe trained and evaluated on Endoscapes — with
   the caveat that Endoscapes ships at 1 fps, so the transfer conflates corpus
   change with temporal-extent change.
5. **Dissertation writing**, targeting Chapter 2.
