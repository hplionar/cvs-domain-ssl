# Weeks 1–2

| | |
|---|---|
| **Date** | 6 August 2026 |
| **Period covered** | 23 July – 5 August 2026 |
| **Dissertation** | [Overleaf project](https://www.overleaf.com/project/69ae1aa3586d2832c13190e6) |
| **Code** | [github.com/hplionar/cvs-domain-ssl](https://github.com/hplionar/cvs-domain-ssl) |

---

## 1. Status

**Kaya GPU partition is still down as of 5 August 2026, 21:00.** The expected
restart on 4 August did not restore GPU access. Login nodes, storage and the
CPU partition are available.

Consequence: no GPU experiments have run on the cluster. All development and
validation has been done on a local RTX 4060 (8 GiB), which is sufficient for
correctness but not for the full pretraining runs.

Everything needed to submit is in place — environment built on group storage
with the correct PyTorch build, datasets transferred, 262 tests passing on both
machines, Slurm scripts written. The first job can be submitted within minutes
of the partition returning.

---

## 2. What was built

Both self-supervised training arms now exist and have been validated end to end
on real surgical video.

| Component | Status |
|---|---|
| Encoder interface + wrappers (VideoMAE, V-JEPA 2, ViT-MAE, DINOv3) | complete |
| Cached-feature extraction and probe protocol | complete |
| Mean-pool and attention-MIL heads | complete |
| **VideoMAE continued pretraining** | complete, validated |
| **V-JEPA 2 continued pretraining** | complete, validated |
| Corpus analysis, leakage registry, diagnostics | complete |
| Test suite | 262 passing, offline |

### Validation evidence

**VideoMAE.** On an overfitting check — 8 clips, deterministic input, 400 steps
— loss fell **0.601 → 0.355** monotonically. The objective, masking, optimiser
and fp16 path all work.

**V-JEPA 2.** Loss fell **0.238 → 0.184** over 200 steps, and a separate
collapse check confirmed the representation retained its variance and
dimensionality (rank 12.7 → 12.4 uncentred, variance ratio 0.96). This matters
because a falling JEPA loss can also mean the target encoder is degenerating
toward a constant, which would look like success and be worthless.

### Defects found and fixed

Six issues were identified during development, four of which would have
produced plausible but wrong results:

1. **VideoMAE attention biases silently zeroed on load.** The checkpoint stores
   them in the BEiT layout (`q_bias`/`v_bias`); transformers 5.x expects
   `query/key/value.bias` and ships no conversion, so trained biases were
   replaced with newly initialised ones.
2. **The repair was applied to the baseline path but not the pretraining path**,
   which would have measured adaptation gain between an encoder using the
   published weights and one that started from different ones.
3. **Pretraining and evaluation were not connected.** `encoder_final.pt` could
   not be loaded by the feature extractor, so adaptation gain could not be
   computed at all.
4. **Learning-curve selection took the maximum over 24 configurations**, which
   biases every point upward, most strongly at small data fractions — exactly
   where a low-label advantage would be claimed.
5. Out-of-memory in clip loading (10.5 GiB → 0.19 GiB after fixing).
6. Pooling allocated 24.5 GiB in one block for video-scale caches.

---

## 3. External baseline reproduction

The two strongest published segmentation-free results were assessed for
reproducibility. Neither can be fully reproduced from its public release.

| Method | Year | Main technique | C1 AP | C2 AP | C3 AP | Published mAP | Reproduced mAP | Status | Repository |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| SwinCVS | 2025 | Transfer learning + LSTM | 65.02 | 61.38 | 75.95 | 67.45 | **67.71** (E2E, seed 5) | Released checkpoint verified; Frozen variant awaiting Kaya | [franeknowak/SwinCVS](https://github.com/franeknowak/SwinCVS) |
| SMIL | 2026 | Video self-distillation + MIL | 70.57 | 69.09 | 72.30 | 70.66 | — | **Cannot be reproduced** | [Zhang-Yutao/SMIL-Framework](https://github.com/Zhang-Yutao/SMIL-Framework) |

Both evaluated on Endoscapes, official test split.

### 3.1 SwinCVS — released-checkpoint verification

Run locally on the RTX 4060. No training performed, so this is checkpoint
verification rather than training reproduction.

**Reproduced exactly**, to four decimal places on every per-criterion value:

| Artefact | Metric | Published | Reproduced |
|---|---|---:|---:|
| E2E SwinCVS (seed 5) | mAP | 67.71% | **67.71%** |
| Standalone SwinV2 backbone (seed 4) | mAP | 65.80% | **65.80%** |
| Standalone SwinV2 backbone (seed 4) | mean BAcc | 68.03% | **68.03%** |

1,796 test sequences for E2E, 1,799 annotated frames for the backbone — the
latter matching the supplementary material's count.

**Two errors found in the released metric code.** In `scripts/f_metrics.py` the
aggregate was computed as `(C1_recall + C2_recall + C3_recall) / 3` — mean
*recall*, not mean *balanced accuracy*. In `SwinCVS.py` the printed value was
`(C1_bacc + C1_bacc + C3_bacc) / 3`, counting C1 twice and omitting C2. Using
their own reported values, (0.7191 + 0.7191 + 0.5404)/3 = 0.6595, which is
exactly what the released code prints; the correct figure is 0.6212.

| Metric | Official | Corrected |
|---|---:|---:|
| Average balanced accuracy | 0.6595 | **0.6212** |
| Mean average precision | 0.6771 | 0.6771 |

**mAP is unaffected** — it is computed independently via
`average_precision_score` — so the published mAP results stand. Only the
balanced-accuracy reporting is wrong. Corrections are on a separate branch,
leaving the authors' implementation intact.

**The Frozen variant cannot be verified.** The paper's strongest result (67.45%
± 0.41% mAP) requires a complete Frozen checkpoint, and none of the three
released weight files is one: the Endoscapes-trained backbone lacks the trained
LSTM and temporal classifier. Reproducing it requires training those components
with the backbone frozen, following the five-seed protocol — planned for Kaya.

### 3.2 SMIL — cannot be reproduced

Three blockers, all visible in the repository:

1. **Imported modules are absent.**
   [`timesformer.py`](https://github.com/Zhang-Yutao/SMIL-Framework/blob/main/timesformer.py)
   imports `models.helpers` and `models.vit_utils`;
   [`train.py`](https://github.com/Zhang-Yutao/SMIL-Framework/blob/main/train.py)
   imports `networks.vit_seg_modeling` and `networks.ori_vit_seg_modeling`.
   Neither directory exists — the repository is seven flat files. The code fails
   on import.
2. **Weights not released.** The README states they will follow publication; the
   paper is published.
3. **Data split files absent.** The README's first instruction is to edit
   `trainlist.txt` and `testlist.txt`; neither is in the repository.

## 4. Dataset characteristics

All figures below are measured, not quoted.

### SAGES CVS Challenge 2024

| Property | Value |
|---|---|
| Videos | 700 train, 300 test |
| Duration | 90.0 s, 30 fps, uniform |
| Frames per video | 2,700 |
| Labelled frames | 18 per video (every 5 s) |
| **Labelled fraction** | **0.67%** |
| Total frames, train split | **1.9 million** |
| Resolution | heterogeneous: 1280×720, 720×576, 854×480, 720×480, 640×360, 640×480, 320×240 |
| Resolution entropy | 1.23 bits over 7 distinct resolutions |
| Licence | CC BY-NC 4.0 |

**The official test set is labelled and public** — 300 videos, 5,400 frames,
with prevalences (C1 14.5%, C2 25.2%, C3 13.3%) consistent with training. This
allows evaluation on 300 held-out videos rather than the 70 carved from training,
and makes results comparable to the challenge paper (arXiv 2509.17100).

**Inter-rater agreement** (Fleiss' kappa over three raters, all 700 videos):

| Criterion | κ | Interpretation | Prevalence | Raters disagreed |
|---|---:|---|---:|---:|
| C1 | 0.474 | moderate | 12.6% | **20.1%** |
| C2 | 0.540 | moderate | 24.1% | **26.1%** |
| C3 | 0.475 | moderate | 15.1% | **22.1%** |

Mean rater confidence 0.637 (sd 0.280).

Three trained surgeons following a published protocol fail to agree unanimously
on roughly one frame in five. This is direct empirical support for the
precondition framing in the introduction — and it sets a ceiling: a model
trained on majority-vote labels is reproducing a judgement the annotators
themselves reproduce about 80% of the time.

The consensus rule was verified as **majority vote**, checked against the source
per-rater columns across 266 non-trivial criterion-videos with zero
disagreements.

### Cholec80

| Property | Value |
|---|---|
| Videos | 80 full procedures, 25 fps |
| Duration | ~29 min average |
| Resolution | 77 at 854×480, 3 at 1920×1080 |
| Phases | 7, frame-level annotated |
| **Calot dissection fraction** | mean **39.8%**, sd 13.4%, range 11.6–75.7% |
| Licence | CC BY-NC-SA 4.0 |

The Calot fraction matters: at ~12 minutes of CVS-relevant footage per
procedure, Cholec80 offers *depth* per procedure where SAGES offers *breadth*
across patients.

### Leakage

Endoscapes and Cholec80 were both collected at IHU Strasbourg and **share six
videos**. Critically, **Cholec80 video 66 is Endoscapes video 121**, which sits
in the validation split. Endoscapes-test is clean.

This is now enforced in code: a corpus containing an evaluation procedure is
refused, with the offending identifier mapping and its source printed. The
module also raises for dataset pairs with no published overlap analysis, since
silence about an unchecked pair reads identically to silence about a clean one.

### Available but not yet used

- MultiBypass140: 39 Bern procedures downloaded (file 1 of 5)
- Endoscapes: complete, 201 videos at 1 fps

---

## 5. Discussion points

### 5.1 Where SMIL's performance actually comes from

Wang et al. (2026), IJCARS. Their Table 2 ablation is the most useful result in
the paper, and it is not the headline number.

**Their ablation, on the official Endoscapes test split:**

| Configuration | C1 mAP | C2 mAP | C3 mAP | **Avg mAP** | Avg Bacc | F1 |
|---|---:|---:|---:|---:|---:|---:|
| SSL only (no MIL, linear head) | 52.04 | 46.34 | 62.61 | **53.67** | 62.96 | 39.06 |
| MIL only (no SSL, Endo-FM ViT) | 27.25 | 23.63 | 34.97 | **28.62** | 50.00 | 32.67 |
| SSL + MIL (full SMIL) | 70.57 | 69.09 | 72.30 | **70.66** | 77.50 | 64.93 |

**Aggregation contributes +16.99 mAP.** Adding MIL to their self-supervised
representation moves it from 53.67 to 70.66.

For scale, SMIL's entire margin over the previous best segmentation-free method
is **+3.21 mAP** (SwinCVS Frozen, 67.45). So the aggregation mechanism
contributes roughly **five times** the published improvement.

**Their representation alone is not strong.** SSL-only at 53.67 mAP sits *below*
SwinCVS Frozen (67.45, −13.78) and below DeepCVS (60.2, −6.53). The
state-of-the-art result comes primarily from how patch features are aggregated,
not from a superior encoder.

**A methodological observation.** Their self-supervised pretraining used
Endoscapes frames, which are released at 1 fps. To construct video clips they
"repeat each frame five times to achieve 5 FPS, resulting in sequences of
identical frames". Their video transformer therefore never observes motion
during pretraining, despite the paper's claim to internalise "spatiotemporal
dynamics such as instrument–tissue interactions". Whatever the representation
learned, it was not from temporal change.

### 5.2 Why SAGES for continued SSL

SMIL's pretraining used Endoscapes, which is released at 1 fps. To construct
video clips they repeated each frame five times, producing sequences of
identical frames. Their video transformer never observed motion during
pretraining. SAGES removes that constraint, and differs on four axes that
matter.

| | Endoscapes (SMIL's corpus) | SAGES |
|---|---|---|
| Format | frames at 1 fps | **native MP4, 30 fps** |
| Temporal content | synthesised by repetition | real motion |
| Frames per procedure | ~290 (1 fps) | **2,700** |
| Procedures | 201 | **700 train + 300 test** |
| Framing | full CVS-relevant segment | **90 s centred on the clipping moment** |
| Resolution | uniform 480×854 | **7 distinct: 1280×720 to 320×240** |
| Institutions | single (IHU Strasbourg) | multi-centre |

Four reasons this matters:

- **Real motion.** At 30 fps a 16-frame clip spans 2.1 s of genuine movement —
  and it is the input geometry VideoMAE and V-JEPA 2 were pretrained on.
- **Precise window.** 90 s centred on the clipping moment, so every frame is
  task-relevant. Cholec80 averages only 39.8% Calot dissection per procedure.
- **Diversity.** Seven resolutions in a 50-video sample. Multi-centre appearance
  variation inside one corpus, which bears on generalisation.
- **Scale and licence.** 1.9M training frames against Endoscapes' 58,000, under
  CC BY-NC 4.0. No access request outstanding.

### 5.3 What this implies

**Aggregation should be a research question, not a fixed component.** A +17 mAP
effect is larger than any difference between published methods, and it has been
shown once, in a paper whose code does not run.

**There is headroom on the representation.** SMIL reaches 70.66 from a 53.67
backbone trained on temporally static clips. A better representation with the
same aggregation should do better.

**Proposal.** Two aggregation heads fixed in advance — mean pooling and gated
attention MIL — applied identically to both objectives. That gives a 2×2 of
objective by aggregation, at a cost of minutes rather than GPU-hours, since both
read the same cached features.

The heads are chosen now and not revisited after seeing results. Otherwise it
becomes an architecture search rather than a comparison.

**Questions.** Two heads or three? And is reimplementing SMIL's
global-plus-local fusion worth it, given their code does not run and it would
have to be rebuilt from the paper description?

## 6. Next

**On GPU availability**

1. 50-step validation job on a V100 to confirm fp16 behaviour and measure
   throughput, then submit VideoMAE and V-JEPA 2 continued pretraining on SAGES
2. Submit the SwinCVS Frozen training job

**Independent of GPU**

3. Dissertation writing — target is to finish Chapter 2 (literature review)
   within the next two weeks