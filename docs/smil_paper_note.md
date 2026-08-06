# SMIL (Wang et al., 2026) — reading notes, with page references

Wang H., Zhang Y., Yang Y., Zhu Y., Xu R. "CVS assessment via distillation-based
self-supervised and multiple instance learning in laparoscopic cholecystectomy."
*IJCARS*, DOI 10.1007/s11548-026-03580-9. Received 19 Jun 2025, accepted 27 Jan 2026.
Hefei University of Technology. Code: github.com/Zhang-Yutao/SMIL-Framework (does not import).

Read from the PDF on 2026-08-06. Page numbers are the PDF's.

## What the backbone actually is

**Not DINOv2. Not DINOv3.** Page 4, *Implementation details*: *"For SSL pretraining, we
used a ViT-Base (patch size 16) initialized with Endo-FM weights as the teacher."*
Page 2: *"A video transformer is first pretrained using self-distillation with no labels
(DINO) [20] on LC videos."*

So: ViT-B/16 → Endo-FM initialisation → DINO objective. Endo-FM is MICCAI'23,
github.com/med-air/Endo-FM, and is **public and ungated**. If a real reproduction is
ever attempted, Endo-FM is the checkpoint to start from — closer to SMIL than any DINO
release and with no access queue.

Hardware, same paragraph: *"four NVIDIA RTX 3090 GPUs"*, 100 epochs SSL + 350 epochs
SMIL training. Not a single-GPU job.

## Head specification (matches what was implemented)

Page 4, *SMIL Training*: *"The global representation h_ctx and the MIL feature are
concatenated into a fused vector h_fused = [h_ctx ; h_MIL] ∈ R^2D"*, Eq. (4) linear +
sigmoid, Eq. (5) sum of three BCE terms. Page 6, Eq. (6): `h_MIL = Σ αᵢhᵢ`, softmax-
normalised. Page 4: *"The MIL module used a 512-dimensional hidden layer and a dropout
rate of 0.1."*

At SMIL-training and inference time the model *"operates on a per-frame basis, processing
one frame at a time"* (p. 4), and Fig. 3 shows a 16×16 patch grid — so the MIL branch
aggregates spatial patches of a single frame. That is exactly what `AttentivePoolHead`
does over patch tokens, which makes the frozen-feature test bed a fair analogue.

## Table 2 (p. 6) — the important finding

| Row | mAP Avg | Bacc Avg | F1 |
|---|---:|---:|---:|
| SSL | 53.67 | 62.96 | 39.06 |
| MIL | 28.62 | **50.00** | 32.67 |
| SSL+MIL | 70.66 | 77.50 | 64.93 |

**The +16.99 aggregation reading is correct.** Page 5: *"We removed the MIL branch and
fed the preprocessed input directly into the pretrained student video transformer,
followed by a linear classifier"* — that is the 53.67 row. The 70.66 row is the full
two-branch fusion. Same backbone, same features. So the delta is genuinely attributable
to adding the MIL branch alongside the global one.

**But the MIL row is a failed run, not a measurement.** Balanced accuracy is exactly
50.00 on C1, C2 and C3; the paper concedes it (p. 5: *"a flat 50.0% Bacc across all
categories"*). mAP 28.62 sits at the base rate stated on p. 4 (*"approximately 30 percent
of frames contain visible CVS criteria"*). That model predicted a constant.

**And that row is not what its label suggests.** Page 5: *"we removed the SSL pretraining
stage and replaced the pretrained student video transformer with a ViT initialized from
Endo-FM."* It is "no SSL + MIL", not "SSL backbone + MIL branch only".

**Consequence: the paper contains no experiment isolating fusion from attention-MIL on a
working encoder.** There is no row for "SSL backbone, MIL branch, no global branch". That
is precisely the arm the three-head comparison adds. This is the strongest justification
for the experiment — stronger than "the effect has only been demonstrated once".

## Other weaknesses worth citing in the dissertation

- **No variance anywhere.** Every number in Tables 1–4 is a single run. No seeds, no
  repetitions, no error bars. +16.99 has no uncertainty attached to it.
- **Baselines are quoted, not re-run** (p. 4): *"using official results from SwinCVS,
  DeepCVS, and LG-CVS ... without re-implementation."*
- **The "spatiotemporal" claim rests on duplicated frames.** Page 4: *"we constructed
  video clips by concatenating the released frames and repeating each frame five times
  to achieve 5 FPS, resulting in sequences of identical frames."* Endoscapes ships 1 fps;
  the 5 fps is manufactured by duplication. There is no sub-second motion in the
  pretraining data, which sits awkwardly against the Discussion's claim of internalising
  *"instrument–tissue interactions"*.
- Table 1 numbers are on the official **test** split. The cached-probe protocol here
  selects on **val**, so the two are not directly comparable regardless.

## Bearing on this project

The encoder in the head comparison is a **held-constant control**, not a reproduction
target. Its only required property is a `[CLS]` token carrying a real global summary, so
that `h_ctx` is not empty. Both DINOv2 and DINOv3 satisfy that; neither is what SMIL
used. Do not describe any of this as a reproduction of SMIL.