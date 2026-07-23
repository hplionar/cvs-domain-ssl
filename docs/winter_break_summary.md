**Project:** Domain-Adaptive Visual Pretraining for Critical View of Safety Assessment in Laparoscopic Cholecystectomy  
**Researcher:** Hans Lionar  
**Repository:** `https://github.com/hplionar/cvs-domain-ssl`


## 1. Task

Automated Critical View of Safety (CVS) assessment is formulated as a **multi-label classification problem**.

For each image or video clip, the model predicts:

`y = [C1, C2, C3] ∈ {0,1}³`

where:

- **C1:** two and only two tubular structures are connected to the gallbladder;
- **C2:** the hepatocystic triangle is cleared of fat and connective tissue;
- **C3:** the lower third of the gallbladder is detached from the cystic plate.

![The three Critical View of Safety criteria](assets/figures/cvs_criteria.jpg)

*Figure 1. The three CVS criteria: identification of two tubular structures, clearance of the hepatocystic triangle, and exposure of the lower third of the cystic plate.*

The input is either:

- one RGB image: `H × W × 3`; or
- an ordered video clip: `T × H × W × 3`.

CVS assessment is not ordinary image classification. The model must recognise clinically meaningful anatomical evidence and determine whether each safety criterion has been visually achieved.

## 2. Main Methodological Directions

```text
Automated CVS Assessment
│
├── A. Explicit anatomy-guided methods
│   │
│   ├── DeepCVS (2022)
│   │   └── anatomy segmentation → CVS classifier
│   │
│   ├── LG-CVS (2024)
│   │   └── detected anatomy → object-centric graph → GNN
│   │
│   └── SV2LSTG (2023)
│       └── spatiotemporal anatomical graph
│
└── B. Segmentation-free methods
    │
    ├── SwinCVS (2025)
    │   └── pretrained image encoder → LSTM → CVS classifier
    │
    ├── SMIL (2026)
    │   └── surgical video self-distillation → MIL
    │
    └── Proposed direction
        └── pretrained image or video encoder
            → continued surgical-domain SSL
            → multi-label CVS classifier
```

The main distinction is how anatomical information is represented.

- **Anatomy-guided methods** explicitly represent structures and their relationships through segmentation masks, bounding boxes, or object-centric graphs.
- **Segmentation-free methods** do not receive these intermediate representations. The encoder must instead learn CVS-relevant anatomical evidence implicitly as latent image or video features.

## 3. Representative CVS Results

| Method | Year | Category | Dataset / setting | Main technique | C1 AP | C2 AP | C3 AP | Mean AP |
|---|---:|---|---|---|---:|---:|---:|---:|
| DeepCVS | 2022 | Anatomy-guided | Endoscapes; reimplemented baseline | Segmentation + classifier | 65.90 | 52.60 | 61.80 | 60.20 |
| LG-CVS | 2024 | Anatomy-guided | Endoscapes; segmentation setting | Object detection + latent graph | 69.50 | 60.70 | 71.80 | 67.30 |
| SV2LSTG | 2023 | Anatomy-guided | Endoscapes; reproduced evaluation | Spatiotemporal anatomy graph | 68.71 | 59.72 | 65.82 | 64.68 |
| SwinCVS | 2025 | Segmentation-free | Endoscapes | Transfer learning + LSTM | 65.02 | 61.38 | **75.95** | 67.45 |
| SMIL | 2026 | Segmentation-free | Endoscapes | Video self-distillation + MIL | **70.57** | **69.09** | 72.30 | **70.66** |

These results suggest that segmentation-free methods can compete with anatomy-guided pipelines when supported by strong pretrained representations and appropriate spatial or temporal aggregation.

However, this is not a perfectly controlled comparison because the methods use different architectures and supervision settings.

## 4. Why Domain-Adaptive Self-Supervised Learning?

Large pretrained encoders provide useful transferable features, but their original pretraining data differ substantially from laparoscopic surgery.

The proposed approach continues self-supervised pretraining on unlabelled surgical images or videos before downstream CVS training.

### 4.1 Conceptual Motivation

The idea follows the principle of **continued pretraining under domain shift** introduced by Gururangan et al. (2020).

<p align="center">
  <img
    src="assets/figures/gururangan-dont_stop_pretraining-data_distributions_illustration.png"
    alt="Illustration of continued pretraining under domain shift"
    width="500"
  >
</p>

<p align="center">
  <em>
    Figure 2. Relationship between the original pretraining domain,
    target domain, and downstream task distribution.
    Reproduced from Gururangan et al. (2020), Figure 1.
  </em>
</p>

For this project:

- **original pretraining domain:** natural images or broad video collections;
- **target domain:** laparoscopic and broader surgical visual data;
- **downstream task:** CVS criterion prediction.

```text
Broad visual pretraining
        ↓
Transferable representation
        ↓
Continued SSL on surgical data
        ↓
Surgically adapted representation
        ↓
CVS classification
```

### 4.2 Main Benefits

#### Lower annotation burden

Continued SSL uses unlabelled surgical images or videos. It does not require segmentation masks, bounding boxes, or anatomical graphs during pretraining.

CVS labels are still needed for downstream training and evaluation, but the larger pretraining corpus can remain unlabelled.

#### Scalability

Surgical videos contain substantially more unlabelled frames than expert-labelled CVS timepoints.

```text
Large collection of unlabelled surgical videos
                    ↓
Continued self-supervised pretraining
                    ↓
Surgically adapted encoder
                    ↓
Smaller labelled CVS dataset
```

Additional videos can therefore contribute to representation learning without requiring dense frame-level annotation.

#### Reduction of the pretraining domain gap

Laparoscopic surgery contains recurring patterns that differ from natural images and general action videos:

- deformable tissue;
- fine-grained anatomical boundaries;
- instruments and instrument–tissue interaction;
- smoke, bleeding, and specular reflections;
- constrained viewpoints and camera motion;
- temporary occlusion;
- gradual anatomical exposure.

Continued SSL may make the encoder more responsive to these surgical visual and temporal regularities.

#### Reduced dependence on intermediate pipelines

Anatomy-guided pipeline:

```text
Image
→ segmentation or object detection
→ graph construction or anatomical rules
→ CVS prediction
```

Proposed segmentation-free pipeline:

```text
Image or video
→ learned encoder representation
→ CVS prediction
```

This does not remove engineering, but it reduces dependence on separate segmentation, detection, graph-construction, and rule-based systems.

### 4.3 Evidence from Previous Studies

| Study | SSL objective | Relevant finding | Link |
|---|---|---|---|
| **Gururangan et al. (2020)** | Continued masked-language modelling through domain-adaptive and task-adaptive pretraining | Established the general principle that continuing pretraining on unlabelled data closer to the downstream domain can improve task performance | [Paper](https://aclanthology.org/2020.acl-main.740/) |
| **Azizi et al. (2021)** | SimCLR contrastive learning and Multi-Instance Contrastive Learning (MICLe) | Broad-domain SSL followed by continued SSL on unlabelled medical images improved dermatology and chest X-ray classification | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Azizi_Big_Self-Supervised_Models_Advance_Medical_Image_Classification_ICCV_2021_paper.html) |
| **Ramesh et al. (2023)** | MoCo v2, SimCLR, DINO, and SwAV | Demonstrated that appropriately adapting SSL methods to surgical video frames improved transfer to surgical phase recognition and tool-presence detection | [Paper](https://doi.org/10.1016/j.media.2023.102844) |
| **SurgeNetXL (Jaspers et al., 2026)** | DINO teacher–student self-distillation on surgical video frames | Large-scale surgical-domain pretraining improved transfer across semantic segmentation, phase recognition, and CVS classification, while also highlighting the importance of dataset scale and composition | [Paper](https://doi.org/10.1016/j.media.2025.103873) |
| **SurgRec (Lu et al., 2026)** | SurgRec-MAE masked spatiotemporal reconstruction and SurgRec-JEPA latent-representation prediction | Demonstrated scalable surgical video pretraining across a large multi-source corpus and provided a direct precedent for comparing masked reconstruction with latent prediction | [Preprint](https://arxiv.org/abs/2603.29966) |
| **SMIL (Wang et al., 2026)** | Label-free video teacher–student self-distillation followed by multiple-instance learning | Demonstrated strong segmentation-free CVS prediction by combining self-supervised spatiotemporal features with global and local feature aggregation | [Paper](https://doi.org/10.1007/s11548-026-03580-9) |

Together, these studies support the hypothesis that continued surgical pretraining may improve downstream transfer.

However, improvement is not guaranteed. It may depend on:

- the starting checkpoint;
- SSL objective;
- pretraining-data scale and diversity;
- adaptation duration;
- downstream task;
- redundancy in the adaptation corpus.

This motivates comparing:

```text
LC / CVS-specific surgical data
versus
broader minimally invasive surgical data
versus
a balanced mixture of both
```

## 5. Why MAE and JEPA?

The project compares two complementary SSL objective families at both image and video levels.

| Input level | Masked reconstruction | Latent prediction |
|---|---|---|
| Image | MAE | I-JEPA |
| Video | VideoMAE | V-JEPA |

```text
Image level
├── MAE
│   └── reconstruct masked image pixels
└── I-JEPA
    └── predict masked image representations

Video level
├── VideoMAE
│   └── reconstruct masked spatiotemporal regions
└── V-JEPA
    └── predict masked spatiotemporal representations
```

### 5.1 Starting Checkpoints

| Model | Original SSL data | Original training objective |
|---|---|---|
| MAE | ImageNet-1K | Masked image-pixel reconstruction |
| I-JEPA | ImageNet-1K or ImageNet-22K | Prediction of masked image representations |
| VideoMAE | Kinetics-400 or Something-Something V2, depending on checkpoint | Masked video reconstruction |
| V-JEPA | VideoMix2M | Masked video-representation prediction |

**VideoMix2M** is the combined video collection used for V-JEPA pretraining. It includes HowTo100M, Kinetics-400/600/700, and Something-Something V2.

In this project, training begins from released SSL-pretrained checkpoints rather than random initialisation:

```text
Released SSL-pretrained checkpoint
        ↓
Continue the same objective
on unlabelled surgical data
        ↓
Surgically adapted encoder
        ↓
CVS evaluation
```

### 5.2 Selection Rationale

MAE and JEPA are prioritised because they provide:

- corresponding image and video formulations;
- two clearly different learning objectives;
- direct use of unlabelled images or clips;
- no explicit negative-sample mechanism;
- a controlled comparison of static and temporal representation learning.

Contrastive methods such as SimCLR and MoCo remain valid alternatives, but require decisions about positive views, negative samples, augmentations, batch sizes, or queues. Visually similar surgical frames may also become false negatives.

The main justification is therefore not only the absence of explicit negatives. MAE and JEPA provide the clearest experimental structure:

1. **MAE versus I-JEPA:** reconstruction versus latent prediction for images.
2. **VideoMAE versus V-JEPA:** reconstruction versus latent prediction for videos.
3. **Image versus video:** static evidence versus spatiotemporal evidence.

### Research Hypothesis

> Continued self-supervised pretraining on surgical images or videos will adapt pretrained encoders toward laparoscopic anatomical and temporal regularities, making CVS-relevant evidence more accessible to a segmentation-free multi-label classifier.

## 6. CVS Datasets

| Dataset | Main characteristics | Strength | Limitation | Proposed role |
|---|---|---|---|---|
| Endoscapes | 201 procedures; 58,813 ordered 1-FPS frames; 11,090 CVS-labelled frames | Established benchmark and comparison with previous work | Public release does not preserve dense video | Initial image-level and sparse temporal benchmark |
| SAGES | 1,000 raw 90-second clips; 18 labelled timepoints per clip; multi-institutional | Dense temporal information and greater clinical variability | Newer benchmark with fewer directly comparable published results | Native video modelling and continued video SSL |

### 6.1 Endoscapes

The original recordings were standardised to 30 FPS at:

`H × W = 480 × 854`

The public release contains ordered 1-FPS frames rather than the original dense video.

Therefore, Endoscapes is suitable for:

- pipeline validation;
- frame-level benchmarking;
- comparison with previous CVS methods;
- optional sparse temporal evaluation.

Its limitation is that VideoMAE and V-JEPA cannot observe most short-term motion occurring between the released frames.

### 6.2 SAGES

SAGES provides raw 90-second MP4 clips with CVS labels every five seconds.

Each clip contains:

- 18 labelled timepoints;
- approximately 2,700 frames at 30 FPS.

![SAGES CVS annotation density](assets/figures/sages_annotation_density.jpg)

*Figure 3. A 90-second SAGES clip contains approximately 2,700 frames at 30 FPS, but only 18 CVS-labelled timepoints.*

SAGES supports:

- frame-level modelling at labelled timepoints;
- native video modelling using temporally ordered clips;
- continued SSL using unlabelled frames;
- evaluation under greater institutional and visual variability.

### 6.3 Proposed Dataset Roles

```text
Endoscapes
→ initial pipeline validation
→ literature-comparable image benchmark
→ optional sparse temporal evaluation

SAGES
→ parallel investigation
→ native video modelling
→ continued surgical-domain SSL
→ evaluation under greater variability
```

The project will begin with Endoscapes while investigating SAGES in parallel.

## 7. Experimental and Implementation Plan

The experiments use a staged design. A **25% surgical-data pilot** is used before any model is scaled to the full continued-pretraining corpus.

### 7.1 Candidate SSL Models

The project compares masked reconstruction and latent prediction at image and video levels:

| Input | Masked reconstruction | Latent prediction |
|---|---|---|
| Image | MAE | I-JEPA |
| Video | VideoMAE | V-JEPA |

All models will start from released self-supervised checkpoints:

| Model | Proposed checkpoint | Original SSL data | Pooled representation dimension |
|---|---|---|---:|
| **MAE** | ViT-B/16 | ImageNet-1K | `768` |
| **I-JEPA** | ViT-H/14 | ImageNet-1K | `1280` |
| **VideoMAE** | ViT-B/16 | Kinetics-400 | `768` |
| **V-JEPA** | ViT-L/16 | VideoMix2M | `1024` |

During continued SSL:

```text
MAE
→ masked image reconstruction

I-JEPA
→ masked image-representation prediction

VideoMAE
→ masked video reconstruction

V-JEPA
→ masked video-representation prediction
```

The checkpoints differ in architecture, model size, input type, and original pretraining data. The experiment will therefore be interpreted as a **practical pretrained-system comparison**, not a perfectly controlled comparison of SSL objectives.

The decoder or predictor used during SSL is separate from the downstream CVS classification head.

---

### 7.2 CVS Classification Head

Every encoder will initially use the same lightweight classifier design:

```text
Encoder output tokens: N × d
        ↓
Mean pooling over N tokens
        ↓
Pooled representation: d
        ↓
LayerNorm(d)
        ↓
Dropout
        ↓
Linear(d → 3)
        ↓
C1, C2, and C3 logits
```

Here:

- `N` is the number of spatial or spatiotemporal tokens;
- `d` is the representation dimension;
- image models pool over spatial patch tokens;
- video models pool over spatial and temporal tubelet tokens.

Mean pooling averages the final contextual token representations, not the original image pixels.

| Backbone | Classification head |
|---|---|
| ViT-B | `LayerNorm(768) → Dropout → Linear(768, 3)` |
| ViT-L | `LayerNorm(1024) → Dropout → Linear(1024, 3)` |
| ViT-H | `LayerNorm(1280) → Dropout → Linear(1280, 3)` |

Because CVS assessment is multi-label classification:

- three independent logits are produced;
- training uses `BCEWithLogitsLoss`;
- sigmoid converts the logits into probabilities;
- softmax is not used because multiple CVS criteria may be positive simultaneously.

#### Why use linear probing?

Linear probing still involves training, but only the lightweight CVS classification head is updated while the pretrained encoder remains frozen.

It is commonly used in SSL research because it:

- measures how accessible downstream information already is in the pretrained representation;
- provides a consistent evaluation protocol across different encoders;
- requires less compute than full fine-tuning;
- reduces the risk that a powerful classifier hides differences in encoder quality.

The primary comparison will use the common normalised linear probe. Attentive pooling or an MLP head may later be evaluated for the selected finalists.

---

### 7.3 Plan A: Full Experimental Plan

Plan A applies continued SSL to **all four models** using the same 25% surgical-data pilot before selecting the strongest image and video models.

<p align="center">
  <img
    src="assets/figures/cvs_ssl_plan_a.png"
    alt="Plan A full four-model continued-SSL pilot"
    width="650"
  >
</p>

<p align="center">
  <em>
    Figure 4. Plan A: all four candidate models undergo continued SSL using the 25% surgical-data pilot before model selection and full-corpus scaling.
  </em>
</p>

#### Stage 1 — Original-checkpoint baselines

For MAE, I-JEPA, VideoMAE, and V-JEPA:

- load the released SSL-pretrained checkpoint;
- freeze the encoder;
- train only the common CVS classification head;
- record validation mAP and per-criterion AP.

Evaluation datasets:

```text
Image models
→ Endoscapes and SAGES

Video models
→ SAGES
```

These results establish the transfer-learning baselines before surgical-domain adaptation.

#### Stage 2 — Continued SSL using the 25% pilot

All four models will undergo continued SSL using the same underlying 25% subset of the collected surgical procedures.

```text
Selected surgical procedures
├── MAE / I-JEPA
│   └── sample individual frames
└── VideoMAE / V-JEPA
    └── sample temporally ordered clips
```

The subset will be selected at the procedure or source-video level and stratified across the included datasets.

The following will be recorded:

- selected datasets and procedures;
- number of sampled images or clips;
- input resolution;
- temporal sampling configuration;
- optimisation updates;
- GPU hours and peak memory;
- model-specific masking settings.

The image and video tracks will be made as comparable as practical, although identical compute is not possible because the models differ in size and input format.

#### Stage 3 — Evaluation of adapted encoders

Each adapted encoder will be frozen and evaluated using the same protocol as its original checkpoint.

```text
Adaptation gain
=
adapted validation mAP
− original-checkpoint validation mAP
```

This stage determines:

- whether continued surgical SSL improves each model;
- whether reconstruction or latent prediction benefits more;
- the strongest adapted image model;
- the strongest adapted video model.

#### Stage 4 — Partial supervised fine-tuning

The strongest adapted image and video models will be partially fine-tuned using CVS labels.

The initial strategy will unfreeze:

- the final 25% of transformer blocks;
- the final encoder normalisation layer;
- the CVS classification head.

```text
ViT-B: final 3 of 12 blocks
ViT-L: final 6 of 24 blocks
ViT-H: final 8 of 32 blocks
```

The finalists will be compared using:

- validation mAP;
- per-criterion AP;
- computational cost;
- memory requirements;
- implementation stability.

#### Stage 5 — Full-corpus scaling

One strongest feasible model will be selected and continued on 100% of the collected surgical corpus.

```text
Four models adapted on the 25% pilot
        ↓
Select image and video winners
        ↓
Partially fine-tune both finalists
        ↓
Select one strongest feasible model
        ↓
Continue SSL on the full corpus
```

#### Stage 6 — Final evaluation

The final model will be compared with:

- its original released checkpoint;
- its 25% surgical-pilot checkpoint;
- its full-corpus adapted checkpoint.

Validation mAP will be used for model selection. The test set will remain locked until the final configuration has been selected.

Primary metrics:

- mean AP;
- AP for C1, C2, and C3.

Secondary metrics:

- balanced accuracy;
- ROC-AUC.

---

### 7.4 Plan B: Reduced-Scope Fallback

Plan B will be used if compute, data access, implementation time, or training stability becomes limiting.

<p align="center">
  <img
    src="assets/figures/cvs_ssl_plan_b.png"
    alt="Plan B reduced-scope fallback"
    width="650"
  >
</p>

<p align="center">
  <em>
    Figure 5. Plan B: one model is selected before continued SSL and is evaluated using a single 25% balanced mixed-data pilot.
  </em>
</p>

Plan B consists of:

1. evaluate the original MAE, I-JEPA, VideoMAE, and V-JEPA checkpoints using frozen linear probing;
2. select one model using validation performance and computational feasibility;
3. continue SSL using a 25% balanced mixed surgical corpus;
4. compare the adapted checkpoint with its original baseline;
5. partially fine-tune the selected model using CVS labels;
6. scale to the full corpus only if validation performance improves.

```text
Evaluate four original checkpoints
        ↓
Select one strongest feasible model
        ↓
Continue SSL on a 25% mixed corpus
        ↓
Evaluate adaptation gain
        ↓
Partial supervised fine-tuning
        ↓
Scale to 100% only if successful
```

The main difference is:

```text
Plan A
→ continue SSL for all four models on the 25% pilot
→ select image and video winners
→ scale one final model

Plan B
→ select one model before continued SSL
→ run one 25% mixed-data pilot
→ scale only if the pilot improves performance
```

## 8. Proposed Surgical Data Collection for Continued SSL

The continued-SSL corpus will be organised into:

1. an LC / CVS-specific pool;
2. a broader minimally invasive surgery pool;
3. a balanced mixture of both pools.

All candidate datasets remain subject to access permission, data-use agreements, licensing, storage, and preprocessing feasibility.

### 8.1 LC / CVS-Specific Pool

| Dataset | Procedure or content | Proposed role | Priority and access note | Link |
|---|---|---|---|---|
| **SAGES CVS Challenge** | 1,000 90-second laparoscopic cholecystectomy clips centred on the period before clipping | Primary source of LC-specific raw video and CVS-relevant temporal content | **Core if permitted.** Available in the current project, but its data-use agreement must explicitly allow continued SSL outside the original challenge task | [Official SAGES dataset page](https://www.cvschallenge.org/the-challenge-2) |
| **Cholec80** | 80 full laparoscopic cholecystectomy procedures recorded at 25 FPS | Full-procedure LC video and wider workflow coverage | **Core, subject to access** | [CAMMA dataset page](https://camma.unistra.fr/datasets/) |
| **HeiChole** | 33 laparoscopic cholecystectomy videos from three centres | Additional multi-centre LC variability | **Secondary; public access may be limited** | [Official HeiChole Synapse page](https://www.synapse.org/Synapse:syn18824884/wiki/) |
| **hSDB-instrument** | Instrument-localisation data from 24 laparoscopic cholecystectomy cases | Optional LC expansion if temporally ordered frames or source videos are available | **Optional; temporal continuity must be verified** | [Official hSDB-instrument page](https://hsdb-instrument.github.io/) |

This pool contains patterns closely related to CVS assessment:

- gallbladder dissection;
- hepatocystic-triangle exposure;
- cystic structures;
- clipping preparation;
- instrument–tissue interaction;
- progressive anatomical exposure.

SAGES and Cholec80 are the highest-priority candidates. HeiChole and hSDB-instrument will be included only if access and data format are suitable.

### 8.2 General Minimally Invasive Surgery Pool

| Dataset | Procedure or content | Proposed role | Priority and access note | Link |
|---|---|---|---|---|
| **MultiBypass140** | 140 full laparoscopic Roux-en-Y gastric bypass procedures from two centres | Large source of broader laparoscopic anatomy, workflow, and multi-centre variability | **Core** | [Official MultiBypass140 repository](https://github.com/CAMMA-public/MultiBypass140) |
| **AutoLaparo** | 21 full laparoscopic hysterectomy procedures | Additional laparoscopic anatomy, camera motion, tissue interaction, and instrument variability | **Core, subject to access request** | [Official AutoLaparo page](https://autolaparo.github.io/) |
| **PSI-AVA** | Robot-assisted radical prostatectomy videos | Additional procedural structure, instruments, and atomic surgical actions | **Secondary; access must be confirmed** | [Dataset paper](https://arxiv.org/abs/2212.04582) |
| **SAR-RARP50** | 50 robotic prostatectomy suturing video segments | Fine-grained robotic instrument motion and surgical actions | **Secondary** | [Official UCL dataset project](https://rdr.ucl.ac.uk/projects/SAR-RARP50_Segmentation_of_surgical_instrumentation_and_Action_Recognition_on_Robot-Assisted_Radical_Prostatectomy_Challenge/191091) |
| **SurgicalActions160** | 160 short clips representing 16 actions in gynaecological laparoscopy | Small source of additional action diversity | **Optional; too small to be a primary pretraining corpus** | [Official SurgicalActions160 page](https://ftp.itec.aau.at/datasets/SurgicalActions160/) |

This pool provides greater variation in:

- procedure-specific anatomy;
- instruments;
- surgical actions;
- camera movement;
- smoke, bleeding, and occlusion;
- acquisition conditions;
- laparoscopic and robotic viewpoints.

The preferred initial general-surgery sources are:

```text
MultiBypass140
+
AutoLaparo
```

PSI-AVA and SAR-RARP50 may be added if robotic-surgery diversity is considered useful. SurgicalActions160 will remain optional because of its limited size.

### 8.3 Initial Collection Priority

```text
Priority 1
├── SAGES
├── Cholec80
├── MultiBypass140
└── AutoLaparo

Priority 2
├── HeiChole
├── PSI-AVA
└── SAR-RARP50

Priority 3
├── hSDB-instrument
└── SurgicalActions160
```

### 8.4 Balanced Mixed Pool

The mixed corpus will use equal sampling budgets from the two pools:

```text
50% LC / CVS-specific samples
+
50% broader minimally invasive surgery samples
```

The balance refers to the number of sampled images or clips, not the number of source videos or total available frames.

This prevents a large dataset from dominating solely because it contains more video.

### 8.5 Matched Image and Video Views

Where raw video is available, image and video models will receive matched views from the same source procedures:

```text
Source surgical video
├── Image SSL
│   └── sample one frame from a source clip
└── Video SSL
    └── sample an ordered sequence from the same source clip
```

This improves comparability because both model types observe the same underlying procedures.

Frame-only datasets may be used as optional image-SSL expansion, but they will not be included in the primary matched image-versus-video comparison.

### 8.6 Data Quality, Licensing, and Leakage Control

Before continued pretraining, the pipeline will:

- verify each dataset licence and data-use agreement;
- record whether commercial or non-commercial restrictions apply;
- standardise video decoding, colour format, and resolution handling;
- define common spatial and temporal sampling procedures;
- detect corrupted or unreadable videos;
- identify duplicated or overlapping surgical procedures;
- record dataset, institution, procedure, and video identifiers;
- prevent downstream validation or test data from entering SSL pretraining.

If a dataset is used for both continued SSL and downstream evaluation:

```text
Training cases
→ may be used for continued SSL

Validation cases
→ excluded from continued SSL

Test cases
→ excluded from continued SSL
```

For example, if SAGES is used for downstream CVS evaluation, only the designated training videos may contribute to continued SSL.

Datasets derived from the same source procedures will be treated as one overlap group. This is particularly important for Cholec80-derived datasets such as CholecT variants, which may contain the same original surgeries.

### 8.7 Purpose of the Composition Study

The experiment does not assume that the most procedure-specific data will automatically produce the best representation.

It tests whether CVS benefits more from:

1. **procedural relevance** from LC/CVS-specific data;
2. **greater visual and procedural diversity** from broader surgical data; or
3. **a balance of relevance and diversity** from the mixed corpus.

---

## 9. Questions for Supervisor Approval

1. **Dataset roles:**  
   Is it appropriate to use both Endoscapes and SAGES for image-level CVS evaluation, with Endoscapes as the literature-comparable benchmark and SAGES as an additional benchmark with greater visual variability?

2. **Experiment plan:**  
   Do you approve the proposed experimental plan, using Plan A as the main approach and Plan B as the fallback if time, compute, or dataset access becomes limiting?