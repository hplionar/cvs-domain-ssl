# Implementation Plan: Domain-Adaptive Visual Pretraining for CVS Assessment

**Researcher:** Hans Lionar
**Repository:** `https://github.com/hplionar/cvs-domain-ssl`
**Intended path:** `docs/implementation_plan.md`
**Status:** Supersedes §7 of `docs/winter_break_summary.md`. Sections 1–6 and 8–9 of that document remain current.

---

## 1. Purpose and Scope

This document specifies the experimental design, evaluation protocol, and engineering architecture for the implementation phase. It differs from `winter_break_summary.md` in four respects:

1. The four-model Plan A / Plan B structure is replaced by a **two-track** structure separating the scientific claim from the performance ambition.
2. The candidate model set is revised in light of checkpoint availability and V100 constraints.
3. A **cached-feature architecture** is adopted for all frozen-encoder evaluation.
4. Corrections are specified for defects identified in the existing baseline implementation.

---

## 2. Two-Track Structure

The project pursues two objectives that must not be allowed to contaminate one another.

### Track 1 — Controlled Comparison (the scientific claim)

Frozen encoders evaluated under an identical cached-feature probe, with **adaptation gain** as the primary outcome:

```text
adaptation gain = adapted mAP − original-checkpoint mAP
```

Track 1 is valid at any absolute mAP. Its contribution is that no published work compares masked reconstruction against latent prediction at matched capacity under an identical probing protocol; existing comparisons contrast whole systems differing along several axes simultaneously.

### Track 2 — Best Achievable System (the ambition)

Adapted encoder combined with attentive aggregation and partial supervised fine-tuning, evaluated on the official Endoscapes test split against published results. The reference target is SMIL at 70.66 mAP.

### Isolation Requirement

**The Track 1 probe protocol must be fixed and version-controlled before any Track 2 tuning begins.** If Track 2 fails to reach competitive absolute performance, Track 1 must remain independently reportable. Any hyperparameter, head architecture, or preprocessing decision made while inspecting Track 2 results is prohibited from propagating backwards into Track 1.

---

## 3. Corrections to Existing Implementation

The following defects were identified by audit of the repository at commit `added winter break summary` (23 July 2026). They must be resolved before any result enters the dissertation.

The split manifests were verified as clean: Endoscapes contains 201 videos partitioned 120/41/40 with no video appearing in more than one split; SAGES contains 700 videos partitioned 560/70/70, all drawn from the official training partition, likewise with no cross-split contamination. `eval/metrics.py` computes per-criterion balanced accuracy correctly and does not exhibit the aggregation defect found in SwinCVS.

| ID | Severity | Location | Defect | Resolution |
|---|---|---|---|---|
| **F1** | High | `data/sages_sequence_datasets.py:165–170` | A seed is drawn from the global torch RNG and then used to reset that same RNG, producing a self-referential orbit. Measured cycle lengths of 7,443 and 34,839 draws; the SAGES training split performs 10,080 draws per epoch. `data/sequence_datasets.py:217` draws from Python's `random` and so avoids orbit collapse, but still resets the global torch RNG on every item. | Replace with `torch.random.fork_rng()`, or sample transform parameters explicitly once and apply them deterministically to all frames. Obviated entirely by the cached-feature architecture (§4). |
| **F2** | High | `data/sages_sequence_datasets.py:184–187` | Boundary clamping produces degenerate clips: 22.2% of all clips contain duplicated frames and 5.6% consist of five identical frames, in every split. Separately, a 5-frame clip at 5-second spacing spans 20 seconds while the label describes only the final timepoint. | Exclude or explicitly pad boundary cases; adopt attentive rather than uniform temporal pooling (§9). |
| **F3** | High | `metadata/experiments/baseline_test_evaluation_summary.md` | The conclusion that weighted BCE confirms the importance of class imbalance is a threshold artefact. Threshold-free metrics are unchanged or worse while only balanced accuracy at a fixed 0.5 cutoff improves. | Tune the decision threshold per criterion on validation for both losses, then compare balanced accuracy at the tuned threshold. Rewrite the summary. |
| **F4** | Medium-high | `models/encoders/dinov2_encoder.py:64` | The encoder returns the CLS token alone. §7.2 of the plan specifies mean pooling over tokens. The pooling branch at `models/cvs_model.py:104` is dead code. All six existing baselines are therefore under-powered relative to both the documented design and the DINOv2 linear-evaluation protocol. | Return full token grids from all encoders; perform pooling in the head. |
| **F5** | Medium | `data/transforms.py:76–81`; `train/train_cvs.py` | `exp001` peaks at epoch 2 and `exp002` at epoch 1 of 5, indicating oscillation rather than convergence at `lr=1e-3` with no schedule. Hue jitter is applied to laparoscopic imagery, perturbing the colour cue on which C2 (clearance of fat and connective tissue) substantially depends. | Resolve by grid search on cached features (§8). Remove colour jitter from the supervised probe transform. |
| **F6** | Low | `train/train_cvs.py:238–246` | `save_checkpoint` serialises the frozen encoder into every checkpoint, approximately 350 MB of constants per file. | Save head state only, plus an encoder identifier. |
| **F7** | Low | `data/sages_datasets.py:18–34` | `REQUIRED_COLUMNS` omits `official_split`, which `__getitem__` reads at line 122. Latent `KeyError` against any future manifest. | Add the column to the requirement set. |
| **F8** | Low | `data/transforms.py:52` | `GaussianBlur` is applied unconditionally rather than with a probability. | Wrap in `RandomApply`. |

Additionally, no `autocast` is used anywhere, and both `DataLoader` constructions omit `persistent_workers` and `prefetch_factor`.

---

## 4. Cached-Feature Architecture

A frozen encoder is a deterministic function. Re-running it every epoch to train a head of a few thousand parameters is pure waste, and any stochastic variation between the baseline run and the adapted run contaminates the adaptation gain, which is a *difference* of two measurements.

### Decision: cache full token grids, not pooled vectors

Pooled vectors destroy spatial and temporal layout irreversibly. Token grids are a strict superset from which pooled vectors can be recomputed in microseconds. The cost is disk space alone.

| Cache | Shape per sample | Endoscapes train (6,960) | SAGES train (10,080) |
|---|---|---|---|
| Pooled vector, fp16 | `[768]` | 10.7 MB | 15.5 MB |
| Image token grid, fp16 | `[197, 768]` | ~2.1 GB | ~3.1 GB |
| Video token grid, fp16 | `[1568, 768]` | n/a | ~24 GB |

Video token counts assume 16 frames, patch 16, tubelet 2, at 224 px.

### Consequences

- The linear probe and the attentive probe read from the same cache. No re-extraction is required when the head changes.
- Probe training becomes a CPU-affordable operation of seconds, which makes the grid search (§8) and the three-seed repetition (§8) tractable.
- Caching a single deterministic view removes random augmentation from the probe stage. This is the standard frozen linear-probe protocol in the SSL literature and is consistent with finding F5. If augmentation is later judged necessary, cache *K* deterministic augmented views per sample.

---

## 5. Experiment Matrix

| ID | Question | Arms | Prerequisite |
|---|---|---|---|
| **E1** | Which checkpoints merit adaptation? | All candidates, frozen, identical cached probe | None — run first |
| **E2** | Does reconstruction or latent prediction adapt better? | VideoMAE vs V-JEPA 2, SAGES train only, fixed budget | E1 |
| **E3** | Does corpus composition matter? | LC-specific / general MIS / balanced mixture, fixed budget | E2 winner; external dataset access |
| **E4** | Does adaptation budget matter? | 1× vs 4× sample budget | E3 winner |
| **E5** | Does aggregation matter? | Uniform pooling vs attentive pooling, applied to both arms | Cached features only |

**E1, E2 and E5 constitute a complete and defensible dissertation.** They depend only on data already held, they answer the stated research question, and they are independent of any external data-use agreement resolving in time. E3 and E4 are upside.

Given an 8,000-word limit, plan to deliver E1, E2 and E5; treat E3 as the stretch objective and E4 as contingent.

---

## 6. Defining the Corpus Fraction

The original exp2–exp5 proposal confounded corpus *composition* with corpus *quantity*, so a difference between arms could not be attributed to either.

### Resolution

**Hold the SSL sample budget fixed across all composition arms.** Define the budget as a fixed number of optimisation steps multiplied by batch size, i.e. a fixed number of clips observed. Vary only the pool from which clips are drawn. Composition then becomes the sole manipulated variable, and quantity is tested separately in E4.

### Sampling unit

Sample at the **procedure level**, stratified by source dataset, with a **fixed clip quota per procedure**.

- Not total frames: Cholec80 procedures exceed thirty minutes while SAGES clips are ninety seconds, so a frame-count criterion would allow one dataset to dominate silently.
- Not raw video count, for the same reason.

Procedure-level sampling with a per-procedure quota makes the candidate corpora equal in size by construction, and isolates patient-level and institutional diversity as the variable actually under study.

### Leakage control

Only designated training procedures may contribute to continued SSL. Validation and test procedures are excluded from pretraining. This must be enforced by an assertion in the corpus-construction script, not by convention:

```text
assert set(ssl_video_ids) & (set(val_video_ids) | set(test_video_ids)) == set()
```

Datasets derived from shared source procedures are treated as a single overlap group. This applies particularly to Cholec80-derived collections.

---

## 7. Evaluation Protocol

Fixed before any Track 2 work begins.

**Primary metric:** mean average precision across C1, C2, C3.
**Secondary metrics:** per-criterion AP, ROC-AUC, balanced accuracy at a validation-tuned per-criterion threshold.

**Threshold policy.** Balanced accuracy is reported only at thresholds selected on the validation split, never at a fixed 0.5 cutoff, for the reason given in F3.

**Model selection.** Validation mAP only. The test split remains locked until the final configuration is selected. Note that the existing test evaluations of 9 July 2026 must be treated as having consumed one look at the test set; this should be stated in the methods section.

**Statistical reporting.** Every configuration entering a results table is run with **three seeds minimum**, reported as mean ± standard deviation. Adaptation gain is a difference of two noisy quantities: if seed variance on validation mAP is on the order of ±0.015 and the measured gain is 0.008, no effect has been demonstrated. On cached features this repetition is nearly free.

**Effective sample size.** Endoscapes provides 6,960 annotated training frames drawn from only **120 training videos**. Frames within a procedure are strongly correlated in patient, anatomy, camera, and illumination. The effective sample size is far closer to 120 than to 6,960. Two consequences follow: full fine-tuning of an 86M-parameter encoder will overfit substantially faster than the frame count suggests, and validation mAP itself carries high variance because it rests on 41 videos.

---

## 8. Hyperparameter Protocol

Three tiers with distinct budgets and distinct rules.

### Tier 1 — Probe hyperparameters

Searched exhaustively on cached features.

```text
lr      ∈ {1e-4, 3e-4, 1e-3, 3e-3}
wd      ∈ {0, 1e-4, 1e-2}
dropout ∈ {0.0, 0.1}
```

**Rule: the identical grid is run for every encoder, and each encoder's best result is reported.** A single fixed learning rate across arms is not acceptable. Feature norms differ materially between reconstruction-trained and latent-prediction-trained encoders, so a fixed learning rate silently advantages whichever feature scale it happens to suit, and that confound propagates directly into adaptation gain. Equal search effort per arm is the defensible protocol.

### Tier 2 — SSL pretraining hyperparameters

**Not searched.** A single run costs twelve or more GPU-hours. Use the published recipe from each upstream repository, adjust batch size to fit V100 memory, scale the learning rate by the linear scaling rule, and change nothing else. State explicitly in the methods section that SSL hyperparameters were not tuned, and why.

### Tier 3 — Fine-tuning hyperparameters

The dominant hyperparameter for ViT fine-tuning is layer-wise learning-rate decay, not base learning rate.

```text
llrd            ∈ {0.65, 0.75, 0.9}
unfrozen blocks ∈ {3, 6, 12}
```

All else fixed. Partial unfreezing with LLRD is preferred over full fine-tuning given the effective sample size noted in §7.

---

## 9. Aggregation Heads

Two heads, both trained on the same cached token grids, both applied identically to every arm.

### Head A — Linear probe

```text
token grid [N, d] → mean over N → LayerNorm(d) → Dropout → Linear(d, 3)
```

### Head B — Attentive probe (attention-based MIL)

Given instance embeddings `h_1 … h_N` from the token grid:

```text
a_i = softmax_i( wᵀ tanh(V h_i) )
z   = Σ_i a_i h_i
     → LayerNorm(d) → Dropout → Linear(d, 3)
```

### Rationale

Uniform mean pooling is exactly this model with attention weights frozen at `1/N`. Attentive pooling is therefore a strict generalisation costing a few thousand additional parameters. It addresses F2 directly, since learned attention can downweight stale timepoints in a 20-second clip whose label describes only its final frame. The attention weights are additionally interpretable as a saliency map over patches and timepoints, which supports a figure showing *where* the model attends for each criterion.

This is also the established protocol rather than an invention: V-JEPA 2's published evaluation uses an attentive probe on frozen representations, and SMIL's gains derive from an equivalent MIL aggregation mechanism.

**Constraint: attentive pooling is fixed for both arms from the outset.** Introducing it to one arm after observing results would destroy the comparison. Fixed in advance, it strengthens both arms while leaving the difference between them interpretable.

---

## 10. Model Inventory

### Frozen baselines (E1)

| Objective family | Level | Checkpoint | Params | Dim |
|---|---|---|---:|---:|
| Self-distillation | Image | DINOv2 ViT-B/14 | 86M | 768 |
| Self-distillation | Image | DINOv3 ViT-B/16 | 86M | 768 |
| Masked reconstruction | Image | MAE ViT-B/16 (IN1K) | 86M | 768 |
| Masked reconstruction | Video | VideoMAE ViT-B/16 (K400) | 87M | 768 |
| Masked reconstruction | Video | VideoMAE ViT-L/16 (K400) | 304M | 1024 |
| Latent prediction | Video | V-JEPA 2.1 ViT-B (distilled) | 87M | 768 |
| Latent prediction | Video | V-JEPA 2 ViT-L | 300M | 1024 |

Feature extraction is cheap, so the full inventory appears in E1 regardless of which arms proceed to adaptation. This yields the capacity trend at no additional cost.

### Adaptation arms (E2)

**Staged, not doubled.**

1. **Pilot: ViT-B vs ViT-B.** Validates the complete pipeline end to end at approximately one third of the compute.
2. **Headline: ViT-L vs ViT-L.** Runs if budget permits.

The ViT-L pair is scientifically cleaner, not merely larger. The V-JEPA 2.1 ViT-B checkpoint is a *distilled student* produced from ViT-g, whereas the natively pretrained V-JEPA 2 encoders begin at ViT-L. Resuming the JEPA objective on a distilled student is not equivalent to resuming it on an encoder trained under that objective: no corresponding EMA target-encoder checkpoint exists, so the teacher must be initialised from the student and the predictor from scratch. VideoMAE carries no equivalent complication, since its decoder is discarded and reinitialised by design. At ViT-L both encoders are natively pretrained under their own objectives and the asymmetry disappears.

If only the ViT-B pair is completed, the distillation asymmetry must be stated in the limitations regardless of the direction of the result.

### Exclusions and their justification

**Image-level JEPA is excluded.** The official I-JEPA release comprises ViT-H/14, ViT-H/16 at 448 px, and ViT-g/16. The smallest is 632M parameters, roughly seven times ViT-B per step, and breaks capacity matching against every other arm. Third-party ViT-B ports exist but are not official checkpoints, which undermines the released-checkpoint framing on which the design rests. The reconstruction-versus-prediction comparison is therefore located at video level, where the match is exact. This is stated explicitly as a scope limitation.

**DINOv3 adaptation is optional and capacity-unmatched.** DINO-style self-distillation depends on batch statistics for centering and sharpening, so gradient accumulation does not fully substitute for a large true batch, and multi-crop multiplies activation memory. Neither Colab Pro+ (A100 sessions that are neither guaranteed nor long enough for multi-day pretraining) nor AWS `g5.xlarge` (approximately USD 170 per week) is justified for an arm that does not answer the stated research question, which concerns reconstruction versus latent prediction. If a DINO trend indicator is wanted, adapt **DINOv3 ViT-S/16 (21M)** on a single V100 and label it explicitly as capacity-unmatched. DINOv3 ViT-B remains in E1 as a frozen baseline in all cases.

---

## 11. SSL Run Engineering

### Wall-clock segmentation

Jobs are limited to twelve hours. Use signal-based requeueing:

```bash
#SBATCH --signal=B:USR1@600
# trap USR1 → write checkpoint → scontrol requeue $SLURM_JOB_ID
```

The simpler alternative is a serialised job array, `--array=1-12%1`, with each element resuming from the same checkpoint file.

### Checkpoint contents

A resumable checkpoint must contain, without exception:

- context encoder weights
- **EMA target encoder weights** (JEPA and DINO — omitting this silently resets the teacher)
- predictor or decoder weights
- optimizer state
- **`GradScaler` state**
- LR scheduler state and global step counter
- RNG states

Omitting the `GradScaler` state causes the loss scale to re-warm from its initial value on every resume, producing a loss spike every twelve hours that is indistinguishable from genuine training instability.

### V100 constraints

The Kaya V100s are Volta, SM 7.0.

- `bf16` is unavailable. Use `fp16` autocast with `GradScaler` throughout. Code developed on the local RTX 4060 (Ada, SM 8.9) will run `bf16` successfully and then fail or silently degrade on Kaya; write `fp16` from the outset.
- FlashAttention-2 requires SM 8.0 and will not build. Use PyTorch `scaled_dot_product_attention`, which selects a memory-efficient kernel supported on Volta.
- Enable gradient checkpointing for ViT-L adaptation runs.

### Data layout

Group storage is Lustre, which handles a small number of large files well and a large number of small files poorly. Extracting SAGES to millions of individual JPEG files will make metadata operations the bottleneck and starve the GPU irrespective of model throughput.

- Shard the corpus into tar archives in WebDataset format, sized at approximately 1–2 GB, read sequentially.
- Pre-extract clips once at the target frame rate and resolution. Decoding MP4 on the fly with `decord` at 30 fps with eight CPUs per GPU will starve a V100.

---

## 12. Instrumentation

**Record of results.** `history.json` per run, aggregated by `scripts/collect_experiment_results.py`. This is the authoritative source for every number entering the dissertation: deterministic, diffable, and version-controlled. TensorBoard is not used.

**Early stopping.** On validation mAP, never on validation loss — the loss changes scale entirely under `pos_weight` and is not comparable across configurations. Applied to SSL pretraining and fine-tuning, where an epoch costs GPU-hours. Not applied to the cached probe, where the full schedule is affordable and the best epoch can be selected post hoc.

**Training curves.** Metric versus epoch, for monitoring and for diagnosing overfitting during fine-tuning. Plotting utility reads `history.json`.

**Learning curves.** Test mAP versus fraction of labelled training data, at {10%, 25%, 50%, 100%}, for adapted against baseline encoders. This is a dissertation figure, not monitoring: if domain-adaptive pretraining is effective, its advantage should be largest in the low-label regime. Subsample at the **video level**, not the frame level, for the reason given in §7. On cached features the entire sweep costs minutes.

---

## 13. Implementation Order

| Order | Artefact | Purpose |
|---|---|---|
| 1 | `models/encoders/base_encoder.py` | Abstract interface returning `[B, N, D]` token grids |
| 2 | `models/encoders/{videomae,vjepa2,dinov3,mae}_encoder.py` | Encoder wrappers against that interface |
| 3 | `scripts/extract_features.py` | Token-grid extraction to sharded fp16 storage |
| 4 | `models/heads/attentive_head.py` | Attention-based MIL head |
| 5 | `train/train_probe_cached.py` | Grid search and three-seed repetition over cached features |
| 6 | `eval/tune_thresholds.py` | Per-criterion validation threshold selection (resolves F3) |
| 7 | `scripts/plot_curves.py` | Training and learning curves from `history.json` |
| 8 | `configs/ssl/*.yaml` | SSL configurations, version-controlled in this repository |
| 9 | `scripts/*.sbatch` | Requeue-capable SSL launch scripts |

Items 1–7 are executable locally on the RTX 4060 and require no cluster access. Item 3 requires GPU time proportional to a single forward pass over each dataset.

---

## 14. Open Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Adaptation gain falls within seed variance | Track 1 claim unsupported | Three-seed minimum; report variance; a null result under a controlled protocol remains reportable |
| External dataset access does not arrive in time | E3 unachievable | E1, E2, E5 depend only on data already held |
| ViT-L adaptation exceeds available GPU hours | Headline comparison unavailable | ViT-B pilot runs first and is independently reportable |
| V-JEPA 2.1 ViT-B distillation asymmetry | Confounds objective comparison at ViT-B | Stage to ViT-L where both encoders are natively pretrained; state in limitations otherwise |
| Track 2 fails to approach 70 mAP | Performance ambition unmet | Track 1 protocol frozen in advance and independently reportable |