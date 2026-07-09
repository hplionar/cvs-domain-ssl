# SAGES Test Evaluation Summary

Date: 8 July 2026

The best validation checkpoints from Exp003–Exp006 were evaluated on the held-out SAGES internal test split. The split contains 70 videos and 1,260 labelled frame/timepoint samples.

| Experiment | Input | Loss | Test mAP | Test mean AUC | Test mean BAcc |
| --- | --- | --- | ---: | ---: | ---: |
| Exp003 | Single frame | Normal BCE | 0.4141 | 0.7252 | 0.5794 |
| Exp004 | Single frame | Weighted BCE | 0.4220 | 0.7323 | 0.6605 |
| Exp005 | 5-frame sparse clip mean-pool | Normal BCE | 0.3636 | 0.6807 | 0.5449 |
| Exp006 | 5-frame sparse clip mean-pool | Weighted BCE | 0.3661 | 0.6869 | 0.6264 |

## Interpretation

The strongest SAGES test-set baseline is Exp004: frozen DINOv2-B single-frame input with weighted BCE.

The 5-frame sparse clip mean-pooling baselines underperform the single-frame baselines on test mAP and mean balanced accuracy. This suggests that simple averaging over nearby sparse-frame features does not provide effective temporal modelling for CVS prediction. Temporal context may still be useful, but it likely requires an explicit temporal model or video representation learner rather than naive feature averaging.

Weighted BCE improves balanced accuracy in both frame and clip settings, confirming that class imbalance remains important for CVS criterion prediction.
