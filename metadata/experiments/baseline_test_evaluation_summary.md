# Baseline Test Evaluation Summary

Date: 9 July 2026

This summary records held-out test-set results for the frozen DINOv2-B CVS baselines.

| Experiment | Dataset | Input | Loss | Test mAP | Test mean AUC | Test mean BAcc |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Exp001 | Endoscapes | Single frame | Normal BCE | 0.4142 | 0.7030 | 0.5293 |
| Exp002 | Endoscapes | Single frame | Weighted BCE | 0.4066 | 0.6977 | 0.6243 |
| Exp003 | SAGES | Single frame | Normal BCE | 0.4141 | 0.7252 | 0.5794 |
| Exp004 | SAGES | Single frame | Weighted BCE | 0.4220 | 0.7323 | 0.6605 |
| Exp005 | SAGES | 5-frame sparse clip mean-pool | Normal BCE | 0.3636 | 0.6807 | 0.5449 |
| Exp006 | SAGES | 5-frame sparse clip mean-pool | Weighted BCE | 0.3661 | 0.6869 | 0.6264 |

## Interpretation

The strongest Endoscapes mAP baseline is Exp001, using single-frame DINOv2-B with normal BCE. The strongest Endoscapes balanced-accuracy baseline is Exp002, using weighted BCE.

The strongest SAGES baseline is Exp004, using single-frame DINOv2-B with weighted BCE.

The 5-frame sparse clip mean-pooling baselines underperform the single-frame SAGES baselines. This suggests that naive temporal averaging does not provide effective temporal modelling for CVS prediction and may dilute target-frame evidence.

Weighted BCE consistently improves mean balanced accuracy, confirming that class imbalance is important for CVS criterion prediction.
