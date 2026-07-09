# Endoscapes Test Evaluation Summary

Date: 9 July 2026

The best validation checkpoints from Exp001 and Exp002 were evaluated on the held-out Endoscapes test split. The test split contains 1,799 manually annotated CVS keyframes.

| Experiment | Input | Loss | Test mAP | Test mean AUC | Test mean BAcc |
| --- | --- | --- | ---: | ---: | ---: |
| Exp001 | Single frame | Normal BCE | 0.4142 | 0.7030 | 0.5293 |
| Exp002 | Single frame | Weighted BCE | 0.4066 | 0.6977 | 0.6243 |

## Interpretation

Exp001 achieved slightly higher test mAP, while Exp002 achieved substantially higher mean balanced accuracy.

This confirms the same pattern observed during validation: positive-class weighting improves threshold-balanced classification behaviour, but it may slightly reduce ranking-based average precision. For reporting, mAP should remain the primary ranking metric, while mean balanced accuracy should be reported as an important secondary metric.
