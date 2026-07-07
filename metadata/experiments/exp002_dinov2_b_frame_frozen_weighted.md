# exp002_dinov2_b_frame_frozen_weighted

Date: 30 June 2026  
Encoder: DINOv2-B  
Input: single annotated Endoscapes frame  
Training: frozen encoder + linear CVS head  
Loss: BCEWithLogitsLoss with positive class weighting  
Train split: Endoscapes manually annotated train keyframes  
Validation split: Endoscapes manually annotated validation keyframes  
GPU: Tesla V100-PCIE-16GB  
Trainable parameters: 3,843  

## Positive class weighting

Training positive counts: [1088, 780, 1245]  
Training negative counts: [5872, 6180, 5715]  
pos_weight: [5.397059, 7.923077, 4.5903616]  

## Best by mAP

Epoch: 1  
mAP: 0.4771  
mean AUC: 0.8104  
mean BAcc: 0.7157  

C1 AP: 0.4238  
C2 AP: 0.5152  
C3 AP: 0.4922  

## Best by mean BAcc

Epoch: 5  
mAP: 0.4653  
mean AUC: 0.8191  
mean BAcc: 0.7226  

C1 BAcc: 0.6934  
C2 BAcc: 0.7428  
C3 BAcc: 0.7315  

## Interpretation

Compared with exp001 normal BCE, weighted BCE slightly reduced best mAP but substantially improved mean balanced accuracy.

Exp001 normal BCE achieved best mAP of 0.4834 and best mean BAcc of 0.5976. Exp002 weighted BCE achieved best mAP of 0.4771 and best mean BAcc of 0.7226.

This suggests that positive class weighting changes the decision behaviour of the linear head and improves sensitivity to rare positive CVS criteria. However, it does not improve ranking-based performance in this short frozen DINOv2-B baseline.

The result is useful because CVS labels are imbalanced. For future experiments, mAP should remain the main comparison metric, while balanced accuracy can be reported as a secondary threshold-based metric.
