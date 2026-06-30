# exp001_dinov2_b_frame_frozen

Date: 26 June 2026  
Encoder: DINOv2-B  
Input: single frame  
Training: frozen encoder + linear CVS head  
Loss: BCEWithLogitsLoss  
Train split: Endoscapes manually annotated train keyframes  
Validation split: Endoscapes manually annotated validation keyframes  
GPU: Tesla V100-PCIE-16GB  
Trainable parameters: 3,843  

## Best by mAP

Epoch: 2  
mAP: 0.4834  
mean AUC: 0.8179  
mean BAcc: 0.5712  

C1 AP: 0.4366  
C2 AP: 0.5244  
C3 AP: 0.4891  

## Best by mean BAcc

Epoch: 5  
mAP: 0.4686  
mean AUC: 0.8198  
mean BAcc: 0.5976  

C1 BAcc: 0.5690  
C2 BAcc: 0.5952  
C3 BAcc: 0.6285  

## Interpretation

This run is the first successful engineering baseline. It validates the complete supervised downstream CVS pipeline using a frozen generic SSL image encoder. Performance is modest, which motivates imbalance-aware training, temporal modelling, stronger heads, and later V-JEPA/domain-adaptive SSL.
