# Dimension Transformation for Endoscapes and SAGES

## 1. Random crop and centre crop

Both datasets contain wide laparoscopic frames, while the pretrained models usually expect a fixed square input such as `224 × 224`.

### Random crop

A **random crop** selects a different `224 × 224` region each time an image is used during training.

```text
Resized image
approximately 256 × 455
        ↓
Randomly choose one 224 × 224 region
        ↓
Model input: 224 × 224
```

This acts as data augmentation because the model sees slightly different regions of the same frame across training epochs.

### Centre crop

A **centre crop** takes the `224 × 224` region from the middle of the resized image.

```text
Resized image
approximately 256 × 455
        ↓
Take the central 224 × 224 region
        ↓
Model input: 224 × 224
```

Centre cropping is normally used during validation and testing because it is deterministic: the same frame always produces the same input.

> Random crop is usually used for training, while centre crop is usually used for validation and testing.

---

## 2. Image-model transformation

### Endoscapes example

```text
Original frame
480 × 854 × 3
        ↓
Resize shorter side to about 256
approximately 256 × 455 × 3
        ↓
Random crop or centre crop
224 × 224 × 3
        ↓
Split into 16 × 16 patches
14 × 14 = 196 patches
        ↓
ViT-B encoder
196 × 768 contextual tokens
        ↓
Mean pooling over 196 tokens
768-dimensional representation
        ↓
Linear(768 → 3)
        ↓
C1, C2, and C3 logits
```

### SAGES example

```text
Original frame
720 × 1280 × 3
        ↓
Resize shorter side to about 256
approximately 256 × 455 × 3
        ↓
Random crop or centre crop
224 × 224 × 3
        ↓
Patch embedding and transformer encoder
N × d contextual tokens
        ↓
Mean pooling
d-dimensional representation
        ↓
Linear(d → 3)
```

After preprocessing, Endoscapes and SAGES frames have the same model input size even though their original resolutions differ.

---

## 3. Video-model transformation

Suppose a SAGES clip contains 16 sampled frames:

```text
Original clip
16 × 720 × 1280 × 3
        ↓
Resize and crop every frame
16 × 224 × 224 × 3
        ↓
Create spatiotemporal tubelets
        ↓
Video transformer encoder
N × d contextual tokens
        ↓
Mean pooling over all spatial and temporal tokens
d-dimensional video representation
        ↓
Linear(d → 3)
        ↓
C1, C2, and C3 logits
```

For VideoMAE ViT-B, `d = 768`.  
For V-JEPA ViT-L, `d = 1024`.

---

## 4. Model-specific dimensions

| Model | Encoder output before pooling | Pooled representation | Classifier |
|---|---|---|---|
| MAE ViT-B | `N × 768` | `768` | `Linear(768, 3)` |
| I-JEPA ViT-H | `N × 1280` | `1280` | `Linear(1280, 3)` |
| VideoMAE ViT-B | `N × 768` | `768` | `Linear(768, 3)` |
| V-JEPA ViT-L | `N × 1024` | `1024` | `Linear(1024, 3)` |

Here:

- `N` is the number of image patches or video tubelets;
- `d` is the feature dimension of each token;
- mean pooling reduces `N × d` into one `d`-dimensional vector;
- the final linear layer converts that vector into three CVS logits.

---

## 5. Overall summary

```text
Raw Endoscapes or SAGES image/video
        ↓
Resize
        ↓
Random crop for training
or centre crop for evaluation
        ↓
Create patches or tubelets
        ↓
Transformer encoder produces N × d tokens
        ↓
Mean pooling produces one d-dimensional representation
        ↓
Linear(d → 3)
        ↓
Predict C1, C2, and C3
```
