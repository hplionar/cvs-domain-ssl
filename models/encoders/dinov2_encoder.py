from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


DINO_MODEL_NAMES = {
    "small": "dinov2_vits14",
    "base": "dinov2_vitb14",
    "large": "dinov2_vitl14",
    "giant": "dinov2_vitg14",
}


class DINOv2Encoder(nn.Module):
    """
    DINOv2 image encoder wrapper.

    Input:
        images: Tensor of shape [B, 3, H, W]

    Output:
        features: Tensor of shape [B, feature_dim]

    Notes:
        This wrapper uses torch.hub to load the official DINOv2 models.
        For the first baseline, use variant='base'.
    """

    def __init__(
        self,
        variant: Literal["small", "base", "large", "giant"] = "base",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if variant not in DINO_MODEL_NAMES:
            raise ValueError(
                f"Unknown DINOv2 variant: {variant}. "
                f"Available variants: {list(DINO_MODEL_NAMES.keys())}"
            )

        self.variant = variant
        self.model_name = DINO_MODEL_NAMES[variant]

        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=pretrained,
        )

        self.feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        feature_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        return feature_dims[self.variant]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder(x)

        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected DINOv2 output to be a Tensor, got {type(output)}")

        if output.ndim != 2:
            raise ValueError(
                f"Expected DINOv2 features with shape [B, D], got {tuple(output.shape)}"
            )

        return output