from __future__ import annotations

import torch
import torch.nn as nn


class LinearCVSHead(nn.Module):
    """
    Linear multi-label classification head for CVS criterion prediction.

    Input:
        features: Tensor of shape [batch_size, feature_dim]

    Output:
        logits: Tensor of shape [batch_size, 3]
                corresponding to C1, C2, and C3.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        layers = []

        if use_layernorm:
            layers.append(nn.LayerNorm(input_dim))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(input_dim, num_labels))

        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape [batch_size, feature_dim], "
                f"but got {tuple(features.shape)}"
            )

        return self.head(features)