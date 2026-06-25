from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.heads.linear_head import LinearCVSHead


class CVSModel(nn.Module):
    """
    Generic CVS classifier wrapper.

    The encoder is expected to produce image or clip features.
    The head maps those features to 3 CVS logits: C1, C2, C3.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_labels: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.feature_dim = feature_dim
        self.freeze_encoder = freeze_encoder

        self.head = LinearCVSHead(
            input_dim=feature_dim,
            num_labels=num_labels,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )

        if freeze_encoder:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()
        self.freeze_encoder = False

    def train(self, mode: bool = True) -> "CVSModel":
        """
        Keep encoder in eval mode when frozen, while allowing the head to train.
        """
        super().train(mode)

        if self.freeze_encoder:
            self.encoder.eval()

        return self

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract a [batch_size, feature_dim] tensor from different encoder output styles.
        """

        if hasattr(self.encoder, "forward_features"):
            output: Any = self.encoder.forward_features(x)
        else:
            output = self.encoder(x)

        # Hugging Face-style output object
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            features = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            features = output.last_hidden_state
        elif isinstance(output, dict):
            if "features" in output:
                features = output["features"]
            elif "pooler_output" in output:
                features = output["pooler_output"]
            elif "last_hidden_state" in output:
                features = output["last_hidden_state"]
            else:
                raise ValueError(
                    f"Encoder returned a dict without a recognised feature key: "
                    f"{list(output.keys())}"
                )
        elif isinstance(output, (tuple, list)):
            features = output[0]
        else:
            features = output

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Expected encoder features to be a torch.Tensor, got {type(features)}"
            )

        # If encoder returns token features [B, tokens, D], mean-pool tokens.
        if features.ndim == 3:
            features = features.mean(dim=1)

        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape [batch_size, feature_dim], "
                f"but got {tuple(features.shape)}"
            )

        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.feature_dim}, "
                f"got {features.shape[1]}"
            )

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                features = self._extract_features(x)
        else:
            features = self._extract_features(x)

        logits = self.head(features)
        return logits