"""Depth-weighted pooling head.

Reads pooled representations from several encoder layers and learns how much
each criterion should draw on each depth.

Motivation
----------
Feature extraction previously kept the final layer only. Every SAGES CVS
Challenge submission finds C2 markedly easier than C1 and C3 -- Farm scores
85.30 against 56.65 and 65.32, and the ordering holds across all thirteen
entries. C2 (hepatocystic triangle cleared) is a judgement about tissue absence
and texture; C1 (two and only two structures) requires counting and C3 (lower
third detached from the cystic plate) is a spatial relation. If those are
resolved at different levels of abstraction, a last-layer probe discards the
evidence for at least one of them.

This head makes that testable: after training, the learned weights read directly
as "C2 draws on depth 0.5, C1 on depth 1.0", or show that they do not differ.

Design
------
Spatial pooling happens first, so each layer becomes one vector per sample, and
the learned weights are over depth alone. Weighting tokens *and* depths jointly
would be more expressive but would not yield an interpretable per-criterion
depth profile, which is the point of the experiment. This ordering follows ELMo
(Peters et al., 2018), which computes a softmax-weighted sum of layer
representations with learned scalar weights; DenseFormer (Pagliardini et al.,
2024) does the same with fixed input-independent coefficients.

Per-layer normalisation is not optional. Measured on a VideoMAE ViT-B cache over
SAGES, pooled layer activations have standard deviations of 2.27, 2.39, 2.72 and
3.97 at relative depths 0.25 to 1.0. Combining them unnormalised would let the
deepest layer dominate any weighted sum before a single gradient step, so the
learned weights would describe magnitude rather than usefulness.

Note also that these layers are *pre*-LayerNorm block outputs, whereas the
``tokens.npy`` cache holds ``last_hidden_state``, which is post-LayerNorm. They
correlate at 0.96 but differ in scale by more than sixfold, so the ``last``
arm below uses the layer stack's own final entry rather than ``tokens.npy``.
Mixing the two would confound depth with normalisation.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DepthMode = Literal["last", "concat", "learned"]


class DepthHeadOutput(NamedTuple):
    """Logits, and the depth weights that produced them.

    ``weights`` is ``[num_criteria, num_layers]`` for the learned mode and
    ``None`` otherwise. It is returned rather than logged internally so that the
    probe can record it alongside the metrics: the weights are the result of
    this experiment, not a diagnostic of it.
    """

    logits: torch.Tensor
    weights: torch.Tensor | None = None


class DepthWeightedHead(nn.Module):
    """Classify from pooled per-layer representations.

    Parameters
    ----------
    feature_dim:
        Encoder width ``D``.
    num_layers:
        Number of cached depths ``L``.
    mode:
        ``"last"``     use only the deepest cached layer. Reproduces an ordinary
                       probe and is the control arm.
        ``"concat"``   concatenate all layers into a ``L*D`` vector. Gives the
                       classifier access to every depth but with ``L`` times the
                       parameters, so a win here is confounded with capacity.
        ``"learned"``  softmax-weighted sum over depths, learned separately per
                       criterion. Same parameter count as ``"last"`` apart from
                       ``num_criteria * num_layers`` weights, so a win is
                       attributable to depth selection rather than capacity.
    num_criteria:
        Three for CVS.
    dropout:
        Applied to the combined vector before the classifier, matching the other
        heads in this project.
    """

    def __init__(
        self,
        feature_dim: int,
        num_layers: int,
        *,
        mode: DepthMode = "learned",
        num_criteria: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if mode == "learned" and num_layers == 1:
            raise ValueError(
                "mode='learned' with a single cached layer has nothing to "
                "choose between; use mode='last'."
            )

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.mode = mode
        self.num_criteria = num_criteria

        # One LayerNorm per depth, not one shared across them. A shared norm
        # computes a single mean and variance over the concatenated layers, so
        # the layer with the largest activation scale sets the statistics and
        # compresses the others -- which is precisely the imbalance being
        # corrected.
        self.norms = nn.ModuleList(nn.LayerNorm(feature_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

        if mode == "last":
            self.classifier = nn.Linear(feature_dim, num_criteria)
            self.depth_logits = None
        elif mode == "concat":
            self.classifier = nn.Linear(feature_dim * num_layers, num_criteria)
            self.depth_logits = None
        elif mode == "learned":
            # Zero-initialised, so training starts from a uniform average over
            # depths. Any departure from uniform is then something the data
            # drove, which is what makes the final weights interpretable.
            self.depth_logits = nn.Parameter(torch.zeros(num_criteria, num_layers))
            # One classifier per criterion: each criterion mixes depths
            # differently, so a shared classifier would see a different input
            # distribution per output and the weights would not be comparable.
            self.classifier = nn.Linear(feature_dim, num_criteria)
        else:
            raise ValueError(f"Unknown mode {mode!r}.")

    def depth_weights(self) -> torch.Tensor | None:
        """Softmax weights over depth, ``[num_criteria, num_layers]``."""
        if self.depth_logits is None:
            return None
        return F.softmax(self.depth_logits, dim=-1)

    def forward(self, layers: torch.Tensor) -> DepthHeadOutput:
        """``layers`` is ``[B, L, D]``: already pooled over tokens."""
        if layers.ndim != 3:
            raise ValueError(
                f"Expected pooled layers [B, L, D], got shape {tuple(layers.shape)}. "
                f"Spatial pooling happens before this head, not inside it."
            )
        if layers.shape[1] != self.num_layers:
            raise ValueError(
                f"Head built for {self.num_layers} layers but received "
                f"{layers.shape[1]}."
            )
        if layers.shape[2] != self.feature_dim:
            raise ValueError(
                f"Head built for width {self.feature_dim} but received "
                f"{layers.shape[2]}."
            )

        normed = torch.stack(
            [norm(layers[:, i]) for i, norm in enumerate(self.norms)], dim=1
        )

        if self.mode == "last":
            combined = normed[:, -1]
            logits = self.classifier(self.dropout(combined))
            return DepthHeadOutput(logits=logits, weights=None)

        if self.mode == "concat":
            combined = normed.flatten(start_dim=1)
            logits = self.classifier(self.dropout(combined))
            return DepthHeadOutput(logits=logits, weights=None)

        # learned: a separate depth mixture per criterion, then that criterion's
        # row of the classifier. Written as an einsum over all criteria at once
        # rather than a loop, but equivalent to mixing and classifying each
        # criterion independently.
        weights = self.depth_weights()                       # [C, L]
        mixed = torch.einsum("cl,bld->bcd", weights, normed)  # [B, C, D]
        mixed = self.dropout(mixed)
        logits = torch.einsum("bcd,cd->bc", mixed, self.classifier.weight)
        logits = logits + self.classifier.bias
        return DepthHeadOutput(logits=logits, weights=weights)

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, num_layers={self.num_layers}, "
            f"mode={self.mode!r}, num_criteria={self.num_criteria}"
        )

    def head_config(self) -> dict[str, object]:
        """Recorded in results.json so an arm is reconstructable from it."""
        return {
            "head": "depth",
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "num_layers": self.num_layers,
            "num_criteria": self.num_criteria,
            "per_layer_layernorm": True,
            "note": (
                "Layers are pre-LayerNorm block outputs; tokens.npy is "
                "post-LayerNorm last_hidden_state. The two correlate at 0.96 "
                "but differ in scale, so mode='last' uses the layer stack's "
                "final entry rather than tokens.npy."
            ),
        }


__all__ = ["DepthHeadOutput", "DepthMode", "DepthWeightedHead"]
