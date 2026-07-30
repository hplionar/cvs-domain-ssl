"""Pooling heads for CVS classification over token grids.

Encoders in this project emit ``[B, N, D]`` token grids, so aggregation happens
here rather than inside the encoder. Two heads are provided, and the same head
must be applied to every arm of a comparison:

``MeanPoolHead``
    Uniform average over tokens. The parsimonious baseline.

``AttentivePoolHead``
    Gated attention-based multiple instance learning (Ilse et al., 2018). Each
    token is an instance, the grid is a bag, and only the bag carries a CVS
    label. Learned attention weights replace the uniform ``1/N``.

Mean pooling is the special case of attentive pooling with attention frozen at
``1/N``, so the attentive head is a strict generalisation costing a few thousand
parameters. This is verified in ``tests/test_heads.py`` rather than asserted.

Why this matters here specifically. A SAGES clip spans 20 seconds at 5-second
frame spacing while its label describes only the final timepoint, so uniform
averaging dilutes the evidence with earlier dissection states — audit finding F2.
Learned attention can downweight stale tokens. The attention weights are also
directly interpretable as a saliency map over patches or timepoints, which
supports showing *where* the model looks for each criterion.

Protocol constraint: whichever head is used must be fixed before results are
inspected. Introducing attentive pooling to one arm after seeing its numbers
would destroy the comparison.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CRITERIA = 3


class HeadOutput(NamedTuple):
    """Logits, with attention retained for visualisation.

    Attributes
    ----------
    logits:
        ``[B, 3]`` logits for C1, C2, C3. Multi-label, so these go through
        ``BCEWithLogitsLoss`` and sigmoid, never softmax.
    attention:
        ``[B, num_branches, N]`` attention weights summing to one over ``N``,
        or ``None`` for heads without attention. Reshape the trailing axis to
        the encoder's ``token_layout.grid`` to obtain a saliency map.
    """

    logits: torch.Tensor
    attention: torch.Tensor | None = None


def _validate_tokens(tokens: torch.Tensor, expected_dim: int) -> None:
    if tokens.ndim != 3:
        raise ValueError(
            f"Expected token grid of shape [B, N, D], got {tuple(tokens.shape)}. "
            f"Encoders return grids; pooling belongs in the head."
        )
    if tokens.shape[2] != expected_dim:
        raise ValueError(
            f"Token dimension {tokens.shape[2]} does not match head input_dim "
            f"{expected_dim}."
        )


class MeanPoolHead(nn.Module):
    """Uniform mean over tokens, then a linear classifier.

    ``[B, N, D]`` -> mean over N -> LayerNorm -> Dropout -> Linear -> ``[B, 3]``
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int = NUM_CRITERIA,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, tokens: torch.Tensor) -> HeadOutput:
        _validate_tokens(tokens, self.input_dim)
        pooled = tokens.mean(dim=1)
        return HeadOutput(logits=self.classifier(self.dropout(self.norm(pooled))), attention=None)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}"


class AttentivePoolHead(nn.Module):
    """Gated attention MIL pooling, then a linear classifier.

    For token embeddings :math:`h_i`:

    .. math::
        a_i = \\operatorname{softmax}_i\\left(
            w^\\top \\left[\\tanh(V h_i) \\odot \\sigma(U h_i)\\right]
        \\right), \\qquad z = \\sum_i a_i h_i

    The gating term :math:`\\sigma(U h_i)` is the variant from Ilse et al.; it
    lets the network suppress instances that ``tanh`` alone would saturate.

    Parameters
    ----------
    num_branches:
        ``1`` learns a single attention map shared by all three criteria.
        ``3`` learns one map per criterion, which is the more faithful model —
        C1 concerns tubular structures, C2 clearance of the hepatocystic
        triangle, C3 the cystic plate, and these are different regions — and
        yields per-criterion saliency maps. It costs ``2 * hidden_dim`` extra
        parameters. Default is ``1`` for parsimony given the small number of
        independent training videos.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int = NUM_CRITERIA,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        num_branches: Literal[1, 3] = 1,
        use_layernorm: bool = True,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_branches not in (1, num_labels):
            raise ValueError(
                f"num_branches must be 1 (shared) or {num_labels} (per-criterion), "
                f"got {num_branches}."
            )

        self.input_dim = input_dim
        self.num_labels = num_labels
        self.num_branches = num_branches

        # Applied to tokens before attention. Frozen encoder outputs differ in
        # scale between objective families, and without normalisation a single
        # learning rate would suit one arm better than another.
        #
        # CAVEAT: LayerNorm is invariant to per-token magnitude. It maps any
        # constant vector to zero, so a uniformly-large token and a
        # uniformly-zero token become indistinguishable. Real ViT tokens vary
        # across the feature axis so their *pattern* survives, but if the
        # informative cue is activation magnitude rather than direction,
        # pre_norm discards it. Set pre_norm=False to retain it; the probe
        # trainer sweeps this.
        self.pre_norm = nn.LayerNorm(input_dim) if pre_norm else nn.Identity()

        self.attention_v = nn.Linear(input_dim, hidden_dim)
        self.attention_u = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, num_branches)

        self.norm = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if num_branches == 1:
            self.classifier = nn.Linear(input_dim, num_labels)
        else:
            # One classifier vector per criterion, each reading its own pooled
            # representation. Equivalent to a block-diagonal linear layer.
            self.classifier_weight = nn.Parameter(torch.empty(num_labels, input_dim))
            self.classifier_bias = nn.Parameter(torch.zeros(num_labels))
            nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def attention_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[B, num_branches, N]`` weights summing to one over tokens."""
        _validate_tokens(tokens, self.input_dim)
        h = self.pre_norm(tokens)
        gated = torch.tanh(self.attention_v(h)) * torch.sigmoid(self.attention_u(h))
        scores = self.attention_w(gated)              # [B, N, branches]
        return F.softmax(scores, dim=1).transpose(1, 2)  # [B, branches, N]

    def forward(self, tokens: torch.Tensor) -> HeadOutput:
        attention = self.attention_weights(tokens)
        pooled = torch.bmm(attention, tokens)  # [B, branches, D]
        pooled = self.dropout(self.norm(pooled))

        if self.num_branches == 1:
            logits = self.classifier(pooled.squeeze(1))
        else:
            logits = (pooled * self.classifier_weight).sum(-1) + self.classifier_bias

        return HeadOutput(logits=logits, attention=attention)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_branches={self.num_branches}, "
            f"hidden_dim={self.attention_v.out_features}"
        )


def build_head(
    kind: str,
    input_dim: int,
    *,
    num_labels: int = NUM_CRITERIA,
    dropout: float = 0.0,
    hidden_dim: int = 128,
    num_branches: int = 1,
) -> nn.Module:
    """Construct a head by name, so experiment configs carry a string."""
    kind = kind.lower()
    if kind in {"mean", "meanpool", "linear"}:
        return MeanPoolHead(input_dim, num_labels=num_labels, dropout=dropout)
    if kind in {"attentive", "attn", "mil", "abmil"}:
        return AttentivePoolHead(
            input_dim,
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_branches=num_branches,
        )
    raise ValueError(f"Unknown head {kind!r}. Expected 'mean' or 'attentive'.")


__all__ = [
    "AttentivePoolHead",
    "HeadOutput",
    "MeanPoolHead",
    "NUM_CRITERIA",
    "build_head",
]