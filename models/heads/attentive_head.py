"""Pooling heads for CVS classification over token grids.

Encoders in this project emit ``[B, N, D]`` token grids, so aggregation happens
here rather than inside the encoder. Three heads are provided, and the same head
must be applied to every arm of a comparison:

``MeanPoolHead``
    Uniform average over tokens. The parsimonious baseline.

``AttentivePoolHead``
    Gated attention-based multiple instance learning (Ilse et al., 2018). Each
    token is an instance, the grid is a bag, and only the bag carries a CVS
    label. Learned attention weights replace the uniform ``1/N``.

``FusionHead``
    Two parallel branches over the same frozen features: a global branch that
    takes the encoder's ``[CLS]`` token, and a MIL branch identical in form to
    ``AttentivePoolHead``. Their outputs are concatenated into a ``2D`` vector
    and read by one linear classifier.

Mean pooling is the special case of attentive pooling with attention frozen at
``1/N``, so the attentive head is a strict generalisation costing a few thousand
parameters. This is verified in ``tests/test_heads.py`` rather than asserted.

``FusionHead`` is in turn a strict generalisation of *both*: zeroing the global
half of its classifier recovers ``AttentivePoolHead`` exactly, and zeroing the
MIL half recovers a linear probe on ``[CLS]``. Both reductions are verified in
``tests/test_heads.py``. This matters for interpreting a comparison: the fusion
head cannot lose on *training* fit, so any loss on validation is a
generalisation cost, and any win is evidence that the two branches carry
complementary information rather than that one has more capacity.

Provenance of the fusion design
-------------------------------
The two-branch structure follows the Methods section of SMIL (Wang et al., 2026,
IJCARS, DOI 10.1007/s11548-026-03580-9): global ``[CLS]`` context ``h_ctx``,
gated-attention MIL over patch tokens ``h_MIL = sum_i alpha_i h_i``, fused as
``h_fused = [h_ctx ; h_MIL]``, then linear + sigmoid over the three criteria
under BCE. The paper's released code does not import, so the paper text is the
only specification; this implementation is not a reproduction of SMIL and must
not be described as one. SMIL's backbone is DINO self-distillation initialised
from Endo-FM, not any checkpoint used here, so only the *head design* transfers.

Two deliberate departures from the paper, both recorded in ``head_config()``:

- Dropout is applied to the fused vector immediately before the classifier,
  matching the other heads in this module, rather than "inside the MIL module".
  Dropout is a searched hyperparameter, so its placement is a design choice that
  the grid partly absorbs.
- LayerNorm is applied *per branch* before concatenation, not once over the
  concatenated vector. A shared LayerNorm computes one mean and variance across
  both halves, so whichever branch has the larger activation scale dominates the
  statistics and the other is compressed. ``[CLS]`` and pooled patch tokens do
  not have matched norms in a ViT, so this is not a hypothetical. Set
  ``use_layernorm=False`` to remove it entirely.

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

from typing import Any, Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CRITERIA = 3

#: Where ``FusionHead`` gets its global context vector ``h_ctx``.
#:
#: ``"cls"``
#:     The encoder's ``[CLS]`` token, i.e. ``prefix[:, prefix_index]``. This is
#:     what SMIL specifies. Requires an encoder that emits prefix tokens.
#: ``"patch_mean"``
#:     The mean of the patch tokens. The substitute for architectures with no
#:     prefix token at all -- VideoMAE and V-JEPA 2 both return ``prefix=None``.
#:
#: This is a *construction-time* choice, never a runtime fallback. A head built
#: with ``"cls"`` that is handed ``prefix=None`` raises rather than quietly
#: switching, because the two variants are different models and a silent switch
#: would make an encoder comparison incomparable without leaving a trace.
GlobalSource = Literal["cls", "patch_mean"]


def _gated_attention(
    h: torch.Tensor,
    v: nn.Linear,
    u: nn.Linear,
    w: nn.Linear,
) -> torch.Tensor:
    """Gated attention weights, ``[B, N, D] -> [B, branches, N]``.

    Shared by ``AttentivePoolHead`` and ``FusionHead`` so that the MIL branch of
    the fusion head is the same function, not a re-derivation of it. If these
    drifted apart, a fusion-versus-attention comparison would silently be
    measuring two differences instead of one.
    """
    gated = torch.tanh(v(h)) * torch.sigmoid(u(h))
    scores = w(gated)                                # [B, N, branches]
    return F.softmax(scores, dim=1).transpose(1, 2)  # [B, branches, N]


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

    def forward(
        self, tokens: torch.Tensor, prefix: torch.Tensor | None = None
    ) -> HeadOutput:
        """``prefix`` is accepted and ignored.

        Every head takes the same ``(tokens, prefix)`` signature so the probe
        trainer has one call site. This head reads patch tokens only, by
        definition; ``head_config()["uses_prefix"]`` records that.
        """
        _validate_tokens(tokens, self.input_dim)
        pooled = tokens.mean(dim=1)
        return HeadOutput(logits=self.classifier(self.dropout(self.norm(pooled))), attention=None)

    def head_config(self) -> dict[str, Any]:
        """Everything that distinguishes this head from another of its kind.

        Written into ``results.json`` so that two runs can be checked for
        comparability instead of assumed comparable.
        """
        return {
            "kind": "mean",
            "input_dim": self.input_dim,
            "uses_prefix": False,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

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
        return _gated_attention(h, self.attention_v, self.attention_u, self.attention_w)

    def forward(
        self, tokens: torch.Tensor, prefix: torch.Tensor | None = None
    ) -> HeadOutput:
        """``prefix`` is accepted and ignored; see ``MeanPoolHead.forward``."""
        attention = self.attention_weights(tokens)
        pooled = torch.bmm(attention, tokens)  # [B, branches, D]
        pooled = self.dropout(self.norm(pooled))

        if self.num_branches == 1:
            logits = self.classifier(pooled.squeeze(1))
        else:
            logits = (pooled * self.classifier_weight).sum(-1) + self.classifier_bias

        return HeadOutput(logits=logits, attention=attention)

    def head_config(self) -> dict[str, Any]:
        return {
            "kind": "attentive",
            "input_dim": self.input_dim,
            "hidden_dim": self.attention_v.out_features,
            "num_branches": self.num_branches,
            "pre_norm": not isinstance(self.pre_norm, nn.Identity),
            "layernorm": not isinstance(self.norm, nn.Identity),
            "uses_prefix": False,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_branches={self.num_branches}, "
            f"hidden_dim={self.attention_v.out_features}"
        )


class FusionHead(nn.Module):
    """Global context concatenated with gated-attention MIL, then a classifier.

    Following SMIL (Wang et al., 2026, IJCARS), two branches read the same frozen
    encoder output:

    .. math::
        h_{\\text{ctx}} = \\text{prefix}[:, k], \\qquad
        h_{\\text{MIL}} = \\sum_i \\alpha_i h_i, \\qquad
        h_{\\text{fused}} = [\\,h_{\\text{ctx}} \\; ; \\; h_{\\text{MIL}}\\,]

    with :math:`\\alpha` the same gated attention as ``AttentivePoolHead`` and
    :math:`h_{\\text{fused}} \\in \\mathbb{R}^{2D}`. A single linear layer maps
    :math:`h_{\\text{fused}}` to the three criteria; training is BCE on the
    logits, so sigmoid lives in the loss and the metric, not here.

    The motivating claim is that the branches are complementary: ``[CLS]`` is a
    whole-image summary that has seen every patch through self-attention, while
    the MIL branch is a sparse, learned reweighting of patches. Whether that
    complementarity survives on a frozen encoder is an empirical question, which
    is the point of running this head against the other two on one cache.

    Parameters
    ----------
    hidden_dim:
        Width of the gated-attention layer. SMIL states 512; the default here is
        512 for that reason, but a comparison against ``AttentivePoolHead`` must
        use the *same* value in both arms or a fusion win is confounded with
        extra attention capacity.
    global_source:
        ``"cls"`` or ``"patch_mean"``. See ``GlobalSource``. Not a fallback:
        ``"cls"`` with ``prefix=None`` raises.
    prefix_index:
        Which prefix token is ``[CLS]``. Zero for every encoder in this project;
        DINOv3's register tokens follow ``[CLS]`` and are *not* spatial, so they
        must never be pooled as patches or read as context.
    num_branches:
        As ``AttentivePoolHead``. With three branches, ``h_ctx`` is shared across
        branches and only ``h_MIL`` differs, since there is one ``[CLS]`` token.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int = NUM_CRITERIA,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        num_branches: Literal[1, 3] = 1,
        use_layernorm: bool = True,
        pre_norm: bool = True,
        global_source: GlobalSource = "cls",
        prefix_index: int = 0,
    ) -> None:
        super().__init__()
        if num_branches not in (1, num_labels):
            raise ValueError(
                f"num_branches must be 1 (shared) or {num_labels} (per-criterion), "
                f"got {num_branches}."
            )
        if global_source not in ("cls", "patch_mean"):
            raise ValueError(
                f"global_source must be 'cls' or 'patch_mean', got "
                f"{global_source!r}. There is no automatic choice: the two give "
                f"different models and the choice belongs in the run record."
            )
        if prefix_index < 0:
            raise ValueError(f"prefix_index must be non-negative, got {prefix_index}.")

        self.input_dim = input_dim
        self.num_labels = num_labels
        self.num_branches = num_branches
        self.global_source = global_source
        self.prefix_index = prefix_index
        self.fused_dim = 2 * input_dim

        self.pre_norm = nn.LayerNorm(input_dim) if pre_norm else nn.Identity()

        self.attention_v = nn.Linear(input_dim, hidden_dim)
        self.attention_u = nn.Linear(input_dim, hidden_dim)
        self.attention_w = nn.Linear(hidden_dim, num_branches)

        # Separate norms, applied before concatenation. See the module docstring
        # for why a single LayerNorm over the 2D vector is the wrong shape here.
        self.norm_ctx = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        self.norm_mil = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if num_branches == 1:
            self.classifier = nn.Linear(self.fused_dim, num_labels)
        else:
            self.classifier_weight = nn.Parameter(torch.empty(num_labels, self.fused_dim))
            self.classifier_bias = nn.Parameter(torch.zeros(num_labels))
            nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    # -- branches ---------------------------------------------------------

    def attention_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, num_branches, N]`` weights summing to one over tokens."""
        _validate_tokens(tokens, self.input_dim)
        h = self.pre_norm(tokens)
        return _gated_attention(h, self.attention_v, self.attention_u, self.attention_w)

    def global_context(
        self, tokens: torch.Tensor, prefix: torch.Tensor | None
    ) -> torch.Tensor:
        """``h_ctx``, shape ``[B, D]``."""
        if self.global_source == "patch_mean":
            return tokens.mean(dim=1)

        if prefix is None:
            raise ValueError(
                "FusionHead was built with global_source='cls' but received "
                "prefix=None. This encoder emits no prefix token (VideoMAE and "
                "V-JEPA 2 return prefix=None), or the cache was extracted with "
                "--reduction spatial/full, which skips prefix.npy. Build the "
                "head with global_source='patch_mean' to use mean-pooled patch "
                "tokens as the global branch instead, and record that choice: "
                "it is a different model, not the same one."
            )
        if prefix.ndim != 3:
            raise ValueError(
                f"Expected prefix of shape [B, P, D], got {tuple(prefix.shape)}."
            )
        if prefix.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"prefix batch {prefix.shape[0]} does not match tokens batch "
                f"{tokens.shape[0]}."
            )
        if prefix.shape[2] != self.input_dim:
            raise ValueError(
                f"prefix dimension {prefix.shape[2]} does not match head "
                f"input_dim {self.input_dim}."
            )
        if prefix.shape[1] <= self.prefix_index:
            raise ValueError(
                f"prefix_index {self.prefix_index} is out of range for a prefix "
                f"with {prefix.shape[1]} token(s)."
            )
        return prefix[:, self.prefix_index]

    def fused_representation(
        self, tokens: torch.Tensor, prefix: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``([B, num_branches, 2D], [B, num_branches, N])``.

        Concatenation order is ``[h_ctx ; h_MIL]``, so the first ``D`` columns of
        the classifier read the global branch and the last ``D`` read MIL. Tests
        and any post-hoc analysis of branch contribution depend on that order.
        """
        attention = self.attention_weights(tokens)
        h_mil = torch.bmm(attention, tokens)                    # [B, branches, D]
        h_ctx = self.global_context(tokens, prefix)             # [B, D]
        h_ctx = h_ctx.unsqueeze(1).expand(-1, self.num_branches, -1)

        fused = torch.cat([self.norm_ctx(h_ctx), self.norm_mil(h_mil)], dim=-1)
        return fused, attention

    # -- forward ----------------------------------------------------------

    def forward(
        self, tokens: torch.Tensor, prefix: torch.Tensor | None = None
    ) -> HeadOutput:
        fused, attention = self.fused_representation(tokens, prefix)
        fused = self.dropout(fused)

        if self.num_branches == 1:
            logits = self.classifier(fused.squeeze(1))
        else:
            logits = (fused * self.classifier_weight).sum(-1) + self.classifier_bias

        return HeadOutput(logits=logits, attention=attention)

    def head_config(self) -> dict[str, Any]:
        return {
            "kind": "fusion",
            "source": "SMIL (Wang et al., 2026) Methods; not a reproduction",
            "input_dim": self.input_dim,
            "fused_dim": self.fused_dim,
            "hidden_dim": self.attention_v.out_features,
            "num_branches": self.num_branches,
            "global_source": self.global_source,
            "prefix_index": self.prefix_index if self.global_source == "cls" else None,
            "uses_prefix": self.global_source == "cls",
            "pre_norm": not isinstance(self.pre_norm, nn.Identity),
            "layernorm": not isinstance(self.norm_ctx, nn.Identity),
            "layernorm_placement": "per_branch_before_concat",
            "dropout_placement": "fused_vector_before_classifier",
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, fused_dim={self.fused_dim}, "
            f"num_branches={self.num_branches}, "
            f"hidden_dim={self.attention_v.out_features}, "
            f"global_source={self.global_source!r}"
        )


def build_head(
    kind: str,
    input_dim: int,
    *,
    num_labels: int = NUM_CRITERIA,
    dropout: float = 0.0,
    hidden_dim: int = 128,
    num_branches: int = 1,
    global_source: GlobalSource = "cls",
) -> nn.Module:
    """Construct a head by name, so experiment configs carry a string.

    ``hidden_dim`` applies to both ``attentive`` and ``fusion``. It is *not*
    defaulted to SMIL's 512 here: the two attention-based heads must be given
    the same width for a comparison between them to isolate the fusion design,
    and the probe trainer therefore sweeps it identically across both arms.
    ``global_source`` is ignored by the heads that do not read prefix tokens.
    """
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
    if kind in {"fusion", "fused", "smil"}:
        return FusionHead(
            input_dim,
            num_labels=num_labels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_branches=num_branches,
            global_source=global_source,
        )
    raise ValueError(
        f"Unknown head {kind!r}. Expected 'mean', 'attentive' or 'fusion'."
    )


#: Head names whose aggregation is learned, so pooling cannot be precomputed and
#: ``hidden_dim`` is a real hyperparameter. Kept here rather than in the trainer
#: so that adding a head does not require editing two files.
ATTENTION_HEADS = frozenset({"attentive", "attn", "mil", "abmil", "fusion", "fused", "smil"})

#: Head names whose global branch reads encoder prefix tokens.
PREFIX_HEADS = frozenset({"fusion", "fused", "smil"})


__all__ = [
    "ATTENTION_HEADS",
    "AttentivePoolHead",
    "FusionHead",
    "GlobalSource",
    "HeadOutput",
    "MeanPoolHead",
    "NUM_CRITERIA",
    "PREFIX_HEADS",
    "build_head",
]