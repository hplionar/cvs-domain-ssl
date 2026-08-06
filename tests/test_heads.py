"""Tests for pooling heads.

The load-bearing test here is that attentive pooling reduces exactly to mean
pooling when attention is uniform: the claim that it is a strict generalisation
is what justifies applying it to both arms of E2 without biasing the comparison.
"""

from __future__ import annotations

import pytest
import torch

from models.heads.attentive_head import (
    AttentivePoolHead,
    FusionHead,
    HeadOutput,
    MeanPoolHead,
    build_head,
)

B, N, D = 4, 196, 64
P = 2  # [CLS] plus one register token, as DINOv3 emits


@pytest.fixture
def tokens() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, N, D)


@pytest.fixture
def prefix() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, P, D)


# -- shapes ---------------------------------------------------------------


def test_mean_head_shape(tokens):
    out = MeanPoolHead(D)(tokens)
    assert isinstance(out, HeadOutput)
    assert out.logits.shape == (B, 3)
    assert out.attention is None


@pytest.mark.parametrize("branches", [1, 3])
def test_attentive_head_shape(tokens, branches):
    out = AttentivePoolHead(D, num_branches=branches)(tokens)
    assert out.logits.shape == (B, 3)
    assert out.attention.shape == (B, branches, N)


def test_rejects_pooled_input():
    with pytest.raises(ValueError, match=r"\[B, N, D\]"):
        MeanPoolHead(D)(torch.randn(B, D))


def test_rejects_dim_mismatch(tokens):
    with pytest.raises(ValueError, match="does not match head input_dim"):
        MeanPoolHead(128)(tokens)


# -- attention is a valid distribution ------------------------------------


@pytest.mark.parametrize("branches", [1, 3])
def test_attention_sums_to_one_over_tokens(tokens, branches):
    a = AttentivePoolHead(D, num_branches=branches).attention_weights(tokens)
    assert torch.allclose(a.sum(dim=-1), torch.ones(B, branches), atol=1e-5)
    assert (a >= 0).all()


def test_attention_reshapes_to_encoder_grid(tokens):
    """Saliency maps require the trailing axis to match token_layout.grid."""
    a = AttentivePoolHead(D)(tokens).attention
    assert a.reshape(B, 1, 14, 14).shape == (B, 1, 14, 14)


# -- the generalisation claim ---------------------------------------------


def test_attentive_reduces_to_mean_pooling(tokens):
    """With zeroed attention parameters the softmax is uniform, so pooling is
    the mean. Mean pooling is therefore a special case, not a rival."""
    head = AttentivePoolHead(D, use_layernorm=False, pre_norm=False)
    with torch.no_grad():
        for layer in (head.attention_v, head.attention_u, head.attention_w):
            layer.weight.zero_()
            layer.bias.zero_()

    out = head(tokens)
    assert torch.allclose(out.attention, torch.full((B, 1, N), 1.0 / N), atol=1e-6)

    pooled = torch.bmm(out.attention, tokens).squeeze(1)
    assert torch.allclose(pooled, tokens.mean(dim=1), atol=1e-5)


def test_attentive_can_ignore_uninformative_tokens():
    """Sanity check that attention is capable of concentrating.

    Directly addresses audit finding F2: a SAGES clip spans 20 seconds while the
    label describes only its final timepoint, so the head must be able to
    downweight earlier tokens rather than averaging them in.

    The informative token carries a distinct *pattern* across the feature axis,
    not merely a larger magnitude, because pre-attention LayerNorm is invariant
    to per-token magnitude (see test below).
    """
    torch.manual_seed(0)
    head = AttentivePoolHead(D, hidden_dim=32)

    n_tokens = 8
    signal = torch.randn(1, n_tokens, D) * 0.1
    signal[0, -1] = torch.randn(D) * 3.0  # only the last token is distinctive

    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    target = torch.tensor([[1.0, 0.0, 1.0]])
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(signal).logits, target
        )
        loss.backward()
        opt.step()

    final = head(signal).attention[0, 0]
    uniform = 1.0 / n_tokens
    assert final.max() > uniform * 1.5, (
        f"attention stayed near uniform (max {final.max():.4f} vs {uniform:.4f}); "
        f"it cannot downweight uninformative tokens"
    )


def test_pre_norm_discards_per_token_magnitude():
    """Documents a property that silently defeats magnitude-based signals.

    LayerNorm maps any constant vector to zero, so tokens differing only in
    uniform scale are indistinguishable to the attention module. Recorded as a
    test so the behaviour is not rediscovered later as a bug.
    """
    flat = torch.ones(1, 4, D)
    spiked = torch.ones(1, 4, D)
    spiked[0, 0] = 5.0  # one token far louder than its neighbours

    normed = AttentivePoolHead(D, pre_norm=True)
    assert torch.allclose(
        normed.attention_weights(flat), normed.attention_weights(spiked), atol=1e-6
    ), "pre_norm should erase a pure magnitude difference"

    torch.manual_seed(0)
    unnormed = AttentivePoolHead(D, pre_norm=False)
    assert not torch.allclose(
        unnormed.attention_weights(flat), unnormed.attention_weights(spiked), atol=1e-4
    ), "without pre_norm the magnitude difference should be visible"


# -- training behaviour ---------------------------------------------------


@pytest.mark.parametrize("kind,kwargs", [("mean", {}), ("attentive", {}),
                                         ("attentive", {"num_branches": 3})])
def test_gradients_flow(tokens, kind, kwargs):
    head = build_head(kind, D, **kwargs)
    out = head(tokens)
    out.logits.sum().backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_eval_mode_is_deterministic(tokens):
    head = build_head("attentive", D, dropout=0.5).eval()
    with torch.no_grad():
        assert torch.equal(head(tokens).logits, head(tokens).logits)


def test_dropout_is_active_in_training(tokens):
    head = build_head("attentive", D, dropout=0.5).train()
    torch.manual_seed(1)
    a = head(tokens).logits
    torch.manual_seed(2)
    b = head(tokens).logits
    assert not torch.equal(a, b)


def test_parameter_cost_is_small(tokens):
    """Attentive pooling must not become a classifier that masks encoder quality."""
    mean_params = sum(p.numel() for p in MeanPoolHead(768).parameters())
    attn_params = sum(p.numel() for p in AttentivePoolHead(768, hidden_dim=128).parameters())
    assert attn_params - mean_params < 250_000


def test_per_criterion_branches_differ(tokens):
    """Three branches should be free to attend to different regions."""
    torch.manual_seed(0)
    a = AttentivePoolHead(D, num_branches=3)(tokens).attention
    assert not torch.allclose(a[:, 0], a[:, 1], atol=1e-4)


# -- factory --------------------------------------------------------------


# -- fusion head: shapes and the 2D fused representation -------------------


@pytest.mark.parametrize("branches", [1, 3])
def test_fusion_head_shape(tokens, prefix, branches):
    out = FusionHead(D, num_branches=branches)(tokens, prefix)
    assert isinstance(out, HeadOutput)
    assert out.logits.shape == (B, 3)
    assert out.attention.shape == (B, branches, N)


@pytest.mark.parametrize("branches", [1, 3])
def test_fused_representation_is_2d_before_the_classifier(tokens, prefix, branches):
    """The classifier reads 2D, not D. This is the whole structural claim."""
    head = FusionHead(D, num_branches=branches)
    fused, attention = head.fused_representation(tokens, prefix)

    assert head.fused_dim == 2 * D
    assert fused.shape == (B, branches, 2 * D)
    assert attention.shape == (B, branches, N)

    classifier_in = (
        head.classifier.in_features
        if branches == 1
        else head.classifier_weight.shape[1]
    )
    assert classifier_in == 2 * D


def test_fused_halves_are_ctx_then_mil(tokens, prefix):
    """Concatenation order is [h_ctx ; h_MIL].

    Every branch-attribution test below slices on this order, so if it were
    reversed those tests would pass while measuring the opposite branch.
    """
    head = FusionHead(D, use_layernorm=False, pre_norm=False)
    fused, attention = head.fused_representation(tokens, prefix)

    assert torch.allclose(fused[:, 0, :D], prefix[:, 0], atol=1e-5)
    assert torch.allclose(
        fused[:, 0, D:], torch.bmm(attention, tokens).squeeze(1), atol=1e-5
    )


def test_fusion_attention_sums_to_one_over_tokens(tokens):
    a = FusionHead(D).attention_weights(tokens)
    assert torch.allclose(a.sum(dim=-1), torch.ones(B, 1), atol=1e-5)
    assert (a >= 0).all()


# -- fusion head: the two reductions ---------------------------------------


def test_zeroing_the_mil_half_reduces_to_h_ctx_alone(tokens, prefix):
    """With the MIL columns of the classifier zeroed, the head is a linear
    probe on [CLS] and nothing the patch tokens do can change its output."""
    torch.manual_seed(0)
    head = FusionHead(D).eval()
    with torch.no_grad():
        head.classifier.weight[:, D:].zero_()

    baseline = head(tokens, prefix).logits

    torch.manual_seed(7)
    other_tokens = torch.randn(B, N, D) * 5.0
    assert torch.allclose(head(other_tokens, prefix).logits, baseline, atol=1e-5), (
        "logits moved when only the patch tokens changed, so the MIL branch is "
        "still reaching the classifier"
    )

    expected = head.classifier(
        torch.cat([head.norm_ctx(prefix[:, 0]), torch.zeros(B, D)], dim=-1)
    )
    assert torch.allclose(baseline, expected, atol=1e-5)


def test_zeroing_the_ctx_half_reduces_to_attentive_pooling(tokens, prefix):
    """Fusion is a strict generalisation of AttentivePoolHead, not a rival.

    Sharing weights and zeroing the global half must reproduce the attentive
    head's logits exactly. If this drifts, a fusion-versus-attention comparison
    is measuring two changes at once.
    """
    torch.manual_seed(0)
    attentive = AttentivePoolHead(D, hidden_dim=32).eval()
    fusion = FusionHead(D, hidden_dim=32).eval()

    with torch.no_grad():
        for name in ("attention_v", "attention_u", "attention_w"):
            getattr(fusion, name).load_state_dict(getattr(attentive, name).state_dict())
        fusion.pre_norm.load_state_dict(attentive.pre_norm.state_dict())
        fusion.norm_mil.load_state_dict(attentive.norm.state_dict())
        fusion.classifier.weight[:, :D].zero_()
        fusion.classifier.weight[:, D:].copy_(attentive.classifier.weight)
        fusion.classifier.bias.copy_(attentive.classifier.bias)

    assert torch.allclose(
        fusion(tokens, prefix).logits, attentive(tokens).logits, atol=1e-5
    )
    assert torch.allclose(fusion.attention_weights(tokens), attentive.attention_weights(tokens))


# -- fusion head: the global branch ----------------------------------------


def test_global_branch_reads_cls_not_a_register_token(tokens, prefix):
    """DINOv3 emits [CLS, reg...]; index 0 must be the one that is read."""
    head = FusionHead(D, use_layernorm=False, pre_norm=False)
    assert torch.allclose(head.global_context(tokens, prefix), prefix[:, 0], atol=1e-6)

    altered = prefix.clone()
    altered[:, 1:] = 99.0  # registers are not spatial and must not be read
    assert torch.allclose(
        head.global_context(tokens, altered), head.global_context(tokens, prefix)
    )


def test_cls_source_with_no_prefix_raises_rather_than_falling_back(tokens):
    """VideoMAE and V-JEPA 2 return prefix=None. A silent switch to mean-pooled
    patches would make two runs look like the same model when they are not."""
    head = FusionHead(D, global_source="cls")
    with pytest.raises(ValueError, match="global_source='patch_mean'"):
        head(tokens, None)


def test_patch_mean_source_works_without_prefix(tokens):
    head = FusionHead(D, global_source="patch_mean", use_layernorm=False, pre_norm=False)
    out = head(tokens, None)
    assert out.logits.shape == (B, 3)

    fused, _ = head.fused_representation(tokens, None)
    assert torch.allclose(fused[:, 0, :D], tokens.mean(dim=1), atol=1e-5)


def test_patch_mean_source_ignores_a_prefix_if_one_is_supplied(tokens, prefix):
    """The choice is fixed at construction, so handing it a prefix must not
    quietly change the model that the run record says was trained."""
    head = FusionHead(D, global_source="patch_mean").eval()
    with torch.no_grad():
        assert torch.equal(head(tokens, prefix).logits, head(tokens, None).logits)


def test_global_source_is_recorded_in_the_head_config(tokens):
    cls_head = FusionHead(D, global_source="cls")
    mean_head = FusionHead(D, global_source="patch_mean")

    assert cls_head.head_config()["global_source"] == "cls"
    assert cls_head.head_config()["uses_prefix"] is True
    assert mean_head.head_config()["global_source"] == "patch_mean"
    assert mean_head.head_config()["uses_prefix"] is False
    assert mean_head.head_config()["prefix_index"] is None
    assert cls_head.head_config()["fused_dim"] == 2 * D


def test_invalid_global_source_rejected():
    with pytest.raises(ValueError, match="must be 'cls' or 'patch_mean'"):
        FusionHead(D, global_source="auto")


def test_prefix_shape_errors_are_specific(tokens, prefix):
    head = FusionHead(D)
    with pytest.raises(ValueError, match=r"\[B, P, D\]"):
        head(tokens, prefix[:, 0])
    with pytest.raises(ValueError, match="does not match head input_dim"):
        head(tokens, torch.randn(B, P, D * 2))
    with pytest.raises(ValueError, match="batch"):
        head(tokens, prefix[:1])
    with pytest.raises(ValueError, match="out of range"):
        FusionHead(D, prefix_index=5)(tokens, prefix)


# -- fusion head: training behaviour ---------------------------------------


def test_gradients_reach_both_fusion_branches(tokens, prefix):
    """Both halves of the classifier and the attention module must receive
    gradient. A dead branch would make the head silently equivalent to one of
    the two heads it is being compared against."""
    head = FusionHead(D, hidden_dim=32)
    head(tokens, prefix).logits.sum().backward()

    grads = {name: p.grad for name, p in head.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())

    ctx_half = grads["classifier.weight"][:, :D]
    mil_half = grads["classifier.weight"][:, D:]
    assert ctx_half.abs().sum() > 0, "no gradient into the global branch"
    assert mil_half.abs().sum() > 0, "no gradient into the MIL branch"

    for name in ("attention_v.weight", "attention_u.weight", "attention_w.weight"):
        assert grads[name].abs().sum() > 0, f"no gradient into {name}"


def test_gradients_flow_to_attention_with_patch_mean_source(tokens):
    head = FusionHead(D, hidden_dim=32, global_source="patch_mean")
    head(tokens, None).logits.sum().backward()
    for name in ("attention_v", "attention_u", "attention_w"):
        grad = getattr(head, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0


@pytest.mark.parametrize("branches", [1, 3])
def test_fusion_gradients_flow_per_branch(tokens, prefix, branches):
    head = build_head("fusion", D, hidden_dim=32, num_branches=branches)
    head(tokens, prefix).logits.sum().backward()
    grads = [p.grad for p in head.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_fusion_eval_mode_is_deterministic(tokens, prefix):
    head = build_head("fusion", D, dropout=0.5).eval()
    with torch.no_grad():
        assert torch.equal(head(tokens, prefix).logits, head(tokens, prefix).logits)


def test_fusion_dropout_is_active_in_training(tokens, prefix):
    head = build_head("fusion", D, dropout=0.5).train()
    torch.manual_seed(1)
    a = head(tokens, prefix).logits
    torch.manual_seed(2)
    b = head(tokens, prefix).logits
    assert not torch.equal(a, b)


def test_fusion_accepts_features_cast_from_fp16_cache(tokens, prefix):
    out = build_head("fusion", D)(tokens.half().float(), prefix.half().float())
    assert out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()


def test_fusion_parameter_cost_is_small(tokens):
    """The fusion head must not become a classifier that masks encoder quality.

    Against the attentive head at the same width it adds only the extra D
    classifier columns and one LayerNorm.
    """
    attn = sum(p.numel() for p in AttentivePoolHead(768, hidden_dim=512).parameters())
    fusion = sum(p.numel() for p in FusionHead(768, hidden_dim=512).parameters())
    assert fusion - attn == 3 * 768 + 2 * 768, (
        "unexpected parameter delta; the comparison is no longer just fusion"
    )


def test_paper_default_width_is_512():
    """SMIL states a 512-dimensional MIL hidden layer. Recorded so that a change
    to this project's 128 default cannot silently move the class default too."""
    assert FusionHead(D).attention_v.out_features == 512


def test_heads_share_one_call_signature(tokens, prefix):
    """The probe trainer has a single call site, so a head that ignores prefix
    must still accept it."""
    for kind in ("mean", "attentive", "fusion"):
        out = build_head(kind, D, hidden_dim=32)(tokens, prefix)
        assert out.logits.shape == (B, 3)


def test_build_head_aliases():
    assert isinstance(build_head("mean", D), MeanPoolHead)
    assert isinstance(build_head("linear", D), MeanPoolHead)
    assert isinstance(build_head("abmil", D), AttentivePoolHead)
    assert isinstance(build_head("fusion", D), FusionHead)
    assert isinstance(build_head("smil", D), FusionHead)


def test_build_head_passes_hidden_dim_to_both_attention_heads():
    """A fusion win is only attributable to fusion if both arms search the same
    attention widths, so the factory must not special-case one of them."""
    assert build_head("attentive", D, hidden_dim=512).attention_v.out_features == 512
    assert build_head("fusion", D, hidden_dim=512).attention_v.out_features == 512
    assert build_head("fusion", D, hidden_dim=128).attention_v.out_features == 128


def test_build_head_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown head"):
        build_head("transformer", D)


def test_invalid_branch_count_rejected():
    with pytest.raises(ValueError, match="num_branches must be"):
        AttentivePoolHead(D, num_branches=2)


# -- fp16 cache compatibility ---------------------------------------------


def test_accepts_features_cast_from_fp16_cache(tokens):
    """Caches are fp16; the probe casts to float32 before the head."""
    out = build_head("attentive", D)(tokens.half().float())
    assert out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()