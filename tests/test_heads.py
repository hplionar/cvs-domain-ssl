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
    HeadOutput,
    MeanPoolHead,
    build_head,
)

B, N, D = 4, 196, 64


@pytest.fixture
def tokens() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, N, D)


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


def test_build_head_aliases():
    assert isinstance(build_head("mean", D), MeanPoolHead)
    assert isinstance(build_head("linear", D), MeanPoolHead)
    assert isinstance(build_head("abmil", D), AttentivePoolHead)


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