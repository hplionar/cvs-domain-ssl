"""Tests for V-JEPA 2 pretraining.

The load-bearing tests here concern collapse. `VJEPA2Model.forward` derives its
target from the same encoder in the same pass, with gradients flowing, which
makes the objective satisfiable by driving all representations to a constant.
The EMA target with stop-gradient is what prevents that, so it is tested
directly rather than assumed.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from train.pretrain_vjepa2 import (
    MultiBlockMaskGenerator,
    TargetEncoder,
    jepa_loss,
    momentum_schedule,
)

GRID = (8, 14, 14)
N = GRID[0] * GRID[1] * GRID[2]


# -- masking --------------------------------------------------------------


def test_context_and_target_partition_the_grid():
    context, target = MultiBlockMaskGenerator(GRID)(random.Random(0))
    assert context.numel() + target.numel() == N
    assert set(context.tolist()).isdisjoint(target.tolist())


def test_masking_leaves_usable_context():
    generator = MultiBlockMaskGenerator(GRID, min_context=16)
    for seed in range(20):
        context, target = generator(random.Random(seed))
        assert context.numel() >= 16
        assert target.numel() > 0


def test_blocks_span_the_temporal_axis():
    """A target visible at the same spatial location in another frame reduces
    prediction to copying, as with VideoMAE's tube masking."""
    context, target = MultiBlockMaskGenerator(GRID)(random.Random(3))
    spatial = GRID[1] * GRID[2]
    mask = torch.zeros(N, dtype=torch.bool)
    mask[target] = True
    grid = mask.reshape(GRID[0], spatial)
    for t in range(1, GRID[0]):
        assert torch.equal(grid[0], grid[t]), f"temporal position {t} differs"


def test_masking_is_reproducible():
    a = MultiBlockMaskGenerator(GRID)(random.Random(7))
    b = MultiBlockMaskGenerator(GRID)(random.Random(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_masking_varies_across_draws():
    generator = MultiBlockMaskGenerator(GRID)
    rng = random.Random(0)
    assert not torch.equal(generator(rng)[1], generator(rng)[1])


def test_batch_masks_are_shaped_for_the_predictor():
    context, target = MultiBlockMaskGenerator(GRID).batch(4, random.Random(0))
    assert isinstance(context, list) and isinstance(target, list)
    assert context[0].shape[0] == 4
    assert target[0].shape[0] == 4


def test_more_blocks_mask_more():
    few = MultiBlockMaskGenerator(GRID, num_blocks=2)(random.Random(1))[1].numel()
    many = MultiBlockMaskGenerator(GRID, num_blocks=12)(random.Random(1))[1].numel()
    assert many > few


# -- EMA target -----------------------------------------------------------


class _Encoder(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, pixel_values_videos):
        from types import SimpleNamespace

        return SimpleNamespace(last_hidden_state=self.linear(pixel_values_videos))


def test_target_starts_identical_to_source():
    source = _Encoder()
    target = TargetEncoder(source)
    assert torch.equal(source.linear.weight, target.encoder.linear.weight)


def test_target_requires_no_gradient():
    """Gradients reaching the target would let the loss be minimised by making
    every representation constant."""
    target = TargetEncoder(_Encoder())
    assert all(not p.requires_grad for p in target.encoder.parameters())


def test_target_stays_in_eval_mode():
    target = TargetEncoder(_Encoder())
    target.train(True)
    assert not target.encoder.training


def test_ema_update_moves_toward_source():
    source = _Encoder()
    target = TargetEncoder(source, momentum=0.9)
    before = target.encoder.linear.weight.clone()

    with torch.no_grad():
        source.linear.weight.add_(1.0)
    target.update(source)

    after = target.encoder.linear.weight
    expected = 0.9 * before + 0.1 * source.linear.weight
    assert torch.allclose(after, expected, atol=1e-6)


def test_momentum_one_freezes_the_target():
    source = _Encoder()
    target = TargetEncoder(source)
    before = target.encoder.linear.weight.clone()
    with torch.no_grad():
        source.linear.weight.add_(5.0)
    target.update(source, momentum=1.0)
    assert torch.equal(target.encoder.linear.weight, before)


def test_target_output_carries_no_grad():
    target = TargetEncoder(_Encoder())
    out = target(torch.randn(2, 4, 8, requires_grad=True))
    assert not out.requires_grad


def test_momentum_schedule_rises_to_one():
    assert momentum_schedule(0, 1000, start=0.996) == pytest.approx(0.996, abs=1e-6)
    assert momentum_schedule(1000, 1000, start=0.996) == pytest.approx(1.0, abs=1e-6)
    values = [momentum_schedule(s, 1000, start=0.996) for s in range(0, 1000, 100)]
    assert all(a <= b for a, b in zip(values, values[1:]))


# -- loss -----------------------------------------------------------------


def test_loss_is_zero_on_agreement():
    x = torch.randn(2, 16, 8)
    assert jepa_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-7)


def test_loss_detaches_the_target():
    prediction = torch.randn(2, 16, 8, requires_grad=True)
    target = torch.randn(2, 16, 8, requires_grad=True)
    jepa_loss(prediction, target).backward()
    assert prediction.grad is not None
    assert target.grad is None, "gradient reached the target; collapse is possible"


def test_loss_grows_with_disagreement():
    prediction = torch.zeros(2, 16, 8)
    near = jepa_loss(prediction, torch.full((2, 16, 8), 0.1))
    far = jepa_loss(prediction, torch.full((2, 16, 8), 1.0))
    assert far > near


# -- integration ----------------------------------------------------------


def test_full_step_runs_and_target_is_excluded_from_gradients():
    """End to end on a tiny model: the predictor receives only context, the
    target comes from the EMA copy, and no gradient reaches it."""
    pytest.importorskip("transformers")
    from transformers import VJEPA2Config, VJEPA2Model

    cfg = VJEPA2Config(
        crop_size=224, patch_size=16, frames_per_clip=16, tubelet_size=2,
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, mlp_ratio=1,
        pred_hidden_size=16, pred_num_hidden_layers=1, pred_num_attention_heads=2,
    )
    model = VJEPA2Model(cfg)
    target = TargetEncoder(model.encoder)
    masker = MultiBlockMaskGenerator((8, 14, 14))
    context_mask, target_mask = masker.batch(1, random.Random(0))

    clips = torch.randn(1, 16, 3, 224, 224)
    with torch.no_grad():
        full = target(clips)
        targets = torch.gather(
            full, 1, target_mask[0].unsqueeze(-1).expand(-1, -1, full.shape[-1])
        )

    out = model(pixel_values_videos=clips,
                context_mask=context_mask, target_mask=target_mask)
    prediction = out.predictor_output.last_hidden_state

    assert prediction.shape == targets.shape
    loss = jepa_loss(prediction, targets)
    loss.backward()

    assert all(p.grad is None for p in target.encoder.parameters())
    assert any(p.grad is not None for p in model.encoder.parameters())