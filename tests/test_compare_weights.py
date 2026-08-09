"""Tests for scripts/compare_weights.py."""

from __future__ import annotations

import pytest
import torch

from scripts.compare_weights import block_of, compare


def _state(scale: float = 1.0, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "embeddings.patch_embeddings.projection.weight": torch.randn(4, 4, generator=g) * scale,
        "encoder.layer.0.attention.attention.query.weight": torch.randn(4, 4, generator=g) * scale,
        "encoder.layer.11.output.dense.weight": torch.randn(4, 4, generator=g) * scale,
        "layernorm.weight": torch.ones(4),
    }


def test_identical_states_show_no_movement():
    state = _state()
    result = compare(state, {k: v.clone() for k, v in state.items()})
    assert result["num_moved"] == 0
    assert result["num_unchanged"] == 4


def test_movement_is_detected():
    before = _state()
    after = {k: v + 0.01 for k, v in before.items()}
    result = compare(before, after)
    assert result["num_moved"] == 4
    assert result["num_unchanged"] == 0


def test_relative_change_is_scale_invariant():
    """A 1% change must read as 1% whether the weights are large or small."""
    small = _state(scale=0.01, seed=1)
    large = {k: v * 100 for k, v in small.items()}
    r_small = compare(small, {k: v * 1.01 for k, v in small.items()})
    r_large = compare(large, {k: v * 1.01 for k, v in large.items()})

    def mean(result):
        rows = [r for r in result["parameters"] if not r["was_zero"]]
        return sum(r["rel_change"] for r in rows) / len(rows)

    assert abs(mean(r_small) - mean(r_large)) < 1e-4


def test_zero_norm_parameters_are_flagged_not_divided_by():
    """VideoMAE's attention biases load as zeros; relative change is undefined
    there and must not appear as an infinity or a spurious result."""
    before = {"encoder.layer.0.attention.attention.query.bias": torch.zeros(4)}
    after = {"encoder.layer.0.attention.attention.query.bias": torch.randn(4)}
    result = compare(before, after)
    row = result["parameters"][0]
    assert row["was_zero"] is True
    assert row["rel_change"] != row["rel_change"]  # NaN
    assert result["num_moved"] == 0  # excluded from the moved count


def test_partial_movement_reported_correctly():
    before = _state()
    after = {k: (v + 0.1 if "layer.0" in k else v.clone()) for k, v in before.items()}
    result = compare(before, after)
    assert result["num_moved"] == 1
    assert result["num_unchanged"] == 3


def test_key_differences_reported():
    before = _state()
    after = {k: v for k, v in before.items() if "layernorm" not in k}
    after["decoder.head.weight"] = torch.randn(4, 4)
    result = compare(before, after)
    assert "layernorm.weight" in result["only_in_before"]
    assert "decoder.head.weight" in result["only_in_after"]


def test_shape_mismatch_does_not_raise():
    before = {"w": torch.randn(4, 4)}
    after = {"w": torch.randn(8, 8)}
    assert "error" in compare(before, after)["parameters"][0]


def test_blocks_group_by_layer():
    assert block_of("encoder.layer.7.attention.output.dense.weight") == "layer_07"
    assert block_of("encoder.layer.11.output.dense.bias") == "layer_11"
    assert block_of("embeddings.patch_embeddings.projection.weight") == "embeddings"
    assert block_of("layernorm.weight") == "final_norm"


def test_block_ordering_is_numeric():
    """layer_02 must sort before layer_10, so the depth profile reads correctly."""
    before = {f"encoder.layer.{i}.w": torch.randn(4, 4) for i in (2, 10)}
    after = {k: v + 0.01 for k, v in before.items()}
    blocks = list(compare(before, after)["blocks"])
    assert blocks == ["layer_02", "layer_10"]


def test_block_summary_aggregates():
    before = {
        "encoder.layer.0.a.weight": torch.ones(4, 4),
        "encoder.layer.0.b.weight": torch.ones(4, 4),
    }
    after = {
        "encoder.layer.0.a.weight": torch.ones(4, 4) * 1.1,
        "encoder.layer.0.b.weight": torch.ones(4, 4) * 1.3,
    }
    stats = compare(before, after)["blocks"]["layer_00"]
    assert stats["num_params"] == 2
    assert stats["max_rel_change"] > stats["mean_rel_change"]


# -- architecture detection -----------------------------------------------


def test_detects_videomae():
    from scripts.compare_weights import detect_architecture

    assert detect_architecture("MCG-NJU/videomae-base") == "videomae"
    assert detect_architecture("MCG-NJU/videomae-large") == "videomae"


def test_detects_vjepa():
    from scripts.compare_weights import detect_architecture

    assert detect_architecture("facebook/vjepa2-vitl-fpc16-256-ssv2") == "vjepa2"
    assert detect_architecture("facebook/vjepa2-vitl-fpc64-256") == "vjepa2"


def test_detects_from_payload_when_path_is_uninformative():
    from scripts.compare_weights import detect_architecture

    payload = {"config": {"model": {"checkpoint": "facebook/vjepa2-vitl-fpc16-256-ssv2"}}}
    assert detect_architecture("/tmp/latest.pt", payload) == "vjepa2"


def test_unknown_architecture_is_rejected():
    """Comparing a VideoMAE reference against a V-JEPA checkpoint produces a
    well-formed report with zero shared parameters rather than an error. Failing
    at detection makes that impossible."""
    from scripts.compare_weights import detect_architecture

    with pytest.raises(ValueError, match="Cannot identify the architecture"):
        detect_architecture("some/unknown-model")


def test_block_grouping_handles_both_naming_schemes():
    """VideoMAE nests blocks under encoder.layer.N; V-JEPA names them layer.N."""
    from scripts.compare_weights import block_of

    assert block_of("encoder.layer.7.attention.output.dense.weight") == "layer_07"
    assert block_of("layer.7.attention.query.weight") == "layer_07"
    assert block_of("embeddings.patch_embeddings.proj.weight") == "embeddings"


def test_architecture_specifies_bias_repair_only_for_videomae():
    """V-JEPA stores query, key and value biases under the names transformers
    expects, so the repair is a VideoMAE-specific fix."""
    from scripts.compare_weights import ARCHITECTURES

    assert ARCHITECTURES["videomae"]["repair_bias"] is True
    assert ARCHITECTURES["vjepa2"]["repair_bias"] is False


def test_state_prefixes_differ():
    from scripts.compare_weights import ARCHITECTURES

    assert ARCHITECTURES["videomae"]["state_prefix"] == "videomae."
    assert ARCHITECTURES["vjepa2"]["state_prefix"] == "encoder."