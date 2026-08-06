"""Tests for scripts/compare_heads.py.

The load-bearing tests are the refusals. A comparison script that happily
compares two arms trained on different encoders, different seeds or different
grids will produce a number, and that number will be reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_heads import (
    ComparabilityError,
    equal_grid_entries,
    load_run,
    paired_delta,
    render_markdown,
    select,
    summarise,
    verify_comparable,
)


def make_run(
    *,
    kind: str = "mean",
    encoder: str = "mae_b",
    seeds: tuple[int, ...] = (0, 1, 2),
    configs: list[dict] | None = None,
    grid: dict | None = None,
) -> dict:
    if configs is None:
        configs = [
            {
                "config": {"lr": 1e-3, "weight_decay": 0.0, "dropout": 0.0},
                "mean_map": 0.60,
                "std_map": 0.01,
                "seeds": {str(s): 0.60 for s in seeds},
                "best_epochs": [10] * len(seeds),
            }
        ]
    return {
        "encoder": {"checkpoint_id": encoder},
        "protocol": {"transform": {"image_size": 224}},
        "head": {"kind": kind},
        "search": {
            "grid": grid or {"lr": [1e-3], "weight_decay": [0.0], "dropout": [0.0]},
            "seeds": list(seeds),
            "epochs": 100,
            "patience": 20,
            "pos_weight": False,
        },
        "all_configs": configs,
    }


def configs_with_hidden(values, per_seed):
    """One entry per hidden_dim, each carrying explicit per-seed numbers."""
    out = []
    for hidden in values:
        seeds = per_seed[hidden]
        out.append({
            "config": {"lr": 1e-3, "weight_decay": 0.0, "dropout": 0.0, "hidden_dim": hidden},
            "mean_map": sum(seeds.values()) / len(seeds),
            "std_map": 0.0,
            "seeds": {str(s): v for s, v in seeds.items()},
            "best_epochs": [10] * len(seeds),
        })
    return out


# -- refusals -------------------------------------------------------------


def test_different_encoders_are_rejected():
    runs = {"a": make_run(kind="mean", encoder="mae_b"),
            "b": make_run(kind="fusion", encoder="dinov3_b")}
    with pytest.raises(ComparabilityError, match="encoder"):
        verify_comparable(runs)


def test_different_seeds_are_rejected():
    runs = {"a": make_run(kind="mean", seeds=(0, 1, 2)),
            "b": make_run(kind="fusion", seeds=(0, 1))}
    with pytest.raises(ComparabilityError, match="seeds"):
        verify_comparable(runs)


def test_different_shared_grid_is_rejected():
    runs = {
        "a": make_run(kind="mean", grid={"lr": [1e-3], "weight_decay": [0.0], "dropout": [0.0]}),
        "b": make_run(kind="fusion", grid={"lr": [1e-3, 3e-3], "weight_decay": [0.0], "dropout": [0.0]}),
    }
    with pytest.raises(ComparabilityError, match=r"grid\.lr"):
        verify_comparable(runs)


def test_hidden_dim_may_differ_between_arms():
    """Only the attention heads define hidden_dim, so requiring it to match
    would make the mean arm incomparable with anything by construction."""
    a = make_run(kind="mean", grid={"lr": [1e-3], "weight_decay": [0.0], "dropout": [0.0]})
    b = make_run(kind="fusion", grid={"lr": [1e-3], "weight_decay": [0.0],
                                      "dropout": [0.0], "hidden_dim": [128, 512]})
    verify_comparable({"a": a, "b": b})


def test_two_arms_with_the_same_head_are_rejected():
    runs = {"a": make_run(kind="attentive"), "b": make_run(kind="attentive")}
    with pytest.raises(ComparabilityError, match="differ in the head"):
        verify_comparable(runs)


def test_single_arm_is_rejected():
    with pytest.raises(ComparabilityError, match="At least two"):
        verify_comparable({"a": make_run()})


def test_different_protocol_is_rejected():
    a = make_run(kind="mean")
    b = make_run(kind="fusion")
    b["protocol"] = {"transform": {"image_size": 256}}
    with pytest.raises(ComparabilityError, match="protocol"):
        verify_comparable({"a": a, "b": b})


# -- selection ------------------------------------------------------------


def test_select_takes_the_best_seed_mean_not_the_luckiest_run():
    entries = [
        {"config": {"lr": 1e-3}, "mean_map": 0.51, "std_map": 0.01,
         "seeds": {"0": 0.50, "1": 0.52}},
        {"config": {"lr": 3e-3}, "mean_map": 0.455, "std_map": 0.2,
         "seeds": {"0": 0.61, "1": 0.30}},
    ]
    assert select(entries)["config"] == {"lr": 1e-3}


def test_equal_grid_restricts_attention_arms_only():
    attention = make_run(
        kind="fusion",
        configs=configs_with_hidden([128, 512], {128: {0: 0.60, 1: 0.60}, 512: {0: 0.70, 1: 0.70}}),
    )
    assert len(equal_grid_entries(attention)) == 1
    assert select(equal_grid_entries(attention))["config"]["hidden_dim"] == 128

    mean_arm = make_run(kind="mean")
    assert equal_grid_entries(mean_arm) == mean_arm["all_configs"], (
        "an arm that never searched hidden_dim paid no extra selection bias"
    )


def test_equal_grid_falls_back_when_the_subgrid_is_empty():
    """A run that searched only hidden_dim=512 still reports something, rather
    than silently dropping out of the table."""
    run = make_run(kind="fusion", configs=configs_with_hidden([512], {512: {0: 0.7}}))
    assert len(equal_grid_entries(run)) == 1


# -- paired differences ---------------------------------------------------


def test_paired_delta_pairs_by_seed():
    arm = make_run(kind="fusion", configs=configs_with_hidden([128], {128: {0: 0.72, 1: 0.60, 2: 0.66}}))
    ref = make_run(kind="attentive", configs=configs_with_hidden([128], {128: {0: 0.70, 1: 0.57, 2: 0.65}}))

    paired = paired_delta(arm, ref)
    assert paired["seeds"] == [0, 1, 2]
    assert paired["deltas"] == pytest.approx([0.02, 0.03, 0.01])
    assert paired["mean_delta"] == pytest.approx(0.02)
    assert paired["wins"] == 3


def test_paired_sd_is_smaller_than_unpaired_when_seeds_move_together():
    """The reason for pairing at all: shared seed variance cancels.

    Both arms swing widely across seeds, but the gap between them is stable. An
    unpaired comparison would call this indistinguishable from noise.
    """
    arm = make_run(kind="fusion", configs=configs_with_hidden([128], {128: {0: 0.50, 1: 0.70, 2: 0.90}}))
    ref = make_run(kind="attentive", configs=configs_with_hidden([128], {128: {0: 0.48, 1: 0.68, 2: 0.88}}))

    paired = paired_delta(arm, ref)
    assert paired["sd_delta"] == pytest.approx(0.0, abs=1e-9)

    summary = summarise({"fusion": arm, "attentive": ref}, "attentive")
    unpaired_sd = max(a["sd_map"] for a in summary["arms"])
    assert unpaired_sd > 0.15
    assert paired["mean_delta"] == pytest.approx(0.02)


def test_paired_delta_returns_none_without_shared_seeds():
    arm = make_run(kind="fusion", configs=configs_with_hidden([128], {128: {0: 0.7}}))
    ref = make_run(kind="attentive", configs=configs_with_hidden([128], {128: {9: 0.7}}))
    assert paired_delta(arm, ref) is None


def test_unknown_reference_is_rejected():
    runs = {"a": make_run(kind="mean"), "b": make_run(kind="fusion")}
    with pytest.raises(ComparabilityError, match="not among"):
        summarise(runs, "c")


# -- report ---------------------------------------------------------------


def test_summary_orders_arms_by_mean_map():
    runs = {
        "mean": make_run(kind="mean", configs=configs_with_hidden([128], {128: {0: 0.50, 1: 0.50}})),
        "fusion": make_run(kind="fusion", configs=configs_with_hidden([128], {128: {0: 0.70, 1: 0.70}})),
        "attentive": make_run(kind="attentive", configs=configs_with_hidden([128], {128: {0: 0.60, 1: 0.60}})),
    }
    summary = summarise(runs, "attentive")
    assert [a["arm"] for a in summary["arms"]] == ["fusion", "attentive", "mean"]


def test_markdown_states_the_noise_floor_and_the_scope():
    runs = {
        "attentive": make_run(kind="attentive", configs=configs_with_hidden([128], {128: {0: 0.60, 1: 0.62}})),
        "fusion": make_run(kind="fusion", configs=configs_with_hidden([128], {128: {0: 0.61, 1: 0.64}})),
    }
    text = render_markdown(summarise(runs, "attentive"))
    assert "Noise floor (2 sd)" in text
    assert "not a reproduction of SMIL" in text
    assert "Equal-grid comparison" in text


def test_markdown_marks_an_unresolved_difference_as_unresolved():
    """A difference inside the noise floor must not read as a finding."""
    ref = make_run(kind="attentive", configs=configs_with_hidden([128], {128: {0: 0.60, 1: 0.70}}))
    tiny = make_run(kind="fusion", configs=configs_with_hidden([128], {128: {0: 0.63, 1: 0.68}}))
    rows = [
        line for line in render_markdown(summarise({"attentive": ref, "fusion": tiny}, "attentive")).splitlines()
        if line.startswith("| fusion |") and "±" not in line
    ]
    assert any("| no |" in row for row in rows), rows


def test_load_run_reports_a_missing_results_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="results.json"):
        load_run(tmp_path)


def test_load_run_reads_results_json(tmp_path: Path):
    (tmp_path / "results.json").write_text(json.dumps(make_run()))
    assert load_run(tmp_path)["head"]["kind"] == "mean"
