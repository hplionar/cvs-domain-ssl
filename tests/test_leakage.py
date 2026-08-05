"""Tests for the cross-dataset leakage registry.

The Cholec80 case is the reference scenario: Endoscapes and Cholec80 share six
procedures, one of which is in Endoscapes-val.
"""

from __future__ import annotations

import pytest

from data.leakage import (
    ENDOSCAPES_CHOLEC80,
    CorpusSource,
    LeakageError,
    assert_no_leakage,
    filter_corpus,
    find_overlap,
    is_verified,
    videos_to_exclude,
)

# Endoscapes official splits, as used throughout the project.
ENDOSCAPES_VAL = [str(i) for i in [121, 127, 130, 140]]
ENDOSCAPES_TEST = [str(i) for i in [150, 160, 170]]


# -- registry -------------------------------------------------------------


def test_registry_records_the_documented_pairs():
    assert ENDOSCAPES_CHOLEC80.pairs["66"] == "121"
    assert ENDOSCAPES_CHOLEC80.pairs["67"] == "1"
    assert len(ENDOSCAPES_CHOLEC80.pairs) == 6


def test_registry_carries_provenance():
    assert ENDOSCAPES_CHOLEC80.source.startswith("https://")
    assert ENDOSCAPES_CHOLEC80.checked


def test_lookup_is_order_independent():
    assert find_overlap("cholec80", "endoscapes") is not None
    assert find_overlap("endoscapes", "cholec80") is not None
    assert find_overlap("ENDOSCAPES", "Cholec80") is not None


def test_unlisted_pair_returns_none():
    assert find_overlap("heichole", "endoscapes") is None


def test_verified_distinguishes_clean_from_unchecked():
    """An empty result must not be conflated with an unexamined pair."""
    assert is_verified("cholec80", "endoscapes")
    assert not is_verified("heichole", "endoscapes")


# -- exclusion sets -------------------------------------------------------


def test_excludes_only_procedures_matching_protected_splits():
    """Cholec80 66 duplicates Endoscapes-val 121 and must go. The five
    duplicating Endoscapes-train procedures are legitimate pretraining data."""
    excluded = videos_to_exclude(
        "cholec80", target_dataset="endoscapes", protected_ids=ENDOSCAPES_VAL + ENDOSCAPES_TEST
    )
    assert excluded == {"66"}


def test_cholect50_exclusions():
    excluded = videos_to_exclude(
        "cholect50", target_dataset="endoscapes", protected_ids=ENDOSCAPES_VAL + ENDOSCAPES_TEST
    )
    assert excluded == {"66", "103"}


def test_returns_ids_in_the_source_numbering():
    """A corpus builder filters by the source dataset's own identifiers."""
    excluded = videos_to_exclude(
        "cholec80", target_dataset="endoscapes", protected_ids=["1", "2"]
    )
    assert excluded == {"67", "68"}


def test_unlisted_dataset_yields_no_exclusions():
    assert videos_to_exclude(
        "heichole", target_dataset="endoscapes", protected_ids=ENDOSCAPES_VAL
    ) == set()


# -- enforcement ----------------------------------------------------------


def test_clean_corpus_passes():
    sources = [CorpusSource("cholec80", [str(i) for i in range(1, 60)])]
    assert_no_leakage(
        sources, target_dataset="endoscapes",
        val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
    )


def test_contaminated_corpus_is_rejected_with_the_mapping():
    sources = [CorpusSource("cholec80", ["65", "66", "67"])]
    with pytest.raises(LeakageError) as excinfo:
        assert_no_leakage(
            sources, target_dataset="endoscapes",
            val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
        )
    message = str(excinfo.value)
    assert "66" in message and "121" in message
    assert "camma_dataset_overlaps" in message


def test_self_leakage_is_caught():
    """Endoscapes-train may be pretrained on; its own val must not be."""
    sources = [CorpusSource("endoscapes", ["1", "2", "121"])]
    with pytest.raises(LeakageError, match="own evaluation"):
        assert_no_leakage(
            sources, target_dataset="endoscapes",
            val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
        )


def test_endoscapes_train_videos_are_allowed():
    sources = [CorpusSource("endoscapes", ["1", "2", "3"])]
    assert_no_leakage(
        sources, target_dataset="endoscapes",
        val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
    )


def test_unverified_source_raises_by_default():
    """Silence about an unchecked pair reads identically to silence about a
    clean one, so the unchecked case must announce itself."""
    sources = [CorpusSource("heichole", ["1", "2"])]
    with pytest.raises(LeakageError, match="No published overlap analysis"):
        assert_no_leakage(
            sources, target_dataset="endoscapes",
            val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
        )


def test_unverified_can_be_accepted_explicitly():
    sources = [CorpusSource("heichole", ["1", "2"])]
    assert_no_leakage(
        sources, target_dataset="endoscapes",
        val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
        strict_unverified=False,
    )


def test_documented_overlap_still_caught_when_unverified_relaxed():
    sources = [
        CorpusSource("cholec80", ["66"]),
        CorpusSource("heichole", ["1"]),
    ]
    with pytest.raises(LeakageError, match="duplicate endoscapes"):
        assert_no_leakage(
            sources, target_dataset="endoscapes",
            val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
            strict_unverified=False,
        )


# -- filtering ------------------------------------------------------------


def test_filter_removes_and_reports():
    sources = [
        CorpusSource("cholec80", ["65", "66", "67", "68"]),
        CorpusSource("cholect50", ["66", "103", "110"]),
    ]
    cleaned, removed = filter_corpus(
        sources, target_dataset="endoscapes",
        val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
    )
    assert cleaned[0].video_ids == ["65", "67", "68"]
    assert cleaned[1].video_ids == ["110"]
    assert removed == {"cholec80": ["66"], "cholect50": ["103", "66"]}


def test_filtered_corpus_then_passes_assertion():
    sources = [CorpusSource("cholec80", ["65", "66", "67"])]
    cleaned, _ = filter_corpus(
        sources, target_dataset="endoscapes",
        val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
    )
    assert_no_leakage(
        cleaned, target_dataset="endoscapes",
        val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
    )


def test_filter_reports_nothing_when_clean():
    sources = [CorpusSource("cholec80", ["1", "2", "3"])]
    _, removed = filter_corpus(
        sources, target_dataset="endoscapes",
        val_ids=ENDOSCAPES_VAL, test_ids=ENDOSCAPES_TEST,
    )
    assert removed == {}