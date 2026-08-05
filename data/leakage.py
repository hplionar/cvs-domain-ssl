"""Registry of documented cross-dataset video overlaps, and enforcement.

Several public surgical datasets share source procedures. Endoscapes and
Cholec80 were both collected at IHU Strasbourg and contain the *same videos*
under different identifiers. Combining them naively places evaluation
procedures into the pretraining corpus, which inflates results in a way that no
amount of careful splitting within either dataset would reveal.

This module records overlaps that have been published, so that corpus
construction can exclude them mechanically rather than by recollection.

Scope and its limits
--------------------
This registry covers only overlaps somebody has *documented*. Absence from it is
not evidence of independence: no published analysis exists for HeiChole,
MultiBypass140, or AutoLaparo against Endoscapes. For any dataset pair not
listed here, run empirical near-duplicate detection before combining them, and
add the result to this file with its provenance.

Every entry carries a source and a date so that a reader can verify it, and so
that a later revision of the upstream analysis can be reconciled against what
was actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Overlap:
    """A documented set of shared procedures between two datasets.

    ``pairs`` maps an identifier in ``dataset_a`` to the identifier of the same
    physical procedure in ``dataset_b``. Identifier schemes are not assumed to
    coincide across datasets even where the numbers look similar.
    """

    dataset_a: str
    dataset_b: str
    pairs: dict[str, str]
    source: str
    checked: str
    note: str = ""

    def a_ids(self) -> set[str]:
        return set(self.pairs)

    def b_ids(self) -> set[str]:
        return set(self.pairs.values())


CAMMA_SOURCE = "https://github.com/CAMMA-public/camma_dataset_overlaps"
CAMMA_CHECKED = "2026-07-30"


#: Endoscapes and Cholec80 share five procedures appearing in Endoscapes-train
#: and one appearing in Endoscapes-val. Endoscapes-test is clean.
ENDOSCAPES_CHOLEC80 = Overlap(
    dataset_a="cholec80",
    dataset_b="endoscapes",
    pairs={
        # Cholec80 id -> Endoscapes id
        "67": "1",
        "68": "2",
        "70": "3",
        "71": "4",
        "72": "7",
        "66": "121",  # Endoscapes-val
    },
    source=CAMMA_SOURCE,
    checked=CAMMA_CHECKED,
    note=(
        "All six are in the Cholec80 test split. Endoscapes 1, 2, 3, 4, 7 are in "
        "Endoscapes-train; Endoscapes 121 is in Endoscapes-val. Endoscapes-test "
        "has no overlap with Cholec80."
    ),
)


ENDOSCAPES_CHOLECT50 = Overlap(
    dataset_a="cholect50",
    dataset_b="endoscapes",
    pairs={
        "68": "2",
        "70": "3",
        "96": "10",
        "110": "33",
        "66": "121",  # Endoscapes-val
        "103": "127",  # Endoscapes-val
    },
    source=CAMMA_SOURCE,
    checked=CAMMA_CHECKED,
    note=(
        "All six are in the CholecT50 train split. Endoscapes 121 and 127 are in "
        "Endoscapes-val. Endoscapes-test has no overlap with CholecT50."
    ),
)


#: Cholec80 and CholecT50 overlap extensively; CholecT50 draws from the same
#: source collection. Recorded for completeness, since a corpus combining both
#: would double-count those procedures and skew the effective sampling weight.
CHOLEC80_CHOLECT50_COUNT = 45


REGISTRY: tuple[Overlap, ...] = (ENDOSCAPES_CHOLEC80, ENDOSCAPES_CHOLECT50)


#: Datasets for which no overlap analysis against Endoscapes has been published.
#: Presence here is a prompt to run detection, not a claim of independence.
UNVERIFIED_AGAINST_ENDOSCAPES: tuple[str, ...] = (
    "heichole",
    "multibypass140",
    "autolaparo",
    "dresden",
    "psi-ava",
    "sages",
)


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------


def find_overlap(dataset_a: str, dataset_b: str) -> Overlap | None:
    """Return the recorded overlap for a pair, in either order."""
    a, b = dataset_a.lower(), dataset_b.lower()
    for entry in REGISTRY:
        if {entry.dataset_a, entry.dataset_b} == {a, b}:
            return entry
    return None


def is_verified(dataset_a: str, dataset_b: str) -> bool:
    """Whether *any* published analysis covers this pair.

    Distinguishes "checked, no overlap" from "never checked", which a bare
    empty result would conflate.
    """
    return find_overlap(dataset_a, dataset_b) is not None


def videos_to_exclude(
    source_dataset: str,
    *,
    target_dataset: str,
    protected_ids: Iterable[str],
) -> set[str]:
    """Identifiers in ``source_dataset`` that duplicate a protected procedure.

    ``protected_ids`` are identifiers in ``target_dataset`` that must not appear
    in a pretraining corpus, normally the union of its validation and test
    splits.

    Returns identifiers in the *source* dataset's own numbering, which is what a
    corpus builder needs in order to filter.
    """
    overlap = find_overlap(source_dataset, target_dataset)
    if overlap is None:
        return set()

    protected = {str(i) for i in protected_ids}
    source = source_dataset.lower()

    if overlap.dataset_a == source:
        return {a for a, b in overlap.pairs.items() if b in protected}
    return {b for a, b in overlap.pairs.items() if a in protected}


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------


@dataclass
class CorpusSource:
    """One dataset's contribution to a pretraining corpus."""

    name: str
    video_ids: list[str] = field(default_factory=list)


class LeakageError(ValueError):
    """Raised when a pretraining corpus contains an evaluation procedure."""


def assert_no_leakage(
    sources: Iterable[CorpusSource],
    *,
    target_dataset: str,
    val_ids: Iterable[str],
    test_ids: Iterable[str],
    strict_unverified: bool = True,
) -> None:
    """Refuse a corpus that contains evaluation procedures.

    Two checks are applied:

    1. Direct containment. A source drawn from the target dataset must not
       include its own validation or test identifiers.
    2. Documented cross-dataset overlap. A source from another dataset must not
       include procedures known to duplicate protected ones.

    ``strict_unverified`` additionally raises when a source has never been
    checked against the target. This is deliberately noisy: silence about an
    unverified pair is indistinguishable from silence about a clean one, and the
    Cholec80 case shows that the intuition "different dataset, different data"
    is unreliable.
    """
    protected = {str(i) for i in val_ids} | {str(i) for i in test_ids}
    target = target_dataset.lower()
    problems: list[str] = []
    unverified: list[str] = []

    for source in sources:
        name = source.name.lower()
        ids = {str(i) for i in source.video_ids}

        if name == target:
            direct = ids & protected
            if direct:
                problems.append(
                    f"  {source.name}: {len(direct)} of its own evaluation "
                    f"procedures are in the corpus: {sorted(direct)[:10]}"
                )
            continue

        overlap = find_overlap(name, target)
        if overlap is None:
            unverified.append(source.name)
            continue

        offending = ids & videos_to_exclude(
            name, target_dataset=target, protected_ids=protected
        )
        if offending:
            mapped = {
                v: overlap.pairs.get(v) or
                   next((a for a, b in overlap.pairs.items() if b == v), "?")
                for v in sorted(offending)
            }
            problems.append(
                f"  {source.name}: {len(offending)} procedures duplicate "
                f"{target_dataset} evaluation data: "
                + ", ".join(f"{k} = {target_dataset} {v}" for k, v in mapped.items())
                + f"\n    source: {overlap.source} (checked {overlap.checked})"
            )

    if problems:
        raise LeakageError(
            "Pretraining corpus contains evaluation procedures:\n"
            + "\n".join(problems)
            + "\n\nExclude these before pretraining. Results from a contaminated "
            "corpus are not reportable."
        )

    if strict_unverified and unverified:
        raise LeakageError(
            "No published overlap analysis exists for these sources against "
            f"{target_dataset}:\n"
            + "\n".join(f"  {name}" for name in unverified)
            + "\n\nRun empirical near-duplicate detection and record the result "
            "in data/leakage.py, or pass strict_unverified=False to proceed "
            "with the risk documented."
        )


def filter_corpus(
    sources: Iterable[CorpusSource],
    *,
    target_dataset: str,
    val_ids: Iterable[str],
    test_ids: Iterable[str],
) -> tuple[list[CorpusSource], dict[str, list[str]]]:
    """Remove leaking procedures, returning the cleaned corpus and what was dropped.

    Preferred over silent filtering: the removal record belongs in the methods
    section, and a corpus that loses more than a handful of procedures deserves
    a second look at whether the datasets are as independent as assumed.
    """
    protected = {str(i) for i in val_ids} | {str(i) for i in test_ids}
    target = target_dataset.lower()
    cleaned, removed = [], {}

    for source in sources:
        name = source.name.lower()
        ids = [str(i) for i in source.video_ids]
        drop = (
            set(ids) & protected
            if name == target
            else videos_to_exclude(name, target_dataset=target, protected_ids=protected)
        )
        keep = [i for i in ids if i not in drop]
        cleaned.append(CorpusSource(name=source.name, video_ids=keep))
        actually_dropped = sorted(drop & set(ids))
        if actually_dropped:
            removed[source.name] = actually_dropped

    return cleaned, removed


__all__ = [
    "CHOLEC80_CHOLECT50_COUNT",
    "CorpusSource",
    "ENDOSCAPES_CHOLEC80",
    "ENDOSCAPES_CHOLECT50",
    "LeakageError",
    "Overlap",
    "REGISTRY",
    "UNVERIFIED_AGAINST_ENDOSCAPES",
    "assert_no_leakage",
    "filter_corpus",
    "find_overlap",
    "is_verified",
    "videos_to_exclude",
]