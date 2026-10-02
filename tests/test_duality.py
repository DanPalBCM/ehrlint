"""Injector/check duality, as a property (SPEC §11).

The one property the whole tool rests on: for every check there is an injector
that plants exactly the leak it looks for, and the check catches it. Stated
over a range of seeds and cohort sizes rather than one fixture, because a check
that only works on 120 subjects at seed 0 is a check that works by accident.

The converse half matters just as much: on clean data **nothing** fires. A
check that fires on everything satisfies the first half trivially.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ehrlint.audit import run_audit
from ehrlint.inject import INJECTORS, inject, inject_many, injector_ids
from ehrlint.schemas.findings import SkippedCheck
from ehrlint.synth.generator import generate

from .conftest import make_context

ALL_IDS = [f"LK{n:03d}" for n in range(1, 11)]

# Hypothesis runs the generator, ten checks, and a full audit per example, so
# the budget is deliberately small and the deadline off. These are slow
# examples by nature, not flaky ones.
DUALITY_SETTINGS = settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def test_every_check_has_an_injector() -> None:
    """SPEC §11. A check with no injector is a check nobody has proven works."""
    assert sorted(INJECTORS) == ALL_IDS
    assert injector_ids() == ALL_IDS


@DUALITY_SETTINGS
@given(
    check_id=st.sampled_from(ALL_IDS),
    seed=st.integers(min_value=0, max_value=50),
    n_subjects=st.integers(min_value=60, max_value=200),
)
def test_an_injected_leak_is_always_caught(check_id: str, seed: int, n_subjects: int) -> None:
    dataset = generate(n_subjects, seed=seed)
    injection = inject(dataset, check_id, n=5, seed=seed)

    report = run_audit(make_context(injection.dataset), [check_id])
    assert not report.was_skipped(check_id), (
        f"{check_id} skipped on its own injected leak: {[s.reason for s in report.skipped]}"
    )

    finding = report.finding(check_id)
    assert finding is not None and finding.fired, (
        f"{check_id} did not fire on {injection.description}"
    )
    missed = sorted(set(injection.subjects) - set(finding.subjects))
    assert not missed, f"{check_id} missed planted subjects {missed[:5]}"


@DUALITY_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=50),
    n_subjects=st.integers(min_value=60, max_value=200),
)
def test_clean_data_fires_nothing_and_skips_nothing(seed: int, n_subjects: int) -> None:
    """The converse half: a check that fires on everything proves nothing."""
    report = run_audit(make_context(generate(n_subjects, seed=seed)))
    assert report.skipped == [], [s.reason for s in report.skipped]
    assert report.fired == [], [(f.check_id, f.message) for f in report.fired]
    assert report.exit_code == 0


@pytest.mark.parametrize("check_id", ALL_IDS)
def test_a_leak_is_caught_by_its_own_check_and_not_only_by_others(check_id: str) -> None:
    """Specificity, not just sensitivity.

    Some leaks genuinely cascade -- moving subjects between splits breaks the
    temporal ordering too -- so other checks firing is expected. What must hold
    is that the check whose leak was planted is among those that fired.
    """
    injection = inject(generate(120, seed=11), check_id, n=5, seed=11)
    report = run_audit(make_context(injection.dataset))
    fired_ids = {f.check_id for f in report.fired}
    assert check_id in fired_ids, f"{check_id} silent; fired instead: {sorted(fired_ids)}"


def test_several_leaks_at_once_are_all_caught() -> None:
    """Injectors compose: a real dataset has more than one problem."""
    dataset, planted = inject_many(
        generate(150, seed=5), ["LK002", "LK004", "LK007", "LK010"], n=4, seed=5
    )
    report = run_audit(make_context(dataset))
    for check_id, subjects in planted.items():
        finding = report.finding(check_id)
        assert finding is not None and finding.fired, check_id
        assert set(subjects) <= set(finding.subjects), check_id


def test_injectors_plant_on_disjoint_subjects() -> None:
    """Otherwise one leak masks another and the ground truth is wrong."""
    _, planted = inject_many(generate(200, seed=3), ALL_IDS, n=4, seed=3)
    seen: dict[str, str] = {}
    for check_id, subjects in planted.items():
        for subject in subjects:
            # LK009's duplicate ids are created by the injector itself, so they
            # cannot have been claimed by an earlier one.
            if subject.endswith("-DUP"):
                continue
            assert subject not in seen, f"{subject} carries both {seen[subject]} and {check_id}"
            seen[subject] = check_id


def test_all_ten_leaks_in_one_dataset_are_all_caught() -> None:
    """The strongest form of the duality property."""
    dataset, planted = inject_many(generate(240, seed=13), ALL_IDS, n=4, seed=13)
    report = run_audit(make_context(dataset))
    silent = [
        check_id
        for check_id in ALL_IDS
        if not (report.finding(check_id) and report.finding(check_id).fired)  # type: ignore[union-attr]
    ]
    assert not silent, f"silent with all ten leaks planted: {silent}"
    for check_id, subjects in planted.items():
        finding = report.finding(check_id)
        assert finding is not None
        missed = sorted(set(subjects) - set(finding.subjects))
        assert not missed, f"{check_id} missed {missed[:5]}"


def test_exclude_is_honoured_by_every_injector() -> None:
    dataset = generate(120, seed=1)
    everyone = set(dataset.events["subject_id"].unique().to_list())
    for check_id in ALL_IDS:
        first = inject(dataset, check_id, n=3, seed=1)
        second = inject(dataset, check_id, n=3, seed=1, exclude=set(first.subjects))
        planted_again = {s for s in second.subjects if s in everyone}
        assert not (planted_again & set(first.subjects)), check_id


def test_an_exhausted_subject_pool_is_an_error_not_a_silent_no_op() -> None:
    """A "leak" that was never planted would make a duality test pass vacuously."""
    from ehrlint.exceptions import EhrlintError

    dataset = generate(20, seed=0)
    everyone = set(dataset.events["subject_id"].unique().to_list())
    with pytest.raises(EhrlintError, match=r"no subject is left|already carries"):
        inject(dataset, "LK002", n=3, seed=0, exclude=everyone)


def test_an_injector_reports_the_subjects_it_actually_touched() -> None:
    """The ground truth a duality test compares against must itself be right."""
    dataset = generate(100, seed=2)
    before = set(dataset.events["subject_id"].unique().to_list())
    injection = inject(dataset, "LK009", n=3, seed=2)
    after = set(injection.dataset.events["subject_id"].unique().to_list())
    added = after - before
    assert added <= set(injection.subjects)
    assert injection.subjects


def test_injecting_does_not_mutate_the_input_dataset() -> None:
    """Otherwise one injector test would contaminate the next."""
    dataset = generate(100, seed=4)
    snapshot = dataset.events.clone()
    inject(dataset, "LK002", n=3, seed=4)
    assert dataset.events.equals(snapshot)


def test_an_unknown_injector_lists_the_available_ones() -> None:
    from ehrlint.exceptions import EhrlintError

    with pytest.raises(EhrlintError, match="LK001"):
        inject(generate(20, seed=0), "LK404")


@pytest.mark.parametrize("check_id", ALL_IDS)
def test_a_check_either_returns_findings_or_a_reasoned_skip(check_id: str) -> None:
    """There is no third outcome. Silence is the failure mode this forbids."""
    from ehrlint.checks.base import get_check

    result = get_check(check_id).run(make_context(generate(80, seed=7)))
    if isinstance(result, SkippedCheck):
        assert result.reason.strip()
    else:
        assert result, f"{check_id} returned an empty list rather than a pass or a skip"
