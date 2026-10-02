"""The §4.5 performance target, measured rather than asserted in prose.

SPEC §4.5: the core checks run on 100k subjects / 10M events in under five
minutes on a 16GB laptop. The full-scale run is marked `slow` and excluded
from the default suite; a smaller run executes everywhere and extrapolates, so
a regression that makes a check quadratic is caught by the ordinary test run
rather than only in a nightly job.
"""

from __future__ import annotations

import time

import pytest

from ehrlint.audit import run_audit
from ehrlint.synth.generator import generate

from .conftest import make_context

#: SPEC §4.5.
TARGET_SUBJECTS = 100_000
TARGET_EVENTS = 10_000_000
TARGET_SECONDS = 300.0

#: The generator makes roughly 16 events per subject, so the event half of the
#: target needs far more subjects than the subject half. Both are measured,
#: because 100k subjects alone only reaches ~1.6M events -- which would let a
#: tool that is slow per *event* pass a test named after the target.
EVENT_TARGET_SUBJECTS = 640_000


def timed_audit(n_subjects: int) -> tuple[float, int, int]:
    """Audit a generated cohort, returning (seconds, n_subjects, n_events)."""
    dataset = generate(n_subjects, seed=0)
    ctx = make_context(dataset)
    clock = time.perf_counter()
    report = run_audit(ctx)
    elapsed = time.perf_counter() - clock
    assert report.skipped == [], [s.reason for s in report.skipped]
    return elapsed, ctx.n_subjects, ctx.events.height


def test_a_moderate_cohort_is_fast_enough_to_extrapolate() -> None:
    """5,000 subjects must audit in a few seconds.

    The budget is generous on purpose: this test guards against an accidental
    quadratic, not against a 20% slowdown. A per-subject cost high enough to
    fail here could never meet the §4.5 target.
    """
    elapsed, subjects, events = timed_audit(5_000)
    per_subject_ms = elapsed / subjects * 1000
    budget_ms = TARGET_SECONDS / TARGET_SUBJECTS * 1000
    assert per_subject_ms < budget_ms, (
        f"{per_subject_ms:.3f} ms/subject over {subjects} subjects and {events} "
        f"events; the §4.5 target allows {budget_ms:.3f} ms/subject"
    )


def test_the_cost_per_subject_does_not_grow_with_the_cohort() -> None:
    """A quadratic check passes a small-cohort budget and misses the target.

    Comparing two sizes catches it where a single measurement cannot. The
    allowance is wide because process noise at these sizes is large; a genuine
    quadratic would show a factor near four, not near two.
    """
    small_elapsed, small_n, _ = timed_audit(2_000)
    large_elapsed, large_n, _ = timed_audit(8_000)

    small_rate = small_elapsed / small_n
    large_rate = large_elapsed / large_n
    assert large_rate < small_rate * 2.5, (
        f"per-subject cost grew from {small_rate * 1000:.3f} to "
        f"{large_rate * 1000:.3f} ms between {small_n} and {large_n} subjects"
    )


@pytest.mark.slow
def test_the_subject_half_of_the_target_is_met() -> None:
    """100k subjects under five minutes.

    Marked `slow` and run on a schedule rather than on every push, since
    generating the cohort alone takes a while.
    """
    elapsed, subjects, events = timed_audit(TARGET_SUBJECTS)
    assert elapsed < TARGET_SECONDS, (
        f"{elapsed:.1f}s for {subjects} subjects and {events} events; "
        f"the §4.5 target is {TARGET_SECONDS:.0f}s"
    )
    print(f"\n§4.5 subjects: {subjects} subjects, {events} events, {elapsed:.1f}s")


@pytest.mark.slow
def test_the_event_half_of_the_target_is_met() -> None:
    """10M events under five minutes, which is the binding half."""
    elapsed, subjects, events = timed_audit(EVENT_TARGET_SUBJECTS)
    assert events >= TARGET_EVENTS, (
        f"only {events} events generated; this test is meant to measure {TARGET_EVENTS}"
    )
    assert elapsed < TARGET_SECONDS, (
        f"{elapsed:.1f}s for {subjects} subjects and {events} events; "
        f"the §4.5 target is {TARGET_SECONDS:.0f}s"
    )
    print(f"\n§4.5 events: {subjects} subjects, {events} events, {elapsed:.1f}s")
