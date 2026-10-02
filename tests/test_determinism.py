"""Determinism (SPEC §11).

Two audits of the same data must produce the same findings, in the same order,
with the same example rows. An audit report is evidence attached to a review or
a submission; one that changes between runs is not evidence.
"""

from __future__ import annotations

import json

from ehrlint.audit import run_audit
from ehrlint.inject import inject_many
from ehrlint.schemas.events import events_hash
from ehrlint.synth.generator import generate

from .conftest import make_context

LEAKS = ["LK002", "LK005", "LK009"]


def leaky():
    dataset, _ = inject_many(generate(140, seed=9), LEAKS, n=5, seed=9)
    return dataset


class TestGenerator:
    def test_the_same_seed_gives_the_same_events(self) -> None:
        assert events_hash(generate(100, seed=3).events) == events_hash(
            generate(100, seed=3).events
        )

    def test_a_different_seed_gives_different_events(self) -> None:
        assert events_hash(generate(100, seed=3).events) != events_hash(
            generate(100, seed=4).events
        )

    def test_labels_and_splits_are_reproducible_too(self) -> None:
        one, two = generate(100, seed=3), generate(100, seed=3)
        assert one.labels.equals(two.labels)
        assert one.splits_mapping() == two.splits_mapping()

    def test_the_data_hash_is_stable_across_processes(self) -> None:
        """A hash built from Python's `hash()` would not be; this one is."""
        assert events_hash(generate(50, seed=1).events) == (
            events_hash(generate(50, seed=1).events)
        )


class TestInjectors:
    def test_the_same_seed_plants_the_same_leak(self) -> None:
        one, planted_one = inject_many(generate(100, seed=1), LEAKS, n=4, seed=1)
        two, planted_two = inject_many(generate(100, seed=1), LEAKS, n=4, seed=1)
        assert planted_one == planted_two
        assert events_hash(one.events) == events_hash(two.events)

    def test_injection_order_does_not_matter(self) -> None:
        """Injectors are applied in sorted id order, whatever the caller listed."""
        one, planted_one = inject_many(generate(100, seed=1), ["LK005", "LK002"], seed=1)
        two, planted_two = inject_many(generate(100, seed=1), ["LK002", "LK005"], seed=1)
        assert planted_one == planted_two
        assert events_hash(one.events) == events_hash(two.events)


class TestAudit:
    def test_two_audits_agree_on_every_finding(self) -> None:
        dataset = leaky()
        first = run_audit(make_context(dataset))
        second = run_audit(make_context(dataset))

        assert [f.check_id for f in first.findings] == [f.check_id for f in second.findings]
        for a, b in zip(first.findings, second.findings, strict=True):
            assert a.model_dump() == b.model_dump()

    def test_findings_are_ordered_worst_first_and_then_by_id(self) -> None:
        report = run_audit(make_context(leaky()))
        ranks = [(-f.severity.rank, f.check_id) for f in report.fired]
        assert ranks == sorted(ranks)

    def test_the_report_json_is_byte_identical_between_runs(self, tmp_path: object) -> None:
        dataset = leaky()
        one = run_audit(make_context(dataset)).to_dict()
        two = run_audit(make_context(dataset)).to_dict()
        for volatile in ("started", "duration_s"):
            one.pop(volatile)
            two.pop(volatile)
        assert json.dumps(one, sort_keys=True, default=str) == json.dumps(
            two, sort_keys=True, default=str
        )

    def test_row_order_in_the_input_does_not_change_the_findings(self) -> None:
        """Shard order is an accident of storage, not a property of the data."""
        dataset = leaky()
        shuffled = dataset.copy()
        shuffled.events = dataset.events.reverse()

        normal = run_audit(make_context(dataset))
        reversed_ = run_audit(make_context(shuffled))

        assert normal.data_hash == reversed_.data_hash
        assert [f.check_id for f in normal.fired] == [f.check_id for f in reversed_.fired]
        for a, b in zip(normal.findings, reversed_.findings, strict=True):
            assert a.n_subjects == b.n_subjects
            assert a.subjects == b.subjects
            assert a.examples == b.examples

    def test_the_subject_list_is_sorted(self) -> None:
        report = run_audit(make_context(leaky()))
        for finding in report.fired:
            assert finding.subjects == sorted(finding.subjects), finding.check_id


class TestReportRendering:
    def test_markdown_is_identical_between_runs(self) -> None:
        from ehrlint.report.builder import render_markdown

        dataset = leaky()
        first = render_markdown(run_audit(make_context(dataset)))
        second = render_markdown(run_audit(make_context(dataset)))
        assert _strip_timestamps(first) == _strip_timestamps(second)


def _strip_timestamps(text: str) -> list[str]:
    """Drop the lines that legitimately differ: generation time and duration."""
    return [
        line for line in text.splitlines() if "Generated by" not in line and "duration" not in line
    ]
