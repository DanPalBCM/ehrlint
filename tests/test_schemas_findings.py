"""Findings: the evidence rule, determinism, and the rate denominator."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ehrlint.schemas.findings import (
    Finding,
    Severity,
    SkippedCheck,
    make_finding,
    write_findings_jsonl,
)


def offending(n: int = 3) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": [f"S{i:03d}" for i in range(n)],
            "time": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)],
            "code": ["E11.9"] * n,
        }
    )


class TestSeverity:
    def test_ranks_error_above_warning_above_info(self) -> None:
        assert Severity.ERROR.rank > Severity.WARNING.rank > Severity.INFO.rank

    @pytest.mark.parametrize(
        ("severity", "threshold", "expected"),
        [
            (Severity.ERROR, Severity.WARNING, True),
            (Severity.WARNING, Severity.ERROR, False),
            (Severity.WARNING, Severity.WARNING, True),
            (Severity.INFO, Severity.WARNING, False),
        ],
    )
    def test_at_least_compares_by_rank(
        self, severity: Severity, threshold: Severity, expected: bool
    ) -> None:
        assert severity.at_least(threshold) is expected


class TestEvidenceRule:
    def test_a_count_without_rows_is_rejected(self) -> None:
        """SPEC §0: a count nobody can check is an assertion, not a finding."""
        with pytest.raises(ValueError, match="no example rows"):
            Finding(
                check_id="LK001",
                name="x",
                severity=Severity.ERROR,
                message="m",
                n_subjects=5,
                n_total_subjects=10,
                n_rows=5,
                examples=[],
                example_columns=[],
                query="q",
            )

    def test_an_info_finding_may_be_a_bare_aggregate(self) -> None:
        """Descriptive statistics are aggregates by nature."""
        finding = Finding(
            check_id="LK001",
            name="x",
            severity=Severity.INFO,
            message="m",
            n_subjects=5,
            n_total_subjects=10,
            n_rows=5,
            examples=[],
            example_columns=[],
            query="q",
        )
        assert finding.n_subjects == 5

    def test_validate_evidence_rejects_subjects_absent_from_the_input(
        self, tiny_events: pl.DataFrame
    ) -> None:
        from ehrlint.schemas.events import coerce_events

        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(2),
            n_total_subjects=10,
            query="q",
        )
        with pytest.raises(ValueError, match="do not appear in the input"):
            finding.validate_evidence(coerce_events(tiny_events))

    def test_validate_evidence_accepts_subjects_present_in_the_input(self) -> None:
        from ehrlint.schemas.events import coerce_events

        events = coerce_events(
            pl.DataFrame(
                {
                    "subject_id": ["S000", "S001"],
                    "time": [datetime(2020, 1, 1)] * 2,
                    "code": ["x"] * 2,
                }
            )
        )
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(2),
            n_total_subjects=10,
            query="q",
        )
        finding.validate_evidence(events)  # must not raise


class TestSubjectRate:
    def test_is_none_rather_than_zero_when_the_denominator_is_zero(self) -> None:
        """Zero would read as "no subjects affected", which is a different claim."""
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(0),
            n_total_subjects=0,
            query="q",
        )
        assert finding.subject_rate is None

    def test_divides_affected_by_total(self) -> None:
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(3),
            n_total_subjects=12,
            query="q",
        )
        assert finding.subject_rate == pytest.approx(0.25)


class TestFired:
    def test_a_zero_count_finding_has_not_fired(self) -> None:
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="clean",
            offending=offending(0),
            n_total_subjects=10,
            query="q",
        )
        assert finding.fired is False

    def test_an_info_finding_never_fires(self) -> None:
        """Info is descriptive (SPEC §4.4); it must not fail a CI gate."""
        finding = Finding(
            check_id="LK001",
            name="x",
            severity=Severity.INFO,
            message="m",
            n_subjects=99,
            n_total_subjects=100,
            n_rows=99,
            examples=[],
            example_columns=[],
            query="q",
        )
        assert finding.fired is False


class TestMakeFinding:
    def test_examples_are_the_same_rows_whatever_the_input_order(self) -> None:
        """Sort before truncating, or the evidence depends on shard order."""
        rows = offending(12)
        first = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=rows,
            n_total_subjects=20,
            query="q",
            max_examples=5,
        )
        second = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=rows.reverse(),
            n_total_subjects=20,
            query="q",
            max_examples=5,
        )
        assert first.examples == second.examples

    def test_counts_come_from_the_frame_not_the_examples(self) -> None:
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(40),
            n_total_subjects=100,
            query="q",
            max_examples=5,
        )
        assert finding.n_rows == 40
        assert finding.n_subjects == 40
        assert len(finding.examples) == 5

    def test_the_subject_list_is_capped_but_the_count_is_not(self) -> None:
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(80),
            n_total_subjects=100,
            query="q",
        )
        assert finding.n_subjects == 80
        assert len(finding.subjects) == 50

    def test_example_columns_are_restricted_to_those_requested(self) -> None:
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(3),
            n_total_subjects=10,
            query="q",
            example_columns=["subject_id", "code"],
        )
        assert set(finding.examples[0]) == {"subject_id", "code"}

    def test_a_requested_column_that_is_absent_is_dropped_not_fatal(self) -> None:
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(3),
            n_total_subjects=10,
            query="q",
            example_columns=["subject_id", "nope"],
        )
        assert finding.example_columns == ["subject_id"]

    def test_timestamps_are_serialized_as_strings(self) -> None:
        finding = make_finding(
            "LK001",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(1),
            n_total_subjects=10,
            query="q",
        )
        assert isinstance(finding.examples[0]["time"], str)


class TestSkippedCheck:
    def test_carries_a_reason(self) -> None:
        skip = SkippedCheck(check_id="LK002", name="same_encounter", reason="no encounter_id")
        assert "encounter" in skip.to_dict()["reason"]


def test_findings_jsonl_is_one_json_object_per_line(tmp_path: Path) -> None:
    import json

    findings = [
        make_finding(
            f"LK00{i}",
            name="x",
            severity=Severity.ERROR,
            message="m",
            offending=offending(2),
            n_total_subjects=10,
            query="q",
        )
        for i in (1, 2)
    ]
    path = tmp_path / "findings.jsonl"
    assert write_findings_jsonl(findings, str(path)) == 2
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["check_id"] for line in lines] == ["LK001", "LK002"]


def test_examples_frame_gives_the_documented_ergonomics() -> None:
    """SPEC §5.2 shows `finding.examples.head()`; a frame view provides it."""
    finding = make_finding(
        "LK001",
        name="x",
        severity=Severity.ERROR,
        message="m",
        offending=offending(3),
        n_total_subjects=10,
        query="q",
    )
    assert finding.examples_frame().height == 3
