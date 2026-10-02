"""The audit runner: exit codes, skip accounting, and the crash path."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ehrlint.audit import run_audit
from ehrlint.checks.base import BaseCheck, register
from ehrlint.context import AuditContext
from ehrlint.exceptions import CheckError
from ehrlint.inject import inject_many
from ehrlint.schemas.events import empty_events
from ehrlint.schemas.findings import Finding, Severity, make_finding
from ehrlint.synth.generator import default_task, generate

from .conftest import make_context


def leaky(*check_ids: str, n_subjects: int = 120, seed: int = 6):
    dataset, planted = inject_many(generate(n_subjects, seed=seed), list(check_ids), n=4, seed=seed)
    return make_context(dataset), planted


class TestExitCodes:
    def test_clean_data_exits_zero(self, clean_context: AuditContext) -> None:
        report = run_audit(clean_context)
        assert report.ok is True
        assert report.exit_code == 0

    def test_an_error_exits_one(self) -> None:
        ctx, _ = leaky("LK002")
        assert run_audit(ctx).exit_code == 1

    def test_a_warning_alone_passes_at_the_error_threshold(self) -> None:
        ctx, _ = leaky("LK003")
        report = run_audit(ctx, ["LK003"], fail_on=Severity.ERROR)
        assert report.summary.n_warnings == 1
        assert report.exit_code == 0

    def test_the_same_warning_fails_at_the_warning_threshold(self) -> None:
        ctx, _ = leaky("LK003")
        assert run_audit(ctx, ["LK003"], fail_on=Severity.WARNING).exit_code == 1

    def test_the_threshold_accepts_a_string(self) -> None:
        ctx, _ = leaky("LK003")
        assert run_audit(ctx, ["LK003"], fail_on="warning").exit_code == 1


class TestSkipsAreNotPasses:
    def test_skips_are_counted_separately_from_passes(self) -> None:
        ctx = AuditContext.from_frames(empty_events(), task_spec=default_task())
        report = run_audit(ctx)
        assert report.summary.n_checks_skipped > 0
        assert report.summary.n_checks_run == (
            report.summary.n_checks_requested - report.summary.n_checks_skipped
        )

    def test_a_skipped_check_has_no_finding(self) -> None:
        ctx = AuditContext.from_frames(empty_events(), task_spec=default_task())
        report = run_audit(ctx)
        for skip in report.skipped:
            assert report.finding(skip.check_id) is None

    def test_an_all_skipped_audit_still_exits_zero_but_says_so(self) -> None:
        """Honest, and the reason the report prints skips above findings.

        The exit code answers "did anything fire"; it cannot also answer "was
        the audit complete". That is what the skip count is for, and why a
        caller gating on ehrlint should check it.
        """
        ctx = AuditContext.from_frames(empty_events(), task_spec=default_task())
        report = run_audit(ctx)
        assert report.exit_code == 0
        assert report.summary.n_checks_skipped == report.summary.n_checks_requested

    def test_a_crashing_check_is_recorded_as_skipped_not_as_a_pass(
        self, clean_context: AuditContext
    ) -> None:
        """And it must not abort the other nine checks."""

        class Exploding(BaseCheck):
            id = "LK090"
            name = "exploding"
            severity_default = Severity.ERROR

            def run(self, ctx: AuditContext) -> list[Finding]:
                raise RuntimeError("boom")

        register(Exploding())
        report = run_audit(clean_context)
        assert report.was_skipped("LK090")
        reason = next(s.reason for s in report.skipped if s.check_id == "LK090")
        assert "RuntimeError" in reason and "bug in ehrlint" in reason
        assert report.summary.n_checks_run == report.summary.n_checks_requested - 1
        assert len(report.findings) == len(report.findings)
        assert report.finding("LK002") is not None


class TestEvidenceValidation:
    def test_a_check_citing_rows_that_do_not_exist_is_an_error(
        self, clean_context: AuditContext
    ) -> None:
        """The mechanical guard against a check reporting invented rows."""

        class Fabricating(BaseCheck):
            id = "LK091"
            name = "fabricating"
            severity_default = Severity.ERROR

            def run(self, ctx: AuditContext) -> list[Finding]:
                return [
                    make_finding(
                        self.id,
                        name=self.name,
                        severity=Severity.ERROR,
                        message="invented",
                        offending=pl.DataFrame({"subject_id": ["NOT_A_SUBJECT"]}),
                        n_total_subjects=ctx.n_subjects,
                        query="select 1",
                    )
                ]

        register(Fabricating())
        with pytest.raises(CheckError, match="do not appear in the input"):
            run_audit(clean_context)

    def test_validation_can_be_turned_off(self, clean_context: AuditContext) -> None:
        class Fabricating(BaseCheck):
            id = "LK092"
            name = "fabricating"
            severity_default = Severity.ERROR

            def run(self, ctx: AuditContext) -> list[Finding]:
                return [
                    make_finding(
                        self.id,
                        name=self.name,
                        severity=Severity.ERROR,
                        message="invented",
                        offending=pl.DataFrame({"subject_id": ["NOT_A_SUBJECT"]}),
                        n_total_subjects=ctx.n_subjects,
                        query="select 1",
                    )
                ]

        register(Fabricating())
        report = run_audit(clean_context, validate_evidence=False)
        assert report.finding("LK092") is not None


class TestSelection:
    def test_only_the_requested_checks_run(self, clean_context: AuditContext) -> None:
        report = run_audit(clean_context, ["LK002", "LK005"])
        assert report.summary.n_checks_requested == 2
        assert {f.check_id for f in report.findings} == {"LK002", "LK005"}

    def test_the_selection_is_deduplicated(self, clean_context: AuditContext) -> None:
        report = run_audit(clean_context, ["LK002", "lk002"])
        assert report.summary.n_checks_requested == 1


class TestSummary:
    def test_counts_subjects_and_events_from_the_context(
        self, clean_context: AuditContext, clean_dataset: object
    ) -> None:
        report = run_audit(clean_context)
        assert report.summary.n_subjects == clean_context.n_subjects
        assert report.summary.n_events == clean_context.events.height

    def test_worst_severity_is_the_highest_that_fired(self) -> None:
        ctx, _ = leaky("LK002", "LK003")
        report = run_audit(ctx, ["LK002", "LK003"])
        assert report.summary.worst_severity == "error"

    def test_worst_severity_is_none_when_nothing_fired(self, clean_context: AuditContext) -> None:
        assert run_audit(clean_context).summary.worst_severity is None

    def test_the_markdown_summary_warns_about_skips(self) -> None:
        ctx = AuditContext.from_frames(empty_events(), task_spec=default_task())
        text = run_audit(ctx).summary.to_markdown()
        assert "not" in text and "passed check" in text


class TestFindingsFor:
    def test_returns_every_finding_from_a_multi_finding_check(self) -> None:
        ctx, _ = leaky("LK001")
        report = run_audit(ctx, ["LK001"])
        assert len(report.findings_for("LK001")) == 2

    def test_finding_returns_the_one_that_fired_not_the_first_emitted(self) -> None:
        """Otherwise "what did LK001 find?" answers with the reassuring half."""
        ctx, _ = leaky("LK001")
        report = run_audit(ctx, ["LK001"])
        finding = report.finding("LK001")
        assert finding is not None and finding.fired
        assert finding.severity is Severity.ERROR

    def test_finding_falls_back_to_the_observation_when_nothing_fired(
        self, clean_context: AuditContext
    ) -> None:
        report = run_audit(clean_context, ["LK001"])
        finding = report.finding("LK001")
        assert finding is not None and finding.fired is False


class TestSerialization:
    def test_the_report_json_has_the_spec_keys(self) -> None:
        ctx, _ = leaky("LK002")
        data = run_audit(ctx).to_dict()
        for key in ("task", "data_hash", "findings", "skipped", "summary", "exit_code"):
            assert key in data

    def test_the_report_json_round_trips(self, tmp_path: Path) -> None:
        ctx, _ = leaky("LK002")
        path = run_audit(ctx).write_json(tmp_path / "report.json")
        assert json.loads(path.read_text())["exit_code"] == 1

    def test_findings_jsonl_has_one_line_per_finding(self, tmp_path: Path) -> None:
        ctx, _ = leaky("LK002")
        report = run_audit(ctx)
        path = report.write_findings_jsonl(tmp_path / "findings.jsonl")
        assert len(path.read_text().strip().splitlines()) == len(report.findings)

    def test_the_version_is_recorded(self) -> None:
        from ehrlint import __version__

        ctx, _ = leaky("LK002")
        assert run_audit(ctx).to_dict()["ehrlint_version"] == __version__
