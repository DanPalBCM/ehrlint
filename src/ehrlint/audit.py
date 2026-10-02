"""Running the check battery and assembling a report.

One function, :func:`run_audit`. It resolves the checks, runs each one, records
skips separately from passes, validates that every finding's evidence really
exists in the input, and computes the exit code.

The distinction the whole report hangs on: a **skipped** check is not a
**passed** check. A clean report that quietly skipped six of ten checks is
worse than a dirty one, because it reads as reassurance. Skips are first-class
in the summary and the exit code logic, and the report prints them above the
findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ehrlint.checks.base import Check, resolve_checks
from ehrlint.context import AuditContext
from ehrlint.exceptions import CheckError
from ehrlint.schemas.findings import Finding, Severity, SkippedCheck


@dataclass
class AuditSummary:
    """Counts a reader needs before looking at anything else."""

    n_checks_requested: int = 0
    n_checks_run: int = 0
    n_checks_skipped: int = 0
    n_findings: int = 0
    n_fired: int = 0
    n_errors: int = 0
    n_warnings: int = 0
    n_info: int = 0
    n_subjects: int = 0
    n_events: int = 0
    worst_severity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        return {
            "n_checks_requested": self.n_checks_requested,
            "n_checks_run": self.n_checks_run,
            "n_checks_skipped": self.n_checks_skipped,
            "n_findings": self.n_findings,
            "n_fired": self.n_fired,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "n_info": self.n_info,
            "n_subjects": self.n_subjects,
            "n_events": self.n_events,
            "worst_severity": self.worst_severity,
        }

    def to_markdown(self) -> str:
        """A compact summary table."""
        lines = [
            "| metric | value |",
            "| --- | --- |",
            f"| subjects | {self.n_subjects} |",
            f"| events | {self.n_events} |",
            f"| checks run | {self.n_checks_run} of {self.n_checks_requested} |",
            f"| checks skipped | {self.n_checks_skipped} |",
            f"| findings that fired | {self.n_fired} |",
            f"| errors | {self.n_errors} |",
            f"| warnings | {self.n_warnings} |",
            f"| worst severity | {self.worst_severity or 'none'} |",
        ]
        if self.n_checks_skipped:
            lines.append("")
            lines.append(
                f"> {self.n_checks_skipped} check(s) did not run. A skipped check is not "
                "a passed check -- see the Skipped section for why."
            )
        return "\n".join(lines)


@dataclass
class AuditReport:
    """The complete result of one audit."""

    task: str
    data_hash: str
    findings: list[Finding] = field(default_factory=list)
    skipped: list[SkippedCheck] = field(default_factory=list)
    summary: AuditSummary = field(default_factory=AuditSummary)
    warnings: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    check_metadata: list[dict[str, str]] = field(default_factory=list)
    started: datetime | None = None
    duration_s: float = 0.0
    fail_on: Severity = Severity.ERROR

    @property
    def fired(self) -> list[Finding]:
        """Findings that actually report a problem, worst first."""
        return sorted(
            [f for f in self.findings if f.fired],
            key=lambda f: (-f.severity.rank, f.check_id),
        )

    @property
    def passed(self) -> list[Finding]:
        """Findings from checks that ran and found nothing."""
        return [f for f in self.findings if not f.fired]

    @property
    def ok(self) -> bool:
        """Whether nothing fired at or above ``fail_on``."""
        return not any(f.severity.at_least(self.fail_on) for f in self.fired)

    @property
    def exit_code(self) -> int:
        """SPEC §5.1: 0 when clean at the threshold, 1 otherwise."""
        return 0 if self.ok else 1

    def by_severity(self, severity: Severity) -> list[Finding]:
        """Fired findings at one severity."""
        return [f for f in self.fired if f.severity is severity]

    def findings_for(self, check_id: str) -> list[Finding]:
        """Every finding from one check, worst first.

        A check may emit more than one: LK001 describes post-index events as an
        observation *and* flags a leaky feature matrix as an error.
        """
        key = check_id.strip().upper()
        return sorted(
            [f for f in self.findings if f.check_id == key],
            key=lambda f: (-f.severity.rank, f.name),
        )

    def finding(self, check_id: str) -> Finding | None:
        """The most serious finding from one check, or None if it skipped.

        Returns the one that **fired** where there is one, not the first
        emitted. A caller asking "what did LK001 find?" must not be handed the
        descriptive observation while an error sits behind it.
        """
        candidates = self.findings_for(check_id)
        if not candidates:
            return None
        return next((f for f in candidates if f.fired), candidates[0])

    def was_skipped(self, check_id: str) -> bool:
        """Whether a check skipped."""
        key = check_id.strip().upper()
        return any(s.check_id == key for s in self.skipped)

    def to_dict(self) -> dict[str, Any]:
        """The SPEC §6.4 report JSON."""
        return {
            "task": self.task,
            "data_hash": self.data_hash,
            "findings": [f.to_dict() for f in self.findings],
            "skipped": [s.to_dict() for s in self.skipped],
            "summary": self.summary.to_dict(),
            "warnings": self.warnings,
            "context": self.context,
            "check_metadata": self.check_metadata,
            "fail_on": self.fail_on.value,
            "exit_code": self.exit_code,
            "started": self.started.isoformat() if self.started else None,
            "duration_s": round(self.duration_s, 4),
            "ehrlint_version": _version(),
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the report JSON."""
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return target

    def write_findings_jsonl(self, path: str | Path) -> Path:
        """Write the findings as JSONL (SPEC §6.3)."""
        from ehrlint.schemas.findings import write_findings_jsonl

        write_findings_jsonl(self.findings, str(path))
        return Path(path)

    def write_markdown(self, path: str | Path) -> Path:
        """Write the Markdown report."""
        from ehrlint.report.builder import render_markdown

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_markdown(self), encoding="utf-8")
        return target

    def write_html(self, outdir: str | Path) -> Path:
        """Write the HTML report and its figures."""
        from ehrlint.report.builder import write_report

        return write_report(self, outdir)["html"]

    def write_all(self, outdir: str | Path) -> dict[str, Path]:
        """Write every artifact: HTML, Markdown, JSON, findings JSONL, figures."""
        from ehrlint.report.builder import write_report

        return write_report(self, outdir)

    def to_markdown(self) -> str:
        """Render the full Markdown report as a string."""
        from ehrlint.report.builder import render_markdown

        return render_markdown(self)


def run_audit(
    ctx: AuditContext,
    checks: list[str] | None = None,
    *,
    fail_on: Severity | str = Severity.ERROR,
    validate_evidence: bool = True,
) -> AuditReport:
    """Run the check battery against a context.

    Args:
        ctx: The resolved inputs.
        checks: Check ids to run. None or empty runs all of them.
        fail_on: The severity at which the audit is considered failed.
        validate_evidence: Assert each finding's example subjects exist in the
            input. On by default — it is the mechanical guard against a check
            reporting rows it invented.

    Returns:
        An :class:`AuditReport`.
    """
    import time

    threshold = Severity(fail_on) if isinstance(fail_on, str) else fail_on
    selected: list[Check] = resolve_checks(checks)
    started = datetime.now(UTC)
    clock = time.perf_counter()

    findings: list[Finding] = []
    skipped: list[SkippedCheck] = []

    for check in selected:
        try:
            result = check.run(ctx)
        except Exception as exc:
            # A crashing check is recorded as skipped, not as a pass. Letting
            # it abort the run would lose the other nine checks' results.
            skipped.append(
                SkippedCheck(
                    check_id=check.id,
                    name=check.name,
                    reason=(
                        f"the check raised {type(exc).__name__}: {exc}. This is a bug in "
                        "ehrlint, not a finding about your data"
                    ),
                )
            )
            continue

        if isinstance(result, SkippedCheck):
            skipped.append(result)
            continue

        for finding in result:
            if validate_evidence:
                try:
                    finding.validate_evidence(ctx.events)
                except ValueError as exc:
                    raise CheckError(str(exc), check_id=check.id) from exc
            findings.append(finding)

    duration = time.perf_counter() - clock
    findings.sort(key=lambda f: (f.check_id, f.name))

    fired = [f for f in findings if f.fired]
    worst = max((f.severity for f in fired), key=lambda s: s.rank, default=None)

    summary = AuditSummary(
        n_checks_requested=len(selected),
        n_checks_run=len(selected) - len(skipped),
        n_checks_skipped=len(skipped),
        n_findings=len(findings),
        n_fired=len(fired),
        n_errors=len([f for f in fired if f.severity is Severity.ERROR]),
        n_warnings=len([f for f in fired if f.severity is Severity.WARNING]),
        n_info=len([f for f in findings if f.severity is Severity.INFO]),
        n_subjects=ctx.n_subjects,
        n_events=ctx.events.height,
        worst_severity=worst.value if worst else None,
    )

    return AuditReport(
        task=ctx.task.task,
        data_hash=ctx.data_hash,
        findings=findings,
        skipped=sorted(skipped, key=lambda s: s.check_id),
        summary=summary,
        warnings=ctx.all_warnings(),
        context=ctx.describe(),
        check_metadata=[
            c.metadata() if hasattr(c, "metadata") else {"id": c.id, "name": c.name}
            for c in selected
        ],
        started=started,
        duration_s=duration,
        fail_on=threshold,
    )


def _version() -> str:
    from ehrlint import __version__

    return __version__
