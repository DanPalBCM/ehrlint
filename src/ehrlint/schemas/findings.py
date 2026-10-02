"""Findings: what a check reports, and the evidence it must carry.

The §0 rule this module exists to enforce: **do not fabricate leakage
statistics.** Every finding references the concrete rows that triggered it —
subject ids, event times, feature names — not just a count.

A `Finding` with a non-zero `n_subjects` and no `examples` is a contradiction,
and :meth:`Finding.validate_evidence` raises on it. That is deliberate: a
count without rows is an assertion a user cannot check, and an auditor whose
claims cannot be checked is worse than no auditor.

Findings are also **deterministic**: given the same inputs, the same findings
in the same order with the same examples. Example rows are sorted before
truncation so a run on shuffled input produces identical output. A property
test asserts it.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: How many example rows a finding carries. Enough to see the pattern, few
#: enough that a report stays readable and carries little patient data.
MAX_EXAMPLES = 10


class Severity(StrEnum):
    """What a finding means for the task.

    * ``error`` -- the task is invalid as defined. Post-index features, split
      overlap, a label using future information.
    * ``warning`` -- a suspicious pattern needing human review. Proxies,
      temporal edge cases.
    * ``info`` -- descriptive statistics, not a problem.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def rank(self) -> int:
        """Ordering for threshold comparisons; higher is more severe."""
        return {"info": 0, "warning": 1, "error": 2}[self.value]

    def at_least(self, threshold: Severity) -> bool:
        """Whether this severity meets or exceeds ``threshold``."""
        return self.rank >= threshold.rank


class Finding(BaseModel):
    """One leakage finding, with the evidence that produced it."""

    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1)
    name: str = ""
    severity: Severity
    message: str = Field(min_length=1, description="One sentence a reviewer can act on.")
    n_subjects: int = Field(default=0, ge=0)
    n_total_subjects: int = Field(default=0, ge=0, description="Denominator for `subject_rate`.")
    n_rows: int = Field(default=0, ge=0, description="Offending rows, which may exceed n_subjects.")
    examples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Concrete rows that triggered this finding. Never aggregates.",
    )
    example_columns: list[str] = Field(default_factory=list)
    query: str = Field(
        default="",
        description="The exact logic, in SQL-ish pseudocode, so a reader can "
        "reproduce the finding by hand.",
    )
    subjects: list[str] = Field(
        default_factory=list,
        description="Affected subject ids, truncated. The audit trail for who.",
    )
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def subject_rate(self) -> float | None:
        """Affected fraction of subjects, or None when the denominator is zero.

        ``None`` rather than ``0.0``: a rate with no denominator is undefined,
        and reporting zero would read as "no subjects affected".
        """
        if self.n_total_subjects <= 0:
            return None
        return self.n_subjects / self.n_total_subjects

    @property
    def fired(self) -> bool:
        """Whether this finding reports a problem."""
        return self.severity is not Severity.INFO and self.n_subjects > 0

    @model_validator(mode="after")
    def _evidence_present(self) -> Finding:
        """A non-zero count must be backed by rows.

        SPEC §0: no fabricated statistics. An `info` finding is exempt, since
        descriptive statistics are aggregates by nature.
        """
        if self.severity is not Severity.INFO and self.n_subjects > 0 and not self.examples:
            raise ValueError(
                f"{self.check_id} reports {self.n_subjects} affected subject(s) but "
                "carries no example rows. Every finding must reference the concrete "
                "rows that triggered it (SPEC §0)."
            )
        return self

    def validate_evidence(self, events: pl.DataFrame | None = None) -> None:
        """Assert the example rows exist in the source data.

        Args:
            events: The canonical event table, when the examples are events.

        Raises:
            ValueError: If an example references a subject absent from the
                data, which would mean the finding was constructed rather than
                observed.
        """
        if events is None or not self.subjects:
            return
        from ehrlint.schemas.events import SUBJECT_ID

        if SUBJECT_ID not in events.columns:
            return
        known = set(events[SUBJECT_ID].drop_nulls().unique().to_list())
        unknown = [s for s in self.subjects if s not in known]
        if unknown:
            raise ValueError(
                f"{self.check_id} references subject(s) {unknown[:5]} that do not appear "
                "in the input data"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict, the §6.3 findings-JSONL shape."""
        return {
            "check_id": self.check_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "n_subjects": self.n_subjects,
            "n_total_subjects": self.n_total_subjects,
            "subject_rate": self.subject_rate,
            "n_rows": self.n_rows,
            "examples": self.examples,
            "example_columns": self.example_columns,
            "subjects": self.subjects,
            "query": self.query,
            "detail": self.detail,
        }

    def examples_frame(self) -> pl.DataFrame:
        """The example rows as a frame, for `finding.examples.head()` ergonomics."""
        if not self.examples:
            return pl.DataFrame()
        return pl.DataFrame(self.examples)


class SkippedCheck(BaseModel):
    """A check that could not run, and why.

    Recorded explicitly so a clean report is distinguishable from an
    incomplete one. A skipped check is **not** a pass.
    """

    model_config = ConfigDict(extra="forbid")

    check_id: str
    name: str = ""
    reason: str = Field(min_length=1)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict."""
        return {"check_id": self.check_id, "name": self.name, "reason": self.reason}


def make_finding(
    check_id: str,
    *,
    name: str,
    severity: Severity,
    message: str,
    offending: pl.DataFrame,
    n_total_subjects: int,
    query: str,
    subject_column: str = "subject_id",
    example_columns: list[str] | None = None,
    detail: dict[str, Any] | None = None,
    max_examples: int = MAX_EXAMPLES,
) -> Finding:
    """Build a finding from the rows that triggered it.

    The single constructor every check uses, so evidence handling is uniform:
    counts come from the offending frame, examples are sorted before truncation
    for determinism, and the subject list is derived rather than passed in.

    Args:
        check_id: e.g. ``LK001``.
        name: Human-readable check name.
        severity: Finding severity.
        message: One actionable sentence.
        offending: The rows that triggered the finding. **Empty means the check
            did not fire**, and a zero-count finding is returned.
        n_total_subjects: Denominator for the rate.
        query: SQL-ish pseudocode of the logic.
        subject_column: Which column holds the subject id.
        example_columns: Columns to include in examples. Defaults to all.
        detail: Extra structured context.
        max_examples: How many example rows to carry.

    Returns:
        A :class:`Finding`.
    """
    columns = example_columns or list(offending.columns)
    columns = [c for c in columns if c in offending.columns]

    if offending.height == 0:
        return Finding(
            check_id=check_id,
            name=name,
            severity=severity,
            message=message,
            n_subjects=0,
            n_total_subjects=n_total_subjects,
            n_rows=0,
            examples=[],
            example_columns=columns,
            query=query,
            subjects=[],
            detail=detail or {},
        )

    subject_ids: list[str] = []
    if subject_column in offending.columns:
        subject_ids = sorted({str(s) for s in offending[subject_column].drop_nulls().to_list()})

    # Sort before truncating so the examples are the same rows regardless of
    # input row order. Determinism is a tested property, not a nicety.
    #
    # The sort key has to be **total**. Sorting only by subject and time leaves
    # tied rows in whatever order the frame happened to hold them, and a
    # truncation of a tie picks different rows on different runs -- which was
    # observable in LK008, where one subject contributes several rows that
    # agree on every preferred key. The preferred columns come first so the
    # examples still read in a sensible order; the rest are appended only to
    # break ties.
    preferred = [c for c in (subject_column, "time", "code") if c in offending.columns]
    tiebreak = [
        c for c in offending.columns if c not in preferred and _is_sortable(offending.schema[c])
    ]
    sort_by = preferred + tiebreak
    ordered = offending.sort(sort_by, nulls_last=True) if sort_by else offending

    examples = [
        {k: _jsonable(v) for k, v in row.items()}
        for row in ordered.select(columns).head(max_examples).to_dicts()
    ]

    return Finding(
        check_id=check_id,
        name=name,
        severity=severity,
        message=message,
        n_subjects=len(subject_ids),
        n_total_subjects=n_total_subjects,
        n_rows=offending.height,
        examples=examples,
        example_columns=columns,
        query=query,
        subjects=subject_ids[:50],
        detail=detail or {},
    )


def _is_sortable(dtype: Any) -> bool:
    """Whether a column can be used in a sort key.

    Nested dtypes cannot, and a check is free to put one in its evidence frame.
    """
    return not isinstance(dtype, (pl.List, pl.Array, pl.Struct, pl.Object))


def _jsonable(value: Any) -> Any:
    """Make a Polars cell JSON-serializable without losing information."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def write_findings_jsonl(findings: list[Finding], path: str) -> int:
    """Write findings as JSONL, the §6.3 format. Returns the count."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for finding in findings:
            fh.write(json.dumps(finding.to_dict(), default=str))
            fh.write("\n")
    return len(findings)
