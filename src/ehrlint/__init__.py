"""ehrlint: lint your clinical prediction task before you train on it.

Most published clinical prediction models have a quiet flaw: the features,
labels, or splits leak information from the future. Post-index features,
same-encounter codes that reveal the outcome, treatment proxies,
immortal-time bias, patient overlap across splits -- these are the most common
reasons clinical models fail in deployment, and the most common reasons
reviewers should have rejected them.

ehrlint audits a **finished** dataset for those patterns. Cohort-extraction
systems help you *define* a task; nothing widely used audits one.

Two rules the whole tool is built around:

* **Evidence or nothing.** Every finding references the concrete rows that
  triggered it -- subject ids, event times, feature names. A count without rows
  is an assertion nobody can check.
* **A skipped check is not a passed check.** A check that cannot run says so
  with a reason, and the report prints skips above findings.

Quickstart::

    import ehrlint as el

    ctx = el.AuditContext.from_meds(
        "data/", task_spec="task.yaml", splits_path="splits.json"
    )
    report = el.run_audit(ctx)
    print(report.summary.to_markdown())
    for finding in report.fired:
        print(finding.check_id, finding.severity, finding.subject_rate)
        print(finding.examples_frame().head())   # concrete rows
    report.write_html("report/")
"""

from __future__ import annotations

__version__ = "1.0.0"

from ehrlint import audit, checks, context, exceptions, inject, io, report, schemas, synth
from ehrlint.audit import AuditReport, AuditSummary, run_audit
from ehrlint.checks import (
    BaseCheck,
    Check,
    CheckResult,
    all_checks,
    check_ids,
    get_check,
    register,
    register_check,
    resolve_checks,
)
from ehrlint.context import AuditContext
from ehrlint.exceptions import (
    CheckError,
    EhrlintError,
    InputError,
    RegistryError,
    ReportError,
    SchemaError,
    SplitsError,
    TaskSpecError,
)
from ehrlint.inject import INJECTORS, Injection, injector_ids
from ehrlint.io import FeatureMatrix, read_feature_matrix, read_labels
from ehrlint.io.meds import read_meds_events
from ehrlint.io.omop import read_omop_events
from ehrlint.schemas import (
    CodeSet,
    Duration,
    Finding,
    FollowUp,
    IndexDef,
    OutcomeDef,
    Severity,
    SkippedCheck,
    Splits,
    SplitsSpec,
    TaskSpec,
    coerce_events,
    make_finding,
)

__all__ = [
    "INJECTORS",
    "AuditContext",
    "AuditReport",
    "AuditSummary",
    "BaseCheck",
    "Check",
    "CheckError",
    "CheckResult",
    "CodeSet",
    "Duration",
    "EhrlintError",
    "FeatureMatrix",
    "Finding",
    "FollowUp",
    "IndexDef",
    "Injection",
    "InputError",
    "OutcomeDef",
    "RegistryError",
    "ReportError",
    "SchemaError",
    "Severity",
    "SkippedCheck",
    "Splits",
    "SplitsError",
    "SplitsSpec",
    "TaskSpec",
    "TaskSpecError",
    "__version__",
    "all_checks",
    "audit",
    "check_ids",
    "checks",
    "coerce_events",
    "context",
    "exceptions",
    "get_check",
    "inject",
    "injector_ids",
    "io",
    "make_finding",
    "read_feature_matrix",
    "read_labels",
    "read_meds_events",
    "read_omop_events",
    "register",
    "register_check",
    "report",
    "resolve_checks",
    "run_audit",
    "schemas",
    "synth",
]
