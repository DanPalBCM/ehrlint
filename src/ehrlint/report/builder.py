"""Report generation: a single offline HTML file, Markdown, and JSON.

The HTML is **one self-contained file**: figures are inlined as base64 data
URIs and there is no JavaScript. An audit report gets attached to a review, a
ticket, or a submission, and a report that breaks when its `figures/` directory
is lost is a report that will be seen broken.

The ordering is deliberate and is the opposite of a dashboard's. Warnings and
skipped checks come **before** the findings, because "six of ten checks did
not run" changes how every result below should be read, and a reader who sees
a green summary first will stop there.
"""

from __future__ import annotations

import base64
import io as _io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")  # no display; must precede pyplot
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ehrlint.exceptions import ReportError
from ehrlint.schemas.findings import Severity

if TYPE_CHECKING:
    from ehrlint.audit import AuditReport

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Severity colours, chosen to stay legible in light and dark rendering.
SEVERITY_COLOURS: dict[str, str] = {
    "error": "#C0392B",
    "warning": "#D4A017",
    "info": "#2E86C1",
}

#: Colour-blind-safe neutral for bar fills.
NEUTRAL = "#4C6EF5"


#: Template suffixes whose output is HTML and must be escaped. Checked after
#: stripping the `.j2` marker, because the templates are named
#: `report.html.j2` -- Jinja's own `select_autoescape` looks at the final
#: suffix, sees `.j2`, and would leave escaping off for the HTML template.
HTML_SUFFIXES: tuple[str, ...] = (".html", ".htm", ".xml")

#: Jinja template markers stripped before the suffix check.
TEMPLATE_MARKERS: tuple[str, ...] = (".j2", ".jinja", ".jinja2")


def _should_autoescape(template_name: str | None) -> bool:
    """Whether a template's output needs HTML escaping.

    Escaping has to be on for the HTML report: it prints codes, free-text
    values, and example rows straight from the input, and a single ``<`` in a
    text value would otherwise truncate the page from that point on -- the
    worst possible failure for a tool whose whole job is to be believed about
    what is in your data. It has to be off for Markdown, where ``&lt;`` is just
    wrong.
    """
    if not template_name:
        return False
    name = template_name
    for marker in TEMPLATE_MARKERS:
        if name.endswith(marker):
            name = name[: -len(marker)]
            break
    return name.endswith(HTML_SUFFIXES)


def _env() -> Environment:
    """The Jinja environment for both report templates."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=_should_autoescape,
    )


def _as_data_uri(fig: Any, *, dpi: int = 150) -> str:
    """Render a figure to a base64 PNG data URI, so the HTML stays one file."""
    buffer = _io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plot_severity_summary(report: AuditReport) -> str | None:
    """Horizontal bars of check outcome by severity, including the zeros."""
    counts = {
        "error": report.summary.n_errors,
        "warning": report.summary.n_warnings,
        "passed": len(report.passed),
        "skipped": report.summary.n_checks_skipped,
    }
    if not any(counts.values()):
        return None

    labels = list(counts)
    values = [counts[k] for k in labels]
    colours = [
        SEVERITY_COLOURS["error"],
        SEVERITY_COLOURS["warning"],
        "#2E7D32",
        "#7F8C8D",
    ]

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bars = ax.barh(labels, values, color=colours, edgecolor="white", height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("checks")
    ax.set_title("Audit outcome")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    for bar, value in zip(bars, values, strict=True):
        if value == 0:
            ax.text(
                0.06,
                bar.get_y() + bar.get_height() / 2,
                "none",
                va="center",
                fontsize=8,
                color="#777777",
            )
    ax.set_xlim(0, max(max(values), 1) * 1.15)
    fig.tight_layout()
    return _as_data_uri(fig)


def plot_subject_rates(report: AuditReport) -> str | None:
    """Affected subject rate per fired check.

    A rate, not a count: "42 subjects" means something different in a cohort
    of 100 than in one of 100,000, and the denominator is what a reviewer
    needs.
    """
    fired = [f for f in report.fired if f.subject_rate is not None]
    if not fired:
        return None

    labels = [f"{f.check_id} {f.name}" for f in fired]
    rates = [(f.subject_rate or 0.0) * 100 for f in fired]
    colours = [SEVERITY_COLOURS.get(f.severity.value, NEUTRAL) for f in fired]

    fig, ax = plt.subplots(figsize=(7.2, max(2.2, 0.46 * len(fired) + 1.1)))
    ax.barh(labels, rates, color=colours, edgecolor="white", height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("affected subjects (%)")
    ax.set_title("Affected subject rate by check")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    for i, rate in enumerate(rates):
        ax.text(rate + max(rates) * 0.015, i, f"{rate:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(max(rates), 1.0) * 1.2)
    fig.tight_layout()
    return _as_data_uri(fig)


def _context(report: AuditReport, *, figures: dict[str, str]) -> dict[str, Any]:
    """Build the template context shared by HTML and Markdown."""
    from ehrlint import __version__

    def finding_row(f: Any) -> dict[str, Any]:
        return {
            "check_id": f.check_id,
            "name": f.name,
            "severity": f.severity.value,
            "message": f.message,
            "n_subjects": f.n_subjects,
            "n_total_subjects": f.n_total_subjects,
            "subject_rate": f.subject_rate,
            "rate_pct": None if f.subject_rate is None else f"{f.subject_rate * 100:.2f}%",
            "n_rows": f.n_rows,
            "query": f.query,
            "examples": f.examples,
            "example_columns": f.example_columns,
            "subjects": f.subjects,
            "detail": f.detail,
        }

    metadata = {m["id"]: m for m in report.check_metadata}

    return {
        "version": __version__,
        "task": report.task,
        "data_hash": report.data_hash,
        "summary": report.summary.to_dict(),
        "fired": [finding_row(f) for f in report.fired],
        "passed": [finding_row(f) for f in report.passed],
        "info": [finding_row(f) for f in report.findings if f.severity is Severity.INFO],
        "skipped": [s.to_dict() for s in report.skipped],
        "warnings": report.warnings,
        "context": report.context,
        "metadata": metadata,
        "figures": figures,
        "fail_on": report.fail_on.value,
        "exit_code": report.exit_code,
        "ok": report.ok,
        "started": report.started.strftime("%Y-%m-%d %H:%M UTC") if report.started else None,
        "duration_s": f"{report.duration_s:.2f}",
        "severity_colours": SEVERITY_COLOURS,
    }


def render_markdown(report: AuditReport, *, figures: dict[str, str] | None = None) -> str:
    """Render the Markdown report."""
    try:
        template = _env().get_template("report.md.j2")
    except Exception as exc:  # pragma: no cover
        raise ReportError(f"could not load the Markdown template: {exc}") from exc
    return template.render(**_context(report, figures=figures or {}))


def render_html(report: AuditReport, *, embed_figures: bool = True) -> str:
    """Render the HTML report as a single self-contained string."""
    figures: dict[str, str] = {}
    if embed_figures:
        severity = plot_severity_summary(report)
        rates = plot_subject_rates(report)
        if severity:
            figures["severity"] = severity
        if rates:
            figures["rates"] = rates
    try:
        template = _env().get_template("report.html.j2")
    except Exception as exc:  # pragma: no cover
        raise ReportError(f"could not load the HTML template: {exc}") from exc
    return template.render(**_context(report, figures=figures))


def write_report(
    report: AuditReport, outdir: str | Path, *, embed_figures: bool = True
) -> dict[str, Path]:
    """Write every report artifact into ``outdir``.

    Produces ``report.html`` (one self-contained file), ``report.md``,
    ``report.json``, and ``findings.jsonl``.

    Returns:
        A mapping of artifact name to path.
    """
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    html_path = target / "report.html"
    html_path.write_text(render_html(report, embed_figures=embed_figures), encoding="utf-8")
    written["html"] = html_path

    md_path = target / "report.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    written["markdown"] = md_path

    written["json"] = report.write_json(target / "report.json")
    written["findings"] = report.write_findings_jsonl(target / "findings.jsonl")
    return written
