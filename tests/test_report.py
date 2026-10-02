"""Report rendering: one offline HTML file, Markdown, JSON, and the ordering."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ehrlint.audit import run_audit
from ehrlint.context import AuditContext
from ehrlint.inject import inject_many
from ehrlint.report.builder import (
    plot_severity_summary,
    plot_subject_rates,
    render_html,
    render_markdown,
    write_report,
)
from ehrlint.schemas.events import empty_events
from ehrlint.synth.generator import default_task, generate

from .conftest import make_context


def leaky_report(*check_ids: str):
    dataset, _ = inject_many(generate(120, seed=8), list(check_ids), n=4, seed=8)
    return run_audit(make_context(dataset))


def skipped_report():
    return run_audit(AuditContext.from_frames(empty_events(), task_spec=default_task()))


class TestSingleOfflineFile:
    def test_the_html_has_no_javascript(self) -> None:
        """SPEC §3.8. A report with scripts will not render where it is read."""
        html = render_html(leaky_report("LK002", "LK005"))
        lowered = html.lower()
        assert "<script" not in lowered
        assert "onclick" not in lowered
        assert "javascript:" not in lowered

    def test_figures_are_inlined_as_data_uris(self) -> None:
        """So the report survives being attached to a ticket without its folder."""
        html = render_html(leaky_report("LK002", "LK005"))
        assert "data:image/png;base64," in html
        assert ".png" not in html.replace("image/png", "")

    def test_no_external_resource_is_referenced(self) -> None:
        html = render_html(leaky_report("LK002"))
        assert "http://" not in html
        assert "https://" not in html

    def test_figures_can_be_left_out(self) -> None:
        html = render_html(leaky_report("LK002"), embed_figures=False)
        assert "data:image/png;base64," not in html


class TestOrdering:
    def test_skips_appear_before_the_findings(self) -> None:
        """ "Six of ten did not run" changes how everything below reads."""
        dataset, _ = inject_many(generate(120, seed=8), ["LK002"], n=4, seed=8)
        data = dataset.copy()
        data.task = data.task.model_copy(
            update={"outcome": data.task.outcome.model_copy(update={"proxy_codes": []})}
        )
        report = run_audit(make_context(data))
        assert report.skipped and report.fired

        markdown = render_markdown(report)
        assert markdown.index("Read this first") < markdown.index("## Findings")

        html = render_html(report)
        assert html.index("Skipped checks") < html.index(">Findings<")

    def test_the_skip_notice_says_a_skip_is_not_a_pass(self) -> None:
        report = skipped_report()
        for text in (render_markdown(report), render_html(report)):
            assert "not" in text and "passed check" in text

    def test_input_warnings_are_shown(self) -> None:
        report = leaky_report("LK002")
        assert report.warnings
        assert "null time" in render_markdown(report)


class TestContent:
    def test_every_fired_finding_prints_its_example_rows(self) -> None:
        report = leaky_report("LK002", "LK004", "LK010")
        markdown = render_markdown(report)
        for finding in report.fired:
            assert finding.check_id in markdown
            assert str(finding.n_subjects) in markdown

    def test_the_query_is_printed_so_a_finding_can_be_reproduced(self) -> None:
        report = leaky_report("LK002")
        markdown = render_markdown(report)
        assert "SELECT" in markdown.upper()

    def test_the_rationale_is_printed(self) -> None:
        report = leaky_report("LK002")
        assert "Why this matters" in render_markdown(report)

    def test_passing_checks_are_listed_with_what_they_verified(self) -> None:
        report = leaky_report("LK002")
        markdown = render_markdown(report)
        assert "found nothing" in markdown
        for finding in report.passed:
            if finding.severity.value != "info":
                assert finding.check_id in markdown

    def test_the_data_hash_is_printed(self) -> None:
        report = leaky_report("LK002")
        assert report.data_hash in render_markdown(report)
        assert report.data_hash in render_html(report)

    def test_the_task_definition_is_printed(self) -> None:
        report = leaky_report("LK002")
        assert "dka_2y" in render_markdown(report)


class TestHtmlEscaping:
    def test_a_value_containing_markup_cannot_truncate_the_page(self) -> None:
        """A single `<` in a code or text value must not swallow the rest.

        The report prints codes and example rows straight from the input, and
        truncating the page from the first angle bracket is the worst failure
        available to a tool whose job is to be believed about what is in your
        data. Exercised at the template level, so the assertion is about the
        renderer and not about which check happens to fire.
        """
        from ehrlint.audit import AuditReport, AuditSummary
        from ehrlint.schemas.findings import Severity, make_finding

        hostile = '<script>alert("x")</script>'
        finding = make_finding(
            "LK002",
            name="same_encounter",
            severity=Severity.ERROR,
            message=f"an outcome code {hostile} shares the index encounter",
            offending=pl.DataFrame({"subject_id": ["S000001"], "code": [hostile]}),
            n_total_subjects=10,
            query=f"SELECT * FROM events WHERE code = '{hostile}'",
        )
        report = AuditReport(
            task="t",
            data_hash="deadbeef",
            findings=[finding],
            summary=AuditSummary(n_checks_requested=1, n_checks_run=1, n_fired=1, n_errors=1),
        )

        html = render_html(report, embed_figures=False)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
        assert html.rstrip().endswith("</html>")

    def test_markdown_is_not_escaped(self) -> None:
        markdown = render_markdown(leaky_report("LK002"))
        assert "&lt;" not in markdown

    def test_the_escaping_rule_sees_through_the_jinja_suffix(self) -> None:
        """`report.html.j2` ends in `.j2`.

        Jinja's own `select_autoescape` looks at the final suffix, finds `.j2`,
        and leaves escaping off -- which is exactly how the HTML template
        shipped unescaped while a test using a bare `.html` name passed.
        """
        from ehrlint.report.builder import _should_autoescape

        assert _should_autoescape("report.html.j2") is True
        assert _should_autoescape("report.html") is True
        assert _should_autoescape("report.md.j2") is False
        assert _should_autoescape("report.md") is False
        assert _should_autoescape(None) is False

    def test_the_shipped_templates_get_the_policy_they_need(self) -> None:
        from ehrlint.report.builder import _env

        env = _env()
        assert env.get_template("report.html.j2").environment.autoescape
        assert env.autoescape("report.html.j2") is True
        assert env.autoescape("report.md.j2") is False


class TestFigures:
    def test_the_severity_summary_renders(self) -> None:
        uri = plot_severity_summary(leaky_report("LK002"))
        assert uri is not None and uri.startswith("data:image/png;base64,")

    def test_the_rate_plot_renders_when_something_fired(self) -> None:
        uri = plot_subject_rates(leaky_report("LK002", "LK005"))
        assert uri is not None

    def test_the_rate_plot_is_omitted_when_nothing_fired(self, clean_context: AuditContext) -> None:
        assert plot_subject_rates(run_audit(clean_context)) is None


class TestWriteReport:
    def test_writes_all_four_artifacts(self, tmp_path: Path) -> None:
        written = write_report(leaky_report("LK002"), tmp_path / "out")
        assert set(written) == {"html", "markdown", "json", "findings"}
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0

    def test_creates_the_directory(self, tmp_path: Path) -> None:
        written = write_report(leaky_report("LK002"), tmp_path / "deep" / "nested")
        assert written["html"].parent.is_dir()

    def test_the_json_is_the_report_dict(self, tmp_path: Path) -> None:
        report = leaky_report("LK002")
        written = write_report(report, tmp_path / "out")
        assert json.loads(written["json"].read_text())["data_hash"] == report.data_hash

    def test_the_html_is_valid_enough_to_parse(self, tmp_path: Path) -> None:
        from html.parser import HTMLParser

        class Strict(HTMLParser):
            def error(self, message: str) -> None:  # pragma: no cover
                raise AssertionError(message)

        html = write_report(leaky_report("LK002", "LK009"), tmp_path / "out")["html"].read_text()
        parser = Strict()
        parser.feed(html)
        parser.close()

    def test_a_clean_report_still_renders(
        self, tmp_path: Path, clean_context: AuditContext
    ) -> None:
        written = write_report(run_audit(clean_context), tmp_path / "clean")
        assert "No check fired" in written["markdown"].read_text()

    def test_an_all_skipped_report_still_renders(self, tmp_path: Path) -> None:
        written = write_report(skipped_report(), tmp_path / "skipped")
        text = written["markdown"].read_text()
        assert "did not run" in text
