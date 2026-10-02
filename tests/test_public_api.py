"""The documented public API (SPEC §5.2).

Every name the README, the docs, and the spec promise is importable from the
top-level package and does what the example shows. A docs example that no
longer runs is a bug report from the future.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ehrlint as el


class TestExports:
    def test_everything_in_dunder_all_is_importable(self) -> None:
        for name in el.__all__:
            assert hasattr(el, name), name

    def test_every_public_class_and_function_is_in_dunder_all(self) -> None:
        """Importable is not the same as exported.

        `TaskSpecError` was importable and missing from `__all__`, which a test
        checking only the other direction could not see.

        Submodules are excluded, and not as a convenience: importing
        `ehrlint.cli` anywhere in the process binds `cli` as an attribute of
        the package, so sweeping `dir()` for modules measures which other test
        files have run rather than what this package exports.
        """
        import types

        exported = set(el.__all__)
        public = {
            name
            for name in dir(el)
            if not name.startswith("_")
            and name != "annotations"  # the `from __future__` import
            and not isinstance(getattr(el, name), types.ModuleType)
        }
        assert public - exported == set(), sorted(public - exported)

    def test_dunder_all_has_no_duplicates(self) -> None:
        # Ordering is ruff's job (RUF022, which sorts constants ahead of
        # classes); uniqueness is not something it checks.
        assert len(el.__all__) == len(set(el.__all__))

    def test_the_version_is_a_release_string(self) -> None:
        assert el.__version__.count(".") == 2

    @pytest.mark.parametrize(
        "name",
        [
            "AuditContext",
            "AuditReport",
            "Finding",
            "Severity",
            "SkippedCheck",
            "Splits",
            "TaskSpec",
            "register_check",
            "run_audit",
        ],
    )
    def test_the_names_the_spec_uses_are_present(self, name: str) -> None:
        assert hasattr(el, name), name


class TestSpecExample:
    """The §5.2 example, executed rather than quoted."""

    def test_the_documented_flow_runs_end_to_end(self, tmp_path: Path) -> None:
        dataset = el.synth.generator.generate(80, seed=0)
        written = dataset.write(tmp_path / "data")

        ctx = el.AuditContext.from_meds(
            tmp_path / "data",
            labels_path=written["labels"],
            splits_path=written["splits_json"],
            task_spec=written["task"],
        )

        report = el.run_audit(ctx, checks=["LK001", "LK002", "LK005"])
        assert report.summary.to_markdown()

        for finding in report.findings:
            assert finding.check_id
            assert isinstance(finding.severity, el.Severity)
            assert finding.subject_rate is None or 0.0 <= finding.subject_rate <= 1.0
            finding.examples_frame()  # the documented `.head()` ergonomics

        html = report.write_html(tmp_path / "report")
        assert html.exists()

    def test_the_custom_check_example_runs(self, tmp_path: Path) -> None:
        dataset = el.synth.generator.generate(40, seed=0)

        @el.register_check(id="LK099", name="my_check", severity="warning")
        def my_check(ctx: el.AuditContext) -> list[el.Finding]:
            """A check of my own."""
            return [
                el.make_finding(
                    "LK099",
                    name="my_check",
                    severity=el.Severity.WARNING,
                    message="nothing to report",
                    offending=ctx.events.head(0),
                    n_total_subjects=ctx.n_subjects,
                    query="SELECT 1",
                )
            ]

        ctx = el.AuditContext.from_frames(
            dataset.events, task_spec=dataset.task, labels=dataset.labels
        )
        report = el.run_audit(ctx, checks=["LK099"])
        assert report.finding("LK099") is not None


class TestSubmodules:
    @pytest.mark.parametrize("name", ["checks", "inject", "io", "report", "schemas", "synth"])
    def test_the_submodules_are_reachable(self, name: str) -> None:
        assert getattr(el, name) is not None

    def test_the_package_docstring_names_the_two_rules(self) -> None:
        """They are the product, so they belong where `help(ehrlint)` shows them."""
        doc = el.__doc__ or ""
        assert "Evidence or nothing" in doc
        assert "skipped check is not a passed check" in doc


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "name",
        [
            "CheckError",
            "InputError",
            "RegistryError",
            "ReportError",
            "SchemaError",
            "SplitsError",
            "TaskSpecError",
        ],
    )
    def test_every_error_derives_from_the_base(self, name: str) -> None:
        """So a caller can catch one class and mean it."""
        assert issubclass(getattr(el, name), el.EhrlintError)

    def test_the_base_is_an_exception(self) -> None:
        assert issubclass(el.EhrlintError, Exception)


def test_the_cli_entry_point_resolves() -> None:
    """The console script in pyproject must name something that exists."""
    import importlib
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    target = pyproject["project"]["scripts"]["ehrlint"]
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute))
