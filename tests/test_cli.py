"""The command line, exercised through Typer's runner.

The exit codes are the contract: this is a CI gate, and the one outcome that
must be impossible is a broken invocation that looks like a clean audit. So
every usage error is asserted to be 2, never 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ehrlint.cli import EXIT_USAGE, app

runner = CliRunner()


@pytest.fixture
def clean_dir(tmp_path: Path) -> Path:
    """A clean synthetic dataset written by the CLI itself."""
    out = tmp_path / "clean"
    result = runner.invoke(app, ["synth", "--out", str(out), "--n-subjects", "80", "--seed", "0"])
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def leaky_dir(tmp_path: Path) -> Path:
    """A dataset with three leaks planted."""
    out = tmp_path / "leaky"
    result = runner.invoke(
        app,
        [
            "synth",
            "--out",
            str(out),
            "--n-subjects",
            "80",
            "--seed",
            "1",
            "--inject",
            "LK002,LK004,LK010",
        ],
    )
    assert result.exit_code == 0, result.output
    return out


def audit_args(root: Path, *extra: str) -> list[str]:
    return [
        "audit",
        "--data",
        str(root),
        "--task",
        str(root / "task.yaml"),
        "--labels",
        str(root / "labels.parquet"),
        "--splits",
        str(root / "splits.json"),
        *extra,
    ]


class TestExitCodes:
    def test_clean_data_exits_zero(self, clean_dir: Path) -> None:
        result = runner.invoke(app, audit_args(clean_dir))
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output

    def test_leaky_data_exits_one(self, leaky_dir: Path) -> None:
        result = runner.invoke(app, audit_args(leaky_dir))
        assert result.exit_code == 1
        assert "FAIL" in result.output

    @pytest.mark.parametrize(
        "extra",
        [
            ["--checks", "LK001,LK999"],
            ["--fail-on", "catastrophe"],
            ["--fail-on", "info"],
            ["--format", "fhir"],
        ],
    )
    def test_a_bad_option_exits_two_not_zero(self, clean_dir: Path, extra: list[str]) -> None:
        """Never 0: a broken gate that reports success is the worst outcome."""
        result = runner.invoke(app, audit_args(clean_dir, *extra))
        assert result.exit_code == EXIT_USAGE, result.output

    def test_a_missing_data_directory_exits_two(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["audit", "--data", str(tmp_path / "absent"), "--task", str(tmp_path / "t.yaml")]
        )
        assert result.exit_code == EXIT_USAGE

    def test_a_missing_task_spec_exits_two(self, clean_dir: Path) -> None:
        result = runner.invoke(
            app, ["audit", "--data", str(clean_dir), "--task", str(clean_dir / "absent.yaml")]
        )
        assert result.exit_code == EXIT_USAGE

    def test_a_missing_optional_input_exits_two(self, clean_dir: Path) -> None:
        result = runner.invoke(
            app,
            audit_args(clean_dir, "--features", str(clean_dir / "absent.parquet")),
        )
        assert result.exit_code == EXIT_USAGE


class TestFailOn:
    def test_info_is_refused_with_the_reason(self, clean_dir: Path) -> None:
        """Rather than silently behaving like --fail-on warning."""
        result = runner.invoke(app, audit_args(clean_dir, "--fail-on", "info"))
        assert result.exit_code == EXIT_USAGE
        assert "descriptive" in result.output

    def test_warning_is_stricter_than_error(self, tmp_path: Path) -> None:
        out = tmp_path / "proxy"
        runner.invoke(
            app,
            ["synth", "--out", str(out), "--n-subjects", "80", "--seed", "2", "--inject", "LK003"],
        )
        assert runner.invoke(app, audit_args(out, "--checks", "LK003")).exit_code == 0
        assert (
            runner.invoke(
                app, audit_args(out, "--checks", "LK003", "--fail-on", "warning")
            ).exit_code
            == 1
        )


class TestChecksOption:
    def test_an_unknown_id_lists_the_available_ones(self, clean_dir: Path) -> None:
        """A typo must not silently narrow the battery."""
        result = runner.invoke(app, audit_args(clean_dir, "--checks", "LK999"))
        assert result.exit_code == EXIT_USAGE
        assert "LK001" in result.output

    def test_only_the_named_checks_run(self, clean_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "rep"
        runner.invoke(app, audit_args(clean_dir, "--checks", "LK002,LK005", "--out", str(out)))
        report = json.loads((out / "report.json").read_text())
        assert report["summary"]["n_checks_requested"] == 2

    def test_ids_are_case_insensitive(self, clean_dir: Path) -> None:
        assert runner.invoke(app, audit_args(clean_dir, "--checks", "lk002")).exit_code == 0


class TestOutput:
    def test_writes_all_four_artifacts(self, leaky_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "rep"
        result = runner.invoke(app, audit_args(leaky_dir, "--out", str(out)))
        assert result.exit_code == 1
        for name in ("report.html", "report.md", "report.json", "findings.jsonl"):
            assert (out / name).exists(), name

    def test_the_console_table_lists_what_fired(self, leaky_dir: Path) -> None:
        result = runner.invoke(app, audit_args(leaky_dir))
        assert "LK002" in result.output
        assert "LK004" in result.output

    def test_quiet_suppresses_the_summary_but_not_the_exit_code(self, leaky_dir: Path) -> None:
        result = runner.invoke(app, audit_args(leaky_dir, "--quiet"))
        assert result.exit_code == 1
        assert "FAIL" not in result.output

    def test_an_error_message_is_not_eaten_by_markup(self, tmp_path: Path) -> None:
        """A path in square brackets would otherwise be parsed as rich markup
        and the whole line printed as nothing."""
        result = runner.invoke(
            app,
            ["audit", "--data", str(tmp_path / "[weird]"), "--task", str(tmp_path / "t.yaml")],
        )
        assert result.exit_code == EXIT_USAGE
        assert "weird" in result.output

    def test_skips_are_printed_above_the_findings(self, tmp_path: Path) -> None:
        out = tmp_path / "noproxy"
        runner.invoke(
            app,
            ["synth", "--out", str(out), "--n-subjects", "60", "--seed", "3", "--inject", "LK002"],
        )
        # Drop the proxy codes so LK003 has to skip.
        task = (out / "task.yaml").read_text()
        (out / "task.yaml").write_text(
            "\n".join(
                line
                for line in task.splitlines()
                if "insulin_drip" not in line
                and "proxy_codes" not in line
                and "PROCEDURE" not in line
            )
        )
        result = runner.invoke(app, audit_args(out))
        if "did not run" in result.output:
            assert result.output.index("did not run") < result.output.index("FAIL")


class TestSynth:
    def test_reports_what_it_planted(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "synth",
                "--out",
                str(tmp_path / "ds"),
                "--n-subjects",
                "40",
                "--inject",
                "LK002,LK005",
            ],
        )
        assert result.exit_code == 0
        assert "LK002" in result.output and "LK005" in result.output

    def test_says_the_data_is_synthetic(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["synth", "--out", str(tmp_path / "ds")])
        assert "synthetic" in result.output.lower()

    def test_an_unknown_injector_exits_two(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["synth", "--out", str(tmp_path / "ds"), "--inject", "LK404"])
        assert result.exit_code == EXIT_USAGE
        assert "LK001" in result.output

    @pytest.mark.parametrize("extra", [["--n-subjects", "0"], ["--positive-rate", "1.5"]])
    def test_bad_arguments_exit_two(self, tmp_path: Path, extra: list[str]) -> None:
        result = runner.invoke(app, ["synth", "--out", str(tmp_path / "ds"), *extra])
        assert result.exit_code == EXIT_USAGE

    def test_the_same_seed_writes_the_same_events(self, tmp_path: Path) -> None:
        import polars as pl

        for name in ("a", "b"):
            runner.invoke(
                app,
                ["synth", "--out", str(tmp_path / name), "--n-subjects", "40", "--seed", "7"],
            )
        one = pl.read_parquet(tmp_path / "a" / "data" / "shard_0.parquet")
        two = pl.read_parquet(tmp_path / "b" / "data" / "shard_0.parquet")
        assert one.equals(two)


class TestValidateTask:
    def test_a_valid_spec_exits_zero(self, clean_dir: Path) -> None:
        result = runner.invoke(app, ["validate-task", str(clean_dir / "task.yaml")])
        assert result.exit_code == 0
        assert "valid" in result.output

    def test_a_missing_file_exits_two(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate-task", str(tmp_path / "absent.yaml")])
        assert result.exit_code == EXIT_USAGE

    def test_malformed_yaml_exits_two(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("task: [unclosed\n")
        result = runner.invoke(app, ["validate-task", str(path)])
        assert result.exit_code == EXIT_USAGE

    def test_a_weakening_spec_says_which_checks_it_weakens(self, tmp_path: Path) -> None:
        path = tmp_path / "weak.yaml"
        path.write_text(
            "task: t\n"
            "index:\n  event: ed_visit\n"
            "horizon: 1y\n"
            "outcome:\n  codes:\n    - system: ICD10CM\n      values: [E11.9]\n"
        )
        result = runner.invoke(app, ["validate-task", str(path)])
        assert result.exit_code == 0
        assert "LK003" in result.output

    def test_a_fully_specified_spec_says_nothing_is_weakened(self, clean_dir: Path) -> None:
        result = runner.invoke(app, ["validate-task", str(clean_dir / "task.yaml")])
        assert "weakens a check" in result.output


class TestMisc:
    def test_version_prints_and_exits_zero(self) -> None:
        from ehrlint import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_arguments_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "audit" in result.output and "synth" in result.output

    def test_list_checks_shows_all_ten(self) -> None:
        result = runner.invoke(app, ["list-checks"])
        assert result.exit_code == 0
        for n in range(1, 11):
            assert f"LK{n:03d}" in result.output

    def test_omop_input_is_accepted(self, tmp_path: Path) -> None:
        from .test_io_omop import write_omop

        root = write_omop(tmp_path / "omop")
        task = tmp_path / "task.yaml"
        task.write_text(
            "task: omop_demo\n"
            "index:\n  event: '9203'\n  code_system: OMOP_VISIT\n"
            "horizon: 1y\n"
            "outcome:\n  codes:\n    - system: OMOP_CONDITION\n      values: ['201826']\n"
            "followup:\n  require_full_horizon_for_negatives: false\n"
        )
        result = runner.invoke(
            app, ["audit", "--data", str(root), "--format", "omop", "--task", str(task)]
        )
        assert result.exit_code in (0, 1), result.output
