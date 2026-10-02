"""The task spec: durations, code sets, consistency, and what it weakens."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ehrlint.exceptions import TaskSpecError
from ehrlint.schemas.task import CodeSet, Duration, TaskSpec
from ehrlint.synth.generator import default_task


class TestDuration:
    @pytest.mark.parametrize(
        ("text", "n", "unit"), [("30d", 30, "d"), ("12w", 12, "w"), ("6m", 6, "m"), ("2y", 2, "y")]
    )
    def test_parses_the_documented_forms(self, text: str, n: float, unit: str) -> None:
        parsed = Duration.parse(text)
        assert parsed is not None
        assert (parsed.n, parsed.unit) == (n, unit)

    def test_is_case_insensitive_on_the_unit(self) -> None:
        assert str(Duration.parse("2Y")) == "2y"

    @pytest.mark.parametrize("text", ["2 years", "", "y2", "-1d", "0d", "two y", "2q"])
    def test_rejects_anything_else_by_naming_the_accepted_forms(self, text: str) -> None:
        """A horizon is load-bearing: a typo must not become a different number."""
        with pytest.raises(TaskSpecError, match="30d"):
            Duration.parse(text)

    def test_parse_is_idempotent(self) -> None:
        once = Duration.parse("2y")
        assert Duration.parse(once) is once

    def test_none_parses_to_none(self) -> None:
        assert Duration.parse(None) is None

    @pytest.mark.parametrize(
        ("text", "days"), [("1d", 1.0), ("1w", 7.0), ("1m", 30.4375), ("1y", 365.25)]
    )
    def test_days_uses_the_documented_conversions(self, text: str, days: float) -> None:
        parsed = Duration.parse(text)
        assert parsed is not None
        assert parsed.days == pytest.approx(days)

    def test_round_trips_through_its_string_form(self) -> None:
        assert str(Duration.parse("2y")) == "2y"
        assert str(Duration.parse("2.5y")) == "2.5y"

    def test_is_frozen(self) -> None:
        duration = Duration.parse("2y")
        assert duration is not None
        with pytest.raises(ValidationError, match=r"frozen|immutable"):
            duration.n = 3  # type: ignore[misc]


class TestCodeSet:
    def test_strips_and_drops_blanks(self) -> None:
        assert CodeSet(values=[" E11.9 ", "", "  "]).values == ["E11.9"]

    def test_rejects_an_all_blank_set(self) -> None:
        with pytest.raises(ValidationError, match="non-blank"):
            CodeSet(values=["", " "])

    def test_rejects_an_empty_set(self) -> None:
        with pytest.raises(ValidationError, match="at least 1 item"):
            CodeSet(values=[])


class TestTaskSpecConsistency:
    def test_requiring_full_horizon_without_a_horizon_is_rejected(self) -> None:
        with pytest.raises(TaskSpecError, match="no horizon is declared"):
            TaskSpec.from_dict(
                {
                    "task": "t",
                    "index": {"event": "x"},
                    "outcome": {"codes": [{"system": "ICD10CM", "values": ["A"]}]},
                    "followup": {"require_full_horizon_for_negatives": True},
                }
            )

    def test_a_one_sided_outcome_window_is_rejected(self) -> None:
        with pytest.raises(TaskSpecError, match="exactly two"):
            TaskSpec.from_dict(
                {
                    "task": "t",
                    "index": {"event": "x"},
                    "horizon": "1y",
                    "outcome": {
                        "codes": [{"system": "ICD10CM", "values": ["A"]}],
                        "window": ["index_date"],
                    },
                }
            )

    def test_an_unknown_key_is_rejected_rather_than_ignored(self) -> None:
        """A typo in a spec key must not silently change the task audited."""
        with pytest.raises(TaskSpecError):
            TaskSpec.from_dict(
                {
                    "task": "t",
                    "index": {"event": "x"},
                    "horzion": "1y",
                    "outcome": {"codes": [{"system": "ICD10CM", "values": ["A"]}]},
                }
            )


class TestTaskSpecWarnings:
    def test_the_default_task_weakens_nothing(self) -> None:
        """The baseline the synthetic data satisfies must be fully auditable."""
        assert default_task().warnings() == []

    def test_a_missing_proxy_list_names_lk003(self) -> None:
        spec = TaskSpec.from_dict(
            {
                "task": "t",
                "index": {"event": "x"},
                "horizon": "1y",
                "outcome": {"codes": [{"system": "ICD10CM", "values": ["A"]}]},
                "splits": {"group_column": "encounter_id"},
            }
        )
        assert any("LK003" in w for w in spec.warnings())

    def test_a_missing_horizon_names_lk010(self) -> None:
        spec = TaskSpec.from_dict(
            {
                "task": "t",
                "index": {"event": "x"},
                "outcome": {"codes": [{"system": "ICD10CM", "values": ["A"]}]},
                "followup": {"require_full_horizon_for_negatives": False},
            }
        )
        assert any("LK010" in w for w in spec.warnings())

    def test_a_systemless_code_set_warns_about_cross_matching(self) -> None:
        spec = TaskSpec.from_dict(
            {
                "task": "t",
                "index": {"event": "x"},
                "horizon": "1y",
                "outcome": {"codes": [{"values": ["A"]}]},
            }
        )
        assert any("cross-match" in w for w in spec.warnings())


class TestTaskSpecIo:
    def test_round_trips_through_yaml(self, tmp_path: Path) -> None:
        spec = default_task()
        path = spec.to_yaml(tmp_path / "task.yaml")
        assert TaskSpec.from_yaml(path).model_dump() == spec.model_dump()

    def test_a_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(TaskSpecError, match="not found"):
            TaskSpec.from_yaml(tmp_path / "absent.yaml")

    def test_malformed_yaml_is_a_task_spec_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("task: [unclosed\n")
        with pytest.raises(TaskSpecError):
            TaskSpec.from_yaml(path)

    def test_describe_covers_the_report_header_fields(self) -> None:
        described = default_task().describe()
        for key in ("task", "index_event", "horizon", "split_strategy"):
            assert key in described
