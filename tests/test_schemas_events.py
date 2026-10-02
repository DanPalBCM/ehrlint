"""The canonical event table."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from ehrlint.exceptions import SchemaError
from ehrlint.schemas.events import (
    CANONICAL_COLUMNS,
    coerce_events,
    empty_events,
    events_hash,
    split_meds_code,
    subjects,
    timed_events,
    validate_events,
)


def test_coerce_adds_optional_columns_and_orders_them() -> None:
    df = pl.DataFrame({"subject_id": [1], "time": [datetime(2020, 1, 1)], "code": ["x"]})
    out = coerce_events(df)
    assert tuple(out.columns) == CANONICAL_COLUMNS


def test_coerce_names_the_missing_required_column() -> None:
    df = pl.DataFrame({"subject_id": [1], "time": [datetime(2020, 1, 1)]})
    with pytest.raises(SchemaError, match="code"):
        coerce_events(df)


@pytest.mark.parametrize("raw", [1, "1", 1.0])
def test_subject_id_is_always_a_string(raw: object) -> None:
    """The normalization that prevents a silent empty join.

    An int64 subject_id in the events and a string one in the labels join to
    nothing, and a join that returns nothing makes every check report clean.
    """
    df = pl.DataFrame({"subject_id": [raw], "time": [datetime(2020, 1, 1)], "code": ["x"]})
    assert coerce_events(df).schema["subject_id"] == pl.String


def test_encounter_id_is_also_a_string() -> None:
    df = pl.DataFrame(
        {
            "subject_id": ["a"],
            "time": [datetime(2020, 1, 1)],
            "code": ["x"],
            "encounter_id": [42],
        }
    )
    assert coerce_events(df).schema["encounter_id"] == pl.String


def test_empty_events_is_already_canonical() -> None:
    empty = empty_events()
    assert empty.height == 0
    assert tuple(empty.columns) == CANONICAL_COLUMNS
    assert coerce_events(empty).equals(empty)


class TestSplitMedsCode:
    def test_splits_on_the_first_separator(self) -> None:
        df = pl.DataFrame({"code": ["ICD10CM//E11.9"]})
        out = split_meds_code(df)
        assert out["code_system"].to_list() == ["ICD10CM"]
        assert out["code"].to_list() == ["E11.9"]

    def test_a_code_with_no_separator_keeps_a_null_system(self) -> None:
        """Not a guessed one: an invented system would cross-match later."""
        out = split_meds_code(pl.DataFrame({"code": ["MEDS_DEATH"]}))
        assert out["code_system"].to_list() == [None]
        assert out["code"].to_list() == ["MEDS_DEATH"]

    def test_only_the_first_separator_splits(self) -> None:
        out = split_meds_code(pl.DataFrame({"code": ["A//B//C"]}))
        assert out["code_system"].to_list() == ["A"]
        assert out["code"].to_list() == ["B//C"]


class TestValidateEvents:
    def test_an_empty_table_says_every_check_will_skip(self) -> None:
        assert any("every check will skip" in w for w in validate_events(empty_events()))

    def test_null_times_are_explained_as_static_facts(self, tiny_events: pl.DataFrame) -> None:
        warnings = validate_events(coerce_events(tiny_events))
        assert any("null time" in w and "static" in w for w in warnings)

    def test_an_absent_encounter_id_names_the_checks_it_disables(self) -> None:
        df = coerce_events(
            pl.DataFrame({"subject_id": ["a"], "time": [datetime(2020, 1, 1)], "code": ["x"]})
        )
        warnings = validate_events(df)
        assert any("LK002" in w and "LK008" in w for w in warnings)

    def test_a_null_subject_id_is_reported(self) -> None:
        df = coerce_events(
            pl.DataFrame(
                {"subject_id": [None], "time": [datetime(2020, 1, 1)], "code": ["x"]},
                schema_overrides={"subject_id": pl.String},
            )
        )
        assert any("null subject_id" in w for w in validate_events(df))

    def test_a_clean_table_warns_about_nothing_but_static_events(
        self, clean_dataset: object
    ) -> None:
        warnings = validate_events(clean_dataset.events)  # type: ignore[attr-defined]
        assert all("null time" in w for w in warnings), warnings


class TestEventsHash:
    def test_is_invariant_to_row_order(self, tiny_events: pl.DataFrame) -> None:
        """Two readers that emit the same rows in different orders must agree."""
        df = coerce_events(tiny_events)
        assert events_hash(df) == events_hash(df.reverse())

    def test_changes_when_a_value_changes(self, tiny_events: pl.DataFrame) -> None:
        df = coerce_events(tiny_events)
        changed = df.with_columns(
            pl.when(pl.col("code") == "E11.65").then(pl.lit("E11.9")).otherwise(pl.col("code"))
        )
        assert events_hash(df) != events_hash(changed)

    def test_changes_when_a_row_is_dropped(self, tiny_events: pl.DataFrame) -> None:
        df = coerce_events(tiny_events)
        assert events_hash(df) != events_hash(df.head(2))


def test_subjects_is_sorted_and_drops_nulls(tiny_events: pl.DataFrame) -> None:
    assert subjects(coerce_events(tiny_events)) == ["A", "B"]


def test_timed_events_excludes_static_rows(tiny_events: pl.DataFrame) -> None:
    """A static fact must not be read as an event at the epoch."""
    df = coerce_events(tiny_events)
    assert df.height == 3
    assert timed_events(df).height == 2
    assert timed_events(df)["time"].null_count() == 0
