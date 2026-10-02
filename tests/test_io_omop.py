"""OMOP CDM ingestion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ehrlint.exceptions import InputError, SchemaError
from ehrlint.io.omop import (
    OMOP_TABLES,
    PERSON_TABLE,
    UNMAPPED_CONCEPT_ID,
    omop_tables_present,
    read_omop_events,
)


def write_omop(root: Path, *, as_csv: bool = False) -> Path:
    """A minimal OMOP extract: two people, four clinical tables."""
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "person": pl.DataFrame(
            {
                "person_id": [1, 2],
                "gender_concept_id": [8507, 8532],
                "race_concept_id": [8527, 8516],
                "year_of_birth": [1970, 1985],
                "birth_datetime": [datetime(1970, 5, 1), datetime(1985, 9, 12)],
            }
        ),
        "visit_occurrence": pl.DataFrame(
            {
                "person_id": [1, 2],
                "visit_occurrence_id": [100, 200],
                "visit_concept_id": [9203, 9203],
                "visit_start_datetime": [datetime(2020, 1, 5), datetime(2020, 2, 7)],
                "visit_end_datetime": [datetime(2020, 1, 6), datetime(2020, 2, 9)],
                "visit_source_value": ["ED", "ED"],
            }
        ),
        "condition_occurrence": pl.DataFrame(
            {
                "person_id": [1, 2],
                "visit_occurrence_id": [100, 200],
                "condition_concept_id": [201826, 0],
                "condition_start_datetime": [datetime(2020, 1, 5), datetime(2020, 2, 7)],
                "condition_end_datetime": [datetime(2020, 3, 1), datetime(2020, 4, 1)],
                "condition_source_value": ["E11.9", "E10.10"],
            }
        ),
        "measurement": pl.DataFrame(
            {
                "person_id": [1, 2],
                "visit_occurrence_id": [100, 200],
                "measurement_concept_id": [3004410, 3004410],
                "measurement_datetime": [datetime(2020, 1, 5), datetime(2020, 2, 7)],
                "value_as_number": [8.1, 11.4],
                "measurement_source_value": ["4548-4", "4548-4"],
            }
        ),
    }
    for name, frame in tables.items():
        if as_csv:
            frame.write_csv(root / f"{name}.csv")
        else:
            frame.write_parquet(root / f"{name}.parquet")
    return root


class TestStartDateChoice:
    def test_every_spec_takes_a_start_date(self) -> None:
        """Using an end date would shift events later and mask post-index leaks."""
        for spec in OMOP_TABLES:
            assert all(
                "start" in column or "_datetime" in column or "_date" in column
                for column in spec.time_column
            ), spec.table
            assert not any("end" in column for column in spec.time_column), spec.table

    def test_the_datetime_column_is_preferred_over_the_date(self) -> None:
        for spec in OMOP_TABLES:
            assert spec.time_column[0].endswith("datetime"), spec.table

    def test_an_end_date_in_the_file_is_not_used(self, tmp_path: Path) -> None:
        events = read_omop_events(write_omop(tmp_path / "omop"))
        conditions = events.filter(pl.col("code_system").str.starts_with("OMOP_CONDITION"))
        assert conditions["time"].min() == datetime(2020, 1, 5)
        assert conditions["time"].max() == datetime(2020, 2, 7)


class TestReadOmopEvents:
    def test_reads_parquet(self, tmp_path: Path) -> None:
        events = read_omop_events(write_omop(tmp_path / "omop"))
        assert events.height > 0
        assert set(events["code_system"].drop_nulls().unique().to_list()) >= {
            "OMOP_CONDITION",
            "OMOP_MEASUREMENT",
            "OMOP_VISIT",
        }

    def test_reads_csv_too(self, tmp_path: Path) -> None:
        events = read_omop_events(write_omop(tmp_path / "csv", as_csv=True))
        assert events.height > 0

    def test_a_missing_path_names_it(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match="not found"):
            read_omop_events(tmp_path / "absent")

    def test_an_empty_directory_lists_the_tables_it_looked_for(self, tmp_path: Path) -> None:
        """A silently empty audit is worse than an error."""
        (tmp_path / "empty").mkdir()
        with pytest.raises(InputError, match="condition_occurrence"):
            read_omop_events(tmp_path / "empty")

    def test_person_id_becomes_subject_id_as_a_string(self, tmp_path: Path) -> None:
        events = read_omop_events(write_omop(tmp_path / "omop"))
        assert events.schema["subject_id"] == pl.String
        assert set(events["subject_id"].unique().to_list()) == {"1", "2"}

    def test_the_visit_id_becomes_the_encounter_id(self, tmp_path: Path) -> None:
        """This is what makes LK002 and LK008 runnable on OMOP input."""
        events = read_omop_events(write_omop(tmp_path / "omop"))
        timed = events.filter(pl.col("code_system").str.starts_with("OMOP_CONDITION"))
        assert sorted(timed["encounter_id"].to_list()) == ["100", "200"]

    def test_a_numeric_value_is_carried_through(self, tmp_path: Path) -> None:
        events = read_omop_events(write_omop(tmp_path / "omop"))
        measurements = events.filter(pl.col("code_system") == "OMOP_MEASUREMENT")
        assert sorted(measurements["numeric_value"].to_list()) == [8.1, 11.4]


class TestUnmappedConcepts:
    def test_concept_id_zero_falls_back_to_the_source_value(self, tmp_path: Path) -> None:
        """Concept 0 is OMOP's "did not map"; keeping it would merge unrelated codes."""
        events = read_omop_events(write_omop(tmp_path / "omop"))
        conditions = events.filter(pl.col("code_system").str.starts_with("OMOP_CONDITION"))
        codes = conditions["code"].to_list()
        assert UNMAPPED_CONCEPT_ID not in codes
        assert "E10.10" in codes

    def test_a_source_code_is_labelled_as_one(self, tmp_path: Path) -> None:
        """A source code and a standard concept must not share a system label.

        Otherwise a task matching `OMOP_CONDITION` codes would silently also
        match ICD strings that were never mapped, which is a cross-vocabulary
        match dressed up as an exact one.
        """
        events = read_omop_events(write_omop(tmp_path / "omop"))
        unmapped = events.filter(pl.col("code") == "E10.10")
        assert unmapped["code_system"].to_list() == ["OMOP_CONDITION_SOURCE"]
        mapped = events.filter(pl.col("code") == "201826")
        assert mapped["code_system"].to_list() == ["OMOP_CONDITION"]


class TestPersonTable:
    def test_demographics_become_static_null_time_events(self, tmp_path: Path) -> None:
        """A birth date is a fact about a person, not an event at the index date."""
        events = read_omop_events(write_omop(tmp_path / "omop"))
        gender = events.filter(pl.col("code_system") == "OMOP_GENDER")
        assert gender.height == 2
        assert gender["time"].null_count() == 2

    def test_the_birth_datetime_stays_a_timed_event(self, tmp_path: Path) -> None:
        events = read_omop_events(write_omop(tmp_path / "omop"))
        births = events.filter(pl.col("code") == "MEDS_BIRTH")
        assert births.height == 2
        assert births["time"].null_count() == 0

    def test_person_can_be_excluded(self, tmp_path: Path) -> None:
        events = read_omop_events(write_omop(tmp_path / "omop"), include_person=False)
        assert events.filter(pl.col("code_system") == "OMOP_GENDER").height == 0

    def test_a_person_table_without_person_id_is_an_error(self, tmp_path: Path) -> None:
        root = write_omop(tmp_path / "omop")
        pl.DataFrame({"id": [1]}).write_parquet(root / f"{PERSON_TABLE}.parquet")
        with pytest.raises(SchemaError, match="person_id"):
            read_omop_events(root)


class TestTablesPresent:
    def test_lists_what_it_found(self, tmp_path: Path) -> None:
        present = omop_tables_present(write_omop(tmp_path / "omop"))
        assert set(present) == {
            "condition_occurrence",
            "measurement",
            "visit_occurrence",
            "person",
        }

    def test_a_missing_directory_lists_nothing(self, tmp_path: Path) -> None:
        assert omop_tables_present(tmp_path / "absent") == []
