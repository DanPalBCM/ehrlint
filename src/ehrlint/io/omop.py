"""OMOP CDM ingestion.

Reads the seven clinical tables SPEC §3.1 lists and converts them to the
canonical event table. Parquet or CSV, whichever is present.

The conversion is a per-table mapping of *which column is the event time* and
*which column is the code*. That choice is the interesting part, and each one
is stated below rather than buried, because picking the wrong date column is
itself a leakage bug: using `condition_end_date` as the event time would place
a diagnosis at its resolution rather than its onset, and every
post-index check downstream would be measuring the wrong thing.

OMOP carries `visit_occurrence_id` on most clinical tables, which is what makes
the same-encounter checks (LK002, LK008) runnable on OMOP input where they
cannot run on plain MEDS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ehrlint.exceptions import InputError, SchemaError
from ehrlint.schemas.events import coerce_events


@dataclass(frozen=True, slots=True)
class OmopTableSpec:
    """How one OMOP table becomes canonical events.

    Attributes:
        table: OMOP table name.
        time_column: The column taken as the event time, in preference order.
        code_column: The concept id column.
        code_system: The vocabulary label written to ``code_system``.
        numeric_column: A numeric value column, if the table has one.
        text_column: A text value column, if the table has one.
        source_value_column: The source code, used when the concept id is 0 --
            OMOP's "no mapping" sentinel.
        required: Whether a missing table is an error.
    """

    table: str
    time_column: tuple[str, ...]
    code_column: str
    code_system: str
    numeric_column: str | None = None
    text_column: str | None = None
    source_value_column: str | None = None
    required: bool = False


#: The tables ehrlint reads, with the column choices made explicit.
#:
#: Start dates are used throughout: a condition's *onset* is when it became
#: true of the patient, which is what a prediction task indexes on. Using an
#: end date would systematically shift events later and mask post-index leaks.
OMOP_TABLES: tuple[OmopTableSpec, ...] = (
    OmopTableSpec(
        table="condition_occurrence",
        time_column=("condition_start_datetime", "condition_start_date"),
        code_column="condition_concept_id",
        code_system="OMOP_CONDITION",
        source_value_column="condition_source_value",
    ),
    OmopTableSpec(
        table="drug_exposure",
        time_column=("drug_exposure_start_datetime", "drug_exposure_start_date"),
        code_column="drug_concept_id",
        code_system="OMOP_DRUG",
        numeric_column="quantity",
        source_value_column="drug_source_value",
    ),
    OmopTableSpec(
        table="measurement",
        time_column=("measurement_datetime", "measurement_date"),
        code_column="measurement_concept_id",
        code_system="OMOP_MEASUREMENT",
        numeric_column="value_as_number",
        text_column="value_source_value",
        source_value_column="measurement_source_value",
    ),
    OmopTableSpec(
        table="procedure_occurrence",
        time_column=("procedure_datetime", "procedure_date"),
        code_column="procedure_concept_id",
        code_system="OMOP_PROCEDURE",
        source_value_column="procedure_source_value",
    ),
    OmopTableSpec(
        table="observation",
        time_column=("observation_datetime", "observation_date"),
        code_column="observation_concept_id",
        code_system="OMOP_OBSERVATION",
        numeric_column="value_as_number",
        text_column="value_as_string",
        source_value_column="observation_source_value",
    ),
    OmopTableSpec(
        table="visit_occurrence",
        time_column=("visit_start_datetime", "visit_start_date"),
        code_column="visit_concept_id",
        code_system="OMOP_VISIT",
        source_value_column="visit_source_value",
    ),
)

#: `person` supplies static demographics rather than timed events.
PERSON_TABLE = "person"

#: OMOP's sentinel for "this source code did not map to a standard concept".
UNMAPPED_CONCEPT_ID = "0"

#: Visit id column, present on most clinical tables. This is what makes the
#: encounter-based checks possible on OMOP input.
VISIT_COLUMN = "visit_occurrence_id"


def _read_table(root: Path, name: str) -> pl.DataFrame | None:
    """Read one OMOP table as parquet or CSV, case-insensitively."""
    for stem in (name, name.upper(), name.capitalize()):
        for suffix, reader in ((".parquet", pl.read_parquet), (".csv", pl.read_csv)):
            path = root / f"{stem}{suffix}"
            if path.exists():
                try:
                    if suffix == ".csv":
                        return pl.read_csv(path, infer_schema_length=10000, try_parse_dates=True)
                    return reader(path)
                except Exception as exc:
                    raise InputError(
                        f"could not read OMOP table: {exc}", path=str(path), table=name
                    ) from exc
    # A directory per table is also a common export shape.
    for stem in (name, name.upper()):
        directory = root / stem
        if directory.is_dir():
            shards = sorted(directory.glob("*.parquet")) or sorted(directory.glob("*.csv"))
            if shards:
                frames = [
                    pl.read_parquet(s)
                    if s.suffix == ".parquet"
                    else pl.read_csv(s, infer_schema_length=10000, try_parse_dates=True)
                    for s in shards
                ]
                return pl.concat(frames, how="diagonal_relaxed")
    return None


def _first_present(df: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def _convert_table(df: pl.DataFrame, spec: OmopTableSpec) -> pl.DataFrame:
    """Convert one OMOP table to canonical events."""
    if "person_id" not in df.columns:
        raise SchemaError(
            f"OMOP table {spec.table} has no person_id column; found {df.columns}",
            table=spec.table,
            column="person_id",
        )

    time_column = _first_present(df, spec.time_column)
    if time_column is None:
        raise SchemaError(
            f"OMOP table {spec.table} has none of the expected date columns "
            f"{list(spec.time_column)}; found {df.columns}",
            table=spec.table,
        )
    if spec.code_column not in df.columns:
        raise SchemaError(
            f"OMOP table {spec.table} has no {spec.code_column} column; found {df.columns}",
            table=spec.table,
            column=spec.code_column,
        )

    code = pl.col(spec.code_column).cast(pl.String, strict=False)
    # OMOP writes concept_id 0 for an unmapped source code. Falling back to the
    # source value keeps the event rather than collapsing every unmapped row
    # onto a single meaningless code "0".
    if spec.source_value_column and spec.source_value_column in df.columns:
        code = (
            pl.when(code.is_null() | (code == UNMAPPED_CONCEPT_ID))
            .then(pl.col(spec.source_value_column).cast(pl.String, strict=False))
            .otherwise(code)
        )

    system = (
        pl.when(
            pl.col(spec.code_column).cast(pl.String, strict=False).is_null()
            | (pl.col(spec.code_column).cast(pl.String, strict=False) == UNMAPPED_CONCEPT_ID)
        )
        .then(pl.lit(f"{spec.code_system}_SOURCE"))
        .otherwise(pl.lit(spec.code_system))
        if spec.source_value_column and spec.source_value_column in df.columns
        else pl.lit(spec.code_system)
    )

    selected = {
        "subject_id": pl.col("person_id").cast(pl.String, strict=False),
        "time": pl.col(time_column).cast(pl.Datetime("us"), strict=False),
        "code": code,
        "code_system": system,
        "numeric_value": (
            pl.col(spec.numeric_column).cast(pl.Float64, strict=False)
            if spec.numeric_column and spec.numeric_column in df.columns
            else pl.lit(None, dtype=pl.Float64)
        ),
        "text_value": (
            pl.col(spec.text_column).cast(pl.String, strict=False)
            if spec.text_column and spec.text_column in df.columns
            else pl.lit(None, dtype=pl.String)
        ),
        "encounter_id": (
            pl.col(VISIT_COLUMN).cast(pl.String, strict=False)
            if VISIT_COLUMN in df.columns
            else pl.lit(None, dtype=pl.String)
        ),
    }
    return df.select([expr.alias(name) for name, expr in selected.items()])


def _convert_person(df: pl.DataFrame) -> pl.DataFrame:
    """Convert `person` demographics to static (null-time) events.

    Static events carry a null time, matching MEDS. They are excluded from
    every time-ordering check, which is why they must *not* be given a
    fabricated timestamp — a birth date is a fact about a person, not an event
    that competes with the index date.
    """
    if "person_id" not in df.columns:
        raise SchemaError(
            "OMOP person table has no person_id column",
            table=PERSON_TABLE,
            column="person_id",
        )

    frames: list[pl.DataFrame] = []
    static_fields = (
        ("gender_concept_id", "OMOP_GENDER"),
        ("race_concept_id", "OMOP_RACE"),
        ("ethnicity_concept_id", "OMOP_ETHNICITY"),
        ("year_of_birth", "OMOP_BIRTH_YEAR"),
    )
    for column, system in static_fields:
        if column not in df.columns:
            continue
        frames.append(
            df.select(
                pl.col("person_id").cast(pl.String, strict=False).alias("subject_id"),
                pl.lit(None, dtype=pl.Datetime("us")).alias("time"),
                pl.col(column).cast(pl.String, strict=False).alias("code"),
                pl.lit(system).alias("code_system"),
                pl.lit(None, dtype=pl.Float64).alias("numeric_value"),
                pl.lit(None, dtype=pl.String).alias("text_value"),
                pl.lit(None, dtype=pl.String).alias("encounter_id"),
            ).drop_nulls("code")
        )

    birth_time = _first_present(df, ("birth_datetime",))
    if birth_time is not None:
        frames.append(
            df.select(
                pl.col("person_id").cast(pl.String, strict=False).alias("subject_id"),
                pl.col(birth_time).cast(pl.Datetime("us"), strict=False).alias("time"),
                pl.lit("MEDS_BIRTH").alias("code"),
                pl.lit(None, dtype=pl.String).alias("code_system"),
                pl.lit(None, dtype=pl.Float64).alias("numeric_value"),
                pl.lit(None, dtype=pl.String).alias("text_value"),
                pl.lit(None, dtype=pl.String).alias("encounter_id"),
            ).drop_nulls("time")
        )

    if not frames:
        return pl.DataFrame(schema=_canonical_dtypes())
    return pl.concat(frames, how="vertical_relaxed")


def _canonical_dtypes() -> dict[str, Any]:
    from ehrlint.schemas.events import CANONICAL_SCHEMA

    return dict(CANONICAL_SCHEMA)


def read_omop_events(data_dir: str | Path, *, include_person: bool = True) -> pl.DataFrame:
    """Read an OMOP CDM extract into the canonical event table.

    Args:
        data_dir: Directory of OMOP tables, as parquet or CSV.
        include_person: Convert `person` demographics to static events.

    Returns:
        A canonical event table.

    Raises:
        InputError: If the path is missing, or no recognized table was found —
            with the names it looked for, since a silently empty audit is worse
            than an error.
    """
    root = Path(data_dir)
    if not root.exists():
        raise InputError("OMOP data path not found", path=str(root))

    frames: list[pl.DataFrame] = []
    found: list[str] = []

    for spec in OMOP_TABLES:
        df = _read_table(root, spec.table)
        if df is None or df.height == 0:
            continue
        found.append(spec.table)
        frames.append(_convert_table(df, spec))

    if include_person:
        person = _read_table(root, PERSON_TABLE)
        if person is not None and person.height:
            found.append(PERSON_TABLE)
            frames.append(_convert_person(person))

    if not frames:
        expected = [s.table for s in OMOP_TABLES] + [PERSON_TABLE]
        raise InputError(
            f"no OMOP tables found under {root}. Looked for {expected} as .parquet, "
            ".csv, or a subdirectory of either",
            path=str(root),
        )

    combined = pl.concat(frames, how="vertical_relaxed")
    return coerce_events(combined, source="omop")


def omop_tables_present(data_dir: str | Path) -> list[str]:
    """Which recognized OMOP tables exist, for the report header."""
    root = Path(data_dir)
    if not root.exists():
        return []
    names = [s.table for s in OMOP_TABLES] + [PERSON_TABLE]
    return [n for n in names if _read_table(root, n) is not None]
