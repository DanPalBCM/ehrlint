"""The canonical event table.

Everything ehrlint audits is reduced to one long-format table:

| column | type | meaning |
| --- | --- | --- |
| `subject_id` | string | the person. **String, always** -- see below |
| `time` | datetime, nullable | when it happened; null for static facts |
| `code` | string | what happened |
| `code_system` | string, nullable | which vocabulary `code` belongs to |
| `numeric_value` | float, nullable | a measured value |
| `text_value` | string, nullable | a free-text value |
| `encounter_id` | string, nullable | for the same-encounter and group checks |

**`subject_id` is normalized to string** even though MEDS types it `int64`.
ehrlint joins subject ids against splits files, feature matrices, and label
tables that come from different tools, and an int/string mismatch between two
of them produces an empty join that looks like *"no leakage found"*. A silent
false-clean is the worst failure mode an auditor can have, so everything is
coerced to string at the boundary and compared as strings throughout.

`encounter_id` and `code_system` are **extensions beyond MEDS**, which has
neither. See ``docs/decisions.md`` D-002. Both are nullable, and the checks
that need them skip with a stated reason rather than silently passing when they
are absent.
"""

from __future__ import annotations

import hashlib
from typing import Any

import polars as pl

#: Column names of the canonical event table.
SUBJECT_ID = "subject_id"
TIME = "time"
CODE = "code"
CODE_SYSTEM = "code_system"
NUMERIC_VALUE = "numeric_value"
TEXT_VALUE = "text_value"
ENCOUNTER_ID = "encounter_id"

#: Columns every canonical event table must have.
REQUIRED_COLUMNS: tuple[str, ...] = (SUBJECT_ID, TIME, CODE)

#: Columns that may be absent. A check needing one skips with a reason.
OPTIONAL_COLUMNS: tuple[str, ...] = (CODE_SYSTEM, NUMERIC_VALUE, TEXT_VALUE, ENCOUNTER_ID)

#: The full column order ehrlint emits.
CANONICAL_COLUMNS: tuple[str, ...] = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)

#: Polars dtypes for the canonical table.
CANONICAL_SCHEMA: dict[str, Any] = {
    SUBJECT_ID: pl.String,
    TIME: pl.Datetime("us"),
    CODE: pl.String,
    CODE_SYSTEM: pl.String,
    NUMERIC_VALUE: pl.Float64,
    TEXT_VALUE: pl.String,
    ENCOUNTER_ID: pl.String,
}

#: MEDS reserves these codes for birth and death. Death matters for the
#: immortal-time and competing-event checks, so the names are kept rather than
#: treated as ordinary codes.
MEDS_BIRTH_CODE = "MEDS_BIRTH"
MEDS_DEATH_CODE = "MEDS_DEATH"

#: The separator MEDS uses to namespace a code inside the code string, e.g.
#: ``ICD10CM//E11.9``. MEDS has no `code_system` column, so this convention is
#: how a system is recovered on ingestion.
MEDS_CODE_SEPARATOR = "//"


def empty_events() -> pl.DataFrame:
    """An empty canonical event table with the right dtypes.

    Returned rather than a bare ``pl.DataFrame()`` so that downstream filters
    and joins behave identically on empty input. An auditor that crashes on an
    empty table is an auditor nobody runs twice.
    """
    return pl.DataFrame(schema=CANONICAL_SCHEMA)


def coerce_events(df: pl.DataFrame, *, source: str = "events") -> pl.DataFrame:
    """Coerce a frame to the canonical schema.

    Adds any missing optional columns as all-null, casts every column to its
    canonical dtype, and orders the columns. ``subject_id`` and
    ``encounter_id`` become **strings**, which is the normalization that
    prevents a silent empty join later.

    Args:
        df: The frame to coerce.
        source: Named in error messages.

    Returns:
        A frame with exactly :data:`CANONICAL_COLUMNS`, in order.

    Raises:
        SchemaError: If a required column is missing, naming which.
    """
    from ehrlint.exceptions import SchemaError

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"canonical event table is missing required column(s) {missing}; "
            f"required: {list(REQUIRED_COLUMNS)}, found: {df.columns}",
            table=source,
        )

    out = df
    for column in OPTIONAL_COLUMNS:
        if column not in out.columns:
            out = out.with_columns(pl.lit(None).cast(CANONICAL_SCHEMA[column]).alias(column))

    casts = []
    for column, dtype in CANONICAL_SCHEMA.items():
        casts.append(pl.col(column).cast(dtype, strict=False).alias(column))

    return out.with_columns(casts).select(list(CANONICAL_COLUMNS))


def split_meds_code(df: pl.DataFrame) -> pl.DataFrame:
    """Recover ``code_system`` from the MEDS ``SYSTEM//code`` convention.

    MEDS has no ``code_system`` column; the convention is to namespace inside
    the code string. This splits on the first separator, leaving ``code`` as
    the local part. A code with no separator keeps a **null** system rather
    than being assigned a guessed one.
    """
    if CODE not in df.columns:
        return df
    has_sep = pl.col(CODE).str.contains(MEDS_CODE_SEPARATOR, literal=True)
    parts = pl.col(CODE).str.splitn(MEDS_CODE_SEPARATOR, 2)
    return df.with_columns(
        pl.when(has_sep).then(parts.struct.field("field_0")).otherwise(None).alias(CODE_SYSTEM),
        pl.when(has_sep).then(parts.struct.field("field_1")).otherwise(pl.col(CODE)).alias(CODE),
    )


def validate_events(df: pl.DataFrame, *, source: str = "events") -> list[str]:
    """Check an event table for problems worth warning about.

    Returns human-readable warnings rather than raising: every one of these is
    survivable, and a completed audit is more useful than a hard stop. An empty
    list means nothing notable.
    """
    warnings: list[str] = []

    if df.height == 0:
        warnings.append(f"{source} is empty, so every check will skip")
        return warnings

    n_null_subject = df[SUBJECT_ID].null_count()
    if n_null_subject:
        warnings.append(
            f"{n_null_subject} event(s) have a null subject_id and cannot be attributed "
            "to anyone; they are excluded from per-subject checks"
        )

    n_null_code = df[CODE].null_count()
    if n_null_code:
        warnings.append(f"{n_null_code} event(s) have a null code")

    n_null_time = df[TIME].null_count()
    if n_null_time:
        warnings.append(
            f"{n_null_time} event(s) have a null time. In MEDS these are static facts "
            "(demographics, birth); they are excluded from time-ordering checks rather "
            "than treated as occurring at the epoch"
        )

    if CODE_SYSTEM in df.columns and df[CODE_SYSTEM].null_count() == df.height:
        warnings.append(
            "no event carries a code_system. Checks that match codes by system fall back "
            "to matching the bare code, which can cross-match between vocabularies"
        )

    if ENCOUNTER_ID in df.columns and df[ENCOUNTER_ID].null_count() == df.height:
        warnings.append(
            "no event carries an encounter_id, so the same-encounter (LK002) and "
            "encounter-group (LK008) checks will skip"
        )

    return warnings


def events_hash(df: pl.DataFrame) -> str:
    """A stable content hash of an event table.

    Recorded in the report so two audits can be compared and a finding can be
    tied to the exact data that produced it. Computed from the shape plus a
    digest of the sorted contents, so it is **invariant to row order** — the
    same data loaded by two different readers hashes the same.
    """
    hasher = hashlib.sha256()
    hasher.update(f"{df.height}x{df.width}".encode())
    hasher.update(",".join(sorted(df.columns)).encode())
    if df.height:
        ordered = df.sort([c for c in (SUBJECT_ID, TIME, CODE) if c in df.columns], nulls_last=True)
        for column in CANONICAL_COLUMNS:
            if column in ordered.columns:
                hasher.update(column.encode())
                values = ordered[column].cast(pl.String, strict=False).to_list()
                hasher.update(repr(values).encode())
    return hasher.hexdigest()[:16]


def subjects(df: pl.DataFrame) -> list[str]:
    """Distinct non-null subject ids, sorted. Deterministic ordering."""
    if df.height == 0 or SUBJECT_ID not in df.columns:
        return []
    return sorted(df[SUBJECT_ID].drop_nulls().unique().to_list())


def timed_events(df: pl.DataFrame) -> pl.DataFrame:
    """Events that carry a timestamp.

    Static events (null ``time``) are excluded from every time-ordering check.
    Treating a null time as the epoch would make every static fact look like a
    pre-index event, which is both wrong and the kind of wrong that produces
    confident false findings.
    """
    return df.filter(pl.col(TIME).is_not_null())
