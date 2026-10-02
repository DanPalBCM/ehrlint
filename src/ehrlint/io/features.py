"""The optional feature matrix.

When a user supplies the actual training features, ehrlint can check them
directly rather than inferring what they must have contained from the event
stream. That is a strictly stronger audit: the event stream tells you what
*could* have leaked, the matrix tells you what *did*.

Expected shape: one row per prediction unit, with a `subject_id` column and
optionally a timestamp column naming when each feature was computed as of.
Everything else is treated as a feature.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ehrlint.exceptions import InputError, SchemaError

#: Column names accepted for the subject identifier.
SUBJECT_CANDIDATES: tuple[str, ...] = (
    "subject_id",
    "person_id",
    "patient_id",
    "subject",
    "id",
)

#: Column names accepted for the as-of timestamp of a feature row.
TIME_CANDIDATES: tuple[str, ...] = (
    "prediction_time",
    "index_date",
    "index_time",
    "as_of",
    "as_of_time",
    "time",
    "timestamp",
)

#: Columns that are identifiers or metadata, not features.
NON_FEATURE_CANDIDATES: tuple[str, ...] = (
    *SUBJECT_CANDIDATES,
    *TIME_CANDIDATES,
    "split",
    "fold",
    "label",
    "y",
    "outcome",
    "target",
    "boolean_value",
    "integer_value",
    "float_value",
    "categorical_value",
    "encounter_id",
    "visit_id",
    "hadm_id",
)


class FeatureMatrix:
    """A training feature matrix, with its identifier columns located.

    Attributes:
        df: The frame, with ``subject_id`` normalized to string.
        subject_column: The resolved subject id column, always renamed to
            ``subject_id``.
        time_column: The as-of timestamp column, or None.
        feature_columns: Everything treated as a feature.
        label_column: A detected label column, or None.
        source: Where it came from.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        subject_column: str,
        time_column: str | None,
        feature_columns: list[str],
        label_column: str | None = None,
        source: str | None = None,
    ) -> None:
        self.df = df
        self.subject_column = subject_column
        self.time_column = time_column
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.source = source

    def __len__(self) -> int:
        return self.df.height

    @property
    def n_features(self) -> int:
        """How many columns are treated as features."""
        return len(self.feature_columns)

    def subjects(self) -> set[str]:
        """Distinct subject ids in the matrix."""
        return set(self.df["subject_id"].drop_nulls().unique().to_list())

    def numeric_features(self) -> list[str]:
        """Feature columns with a numeric dtype."""
        return [c for c in self.feature_columns if self.df.schema[c].is_numeric()]

    def describe(self) -> dict[str, object]:
        """Summary for the report header."""
        return {
            "source": self.source,
            "n_rows": self.df.height,
            "n_features": self.n_features,
            "subject_column": self.subject_column,
            "time_column": self.time_column,
            "label_column": self.label_column,
            "n_subjects": len(self.subjects()),
        }

    def warnings(self) -> list[str]:
        """Things that limit what can be checked on this matrix."""
        out: list[str] = []
        if self.time_column is None:
            out.append(
                "the feature matrix has no as-of timestamp column, so ehrlint cannot "
                "verify from the matrix alone that features predate the index date. "
                f"Add one of {list(TIME_CANDIDATES)} to enable the direct check"
            )
        if not self.feature_columns:
            out.append("no feature columns were identified in the matrix")
        duplicated = self.df.height - self.df["subject_id"].n_unique()
        if duplicated > 0 and self.time_column is None:
            out.append(
                f"{duplicated} row(s) repeat a subject_id with no timestamp to "
                "distinguish them, so per-subject checks treat them as one unit"
            )
        return out


def read_feature_matrix(
    path: str | Path,
    *,
    subject_column: str | None = None,
    time_column: str | None = None,
) -> FeatureMatrix:
    """Read a feature matrix from parquet or CSV.

    Args:
        path: The file.
        subject_column: Override the subject id column.
        time_column: Override the as-of timestamp column.

    Returns:
        A :class:`FeatureMatrix`.

    Raises:
        InputError: If the file is missing or unreadable.
        SchemaError: If no subject id column can be found, listing the names it
            looked for — guessing one would risk auditing the wrong column.
    """
    source = Path(path)
    if not source.exists():
        raise InputError("feature matrix not found", path=str(source))

    try:
        if source.suffix in (".parquet", ".pq"):
            df = pl.read_parquet(source)
        elif source.suffix in (".csv", ".tsv"):
            separator = "\t" if source.suffix == ".tsv" else ","
            df = pl.read_csv(
                source, separator=separator, infer_schema_length=10000, try_parse_dates=True
            )
        else:
            raise InputError(
                f"unsupported feature matrix format {source.suffix!r}; use .parquet or .csv",
                path=str(source),
            )
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"could not read feature matrix: {exc}", path=str(source)) from exc

    return build_feature_matrix(
        df, subject_column=subject_column, time_column=time_column, source=str(source)
    )


def build_feature_matrix(
    df: pl.DataFrame,
    *,
    subject_column: str | None = None,
    time_column: str | None = None,
    source: str | None = None,
) -> FeatureMatrix:
    """Locate the identifier columns in a frame and wrap it."""
    lower = {c.lower(): c for c in df.columns}

    resolved_subject = subject_column
    if resolved_subject is None:
        resolved_subject = next((lower[c] for c in SUBJECT_CANDIDATES if c in lower), None)
    if resolved_subject is None:
        raise SchemaError(
            f"could not find a subject id column in the feature matrix. Looked for "
            f"{list(SUBJECT_CANDIDATES)}; found {df.columns}. Pass --feature-subject-column "
            "to name it explicitly",
            path=source,
            table="features",
        )
    if resolved_subject not in df.columns:
        raise SchemaError(
            f"subject column {resolved_subject!r} is not in the feature matrix; found {df.columns}",
            path=source,
            table="features",
            column=resolved_subject,
        )

    resolved_time = time_column
    if resolved_time is None:
        resolved_time = next((lower[c] for c in TIME_CANDIDATES if c in lower), None)
    if resolved_time is not None and resolved_time not in df.columns:
        raise SchemaError(
            f"time column {resolved_time!r} is not in the feature matrix; found {df.columns}",
            path=source,
            table="features",
            column=resolved_time,
        )

    label = next(
        (lower[c] for c in ("label", "y", "outcome", "target", "boolean_value") if c in lower),
        None,
    )

    casts = [pl.col(resolved_subject).cast(pl.String, strict=False).alias("subject_id")]
    if resolved_time is not None:
        casts.append(
            pl.col(resolved_time).cast(pl.Datetime("us"), strict=False).alias(resolved_time)
        )
    out = df.with_columns(casts)

    non_features = {c.lower() for c in NON_FEATURE_CANDIDATES}
    non_features.add("subject_id")
    if resolved_time:
        non_features.add(resolved_time.lower())
    features = [c for c in out.columns if c.lower() not in non_features]

    return FeatureMatrix(
        df=out,
        subject_column=resolved_subject,
        time_column=resolved_time,
        feature_columns=features,
        label_column=label,
        source=source,
    )


def read_labels(path: str | Path) -> pl.DataFrame:
    """Read a labels table, MEDS-shaped or ehrlint-shaped.

    MEDS `LabelSchema` is ``subject_id, prediction_time, boolean_value,
    integer_value, float_value, categorical_value`` — verified against `meds`
    0.4.1. ehrlint normalizes whichever typed value column is populated into a
    single ``label`` column, and takes ``prediction_time`` as the index date
    when the task spec does not define one from an event code.

    Returns:
        A frame with ``subject_id``, ``index_date``, and ``label``.
    """
    source = Path(path)
    if not source.exists():
        raise InputError("labels file not found", path=str(source))

    try:
        if source.suffix in (".parquet", ".pq"):
            df = pl.read_parquet(source)
        else:
            df = pl.read_csv(source, infer_schema_length=10000, try_parse_dates=True)
    except Exception as exc:
        raise InputError(f"could not read labels: {exc}", path=str(source)) from exc

    lower = {c.lower(): c for c in df.columns}
    subject = next((lower[c] for c in SUBJECT_CANDIDATES if c in lower), None)
    if subject is None:
        raise SchemaError(
            f"labels table has no subject id column; looked for {list(SUBJECT_CANDIDATES)}, "
            f"found {df.columns}",
            path=str(source),
            table="labels",
        )

    time_col = next(
        (lower[c] for c in ("prediction_time", "index_date", "index_time", "time") if c in lower),
        None,
    )
    if time_col is None:
        raise SchemaError(
            "labels table has no prediction_time or index_date column, so index dates "
            f"cannot be read from it; found {df.columns}",
            path=str(source),
            table="labels",
        )

    label_col = next(
        (
            lower[c]
            for c in (
                "boolean_value",
                "label",
                "y",
                "outcome",
                "target",
                "integer_value",
                "float_value",
            )
            if c in lower
        ),
        None,
    )

    exprs = [
        pl.col(subject).cast(pl.String, strict=False).alias("subject_id"),
        pl.col(time_col).cast(pl.Datetime("us"), strict=False).alias("index_date"),
    ]
    if label_col is not None:
        exprs.append(pl.col(label_col).cast(pl.Boolean, strict=False).alias("label"))
    else:
        exprs.append(pl.lit(None, dtype=pl.Boolean).alias("label"))

    return df.select(exprs)
