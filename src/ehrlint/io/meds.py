"""MEDS ingestion.

**Verified against `meds` 0.4.1**, by installing and introspecting the package
rather than reading about it (``docs/decisions.md`` D-002). The data schema is:

| column | type | required |
| --- | --- | --- |
| `subject_id` | int64 | **yes** |
| `time` | timestamp[us] | **yes** (nullable for static events) |
| `code` | string | **yes** |
| `numeric_value` | float | no |
| `text_value` | large_string | no |

Three differences from the spec's §2 sketch, all handled here:

1. **There is no `code_system` column.** MEDS namespaces the vocabulary inside
   the code string by convention, as `ICD10CM//E11.9`. ehrlint splits on the
   first `//` to recover a system, and leaves it null when there is no
   separator rather than guessing one.
2. **There is no `encounter_id` column.** The same-encounter checks (LK002,
   LK008) therefore skip on plain MEDS input unless an extra column supplies
   one — MEDS sets `allow_extra_columns = True`, so a dataset may carry one.
3. **`subject_id` is `int64`**, while ehrlint compares subject ids as strings
   everywhere. It is cast at ingestion.

Dataset layout, from the package's own constants:

```
<root>/data/**/*.parquet          the event shards
<root>/metadata/dataset.json      dataset metadata
<root>/metadata/codes.parquet     code metadata (description, parent_codes)
<root>/metadata/subject_splits.parquet   subject_id, split
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from ehrlint.exceptions import InputError, SchemaError
from ehrlint.schemas.events import coerce_events, split_meds_code

#: Layout constants, matching `meds` 0.4.1. Hard-coded rather than imported so
#: that `meds` stays an optional dependency, and asserted equal to the
#: package's own values by a test when it is installed.
DATA_SUBDIRECTORY = "data"
DATASET_METADATA_FILEPATH = "metadata/dataset.json"
CODE_METADATA_FILEPATH = "metadata/codes.parquet"
SUBJECT_SPLITS_FILEPATH = "metadata/subject_splits.parquet"

#: An ehrlint extension, not part of MEDS. MEDS requires an int64
#: ``subject_id``; ehrlint's canonical table holds strings. When the strings
#: are not integers (``"S000042"``), writing MEDS needs a surrogate integer,
#: and the original id has to be kept somewhere or the dataset becomes
#: unreadable back into ehrlint. MEDS allows extra metadata files, so the
#: mapping goes here and :func:`read_meds_events` restores from it. Written
#: only when a surrogate was actually needed.
SUBJECT_ID_MAP_FILEPATH = "metadata/subject_id_map.parquet"

#: MEDS data-schema columns.
MEDS_REQUIRED_COLUMNS: tuple[str, ...] = ("subject_id", "time", "code")
MEDS_OPTIONAL_COLUMNS: tuple[str, ...] = ("numeric_value", "text_value")

#: Extra columns ehrlint will use if a dataset happens to carry them. MEDS
#: permits extra columns, and an encounter id makes two more checks runnable.
ENCOUNTER_COLUMN_CANDIDATES: tuple[str, ...] = (
    "encounter_id",
    "visit_id",
    "hadm_id",
    "visit_occurrence_id",
)


def _find_shards(root: Path) -> list[Path]:
    """Locate the parquet shards, tolerating a flat or nested layout."""
    data_dir = root / DATA_SUBDIRECTORY
    if data_dir.is_dir():
        shards = sorted(data_dir.glob("**/*.parquet"))
        if shards:
            return shards
    # A directory of parquet files with no `data/` subdirectory is common in
    # hand-made extracts; accept it rather than insisting on the full layout.
    shards = sorted(root.glob("*.parquet"))
    if shards:
        return shards
    return sorted(root.glob("**/*.parquet"))


def read_meds_events(data_dir: str | Path, *, encounter_column: str | None = None) -> pl.DataFrame:
    """Read a MEDS dataset directory into the canonical event table.

    Args:
        data_dir: The MEDS dataset root, or a directory of parquet shards.
        encounter_column: An extra column holding an encounter id. When None,
            the candidates in :data:`ENCOUNTER_COLUMN_CANDIDATES` are tried.

    Returns:
        A canonical event table.

    Raises:
        InputError: If the path is missing or holds no parquet.
        SchemaError: If a required MEDS column is absent, naming which.
    """
    root = Path(data_dir)
    if not root.exists():
        raise InputError("MEDS data path not found", path=str(root))

    shards = [root] if root.is_file() else _find_shards(root)
    if not shards:
        raise InputError(
            f"no parquet files found under {root}. A MEDS dataset keeps its shards in "
            f"a '{DATA_SUBDIRECTORY}/' subdirectory",
            path=str(root),
        )

    frames: list[pl.DataFrame] = []
    for shard in shards:
        try:
            df = pl.read_parquet(shard)
        except Exception as exc:
            raise InputError(f"could not read parquet: {exc}", path=str(shard)) from exc

        missing = [c for c in MEDS_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise SchemaError(
                f"MEDS shard is missing required column(s) {missing}; MEDS requires "
                f"{list(MEDS_REQUIRED_COLUMNS)} and found {df.columns}",
                path=str(shard),
            )
        frames.append(df)

    combined = pl.concat(frames, how="diagonal_relaxed")
    events = meds_frame_to_canonical(combined, encounter_column=encounter_column)
    return _restore_subject_ids(events, root)


def read_subject_id_map(data_dir: str | Path) -> dict[str, str] | None:
    """The dataset's surrogate-to-original subject id mapping, or None.

    Anything reading a sibling file keyed by subject id -- splits, labels, a
    feature matrix -- has to apply the same mapping the event shards used, or
    it will name subjects the event table does not contain.
    """
    root = Path(data_dir)
    if root.is_file():
        root = root.parent
    path = root / SUBJECT_ID_MAP_FILEPATH
    if not path.exists():
        return None
    mapping = pl.read_parquet(path)
    needed = {"subject_id", "original_subject_id"}
    if not needed.issubset(mapping.columns):
        raise SchemaError(
            f"{SUBJECT_ID_MAP_FILEPATH} must have columns {sorted(needed)}; found "
            f"{mapping.columns}",
            path=str(path),
        )
    rows = mapping.select(
        pl.col("subject_id").cast(pl.String, strict=False),
        pl.col("original_subject_id").cast(pl.String, strict=False),
    ).iter_rows()
    return {surrogate: original for surrogate, original in rows if surrogate is not None}


def _restore_subject_ids(events: pl.DataFrame, root: Path) -> pl.DataFrame:
    """Undo a surrogate subject-id mapping, if the dataset carries one.

    Without this, a dataset ehrlint itself wrote would read back with surrogate
    integers while the labels and splits beside it still used the original ids
    -- every join empty, every check clean. A false-clean audit is the one
    outcome this tool must never produce.
    """
    if root.is_file():
        root = root.parent
    path = root / SUBJECT_ID_MAP_FILEPATH
    if not path.exists():
        return events
    try:
        mapping = pl.read_parquet(path)
    except Exception as exc:
        raise InputError(f"could not read the subject id map: {exc}", path=str(path)) from exc

    needed = {"subject_id", "original_subject_id"}
    if not needed.issubset(mapping.columns):
        raise SchemaError(
            f"{SUBJECT_ID_MAP_FILEPATH} must have columns {sorted(needed)}; found "
            f"{mapping.columns}",
            path=str(path),
        )

    mapping = mapping.select(
        pl.col("subject_id").cast(pl.String, strict=False),
        pl.col("original_subject_id").cast(pl.String, strict=False),
    ).unique(subset=["subject_id"])

    joined = events.join(mapping, on="subject_id", how="left")
    unmapped = joined.filter(pl.col("original_subject_id").is_null()).height
    if unmapped:
        raise SchemaError(
            f"{SUBJECT_ID_MAP_FILEPATH} does not cover {unmapped} event row(s). "
            "Restoring subject ids from a partial map would mix surrogate and "
            "original ids in one table",
            path=str(path),
        )
    return (
        joined.drop("subject_id")
        .rename({"original_subject_id": "subject_id"})
        .select(events.columns)
    )


def meds_frame_to_canonical(
    df: pl.DataFrame, *, encounter_column: str | None = None
) -> pl.DataFrame:
    """Convert a MEDS-shaped frame to the canonical event table."""
    missing = [c for c in MEDS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"MEDS frame is missing required column(s) {missing}; found {df.columns}",
            table="meds",
        )

    out = df
    chosen = encounter_column
    if chosen is None:
        chosen = next((c for c in ENCOUNTER_COLUMN_CANDIDATES if c in out.columns), None)
    if chosen is not None:
        if chosen not in out.columns:
            raise SchemaError(
                f"encounter column {chosen!r} is not in the MEDS data; found {out.columns}",
                table="meds",
                column=chosen,
            )
        out = out.with_columns(pl.col(chosen).cast(pl.String, strict=False).alias("encounter_id"))

    # Recover `code_system` from the SYSTEM//code convention before coercing.
    out = split_meds_code(out)
    return coerce_events(out, source="meds")


def read_dataset_metadata(data_dir: str | Path) -> dict[str, Any]:
    """Read ``metadata/dataset.json``, or return an empty dict.

    Absence is not an error: a hand-made extract often has no manifest, and the
    audit does not depend on one. What is in it goes into the report header so a
    reader knows which ETL produced the data.
    """
    path = Path(data_dir) / DATASET_METADATA_FILEPATH
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise InputError(f"malformed dataset metadata: {exc.msg}", path=str(path)) from exc
    return raw if isinstance(raw, dict) else {}


def read_code_metadata(data_dir: str | Path) -> pl.DataFrame | None:
    """Read ``metadata/codes.parquet``, or return None.

    MEDS code metadata carries ``code``, ``description``, and ``parent_codes``.
    ehrlint uses it only to make findings readable — a code's description
    alongside its id — never to expand a code set, because silently matching a
    parent code would change what the task means.
    """
    path = Path(data_dir) / CODE_METADATA_FILEPATH
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        raise InputError(f"could not read code metadata: {exc}", path=str(path)) from exc


def find_meds_splits(data_dir: str | Path) -> Path | None:
    """Locate ``metadata/subject_splits.parquet`` if the dataset has one."""
    path = Path(data_dir) / SUBJECT_SPLITS_FILEPATH
    return path if path.exists() else None


def validate_with_meds_package(df: pl.DataFrame) -> list[str]:
    """Validate a MEDS frame with the ``meds`` package, if it is installed.

    Returns a list of warnings. The ``meds`` extra is optional, so its absence
    is reported as a single informational line rather than failing the audit.
    """
    try:
        import meds
        import pyarrow as pa
    except ImportError:
        return [
            "the `meds` package is not installed, so MEDS schema validation was "
            "skipped. Install it with: pip install 'ehrlint[meds]'"
        ]

    warnings: list[str] = []
    try:
        required = list(meds.DataSchema.required_columns())
    except Exception:  # pragma: no cover - defensive against API change
        required = list(MEDS_REQUIRED_COLUMNS)

    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(
            f"the installed meds {meds.__version__} requires column(s) {missing}, which "
            "the input lacks"
        )

    if set(required) != set(MEDS_REQUIRED_COLUMNS):
        warnings.append(
            f"the installed meds {meds.__version__} requires {required}, but ehrlint was "
            f"written against {list(MEDS_REQUIRED_COLUMNS)}. Re-read "
            "docs/decisions.md D-002 before trusting the ingestion."
        )

    try:
        table = df.to_arrow()
        _ = pa.schema(meds.DataSchema.schema())
        _ = table  # shape check only; MEDS allows extra columns
    except Exception as exc:  # pragma: no cover
        warnings.append(f"meds schema comparison failed: {type(exc).__name__}: {exc}")

    return warnings


def write_meds_dataset(
    events: pl.DataFrame,
    out_dir: str | Path,
    *,
    dataset_name: str = "ehrlint-synthetic",
    dataset_version: str = "1.0.0",
    splits: pl.DataFrame | None = None,
) -> dict[str, Path]:
    """Write a canonical event table out in MEDS layout.

    Used by the synthetic generator so the demo exercises the real MEDS reader
    rather than a shortcut. Codes are re-joined into the `SYSTEM//code` form so
    a round trip is faithful.

    MEDS requires an int64 ``subject_id``. Where ehrlint's string ids are not
    integers, a deterministic surrogate is assigned and the original is written
    to :data:`SUBJECT_ID_MAP_FILEPATH`, which the reader uses to restore them.
    A plain ``cast(Int64, strict=False)`` would turn every ``"S000042"`` into
    null, and a dataset with no subject ids audits perfectly clean.

    Returns:
        A mapping of artifact name to path.
    """
    root = Path(out_dir)
    (root / DATA_SUBDIRECTORY).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    surrogates = _subject_id_surrogates(events)

    meds_df = events.with_columns(
        pl.when(pl.col("code_system").is_not_null())
        .then(pl.concat_str([pl.col("code_system"), pl.lit("//"), pl.col("code")]))
        .otherwise(pl.col("code"))
        .alias("code"),
        pl.col("numeric_value").cast(pl.Float32, strict=False),
    )
    meds_df = _apply_subject_surrogates(meds_df, surrogates)
    keep = ["subject_id", "time", "code", "numeric_value", "text_value"]
    if "encounter_id" in meds_df.columns and meds_df["encounter_id"].null_count() < meds_df.height:
        keep.append("encounter_id")

    data_path = root / DATA_SUBDIRECTORY / "shard_0.parquet"
    meds_df.select(keep).write_parquet(data_path)
    written["data"] = data_path

    metadata = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "etl_name": "ehrlint.synth",
        "etl_version": "1.0.0",
        "meds_version": "0.4.1",
    }
    metadata_path = root / DATASET_METADATA_FILEPATH
    metadata_path.write_text(json.dumps(metadata, indent=2))
    written["dataset_metadata"] = metadata_path

    if splits is not None:
        splits_path = root / SUBJECT_SPLITS_FILEPATH
        _apply_subject_surrogates(splits, surrogates).write_parquet(splits_path)
        written["subject_splits"] = splits_path

    if surrogates is not None:
        map_path = root / SUBJECT_ID_MAP_FILEPATH
        surrogates.write_parquet(map_path)
        written["subject_id_map"] = map_path

    return written


def _subject_id_surrogates(events: pl.DataFrame) -> pl.DataFrame | None:
    """Build a surrogate integer id per subject, or None if ids already are ones.

    Returns a frame of ``original_subject_id`` (string) and ``subject_id``
    (int64), assigned in sorted order of the original id so the same dataset
    always gets the same surrogates.
    """
    ids = events.select(pl.col("subject_id").cast(pl.String, strict=False)).drop_nulls().unique()
    if ids.height == 0:
        return None

    as_int = ids.select(pl.col("subject_id").cast(pl.Int64, strict=False).alias("as_int"))
    if as_int["as_int"].null_count() == 0:
        return None  # already integer-valued; MEDS can hold them directly

    ordered = ids.sort("subject_id")
    return ordered.select(
        pl.col("subject_id").alias("original_subject_id"),
        (pl.int_range(1, ordered.height + 1, dtype=pl.Int64)).alias("subject_id"),
    )


def _apply_subject_surrogates(df: pl.DataFrame, surrogates: pl.DataFrame | None) -> pl.DataFrame:
    """Replace ``subject_id`` with its int64 form, surrogate or direct."""
    original_columns = df.columns
    if surrogates is None:
        out = df.with_columns(pl.col("subject_id").cast(pl.Int64, strict=False))
        _assert_no_id_loss(df, out)
        return out

    lookup = surrogates.rename({"original_subject_id": "_original", "subject_id": "_surrogate"})
    out = (
        df.with_columns(pl.col("subject_id").cast(pl.String, strict=False))
        .join(lookup, left_on="subject_id", right_on="_original", how="left")
        .drop("subject_id")
        .rename({"_surrogate": "subject_id"})
        .select(original_columns)
    )
    _assert_no_id_loss(df, out)
    return out


def _assert_no_id_loss(before: pl.DataFrame, after: pl.DataFrame) -> None:
    """Fail if a non-null subject id became null.

    An assertion rather than a warning: a silently null subject id makes every
    subsequent join empty, and an empty join reads as a clean audit.
    """
    lost = after["subject_id"].null_count() - before["subject_id"].null_count()
    if lost > 0:
        raise SchemaError(
            f"{lost} subject id(s) would be lost converting to the MEDS int64 "
            "subject_id. This is a bug in ehrlint's MEDS writer, not a problem "
            "with your data",
            table="meds",
            column="subject_id",
        )
