"""The audit context: everything a check needs, resolved once.

Index dates are the expensive, fiddly part — they come either from an event
code or from a labels table, and every check needs them — so they are resolved
once here and cached rather than recomputed per check.

This module is an addition to the SPEC §4.1 layout. §5.2 names
`AuditContext.from_meds(...)` as public API but gives it no home; putting it in
`checks/base.py` would make the check protocol import the loaders, and putting
it in `cli.py` would make it unavailable to the Python API. See
``docs/decisions.md`` D-005.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ehrlint.exceptions import InputError, TaskSpecError
from ehrlint.io.features import FeatureMatrix, read_feature_matrix, read_labels
from ehrlint.io.meds import (
    find_meds_splits,
    read_dataset_metadata,
    read_meds_events,
    read_subject_id_map,
    validate_with_meds_package,
)
from ehrlint.io.omop import omop_tables_present, read_omop_events
from ehrlint.schemas.events import (
    CODE,
    CODE_SYSTEM,
    ENCOUNTER_ID,
    SUBJECT_ID,
    TIME,
    events_hash,
    subjects,
    timed_events,
    validate_events,
)
from ehrlint.schemas.splits import Splits
from ehrlint.schemas.task import TaskSpec


@dataclass
class AuditContext:
    """The resolved inputs for one audit.

    Attributes:
        events: The canonical event table.
        task: The task specification.
        splits: Split assignments, if supplied.
        features: A feature matrix, if supplied.
        labels: A labels frame with ``subject_id``, ``index_date``, ``label``.
        source_format: ``meds`` or ``omop``.
        source_path: Where the data came from.
        dataset_metadata: Whatever the MEDS manifest said.
        warnings: Non-fatal problems found while loading.
    """

    events: pl.DataFrame
    task: TaskSpec
    splits: Splits | None = None
    features: FeatureMatrix | None = None
    labels: pl.DataFrame | None = None
    source_format: str = "canonical"
    source_path: str | None = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    _index_cache: pl.DataFrame | None = field(default=None, repr=False, compare=False)

    # -- construction ---------------------------------------------------

    @classmethod
    def from_meds(
        cls,
        data_dir: str | Path,
        *,
        task_spec: str | Path | TaskSpec,
        splits_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        features_path: str | Path | None = None,
        encounter_column: str | None = None,
    ) -> AuditContext:
        """Build a context from a MEDS dataset directory.

        When ``splits_path`` is omitted, the dataset's own
        ``metadata/subject_splits.parquet`` is used if present — that is where
        MEDS keeps them, and requiring a separate file would be busywork.
        """
        events = read_meds_events(data_dir, encounter_column=encounter_column)
        task = _resolve_task(task_spec)

        warnings = validate_events(events, source="meds")
        warnings.extend(validate_with_meds_package(events))

        resolved_splits = None
        if splits_path is not None:
            resolved_splits = _load_splits(splits_path)
        else:
            discovered = find_meds_splits(data_dir)
            if discovered is not None:
                resolved_splits = Splits.from_meds(discovered, id_map=read_subject_id_map(data_dir))
                warnings.append(
                    f"using the dataset's own splits at {discovered.name}; pass "
                    "--splits to override"
                )

        return cls(
            events=events,
            task=task,
            splits=resolved_splits,
            features=read_feature_matrix(features_path) if features_path else None,
            labels=read_labels(labels_path) if labels_path else None,
            source_format="meds",
            source_path=str(data_dir),
            dataset_metadata=read_dataset_metadata(data_dir),
            warnings=warnings,
        )

    @classmethod
    def from_omop(
        cls,
        data_dir: str | Path,
        *,
        task_spec: str | Path | TaskSpec,
        splits_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        features_path: str | Path | None = None,
    ) -> AuditContext:
        """Build a context from an OMOP CDM extract."""
        events = read_omop_events(data_dir)
        task = _resolve_task(task_spec)
        warnings = validate_events(events, source="omop")
        warnings.append("OMOP tables found: " + ", ".join(omop_tables_present(data_dir)))
        return cls(
            events=events,
            task=task,
            splits=_load_splits(splits_path) if splits_path else None,
            features=read_feature_matrix(features_path) if features_path else None,
            labels=read_labels(labels_path) if labels_path else None,
            source_format="omop",
            source_path=str(data_dir),
            warnings=warnings,
        )

    @classmethod
    def from_frames(
        cls,
        events: pl.DataFrame,
        *,
        task_spec: str | Path | TaskSpec,
        splits: Splits | dict[str, list[str]] | None = None,
        labels: pl.DataFrame | None = None,
        features: FeatureMatrix | None = None,
    ) -> AuditContext:
        """Build a context from in-memory frames, for tests and the library API."""
        from ehrlint.schemas.events import coerce_events

        resolved_splits: Splits | None
        if isinstance(splits, dict):
            resolved_splits = Splits.from_mapping(splits, source="<dict>")
        else:
            resolved_splits = splits

        coerced = coerce_events(events, source="frames")
        return cls(
            events=coerced,
            task=_resolve_task(task_spec),
            splits=resolved_splits,
            features=features,
            labels=labels,
            source_format="canonical",
            warnings=validate_events(coerced, source="frames"),
        )

    # -- derived views --------------------------------------------------

    @property
    def n_subjects(self) -> int:
        """Distinct subjects in the event data. The denominator for every rate."""
        return len(self.subject_ids)

    @property
    def subject_ids(self) -> list[str]:
        """Distinct subject ids, sorted."""
        return subjects(self.events)

    @property
    def data_hash(self) -> str:
        """Content hash of the event table, recorded in the report."""
        return events_hash(self.events)

    @property
    def has_encounters(self) -> bool:
        """Whether any event carries an encounter id."""
        return (
            ENCOUNTER_ID in self.events.columns
            and self.events[ENCOUNTER_ID].null_count() < self.events.height
        )

    @property
    def has_code_systems(self) -> bool:
        """Whether any event carries a code system."""
        return (
            CODE_SYSTEM in self.events.columns
            and self.events[CODE_SYSTEM].null_count() < self.events.height
        )

    def timed(self) -> pl.DataFrame:
        """Events with a timestamp. Static events are excluded."""
        return timed_events(self.events)

    def index_dates(self) -> pl.DataFrame:
        """Each subject's index date, resolved once and cached.

        Two sources, in this order:

        1. a labels table, whose ``prediction_time`` is the index date — this
           wins because it is what the model was actually trained against;
        2. the task spec's index event code.

        Returns:
            A frame with ``subject_id`` and ``index_date``. Subjects with no
            resolvable index date are **absent**, not given a fabricated one.
        """
        if self._index_cache is not None:
            return self._index_cache

        resolved: pl.DataFrame
        if self.labels is not None and "index_date" in self.labels.columns:
            resolved = (
                self.labels.select(["subject_id", "index_date"])
                .drop_nulls()
                .unique(subset=["subject_id"], keep="first")
            )
        elif self.task.index.event:
            matched = self._match_codes(
                self.timed(), {self.task.index.event}, self.task.index.code_system
            )
            aggregator = (
                pl.col(TIME).min() if self.task.index.occurrence == "first" else pl.col(TIME).max()
            )
            resolved = (
                matched.group_by(SUBJECT_ID)
                .agg(aggregator.alias("index_date"))
                .rename({SUBJECT_ID: "subject_id"})
            )
        else:
            resolved = pl.DataFrame(
                schema={"subject_id": pl.String, "index_date": pl.Datetime("us")}
            )

        self._index_cache = resolved.sort("subject_id")
        return self._index_cache

    def events_with_index(self) -> pl.DataFrame:
        """Timed events joined to their subject's index date.

        An inner join: a subject with no index date has no task to leak
        relative to, so their events are excluded rather than compared against
        a null.
        """
        index = self.index_dates()
        if index.height == 0:
            return (
                self.timed()
                .with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("index_date"))
                .clear()
            )
        return self.timed().join(index, left_on=SUBJECT_ID, right_on="subject_id", how="inner")

    def _match_codes(self, df: pl.DataFrame, codes: set[str], system: str | None) -> pl.DataFrame:
        """Filter events to a code set, honouring the system when given."""
        if not codes:
            return df.clear()
        predicate = pl.col(CODE).is_in(list(codes))
        if system is not None and self.has_code_systems:
            predicate = predicate & (pl.col(CODE_SYSTEM) == system)
        return df.filter(predicate)

    def match_code_sets(self, df: pl.DataFrame, code_sets: list[Any]) -> pl.DataFrame:
        """Filter events to any of several :class:`CodeSet` objects.

        A code set with no system matches the bare code in any vocabulary,
        which can cross-match; the context records a warning for that rather
        than refusing, since many datasets carry no systems at all.
        """
        if not code_sets:
            return df.clear()
        frames = [self._match_codes(df, set(cs.values), cs.system) for cs in code_sets]
        present = [f for f in frames if f.height]
        if not present:
            return df.clear()
        return pl.concat(present, how="vertical").unique()

    def subject_split(self) -> pl.DataFrame:
        """Subject id to split name, or an empty frame when no splits exist."""
        if self.splits is None:
            return pl.DataFrame(schema={"subject_id": pl.String, "split": pl.String})
        return self.splits.as_frame()

    def describe(self) -> dict[str, Any]:
        """Summary for the report header."""
        return {
            "source_format": self.source_format,
            "source_path": self.source_path,
            "data_hash": self.data_hash,
            "n_events": self.events.height,
            "n_subjects": self.n_subjects,
            "n_timed_events": self.timed().height,
            "n_static_events": self.events.height - self.timed().height,
            "has_encounters": self.has_encounters,
            "has_code_systems": self.has_code_systems,
            "n_index_dates": self.index_dates().height,
            "task": self.task.describe(),
            "splits": self.splits.describe() if self.splits else None,
            "features": self.features.describe() if self.features else None,
            "dataset_metadata": self.dataset_metadata,
        }

    def all_warnings(self) -> list[str]:
        """Loading warnings plus anything the task and splits flag.

        Collected here so the report shows every reason a result might be
        weaker than it looks, in one place above the findings.
        """
        out = list(self.warnings)
        out.extend(self.task.warnings())
        if self.splits is not None:
            out.extend(self.splits.warnings(set(self.subject_ids)))
        if self.features is not None:
            out.extend(self.features.warnings())
        if self.index_dates().height == 0:
            out.append(
                "no index date could be resolved for any subject, so every check that "
                "needs one will skip. Either set `index.event` to a code present in the "
                "data, or supply a labels table with prediction_time"
            )
        elif self.index_dates().height < self.n_subjects:
            missing = self.n_subjects - self.index_dates().height
            out.append(
                f"{missing} of {self.n_subjects} subject(s) have no resolvable index "
                "date and are excluded from index-relative checks"
            )
        return out


def _resolve_task(task_spec: str | Path | TaskSpec) -> TaskSpec:
    """Accept a TaskSpec, a path, or a dict."""
    if isinstance(task_spec, TaskSpec):
        return task_spec
    if isinstance(task_spec, dict):
        return TaskSpec.from_dict(task_spec)
    if isinstance(task_spec, (str, Path)):
        return TaskSpec.from_yaml(task_spec)
    raise TaskSpecError(f"cannot interpret {type(task_spec).__name__} as a task spec")


def _load_splits(path: str | Path) -> Splits:
    """Load splits from JSON or MEDS parquet, by extension."""
    source = Path(path)
    if not source.exists():
        raise InputError("splits file not found", path=str(source))
    if source.suffix in (".parquet", ".pq"):
        return Splits.from_meds(source)
    return Splits.from_json(source)
