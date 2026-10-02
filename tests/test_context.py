"""The audit context: index-date resolution, code matching, and warnings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ehrlint.context import AuditContext
from ehrlint.exceptions import InputError, SplitsError, TaskSpecError
from ehrlint.io.features import build_feature_matrix
from ehrlint.schemas.events import empty_events
from ehrlint.schemas.task import CodeSet, TaskSpec
from ehrlint.synth.generator import OUTCOME_CODES, OUTCOME_SYSTEM, default_task, generate


class TestLoaders:
    def test_from_meds_reads_the_whole_dataset(self, meds_dir: Path) -> None:
        ctx = AuditContext.from_meds(meds_dir, task_spec=meds_dir / "task.yaml")
        assert ctx.n_subjects > 0
        assert ctx.source_format == "meds"

    def test_from_meds_discovers_the_datasets_own_splits(self, meds_dir: Path) -> None:
        """MEDS keeps them in metadata/; requiring a separate file is busywork."""
        ctx = AuditContext.from_meds(meds_dir, task_spec=meds_dir / "task.yaml")
        assert ctx.splits is not None
        assert any("dataset's own splits" in w for w in ctx.all_warnings())

    def test_an_explicit_splits_path_wins(self, meds_dir: Path) -> None:
        ctx = AuditContext.from_meds(
            meds_dir,
            task_spec=meds_dir / "task.yaml",
            splits_path=meds_dir / "splits.json",
        )
        assert ctx.splits is not None
        assert not any("dataset's own splits" in w for w in ctx.all_warnings())

    def test_the_task_spec_may_be_an_object(self) -> None:
        dataset = generate(20, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        assert ctx.task.task == "dka_2y"

    def test_an_unreadable_task_spec_is_reported(self, tmp_path: Path) -> None:
        dataset = generate(10, seed=0)
        with pytest.raises(TaskSpecError):
            AuditContext.from_frames(dataset.events, task_spec=tmp_path / "absent.yaml")

    def test_an_unreadable_splits_file_is_reported(self, tmp_path: Path) -> None:
        dataset = generate(10, seed=0)
        with pytest.raises((SplitsError, InputError)):
            AuditContext.from_meds(
                tmp_path / "absent", task_spec=dataset.task, splits_path=tmp_path / "s.json"
            )

    def test_from_frames_accepts_a_splits_mapping(self) -> None:
        dataset = generate(20, seed=0)
        ctx = AuditContext.from_frames(
            dataset.events, task_spec=dataset.task, splits=dataset.splits_mapping()
        )
        assert ctx.splits is not None and len(ctx.splits) == 3


class TestIndexDates:
    def test_the_labels_prediction_time_wins_over_the_index_event(self) -> None:
        """An explicit label time is the task author's own statement.

        Re-deriving it from an event code would silently audit a different task
        than the one that was trained.
        """
        dataset = generate(30, seed=0)
        shifted = dataset.labels.with_columns(pl.col("index_date") - pl.duration(days=10))
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task, labels=shifted)
        resolved = ctx.index_dates()
        expected = dict(
            zip(
                shifted["subject_id"].to_list(),
                shifted["index_date"].to_list(),
                strict=True,
            )
        )
        for row in resolved.iter_rows(named=True):
            assert row["index_date"] == expected[row["subject_id"]]

    def test_falls_back_to_the_index_event_code(self) -> None:
        dataset = generate(30, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        assert ctx.index_dates().height == dataset.n_subjects

    def test_is_cached_so_a_battery_resolves_once(self) -> None:
        ctx = AuditContext.from_frames(generate(20, seed=0).events, task_spec=default_task())
        assert ctx.index_dates() is ctx.index_dates()

    def test_a_subject_with_no_index_event_is_absent_rather_than_guessed(self) -> None:
        dataset = generate(20, seed=0)
        without = dataset.events.filter(
            ~((pl.col("subject_id") == "S000001") & (pl.col("code") == "ed_visit"))
        )
        ctx = AuditContext.from_frames(without, task_spec=dataset.task)
        resolved = set(ctx.index_dates()["subject_id"].to_list())
        assert "S000001" not in resolved


class TestCodeMatching:
    def test_matches_on_system_and_value(self) -> None:
        dataset = generate(40, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        matched = ctx.match_code_sets(
            ctx.events, [CodeSet(system=OUTCOME_SYSTEM, values=list(OUTCOME_CODES))]
        )
        assert matched.height > 0
        assert set(matched["code"].unique().to_list()) <= set(OUTCOME_CODES)

    def test_a_systemless_code_set_matches_the_bare_code(self) -> None:
        dataset = generate(40, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        matched = ctx.match_code_sets(ctx.events, [CodeSet(values=list(OUTCOME_CODES))])
        assert matched.height > 0

    def test_a_wrong_system_matches_nothing(self) -> None:
        """Otherwise an ICD code would match a LOINC set with the same string."""
        dataset = generate(40, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        matched = ctx.match_code_sets(
            ctx.events, [CodeSet(system="SNOMED", values=list(OUTCOME_CODES))]
        )
        assert matched.height == 0

    def test_an_empty_code_set_list_matches_nothing(self) -> None:
        dataset = generate(20, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        assert ctx.match_code_sets(ctx.events, []).height == 0


class TestDerivedViews:
    def test_timed_excludes_static_events(self) -> None:
        dataset = generate(30, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        assert ctx.timed().height < ctx.events.height
        assert ctx.timed()["time"].null_count() == 0

    def test_events_with_index_attaches_the_index_date(self) -> None:
        dataset = generate(30, seed=0)
        ctx = AuditContext.from_frames(
            dataset.events, task_spec=dataset.task, labels=dataset.labels
        )
        joined = ctx.events_with_index()
        assert "index_date" in joined.columns
        assert joined["index_date"].null_count() == 0

    def test_has_encounters_and_code_systems_are_reported(self) -> None:
        dataset = generate(20, seed=0)
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        assert ctx.has_encounters is True
        assert ctx.has_code_systems is True

    def test_has_encounters_is_false_when_they_are_all_null(self) -> None:
        dataset = generate(20, seed=0)
        stripped = dataset.events.with_columns(pl.lit(None, dtype=pl.String).alias("encounter_id"))
        ctx = AuditContext.from_frames(stripped, task_spec=dataset.task)
        assert ctx.has_encounters is False

    def test_the_data_hash_is_order_invariant(self) -> None:
        dataset = generate(30, seed=0)
        one = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
        two = AuditContext.from_frames(dataset.events.reverse(), task_spec=dataset.task)
        assert one.data_hash == two.data_hash

    def test_subject_split_is_a_long_frame(self) -> None:
        dataset = generate(20, seed=0)
        ctx = AuditContext.from_frames(
            dataset.events, task_spec=dataset.task, splits=dataset.splits_mapping()
        )
        assert set(ctx.subject_split().columns) == {"subject_id", "split"}


class TestWarnings:
    def test_a_split_subject_absent_from_the_events_is_flagged(self) -> None:
        dataset = generate(20, seed=0)
        mapping = dataset.splits_mapping()
        mapping["train"].append("GHOST")
        ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task, splits=mapping)
        assert any("do not appear in the event data" in w for w in ctx.all_warnings())

    def test_a_weakening_task_spec_is_flagged(self) -> None:
        spec = TaskSpec.from_dict(
            {
                "task": "t",
                "index": {"event": "ed_visit"},
                "horizon": "1y",
                "outcome": {"codes": [{"system": "ICD10CM", "values": ["E11.9"]}]},
            }
        )
        ctx = AuditContext.from_frames(generate(20, seed=0).events, task_spec=spec)
        assert any("LK003" in w for w in ctx.all_warnings())

    def test_an_empty_event_table_is_flagged(self) -> None:
        ctx = AuditContext.from_frames(empty_events(), task_spec=default_task())
        assert any("empty" in w for w in ctx.all_warnings())

    def test_the_warnings_are_deduplicated(self) -> None:
        ctx = AuditContext.from_frames(generate(20, seed=0).events, task_spec=default_task())
        warnings = ctx.all_warnings()
        assert len(warnings) == len(set(warnings))


class TestDescribe:
    def test_covers_the_report_header(self) -> None:
        dataset = generate(20, seed=0)
        ctx = AuditContext.from_frames(
            dataset.events,
            task_spec=dataset.task,
            labels=dataset.labels,
            splits=dataset.splits_mapping(),
        )
        described = ctx.describe()
        for key in (
            "source_format",
            "n_events",
            "n_subjects",
            "n_timed_events",
            "n_static_events",
            "has_encounters",
            "task",
        ):
            assert key in described

    def test_includes_the_feature_matrix_when_one_is_supplied(self) -> None:
        dataset = generate(20, seed=0)
        features = pl.DataFrame(
            {
                "subject_id": dataset.labels["subject_id"],
                "prediction_time": dataset.labels["index_date"],
                "age": [40] * dataset.labels.height,
            }
        )
        ctx = AuditContext.from_frames(
            dataset.events,
            task_spec=dataset.task,
            labels=dataset.labels,
            features=build_feature_matrix(features, source="test"),
        )
        assert ctx.describe()["features"]["n_features"] == 1

    def test_is_json_serializable(self) -> None:
        import json

        dataset = generate(20, seed=0)
        ctx = AuditContext.from_frames(
            dataset.events, task_spec=dataset.task, splits=dataset.splits_mapping()
        )
        json.dumps(ctx.describe(), default=str)


def test_a_dataset_with_one_subject_still_builds_a_context() -> None:
    dataset = generate(1, seed=0)
    ctx = AuditContext.from_frames(dataset.events, task_spec=dataset.task)
    assert ctx.n_subjects == 1


def test_subject_ids_are_sorted() -> None:
    ctx = AuditContext.from_frames(generate(30, seed=0).events, task_spec=default_task())
    assert ctx.subject_ids == sorted(ctx.subject_ids)


def test_an_event_table_missing_a_required_column_is_rejected() -> None:
    from ehrlint.exceptions import SchemaError

    with pytest.raises(SchemaError, match="code"):
        AuditContext.from_frames(
            pl.DataFrame({"subject_id": ["a"], "time": [datetime(2020, 1, 1)]}),
            task_spec=default_task(),
        )
