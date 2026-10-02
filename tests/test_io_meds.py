"""MEDS ingestion, and the round trip through the real layout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ehrlint.exceptions import InputError, SchemaError
from ehrlint.io.meds import (
    CODE_METADATA_FILEPATH,
    DATA_SUBDIRECTORY,
    DATASET_METADATA_FILEPATH,
    SUBJECT_ID_MAP_FILEPATH,
    SUBJECT_SPLITS_FILEPATH,
    find_meds_splits,
    meds_frame_to_canonical,
    read_dataset_metadata,
    read_meds_events,
    read_subject_id_map,
    validate_with_meds_package,
    write_meds_dataset,
)
from ehrlint.schemas.events import events_hash
from ehrlint.synth.generator import SyntheticDataset

meds = pytest.importorskip("meds", reason="the `meds` extra is optional")


class TestVerifiedAgainstTheInstalledPackage:
    """SPEC §0: verify the schema against the package, not against memory."""

    def test_layout_constants_match_the_package(self) -> None:
        assert meds.data_subdirectory == DATA_SUBDIRECTORY
        assert meds.dataset_metadata_filepath == DATASET_METADATA_FILEPATH
        assert meds.code_metadata_filepath == CODE_METADATA_FILEPATH
        assert meds.subject_splits_filepath == SUBJECT_SPLITS_FILEPATH

    def test_the_required_columns_are_still_the_ones_ehrlint_expects(self) -> None:
        names = set(meds.DataSchema.schema().names)
        assert {"subject_id", "time", "code"} <= names

    def test_meds_still_has_no_code_system_or_encounter_column(self) -> None:
        """Both absences are why two checks skip on plain MEDS input."""
        names = set(meds.DataSchema.schema().names)
        assert "code_system" not in names
        assert "encounter_id" not in names

    def test_subject_id_is_still_int64(self) -> None:
        """The reason the writer needs a surrogate map for string ids."""
        schema = meds.DataSchema.schema()
        assert str(schema.field("subject_id").type) == "int64"

    def test_numeric_value_is_still_float32(self) -> None:
        """Which makes a MEDS round trip narrow numeric values; documented, not a bug."""
        schema = meds.DataSchema.schema()
        assert str(schema.field("numeric_value").type) == "float"

    def test_the_split_names_are_still_the_ones_aliased(self) -> None:
        from ehrlint.schemas.splits import SPLIT_ALIASES

        assert meds.train_split in SPLIT_ALIASES
        assert meds.tuning_split in SPLIT_ALIASES
        assert meds.held_out_split in SPLIT_ALIASES


class TestReadMedsEvents:
    def test_reads_a_written_dataset(self, meds_dir: Path) -> None:
        events = read_meds_events(meds_dir)
        assert events.height > 0

    def test_a_missing_path_names_it(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match="not found"):
            read_meds_events(tmp_path / "absent")

    def test_a_directory_with_no_parquet_says_where_shards_live(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(InputError, match=DATA_SUBDIRECTORY):
            read_meds_events(tmp_path / "empty")

    def test_a_missing_required_column_is_named(self, tmp_path: Path) -> None:
        root = tmp_path / "broken" / DATA_SUBDIRECTORY
        root.mkdir(parents=True)
        pl.DataFrame({"subject_id": [1], "time": [datetime(2020, 1, 1)]}).write_parquet(
            root / "shard_0.parquet"
        )
        with pytest.raises(SchemaError, match="code"):
            read_meds_events(tmp_path / "broken")

    def test_a_flat_directory_of_parquet_is_accepted(self, tmp_path: Path) -> None:
        """Hand-made extracts often have no data/ subdirectory."""
        pl.DataFrame(
            {"subject_id": [1], "time": [datetime(2020, 1, 1)], "code": ["x"]}
        ).write_parquet(tmp_path / "shard.parquet")
        assert read_meds_events(tmp_path).height == 1

    def test_several_shards_are_concatenated(self, tmp_path: Path) -> None:
        data = tmp_path / DATA_SUBDIRECTORY
        data.mkdir()
        for i in (1, 2):
            pl.DataFrame(
                {"subject_id": [i], "time": [datetime(2020, 1, 1)], "code": ["x"]}
            ).write_parquet(data / f"shard_{i}.parquet")
        assert read_meds_events(tmp_path).height == 2


class TestCodeSystemRecovery:
    def test_recovers_the_system_from_the_namespaced_code(self) -> None:
        df = pl.DataFrame(
            {"subject_id": [1], "time": [datetime(2020, 1, 1)], "code": ["ICD10CM//E11.9"]}
        )
        out = meds_frame_to_canonical(df)
        assert out["code_system"].to_list() == ["ICD10CM"]
        assert out["code"].to_list() == ["E11.9"]

    def test_an_encounter_column_is_picked_up_when_present(self) -> None:
        df = pl.DataFrame(
            {
                "subject_id": [1],
                "time": [datetime(2020, 1, 1)],
                "code": ["x"],
                "hadm_id": [77],
            }
        )
        assert meds_frame_to_canonical(df)["encounter_id"].to_list() == ["77"]

    def test_an_explicitly_named_absent_encounter_column_is_an_error(self) -> None:
        df = pl.DataFrame({"subject_id": [1], "time": [datetime(2020, 1, 1)], "code": ["x"]})
        with pytest.raises(SchemaError, match="nope"):
            meds_frame_to_canonical(df, encounter_column="nope")


class TestSubjectIdHandling:
    """The silent-null bug class: a lost subject id makes every join empty."""

    def test_string_ids_round_trip_through_a_surrogate_map(
        self, tmp_path: Path, clean_dataset: SyntheticDataset
    ) -> None:
        root = tmp_path / "meds"
        write_meds_dataset(clean_dataset.events, root, splits=clean_dataset.splits)
        shard = pl.read_parquet(root / DATA_SUBDIRECTORY / "shard_0.parquet")
        assert shard.schema["subject_id"] == pl.Int64
        assert shard["subject_id"].null_count() == 0

        restored = read_meds_events(root)
        assert sorted(restored["subject_id"].unique().to_list()) == sorted(
            clean_dataset.events["subject_id"].unique().to_list()
        )

    def test_the_map_is_written_only_when_a_surrogate_was_needed(self, tmp_path: Path) -> None:
        integer_ids = pl.DataFrame(
            {
                "subject_id": ["1", "2"],
                "time": [datetime(2020, 1, 1)] * 2,
                "code": ["x"] * 2,
                "code_system": [None, None],
                "numeric_value": [None, None],
                "text_value": [None, None],
                "encounter_id": [None, None],
            }
        )
        written = write_meds_dataset(integer_ids, tmp_path / "ints")
        assert "subject_id_map" not in written
        assert read_subject_id_map(tmp_path / "ints") is None

    def test_the_splits_file_uses_the_same_surrogates_as_the_shards(
        self, tmp_path: Path, clean_dataset: SyntheticDataset
    ) -> None:
        """Different ids in the two files would make LK005 and LK008 vacuous."""
        root = tmp_path / "meds"
        write_meds_dataset(clean_dataset.events, root, splits=clean_dataset.splits)
        shard = pl.read_parquet(root / DATA_SUBDIRECTORY / "shard_0.parquet")
        splits = pl.read_parquet(root / SUBJECT_SPLITS_FILEPATH)
        assert splits["subject_id"].null_count() == 0
        assert set(splits["subject_id"].to_list()) <= set(shard["subject_id"].to_list())

    def test_a_restored_dataset_keeps_every_event(
        self, tmp_path: Path, clean_dataset: SyntheticDataset
    ) -> None:
        root = tmp_path / "meds"
        write_meds_dataset(clean_dataset.events, root)
        assert read_meds_events(root).height == clean_dataset.events.height

    def test_a_partial_map_is_an_error_rather_than_a_silent_mix(self, tmp_path: Path) -> None:
        root = tmp_path / "meds"
        data = root / DATA_SUBDIRECTORY
        data.mkdir(parents=True)
        (root / "metadata").mkdir(parents=True)
        pl.DataFrame(
            {"subject_id": [1, 2], "time": [datetime(2020, 1, 1)] * 2, "code": ["x"] * 2}
        ).write_parquet(data / "shard_0.parquet")
        pl.DataFrame({"subject_id": [1], "original_subject_id": ["S1"]}).write_parquet(
            root / SUBJECT_ID_MAP_FILEPATH
        )
        with pytest.raises(SchemaError, match="does not cover"):
            read_meds_events(root)

    def test_a_malformed_map_names_the_columns_it_needs(self, tmp_path: Path) -> None:
        root = tmp_path / "meds"
        data = root / DATA_SUBDIRECTORY
        data.mkdir(parents=True)
        (root / "metadata").mkdir(parents=True)
        pl.DataFrame(
            {"subject_id": [1], "time": [datetime(2020, 1, 1)], "code": ["x"]}
        ).write_parquet(data / "shard_0.parquet")
        pl.DataFrame({"wrong": [1]}).write_parquet(root / SUBJECT_ID_MAP_FILEPATH)
        with pytest.raises(SchemaError, match="original_subject_id"):
            read_meds_events(root)


class TestRoundTrip:
    def test_everything_but_numeric_precision_survives(
        self, tmp_path: Path, clean_dataset: SyntheticDataset
    ) -> None:
        """MEDS stores numeric_value as float32, so that one column narrows.

        Asserted explicitly rather than hidden behind a tolerance, because the
        narrowing is a property of the standard and anyone comparing two hashes
        across a round trip needs to know it is expected.
        """
        root = tmp_path / "meds"
        write_meds_dataset(clean_dataset.events, root)
        back = read_meds_events(root)

        original = clean_dataset.events
        key = ["subject_id", "time", "code"]
        for column in ("subject_id", "time", "code", "code_system", "text_value", "encounter_id"):
            assert (
                original.sort(key, nulls_last=True)
                .select(column)
                .equals(back.sort(key, nulls_last=True).select(column))
            ), column

        narrowed = original.with_columns(pl.col("numeric_value").cast(pl.Float32).cast(pl.Float64))
        assert events_hash(narrowed) == events_hash(back)


class TestMetadata:
    def test_dataset_metadata_is_read_back(self, meds_dir: Path) -> None:
        metadata = read_dataset_metadata(meds_dir)
        assert metadata["meds_version"]

    def test_an_absent_manifest_is_not_an_error(self, tmp_path: Path) -> None:
        """A hand-made extract often has none, and the audit does not need it."""
        assert read_dataset_metadata(tmp_path) == {}

    def test_a_malformed_manifest_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "metadata").mkdir()
        (tmp_path / DATASET_METADATA_FILEPATH).write_text("{not json")
        with pytest.raises(InputError, match="malformed"):
            read_dataset_metadata(tmp_path)

    def test_splits_are_discovered_in_the_dataset(self, meds_dir: Path) -> None:
        assert find_meds_splits(meds_dir) is not None

    def test_no_splits_file_returns_none(self, tmp_path: Path) -> None:
        assert find_meds_splits(tmp_path) is None


def test_package_validation_runs_and_returns_strings(meds_dir: Path) -> None:
    events = read_meds_events(meds_dir)
    assert all(isinstance(w, str) for w in validate_with_meds_package(events))
