"""Splits: aliases, overlap detection, and the id-mismatch warning."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ehrlint.exceptions import SplitsError
from ehrlint.schemas.splits import SPLIT_ALIASES, Splits, canonical_split_name


class TestCanonicalNames:
    @pytest.mark.parametrize(("raw", "expected"), sorted(SPLIT_ALIASES.items()))
    def test_meds_names_map_to_ehrlints(self, raw: str, expected: str) -> None:
        """MEDS calls them train/tuning/held_out; ehrlint uses train/valid/test."""
        assert canonical_split_name(raw) == expected

    def test_is_case_and_space_insensitive(self) -> None:
        assert canonical_split_name(" Held_Out ") == "test"

    def test_an_unknown_name_is_kept_rather_than_guessed(self) -> None:
        assert canonical_split_name("fold_3") == "fold_3"


class TestFromMapping:
    def test_builds_from_a_plain_mapping(self) -> None:
        splits = Splits.from_mapping({"train": ["a", "b"], "test": ["c"]})
        assert splits.sizes() == {"train": 2, "test": 1}

    def test_coerces_ids_to_strings(self) -> None:
        splits = Splits.from_mapping({"train": [1, 2], "test": [3]})
        assert splits.get("train") == ["1", "2"]

    def test_a_non_list_value_names_the_split(self) -> None:
        with pytest.raises(SplitsError, match="train"):
            Splits.from_mapping({"train": "a"})

    def test_an_empty_mapping_is_an_error(self) -> None:
        with pytest.raises(SplitsError, match="no splits"):
            Splits.from_mapping({})

    def test_two_aliases_of_one_split_merge(self) -> None:
        splits = Splits.from_mapping({"held_out": ["a"], "test": ["b"]})
        assert sorted(splits.get("test")) == ["a", "b"]


class TestOverlaps:
    def test_disjoint_splits_report_no_overlap(self) -> None:
        splits = Splits.from_mapping({"train": ["a"], "test": ["b"]})
        assert splits.overlaps() == {}

    def test_a_shared_subject_is_reported_with_both_names(self) -> None:
        splits = Splits.from_mapping({"train": ["a", "b"], "test": ["b"]})
        assert splits.overlaps() == {("train", "test"): ["b"]}

    def test_the_pair_key_is_ordered_so_the_result_is_deterministic(self) -> None:
        one = Splits.from_mapping({"train": ["x"], "test": ["x"]})
        two = Splits.from_mapping({"test": ["x"], "train": ["x"]})
        assert list(one.overlaps()) == list(two.overlaps())


class TestSplitOf:
    def test_maps_a_subject_to_every_split_holding_it(self) -> None:
        splits = Splits.from_mapping({"train": ["a", "b"], "test": ["b"]})
        assert sorted(splits.split_of()["b"]) == ["test", "train"]


class TestWarnings:
    def test_one_split_names_the_checks_it_disables(self) -> None:
        warnings = Splits.from_mapping({"train": ["a"]}).warnings()
        assert any("LK005" in w and "LK006" in w for w in warnings)

    def test_ids_absent_from_the_events_are_called_a_format_mismatch(self) -> None:
        """The likeliest cause, and the one that silently empties every join."""
        splits = Splits.from_mapping({"train": ["1"], "test": ["2"]})
        warnings = splits.warnings(known_subjects={"S000001", "S000002"})
        assert any("format mismatch" in w for w in warnings)

    def test_events_in_no_split_are_reported(self) -> None:
        splits = Splits.from_mapping({"train": ["a"], "test": ["b"]})
        warnings = splits.warnings(known_subjects={"a", "b", "c"})
        assert any("in no split" in w for w in warnings)

    def test_a_matching_set_warns_about_nothing(self) -> None:
        splits = Splits.from_mapping({"train": ["a"], "test": ["b"]})
        assert splits.warnings(known_subjects={"a", "b"}) == []


class TestIo:
    def test_round_trips_through_json(self, tmp_path: Path) -> None:
        splits = Splits.from_mapping({"train": ["a", "b"], "test": ["c"]})
        path = splits.write_json(tmp_path / "splits.json")
        assert Splits.from_json(path).sizes() == splits.sizes()

    def test_a_missing_json_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(SplitsError):
            Splits.from_json(tmp_path / "absent.json")

    def test_malformed_json_is_a_splits_error(self, tmp_path: Path) -> None:
        path = tmp_path / "splits.json"
        path.write_text("{not json")
        with pytest.raises(SplitsError):
            Splits.from_json(path)

    def test_from_meds_applies_the_name_aliases(self, tmp_path: Path) -> None:
        path = tmp_path / "subject_splits.parquet"
        pl.DataFrame(
            {"subject_id": [1, 2, 3], "split": ["train", "tuning", "held_out"]}
        ).write_parquet(path)
        assert Splits.from_meds(path).names == ["train", "valid", "test"]

    def test_from_meds_applies_a_surrogate_id_map(self, tmp_path: Path) -> None:
        """Splits must name the same subjects the event table does."""
        path = tmp_path / "subject_splits.parquet"
        pl.DataFrame({"subject_id": [1, 2], "split": ["train", "held_out"]}).write_parquet(path)
        splits = Splits.from_meds(path, id_map={"1": "S000001", "2": "S000002"})
        assert splits.get("train") == ["S000001"]
        assert splits.get("test") == ["S000002"]

    def test_a_partial_id_map_is_an_error_not_a_silent_mix(self, tmp_path: Path) -> None:
        path = tmp_path / "subject_splits.parquet"
        pl.DataFrame({"subject_id": [1, 2], "split": ["train", "held_out"]}).write_parquet(path)
        with pytest.raises(SplitsError, match=r"vacuous|not in the dataset"):
            Splits.from_meds(path, id_map={"1": "S000001"})

    def test_from_meds_names_a_missing_column(self, tmp_path: Path) -> None:
        path = tmp_path / "subject_splits.parquet"
        pl.DataFrame({"subject_id": [1]}).write_parquet(path)
        with pytest.raises(SplitsError, match="split"):
            Splits.from_meds(path)

    def test_from_frame_reads_a_two_column_frame(self) -> None:
        df = pl.DataFrame({"subject_id": ["a", "b"], "split": ["train", "test"]})
        assert Splits.from_frame(df).sizes() == {"train": 1, "test": 1}

    def test_json_written_is_sorted_so_two_runs_match(self, tmp_path: Path) -> None:
        splits = Splits.from_mapping({"train": ["b", "a"], "test": ["c"]})
        path = splits.write_json(tmp_path / "s.json")
        assert json.loads(path.read_text())["train"] == ["a", "b"]


def test_as_frame_gives_one_row_per_subject_per_split() -> None:
    splits = Splits.from_mapping({"train": ["a", "b"], "test": ["b"]})
    frame = splits.as_frame()
    assert frame.height == 3
    assert set(frame.columns) == {"subject_id", "split"}
