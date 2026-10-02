"""Feature matrices and label tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ehrlint.exceptions import InputError, SchemaError
from ehrlint.io.features import build_feature_matrix, read_feature_matrix, read_labels


def matrix_frame(subject: str = "subject_id", time: str | None = "prediction_time") -> pl.DataFrame:
    data: dict[str, object] = {
        subject: ["S000001", "S000002"],
        "age": [54, 37],
        "n_prior_visits": [3, 1],
        "label": [True, False],
    }
    if time is not None:
        data[time] = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
    return pl.DataFrame(data)


class TestColumnResolution:
    @pytest.mark.parametrize("name", ["subject_id", "patient_id", "person_id", "SUBJECT_ID"])
    def test_finds_the_subject_column_under_its_common_names(self, name: str) -> None:
        resolved = build_feature_matrix(matrix_frame(subject=name))
        assert resolved.subject_column == name
        assert resolved.subjects() == {"S000001", "S000002"}

    @pytest.mark.parametrize("name", ["prediction_time", "index_date", "as_of"])
    def test_finds_the_time_column_under_its_common_names(self, name: str) -> None:
        assert build_feature_matrix(matrix_frame(time=name)).time_column == name

    def test_no_subject_column_lists_what_it_looked_for(self) -> None:
        """Guessing a column would risk auditing the wrong one."""
        with pytest.raises(SchemaError, match="Looked for"):
            build_feature_matrix(pl.DataFrame({"age": [1], "label": [True]}))

    def test_an_explicit_override_wins(self) -> None:
        df = pl.DataFrame({"mrn_surrogate": ["a"], "age": [1]})
        assert build_feature_matrix(df, subject_column="mrn_surrogate").subject_column == (
            "mrn_surrogate"
        )

    def test_an_override_naming_an_absent_column_is_an_error(self) -> None:
        with pytest.raises(SchemaError, match="nope"):
            build_feature_matrix(matrix_frame(), subject_column="nope")

    def test_no_time_column_is_allowed(self) -> None:
        """A matrix with no as-of column simply makes LK001 unable to use it."""
        assert build_feature_matrix(matrix_frame(time=None)).time_column is None


class TestNormalization:
    def test_subject_ids_become_strings(self) -> None:
        df = pl.DataFrame({"subject_id": [1, 2], "age": [1, 2]})
        assert build_feature_matrix(df).df.schema["subject_id"] == pl.String

    def test_identifier_and_label_columns_are_not_features(self) -> None:
        resolved = build_feature_matrix(matrix_frame())
        assert set(resolved.feature_columns) == {"age", "n_prior_visits"}

    def test_the_label_column_is_located(self) -> None:
        assert build_feature_matrix(matrix_frame()).label_column == "label"

    def test_numeric_features_are_identified(self) -> None:
        resolved = build_feature_matrix(matrix_frame())
        assert set(resolved.numeric_features()) == {"age", "n_prior_visits"}

    def test_describe_reports_the_resolved_columns(self) -> None:
        described = build_feature_matrix(matrix_frame()).describe()
        assert described["n_features"] == 2
        assert described["n_rows"] == 2


class TestReadFeatureMatrix:
    def test_reads_parquet(self, tmp_path: Path) -> None:
        path = tmp_path / "f.parquet"
        matrix_frame().write_parquet(path)
        assert len(read_feature_matrix(path)) == 2

    def test_reads_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "f.csv"
        matrix_frame().write_csv(path)
        assert len(read_feature_matrix(path)) == 2

    def test_a_missing_file_is_an_input_error(self, tmp_path: Path) -> None:
        with pytest.raises(InputError, match="not found"):
            read_feature_matrix(tmp_path / "absent.parquet")

    def test_an_unsupported_suffix_names_the_supported_ones(self, tmp_path: Path) -> None:
        path = tmp_path / "f.xlsx"
        path.write_bytes(b"not a spreadsheet")
        with pytest.raises(InputError, match="parquet"):
            read_feature_matrix(path)

    def test_an_unreadable_file_is_an_input_error(self, tmp_path: Path) -> None:
        path = tmp_path / "f.parquet"
        path.write_bytes(b"this is not parquet")
        with pytest.raises(InputError, match="could not read"):
            read_feature_matrix(path)


class TestReadLabels:
    def test_reads_ehrlints_own_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.parquet"
        pl.DataFrame(
            {
                "subject_id": ["S1", "S2"],
                "index_date": [datetime(2020, 1, 1)] * 2,
                "label": [True, False],
            }
        ).write_parquet(path)
        labels = read_labels(path)
        assert labels.height == 2
        assert "label" in labels.columns

    def test_normalizes_the_meds_typed_value_columns(self, tmp_path: Path) -> None:
        """MEDS LabelSchema spreads the label across typed columns."""
        path = tmp_path / "labels.parquet"
        pl.DataFrame(
            {
                "subject_id": [1, 2],
                "prediction_time": [datetime(2020, 1, 1)] * 2,
                "boolean_value": [True, False],
                "integer_value": [None, None],
                "float_value": [None, None],
            }
        ).write_parquet(path)
        labels = read_labels(path)
        assert labels["label"].to_list() == [True, False]
        assert labels.schema["subject_id"] == pl.String

    def test_a_missing_file_is_an_input_error(self, tmp_path: Path) -> None:
        with pytest.raises(InputError):
            read_labels(tmp_path / "absent.parquet")
